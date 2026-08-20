"""Base bar constructor — shared infrastructure for all bar types.

The ``BaseBarConstructor`` orchestrates an accumulator, threshold estimator,
and trading calendar to produce bars from a streaming tick feed.  It is
abstracted over the specific bar type — subclasses in ``standard/`` and
``information/`` provide the concrete accumulator + estimator wiring.
"""

from __future__ import annotations

import collections
import warnings
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from flowbars.bars.accumulators import BaseAccumulator
from flowbars.calendars import ContinuousCalendar, TradingCalendar
from flowbars.core import Bar, SchemaError, StateValidationError, TickDataError, TickInfo
from flowbars.schema import SchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator, ThresholdEstimator
from flowbars.tick_rule import resolve_tick_signs


def _tick_watermark(tick: TickInfo) -> int:
    """Return the dedup watermark for *tick*.

    Uses ``TickInfo.watermark`` when set (a custom monotonic key such as a
    sequence number), otherwise falls back to ``timestamp``.
    """
    return tick.watermark if tick.watermark is not None else tick.timestamp


class BaseBarConstructor:
    """Shared bar-construction engine for all bar types.

    Wires together an accumulator (Phase 2), a threshold estimator (Phase 3),
    and a trading calendar (Phase 4).  Each bar type is a subclass that
    picks the right components.

    Parameters
    ----------
    accumulator : BaseAccumulator
        The bar-type-specific accumulator (tick, volume, imbalance, …).
    threshold_estimator : ThresholdEstimator
        Fixed (:class:`StaticThresholdEstimator`) for standard bars,
        adaptive (:class:`EWMAThresholdEstimator`) for information-driven bars.
    calendar : TradingCalendar, optional
        Defaults to :class:`ContinuousCalendar` (always-open).
    schema : SchemaMapping, optional
        Required for :meth:`batch`; not used by :meth:`update`.
    watermark : str or None, default ``"timestamp"``
        Dedup key for idempotent resume.  ``"timestamp"`` (default) dedups
        on the tick timestamp; any other string is a column name (batch) or
        ``TickInfo.watermark`` field (streaming) to dedup on.  ``None``
        disables dedup.
    stream_id : str, default ``"default"``
        Opaque identifier validated on state resume.
    strict_ordering : bool, default False
        When True, raise :class:`TickDataError` if a tick arrives with a
        timestamp earlier than the previous tick's (out-of-order input).
    warmup_bars : int, default 0
        Number of initial bars to discard (not returned from
        :meth:`update` / :meth:`batch`).  Bars during warmup still
        trigger callbacks and update the estimator.
    backend : str, default ``"python"``
        ``"python"`` (default, always available) or ``"numba"`` (optional,
        requires ``pip install flowbars[numba]``).  The numba backend
        accelerates :meth:`batch` via a JIT-compiled inner loop.
        :meth:`update` (streaming, single-tick) always uses the Python
        backend regardless of this setting.
    on_bar : callable or None
        Called as ``on_bar(bar)`` after every bar closes (including
        warmup bars and session-boundary force-closes).
    on_threshold_update : callable or None
        Called as ``on_threshold_update(new_value)`` when the threshold
        estimator produces a new value after a bar close.

    Thread safety
    -------------
    Not thread-safe.  Designed for single-threaded streaming or batch use.
    """

    def __init__(
        self,
        accumulator: BaseAccumulator,
        threshold_estimator: ThresholdEstimator,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        warmup_bars: int = 0,
        backend: Literal["python", "numba"] = "python",
        on_bar: Callable[[Bar], None] | None = None,
        on_threshold_update: Callable[[float], None] | None = None,
    ) -> None:
        if warmup_bars < 0:
            raise ValueError(f"warmup_bars must be non-negative, got {warmup_bars}")
        if backend not in ("python", "numba"):
            raise ValueError(f"backend must be 'python' or 'numba', got {backend!r}")

        self._accumulator = accumulator
        self._threshold_estimator = threshold_estimator
        self._calendar = calendar if calendar is not None else ContinuousCalendar()
        self._schema = schema
        self._watermark_key = watermark
        self._stream_id = stream_id
        self._strict_ordering = strict_ordering
        self._warmup_bars = warmup_bars
        self._backend = backend
        self._on_bar = on_bar
        self._on_threshold_update = on_threshold_update

        # Internal state
        self._bars_emitted: int = 0
        self._in_session: bool = True  # True until first boundary tick lands
        self._pending_bars: collections.deque[Bar] = collections.deque()
        self._last_watermark: int | None = None  # dedup watermark of last accepted tick
        self._last_timestamp: int | None = None  # timestamp of last accepted tick (ordering)

    # ── core API ────────────────────────────────────────────────────────

    def update(self, tick: TickInfo) -> Bar | None:
        """Feed one tick.  Return a completed bar, or ``None``.

        Session-boundary force-closes and threshold-driven closes are both
        handled.  During warmup bars are queued for the estimator but not
        returned.
        """
        # 0. Watermark dedup — drop ticks already processed (idempotent resume)
        wm = _tick_watermark(tick)
        if (
            self._watermark_key is not None
            and self._last_watermark is not None
            and wm <= self._last_watermark
        ):
            return None

        # 0b. Ordering check — reject non-monotonic timestamps when enabled
        if (
            self._strict_ordering
            and self._last_timestamp is not None
            and tick.timestamp < self._last_timestamp
        ):
            raise TickDataError(
                f"Out-of-order tick: timestamp {tick.timestamp} < previous {self._last_timestamp}"
            )

        # 1. Drain pending queue first (bar from previous tick)
        if self._pending_bars:
            return self._pending_bars.popleft()

        # 2. Session boundary detection
        is_boundary = self._calendar.is_session_boundary(tick.timestamp)
        if is_boundary:
            if self._accumulator._has_tick and self._in_session:
                # Transition session → boundary: force-close current bar
                self._emit_bar()
            self._in_session = False
        else:
            if self._accumulator._has_tick and not self._in_session:
                # Transition boundary → session: force-close after-hours bar
                self._emit_bar()
            self._in_session = True

        # 3. Add tick to accumulator
        self._accumulator.add_tick(tick)
        self._last_watermark = wm
        self._last_timestamp = tick.timestamp

        # 4. Update threshold estimator (no-op for static, also no-op for EWMA)
        self._threshold_estimator.update(tick)

        # 5. Threshold-based closure
        threshold = self._threshold_estimator.current_threshold
        if self._accumulator.should_close(threshold):
            t_stat, proportion_stat = self._accumulator.get_close_stats()
            bar = self._accumulator.close(threshold)

            old_threshold = threshold
            self._threshold_estimator.on_bar_close(t_stat, proportion_stat)
            new_threshold = self._threshold_estimator.current_threshold

            self._bars_emitted += 1

            if self._on_bar is not None:
                self._on_bar(bar)

            if self._on_threshold_update is not None and old_threshold != new_threshold:
                self._on_threshold_update(new_threshold)

            if self._bars_emitted > self._warmup_bars and self._should_return(bar):
                self._pending_bars.append(bar)

        # 6. Return one bar from queue, or None
        if self._pending_bars:
            return self._pending_bars.popleft()
        return None

    def batch(self, ticks_df: pd.DataFrame) -> pd.DataFrame:
        """Feed all ticks from a DataFrame, return completed bars.

        Parameters
        ----------
        ticks_df : pd.DataFrame
            Ticks with user-mapped column names (see :class:`SchemaMapping`).

        Returns
        -------
        pd.DataFrame
            Bars as a DataFrame with columns: ``bar_id``, ``open``, ``high``,
            ``low``, ``close``, ``volume``, ``dollar_value``, ``vwap``,
            ``num_ticks``, ``open_ts``, ``close_ts``, ``bar_type``.

        Raises
        ------
        ValueError
            If no :class:`SchemaMapping` was provided at construction time.
        """
        if self._schema is None:
            # Auto-create a default schema so convenience batch functions
            # work without explicit SchemaMapping when columns are standard
            self._schema = SchemaMapping(
                {"timestamp": "timestamp", "price": "price", "volume": "volume"}
            )

        timestamps, prices, volumes, sides = self._schema.extract_arrays(ticks_df)

        # Derive signs if not supplied
        sides = resolve_tick_signs(prices, sides)

        # ── watermark pre-filter (idempotent resume, backend-agnostic) ──
        wm_arr = self._resolve_watermark_array(ticks_df, timestamps)
        if wm_arr is not None and self._last_watermark is not None:
            keep = wm_arr > self._last_watermark
            timestamps = timestamps[keep]
            prices = prices[keep]
            volumes = volumes[keep]
            if sides is not None:
                sides = sides[keep]
            wm_arr = wm_arr[keep]

        # ── strict ordering check (fail fast on the whole batch) ────────
        if self._strict_ordering and len(timestamps) > 0:
            if self._last_timestamp is not None and int(timestamps[0]) < self._last_timestamp:
                raise TickDataError(
                    f"Out-of-order tick: timestamp {int(timestamps[0])} < "
                    f"previous {self._last_timestamp}"
                )
            diffs = np.diff(timestamps)
            if diffs.size and np.any(diffs < 0):
                idx = int(np.argmax(diffs < 0)) + 1
                raise TickDataError(
                    f"Out-of-order tick at batch index {idx}: "
                    f"timestamp {int(timestamps[idx])} < "
                    f"previous {int(timestamps[idx - 1])}"
                )

        # ── numba path ──────────────────────────────────────────────────
        if self._backend == "numba":
            result = self._batch_numba(timestamps, prices, volumes, sides, wm_arr)
            # numba bypasses update(); advance the watermark explicitly
            if wm_arr is not None and len(wm_arr) > 0:
                self._last_watermark = int(wm_arr[-1])
            if len(timestamps) > 0:
                self._last_timestamp = int(timestamps[-1])
            return result

        # ── Python path ─────────────────────────────────────────────────
        bars: list[Bar] = []
        n = len(timestamps)
        for i in range(n):
            raw_side = sides[i]
            tick = TickInfo(
                timestamp=int(timestamps[i]),
                price=float(prices[i]),
                volume=float(volumes[i]),
                side=float(raw_side) if not np.isnan(raw_side) else None,
                watermark=int(wm_arr[i]) if wm_arr is not None else None,
            )
            bar = self.update(tick)
            if bar is not None:
                bars.append(bar)

        # Drain remaining pending bars
        while self._pending_bars:
            bars.append(self._pending_bars.popleft())

        # Convert to DataFrame
        if not bars:
            return pd.DataFrame(
                columns=[
                    "bar_id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "dollar_value",
                    "vwap",
                    "num_ticks",
                    "open_ts",
                    "close_ts",
                    "bar_type",
                ]
            )

        return pd.DataFrame(
            [
                {
                    "bar_id": b.bar_id,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "dollar_value": b.dollar_value,
                    "vwap": b.vwap,
                    "num_ticks": b.num_ticks,
                    "open_ts": b.open_ts,
                    "close_ts": b.close_ts,
                    "bar_type": b.bar_type,
                }
                for b in bars
            ]
        )

    # ── numba batch path ──────────────────────────────────────────────────

    def _batch_numba(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
        sides: np.ndarray,
        watermark: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Run the batch bar-construction loop via numba (or fall back to Python).

        Handles static and EWMA thresholds, time-bar special casing, and
        warmup-bar slicing.  Falls back to the Python path when:

        * numba is not installed
        * a session-producing calendar is in use (``has_sessions=True`` —
          boundaries are infrequent callbacks, not worth duplicating in numba)
        * an unsupported calendar type is detected
        """
        from flowbars.bars.numba_backend import (
            _NUMBA_AVAILABLE,
            _bar_data_to_columns,
            _get_compilation_warning,
            numba_batch_ewma,
            numba_batch_static,
        )

        bar_type = self._accumulator._bar_type

        # ── fallback checks ────────────────────────────────────────────
        if not _NUMBA_AVAILABLE:
            _get_compilation_warning()
            return self._batch_python(timestamps, prices, volumes, sides, watermark)

        if self._calendar.has_sessions:
            warnings.warn(
                "This calendar produces session boundaries, which the numba "
                "backend does not support. Falling back to the Python path.",
                stacklevel=2,
            )
            return self._batch_python(timestamps, prices, volumes, sides, watermark)

        # ── determine threshold config ─────────────────────────────────
        estimator = self._threshold_estimator

        if bar_type == "time":
            # Time bars: extract interval_ms and anchor from the accumulator.
            # TimeAccumulator stores these — drop into the instance to read them.
            acc = self._accumulator
            interval_ms: int = getattr(acc, "_interval_ms", 60000)
            anchor: str = getattr(acc, "_anchor", "clock")
            try:
                bar_data, bt = numba_batch_static(
                    bar_type="time",
                    timestamps=timestamps,
                    prices=prices,
                    volumes=volumes,
                    sides=None,
                    threshold=0.0,  # unused by time bars
                    interval_ms=interval_ms,
                    anchor=anchor,
                )
            except Exception:
                warnings.warn(
                    "numba compilation/execution failed for time bars. "
                    "Falling back to the Python path.",
                    stacklevel=2,
                )
                return self._batch_python(timestamps, prices, volumes, sides, watermark)
        elif isinstance(estimator, EWMAThresholdEstimator):
            # Adaptive threshold — use the EWMA-aware numba path
            try:
                bar_data, bt = numba_batch_ewma(
                    bar_type=bar_type,
                    timestamps=timestamps,
                    prices=prices,
                    volumes=volumes,
                    sides=sides,
                    alpha=estimator.alpha,
                    initial_ewa_t=estimator.ewa_t,
                    initial_ewa_proportion=estimator.ewa_proportion,
                )
            except Exception:
                warnings.warn(
                    f"numba compilation/execution failed for {bar_type}. "
                    "Falling back to the Python path.",
                    stacklevel=2,
                )
                return self._batch_python(timestamps, prices, volumes, sides, watermark)
        else:
            # Static threshold (standard bars or info-driven bars with fixed threshold)
            threshold = estimator.current_threshold
            try:
                bar_data, bt = numba_batch_static(
                    bar_type=bar_type,
                    timestamps=timestamps,
                    prices=prices,
                    volumes=volumes,
                    sides=sides if bar_type.startswith(("imbalance_", "run_")) else None,
                    threshold=threshold,
                )
            except Exception:
                warnings.warn(
                    f"numba compilation/execution failed for {bar_type}. "
                    "Falling back to the Python path.",
                    stacklevel=2,
                )
                return self._batch_python(timestamps, prices, volumes, sides, watermark)

        # numba kernel bypasses update() — account for bars emitted (including
        # warmup bars, matching the Python path which counts before the warmup
        # filter). bar_data still contains warmup bars at this point.
        self._bars_emitted += len(bar_data)

        # ── warmup slicing ─────────────────────────────────────────────
        warmup = self._warmup_bars
        if warmup > 0 and len(bar_data) > 0:
            bar_data = bar_data[warmup:]
            # Re-number bar_ids after warmup
            if len(bar_data) > 0:
                bar_data[:, 0] = np.arange(len(bar_data), dtype=np.float64)

        # ── convert to DataFrame ───────────────────────────────────────
        if len(bar_data) == 0:
            return pd.DataFrame(
                columns=[
                    "bar_id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "dollar_value",
                    "vwap",
                    "num_ticks",
                    "open_ts",
                    "close_ts",
                    "bar_type",
                ]
            )

        cols = _bar_data_to_columns(bar_data)
        cols["bar_type"] = bt  # type: ignore[assignment]
        result = pd.DataFrame(cols)

        # ── fire callbacks ─────────────────────────────────────────────
        if self._on_bar is not None:
            for _, row in result.iterrows():
                bar = Bar(
                    bar_id=int(row["bar_id"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    dollar_value=float(row["dollar_value"]),
                    vwap=float(row["vwap"]),
                    num_ticks=int(row["num_ticks"]),
                    open_ts=int(row["open_ts"]),
                    close_ts=int(row["close_ts"]),
                    bar_type=bt,
                )
                self._on_bar(bar)

        return result

    def _batch_python(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
        sides: np.ndarray,
        watermark: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Pure-Python batch loop — used as the fallback from :meth:`_batch_numba`."""
        bars: list[Bar] = []
        n = len(timestamps)
        for i in range(n):
            raw_side = sides[i]
            tick = TickInfo(
                timestamp=int(timestamps[i]),
                price=float(prices[i]),
                volume=float(volumes[i]),
                side=float(raw_side) if not np.isnan(raw_side) else None,
                watermark=int(watermark[i]) if watermark is not None else None,
            )
            bar_result = self.update(tick)
            if bar_result is not None:
                bars.append(bar_result)

        # Drain remaining pending bars
        while self._pending_bars:
            bars.append(self._pending_bars.popleft())

        if not bars:
            return pd.DataFrame(
                columns=[
                    "bar_id",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "dollar_value",
                    "vwap",
                    "num_ticks",
                    "open_ts",
                    "close_ts",
                    "bar_type",
                ]
            )

        return pd.DataFrame(
            [
                {
                    "bar_id": b.bar_id,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "dollar_value": b.dollar_value,
                    "vwap": b.vwap,
                    "num_ticks": b.num_ticks,
                    "open_ts": b.open_ts,
                    "close_ts": b.close_ts,
                    "bar_type": b.bar_type,
                }
                for b in bars
            ]
        )

    def _resolve_watermark_array(
        self, ticks_df: pd.DataFrame, timestamps: np.ndarray
    ) -> np.ndarray | None:
        """Resolve the watermark values for a batch, or ``None`` if disabled.

        ``"timestamp"`` reuses the already-extracted timestamp array; any
        other key is looked up as a literal column name in ``ticks_df``.
        """
        if self._watermark_key is None:
            return None
        if self._watermark_key == "timestamp":
            return timestamps
        col = self._watermark_key
        if col not in ticks_df.columns:
            raise SchemaError(
                f"Watermark column {col!r} not found in input. "
                f"Available columns: {sorted(ticks_df.columns)}"
            )
        return np.asarray(ticks_df[col].to_numpy())

    # ── read-only accessors ─────────────────────────────────────────────

    @property
    def current_bar(self) -> Bar | None:
        """The in-progress bar, or ``None`` if no ticks have been added."""
        return self._accumulator.current_bar

    @property
    def bars_emitted(self) -> int:
        """Total bars emitted so far (including warmup bars)."""
        return self._bars_emitted

    # ── state persistence ───────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the constructor's current state.

        Includes the accumulator state, threshold-estimator state, and
        constructor-level bookkeeping.
        """
        return {
            "schema_version": 1,
            "stream_id": self._stream_id,
            "bar_type": self._accumulator._bar_type,
            "backend": self._backend,
            "watermark_key": self._watermark_key,
            "last_watermark": self._last_watermark,
            "last_timestamp": self._last_timestamp,
            "accumulator": self._accumulator.get_state(),
            "threshold_estimator": self._threshold_estimator.get_state(),
            "bars_emitted": self._bars_emitted,
            "in_session": self._in_session,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore constructor state from a saved state dict.

        Validates stream identity and schema version before applying state.

        Raises
        ------
        StateValidationError
            If *stream_id*, *bar_type*, or *schema_version* is incompatible.
        """
        # Version check
        version = state.get("schema_version", 0)
        if version < 1:
            raise StateValidationError(
                f"State schema version {version} is not supported. Minimum required version is 1."
            )
        if version > 1:
            raise StateValidationError(
                f"State schema version {version} was written by a newer version "
                f"of flowbars. Upgrade the library to load this state."
            )

        # Identity validation
        if state["stream_id"] != self._stream_id:
            raise StateValidationError(
                f"stream_id mismatch: state has {state['stream_id']!r}, "
                f"constructor has {self._stream_id!r}"
            )
        if state["bar_type"] != self._accumulator._bar_type:
            raise StateValidationError(
                f"bar_type mismatch: state has {state['bar_type']!r}, "
                f"accumulator has {self._accumulator._bar_type!r}"
            )
        if state["watermark_key"] != self._watermark_key:
            raise StateValidationError(
                f"watermark_key mismatch: state has {state['watermark_key']!r}, "
                f"constructor has {self._watermark_key!r}"
            )

        # Apply state
        self._accumulator.load_state(state["accumulator"])
        self._threshold_estimator.load_state(state["threshold_estimator"])
        self._bars_emitted = state.get("bars_emitted", 0)
        self._in_session = state.get("in_session", True)
        self._last_watermark = state["last_watermark"]
        self._last_timestamp = state.get("last_timestamp")
        # Restore backend if present (new in state schema v1, added in Phase 9)
        saved_backend = state.get("backend", "python")
        if saved_backend in ("python", "numba"):
            self._backend = saved_backend

        # Clear any pending bars (they should have been drained before saving)
        self._pending_bars.clear()

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        accumulator: BaseAccumulator,
        threshold_estimator: ThresholdEstimator,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
    ) -> BaseBarConstructor:
        """Create a new constructor instance from a saved state dict.

        The caller is responsible for providing the correct accumulator
        and threshold-estimator types (or re-creating them from the
        bar-type information).  Subclasses can wrap this to auto-create
        the right components.

        Parameters
        ----------
        state : dict
            State previously returned by :meth:`get_state`.
        accumulator : BaseAccumulator
            Fresh instance of the correct accumulator type.
        threshold_estimator : ThresholdEstimator
            Fresh instance of the correct estimator type.
        calendar : TradingCalendar, optional
        schema : SchemaMapping, optional

        Returns
        -------
        BaseBarConstructor
            A new constructor with its state restored from *state*.
        """
        stream_id = state["stream_id"]
        warmup_bars = 0  # warmup is user-specified, not persisted
        backend = state.get("backend", "python")
        if backend not in ("python", "numba"):
            backend = "python"
        inst = cls(
            accumulator=accumulator,
            threshold_estimator=threshold_estimator,
            calendar=calendar,
            schema=schema,
            watermark=state["watermark_key"],
            stream_id=stream_id,
            warmup_bars=warmup_bars,
            backend=backend,
        )
        inst.load_state(state)
        return inst

    # ── internal helpers ────────────────────────────────────────────────

    def _should_return(self, bar: Bar) -> bool:  # noqa: B027
        """Return ``True`` if *bar* should be returned to the caller.

        Override in subclasses to filter bars (e.g. ``min_run_length`` for
        run bars).  The default always returns ``True``.
        """
        return True

    def _emit_bar(self) -> None:
        """Force-close the current bar and queue it for return.

        Does NOT call ``on_bar_close`` on the estimator — session-boundary
        closures are artificial and should not influence the adaptive
        threshold.
        """
        threshold = self._threshold_estimator.current_threshold
        bar = self._accumulator.close(threshold)
        self._bars_emitted += 1

        if self._on_bar is not None:
            self._on_bar(bar)

        # Warmup check + subclass filter (e.g. min_run_length)
        if self._bars_emitted > self._warmup_bars and self._should_return(bar):
            self._pending_bars.append(bar)
