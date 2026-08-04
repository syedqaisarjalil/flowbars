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
from flowbars.calendars import ContinuousCalendar, SessionCalendar, TradingCalendar
from flowbars.core import Bar, StateValidationError, TickInfo
from flowbars.schema import SchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator, ThresholdEstimator
from flowbars.tick_rule import resolve_tick_signs


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
    stream_id : str, default ``"default"``
        Opaque identifier validated on state resume.
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
        stream_id: str = "default",
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
        self._stream_id = stream_id
        self._warmup_bars = warmup_bars
        self._backend = backend
        self._on_bar = on_bar
        self._on_threshold_update = on_threshold_update

        # Internal state
        self._bars_emitted: int = 0
        self._in_session: bool = True  # True until first boundary tick lands
        self._pending_bars: collections.deque[Bar] = collections.deque()

    # ── core API ────────────────────────────────────────────────────────

    def update(self, tick: TickInfo) -> Bar | None:
        """Feed one tick.  Return a completed bar, or ``None``.

        Session-boundary force-closes and threshold-driven closes are both
        handled.  During warmup bars are queued for the estimator but not
        returned.
        """
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

        # ── numba path ──────────────────────────────────────────────────
        if self._backend == "numba":
            return self._batch_numba(timestamps, prices, volumes, sides)

        # ── Python path (unchanged) ─────────────────────────────────────
        bars: list[Bar] = []
        n = len(timestamps)
        for i in range(n):
            raw_side = sides[i]
            tick = TickInfo(
                timestamp=int(timestamps[i]),
                price=float(prices[i]),
                volume=float(volumes[i]),
                side=float(raw_side) if not np.isnan(raw_side) else None,
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
    ) -> pd.DataFrame:
        """Run the batch bar-construction loop via numba (or fall back to Python).

        Handles static and EWMA thresholds, time-bar special casing, and
        warmup-bar slicing.  Falls back to the Python path when:

        * numba is not installed
        * a :class:`SessionCalendar` is in use (boundaries are
          infrequent callbacks — not worth duplicating in numba)
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
            return self._batch_python(timestamps, prices, volumes, sides)

        if isinstance(self._calendar, SessionCalendar):
            warnings.warn(
                "SessionCalendar is not supported by the numba backend. "
                "Falling back to the Python path."
            )
            return self._batch_python(timestamps, prices, volumes, sides)

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
                    "Falling back to the Python path."
                )
                return self._batch_python(timestamps, prices, volumes, sides)
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
                    "Falling back to the Python path."
                )
                return self._batch_python(timestamps, prices, volumes, sides)
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
                    "Falling back to the Python path."
                )
                return self._batch_python(timestamps, prices, volumes, sides)

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
                    "bar_id", "open", "high", "low", "close", "volume",
                    "dollar_value", "vwap", "num_ticks", "open_ts", "close_ts",
                    "bar_type",
                ]
            )

        cols = _bar_data_to_columns(bar_data)
        cols["bar_type"] = bt
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
                    "bar_id", "open", "high", "low", "close", "volume",
                    "dollar_value", "vwap", "num_ticks", "open_ts", "close_ts",
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

        # Apply state
        self._accumulator.load_state(state["accumulator"])
        self._threshold_estimator.load_state(state["threshold_estimator"])
        self._bars_emitted = state.get("bars_emitted", 0)
        self._in_session = state.get("in_session", True)
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
            stream_id=stream_id,
            warmup_bars=warmup_bars,
            backend=backend,  # type: ignore[call-arg]
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
