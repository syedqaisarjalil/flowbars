"""Base bar constructor — shared infrastructure for all bar types.

The ``BaseBarConstructor`` orchestrates an accumulator, threshold estimator,
and trading calendar to produce bars from a streaming tick feed.  It is
abstracted over the specific bar type — subclasses in ``standard/`` and
``information/`` provide the concrete accumulator + estimator wiring.
"""

from __future__ import annotations

import collections
import copy
import math
import warnings
from typing import Any, Callable, ClassVar, Literal

import numpy as np
import pandas as pd

from flowbars.bars.accumulators import BaseAccumulator
from flowbars.calendars import ContinuousCalendar, TradingCalendar
from flowbars.core import (
    Bar,
    MinuteInfo,
    SchemaError,
    StateValidationError,
    TickDataError,
    TickInfo,
)
from flowbars.schema import MinuteSchemaMapping, SchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator, ThresholdEstimator
from flowbars.tick_rule import resolve_tick_signs

_OUTPUT_COLUMNS = [
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


def _input_watermark(item: TickInfo | MinuteInfo) -> int:
    """Return the dedup watermark for *item*.

    Uses ``item.watermark`` when set (a custom monotonic key such as a
    sequence number), otherwise falls back to ``timestamp``.
    """
    return item.watermark if item.watermark is not None else item.timestamp


def _states_equal(a: Any, b: Any) -> bool:
    """Compare two state structures, treating NaN floats as equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


def _row_to_bar(row: np.ndarray, bar_type: str) -> Bar:
    """Build a :class:`Bar` from one row of a numba output array."""
    return Bar(
        bar_id=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        dollar_value=float(row[6]),
        vwap=float(row[7]),
        num_ticks=int(row[8]),
        open_ts=int(row[9]),
        close_ts=int(row[10]),
        bar_type=bar_type,
    )


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

    input_kind: ClassVar[str] = "tick"

    def __init__(
        self,
        accumulator: BaseAccumulator,
        threshold_estimator: ThresholdEstimator,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | MinuteSchemaMapping | None = None,
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

        # Snapshots used to detect a resumed/partially-consumed constructor, in
        # which case the numba batch kernels (which assume a zero start) fall
        # back to the Python path.
        self._initial_acc_state = copy.deepcopy(self._accumulator.get_state())
        self._initial_est_state = copy.deepcopy(self._threshold_estimator.get_state())

    # ── core API ────────────────────────────────────────────────────────

    def update(self, item: TickInfo | MinuteInfo) -> Bar | None:
        """Feed one tick or minute.  Return a completed bar, or ``None``.

        Session-boundary force-closes and threshold-driven closes are both
        handled.  During warmup bars are queued for the estimator but not
        returned.
        """
        # 0. Watermark dedup — drop input already processed (idempotent resume)
        wm = _input_watermark(item)
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
            and item.timestamp < self._last_timestamp
        ):
            raise TickDataError(
                f"Out-of-order tick: timestamp {item.timestamp} < previous {self._last_timestamp}"
            )

        # 1. Per-input preparation (minute path derives missing sides here, so
        #    a deduplicated input never corrupts the sign state)
        item = self._prepare_input(item)

        # 2. Session boundary detection
        is_boundary = self._calendar.is_session_boundary(item.timestamp)
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

        # 3. Add input to accumulator
        self._ingest_input(item)
        self._last_watermark = wm
        self._last_timestamp = item.timestamp

        # 4. Update threshold estimator (no-op for static, also no-op for EWMA)
        self._threshold_estimator.update(item)

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
        """Feed all input rows from a DataFrame, return completed bars.

        Parameters
        ----------
        ticks_df : pd.DataFrame
            Ticks (or minute candles) with user-mapped column names (see
            :class:`SchemaMapping` / :class:`MinuteSchemaMapping`).

        Returns
        -------
        pd.DataFrame
            Bars as a DataFrame with columns: ``bar_id``, ``open``, ``high``,
            ``low``, ``close``, ``volume``, ``dollar_value``, ``vwap``,
            ``num_ticks``, ``open_ts``, ``close_ts``, ``bar_type``.

        Raises
        ------
        ValueError
            If no schema was provided and the input columns are not standard.
        """
        if self._schema is None:
            # Auto-create a default schema so convenience batch functions
            # work without an explicit schema when columns are standard.
            self._schema = self._default_schema()

        arrays = self._extract_batch_arrays(ticks_df)
        timestamps = arrays[0]

        # ── watermark pre-filter (idempotent resume, backend-agnostic) ──
        wm_arr = self._resolve_watermark_array(ticks_df, timestamps)
        if wm_arr is not None and self._last_watermark is not None:
            keep = wm_arr > self._last_watermark
            arrays = self._slice_batch_arrays(arrays, keep)
            timestamps = arrays[0]
            wm_arr = wm_arr[keep]

        # ── strict ordering check (fail fast on the whole batch) ────────
        self._check_batch_ordering(timestamps)

        # ── numba path ──────────────────────────────────────────────────
        if self._backend == "numba":
            result = self._batch_numba(arrays, wm_arr)
            # numba bypasses update(); advance the watermark explicitly
            if wm_arr is not None and len(wm_arr) > 0:
                self._last_watermark = int(wm_arr[-1])
            if len(timestamps) > 0:
                self._last_timestamp = int(timestamps[-1])
            return result

        # ── Python path ─────────────────────────────────────────────────
        return self._batch_python(arrays, wm_arr)

    # ── numba batch path ──────────────────────────────────────────────────

    def _batch_numba(
        self,
        arrays: tuple[np.ndarray, ...],
        watermark: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Run the batch bar-construction loop via numba (or fall back to Python).

        Handles static and EWMA thresholds, time-bar special casing, and
        warmup-bar slicing.  Falls back to the Python path when:

        * numba is not installed
        * a session-producing calendar is in use (``has_sessions=True`` —
          boundaries are infrequent callbacks, not worth duplicating in numba)
        * the constructor is not pristine (resumed or partially consumed
          state — the kernels assume a zero start)
        * compilation/execution fails
        """
        from flowbars.bars.numba_backend import (
            _NUMBA_AVAILABLE,
            _get_compilation_warning,
            numba_batch_ewma,
            numba_batch_static,
        )

        if not _NUMBA_AVAILABLE:
            _get_compilation_warning()
            return self._batch_python(arrays, watermark)

        if self._calendar.has_sessions:
            warnings.warn(
                "This calendar produces session boundaries, which the numba "
                "backend does not support. Falling back to the Python path.",
                stacklevel=2,
            )
            return self._batch_python(arrays, watermark)

        if not self._is_pristine():
            warnings.warn(
                "The numba backend requires a pristine constructor (no prior "
                "update()/batch() calls and no restored state). Falling back "
                "to the Python path.",
                stacklevel=2,
            )
            return self._batch_python(arrays, watermark)

        timestamps, prices, volumes, sides = arrays
        bar_type = self._accumulator._bar_type

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
                return self._batch_python(arrays, watermark)
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
                return self._batch_python(arrays, watermark)
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
                return self._batch_python(arrays, watermark)

        return self._finalize_numba_result(bar_data, bt)

    def _finalize_numba_result(self, bar_data: np.ndarray, bar_type: str) -> pd.DataFrame:
        """Turn a numba kernel's raw output into the public bar DataFrame.

        Matches the Python path: every closed bar (including warmup bars)
        fires ``on_bar`` and counts toward ``bars_emitted``; warmup bars are
        then removed without renumbering (their ``bar_id`` values already
        start at ``warmup_bars``); finally the subclass ``_should_return``
        filter is applied.
        """
        from flowbars.bars.numba_backend import _bar_data_to_columns

        self._bars_emitted += len(bar_data)

        if self._on_bar is not None:
            for row in bar_data:
                self._on_bar(_row_to_bar(row, bar_type))

        if self._warmup_bars > 0 and len(bar_data) > 0:
            bar_data = bar_data[self._warmup_bars :]

        if len(bar_data) > 0:
            keep = np.array([self._should_return(_row_to_bar(row, bar_type)) for row in bar_data])
            bar_data = bar_data[keep]

        if len(bar_data) == 0:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)

        cols = _bar_data_to_columns(bar_data)
        cols["bar_type"] = bar_type  # type: ignore[assignment]
        return pd.DataFrame(cols)

    def _batch_python(
        self,
        arrays: tuple[np.ndarray, ...],
        watermark: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Pure-Python batch loop — used as the fallback from :meth:`_batch_numba`."""
        timestamps = arrays[0]
        bars: list[Bar] = []
        for i in range(len(timestamps)):
            bar_result = self.update(self._make_input(arrays, i, watermark))
            if bar_result is not None:
                bars.append(bar_result)

        # Drain remaining pending bars
        while self._pending_bars:
            bars.append(self._pending_bars.popleft())

        if not bars:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)

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

    # ── input abstraction hooks (tick defaults) ───────────────────────────

    def _default_schema(self) -> SchemaMapping | MinuteSchemaMapping:
        """Schema used by :meth:`batch` when none was supplied."""
        return SchemaMapping({"timestamp": "timestamp", "price": "price", "volume": "volume"})

    def _extract_batch_arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, ...]:
        """Extract and validate the per-column arrays from a batch DataFrame."""
        assert isinstance(self._schema, SchemaMapping)
        timestamps, prices, volumes, sides = self._schema.extract_arrays(df)
        sides = resolve_tick_signs(prices, sides)
        return (timestamps, prices, volumes, sides)

    def _slice_batch_arrays(
        self, arrays: tuple[np.ndarray, ...], keep: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Apply a boolean mask to every extracted array (``None`` passes through)."""
        return tuple(a[keep] if a is not None else None for a in arrays)  # type: ignore[misc]

    def _make_input(
        self, arrays: tuple[np.ndarray, ...], i: int, watermark: np.ndarray | None
    ) -> TickInfo | MinuteInfo:
        """Build one ``TickInfo``/``MinuteInfo`` from the batch arrays at row *i*."""
        timestamps, prices, volumes, sides = arrays
        raw_side = sides[i]
        return TickInfo(
            timestamp=int(timestamps[i]),
            price=float(prices[i]),
            volume=float(volumes[i]),
            side=float(raw_side) if not np.isnan(raw_side) else None,
            watermark=int(watermark[i]) if watermark is not None else None,
        )

    def _prepare_input(self, item: TickInfo | MinuteInfo) -> TickInfo | MinuteInfo:
        """Hook for per-input normalization before ingestion (tick: no-op)."""
        return item

    def _ingest_input(self, item: TickInfo | MinuteInfo) -> None:
        """Add one prepared input to the accumulator."""
        self._accumulator.add_tick(item)  # type: ignore[arg-type]

    def _check_batch_ordering(self, timestamps: np.ndarray) -> None:
        """Fail fast if the whole batch is out of order (when enabled)."""
        if not self._strict_ordering or len(timestamps) == 0:
            return
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

    def _is_pristine(self) -> bool:
        """True if no input has been consumed and state is untouched."""
        return (
            self._bars_emitted == 0
            and not self._pending_bars
            and self._last_watermark is None
            and _states_equal(self._accumulator.get_state(), self._initial_acc_state)
            and _states_equal(self._threshold_estimator.get_state(), self._initial_est_state)
        )

    def _extra_state(self) -> dict[str, Any]:  # noqa: B027
        """Subclass-specific state merged into :meth:`get_state`."""
        return {}

    def _load_extra_state(self, state: dict[str, Any]) -> None:  # noqa: B027
        """Restore subclass-specific state from :meth:`load_state`."""

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
        state = {
            "schema_version": 1,
            "input_kind": self.input_kind,
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
        state.update(self._extra_state())
        return state

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
        saved_kind = state.get("input_kind", "tick")
        if saved_kind != self.input_kind:
            raise StateValidationError(
                f"input_kind mismatch: state has {saved_kind!r}, "
                f"constructor has {self.input_kind!r}"
            )
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
        self._load_extra_state(state)
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
        schema: SchemaMapping | MinuteSchemaMapping | None = None,
    ) -> BaseBarConstructor:
        """Create a new constructor instance from a saved state dict.

        The caller is responsible for providing the correct accumulator
        and threshold-estimator types (or re-creating them from the
        bar-type information).  Subclasses can wrap this to auto-create
        the right components.

        The instance is built through an attribute-initialization path rather
        than ``__init__`` so that subclass behavior (e.g. ``min_run_length``
        filtering) is preserved; subclass-specific fields are restored by
        :meth:`_load_extra_state`.

        Parameters
        ----------
        state : dict
            State previously returned by :meth:`get_state`.
        accumulator : BaseAccumulator
            Fresh instance of the correct accumulator type.
        threshold_estimator : ThresholdEstimator
            Fresh instance of the correct estimator type.
        calendar : TradingCalendar, optional
        schema : SchemaMapping or MinuteSchemaMapping, optional

        Returns
        -------
        BaseBarConstructor
            A new constructor with its state restored from *state*.
        """
        backend = state.get("backend", "python")
        if backend not in ("python", "numba"):
            backend = "python"

        inst = cls.__new__(cls)
        inst._accumulator = accumulator
        inst._threshold_estimator = threshold_estimator
        inst._calendar = calendar if calendar is not None else ContinuousCalendar()
        inst._schema = schema
        inst._watermark_key = state["watermark_key"]
        inst._stream_id = state["stream_id"]
        inst._strict_ordering = False
        inst._warmup_bars = 0  # warmup is user-specified, not persisted
        inst._backend = backend
        inst._on_bar = None
        inst._on_threshold_update = None
        inst._bars_emitted = 0
        inst._in_session = True
        inst._pending_bars = collections.deque()
        inst._last_watermark = None
        inst._last_timestamp = None
        inst._initial_acc_state = copy.deepcopy(accumulator.get_state())
        inst._initial_est_state = copy.deepcopy(threshold_estimator.get_state())
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
        self._accumulator.on_session_boundary()
        self._bars_emitted += 1

        if self._on_bar is not None:
            self._on_bar(bar)

        # Warmup check + subclass filter (e.g. min_run_length)
        if self._bars_emitted > self._warmup_bars and self._should_return(bar):
            self._pending_bars.append(bar)
