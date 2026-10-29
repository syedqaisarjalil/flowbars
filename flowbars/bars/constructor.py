"""Base bar constructor — shared infrastructure for all bar types.

The ``BaseBarConstructor`` orchestrates an accumulator, threshold estimator,
and trading calendar to produce bars from a streaming tick feed.  It is
abstracted over the specific bar type — subclasses in ``standard/`` and
``information/`` provide the concrete accumulator + estimator wiring.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import warnings
from typing import Any, Callable, ClassVar, Literal

import numpy as np
import pandas as pd

from flowbars._version import __version__ as _version_string
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
    "partial",
]

# Current state schema.  v2 adds params/params_hash/engine_version plus the
# persisted warmup/ordering settings; v1 states remain loadable.
_STATE_SCHEMA_VERSION = 2
_ENGINE_VERSION = _version_string
# Version of the public output contract (columns + frame attrs).  Consumers
# can pin ``(bar_type, spec_version, params_hash)``.
_OUTPUT_CONTRACT_VERSION = 1


def params_hash(params: dict[str, Any]) -> str:
    """Stable SHA-256 hex digest of a params dict.

    Canonical JSON rules: keys sorted, separators ``(",", ":")`` (no
    whitespace), NaN/Infinity rejected.  Two equal parameter sets therefore
    always hash identically regardless of insertion order.
    """
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Backwards-compatible internal alias (imported by existing tests).
_params_hash = params_hash


def _bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    """Build the public output frame from completed bars."""
    if not bars:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.DataFrame([vars(bar) for bar in bars], columns=_OUTPUT_COLUMNS)


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
        trigger callbacks and update the estimator.  Mutually exclusive
        with *warmup_minutes*.
    warmup_minutes : int or None, default None
        Duration-based alternative to *warmup_bars*: bars that close
        before ``warmup_minutes`` minutes have elapsed since the first
        accepted input are not returned (callbacks still fire and the
        estimator still learns).  Bars keep their original ``bar_id``, so
        the first returned bar is not renumbered.  ``None`` disables it;
        mutually exclusive with *warmup_bars*.
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
    multiplier : float, default 1.0
        Contract multiplier: converts traded units to notional
        (``dollar_value = Σ price × volume × multiplier``).  ``vwap`` stays
        a price.  Must be positive.
    volume_type : str, default ``"base"``
        What the ``volume`` column means: ``"base"``, ``"quote"``,
        ``"shares"``, ``"contracts"``, or ``"tick"``.  Metadata only.

    Thread safety
    -------------
    Not thread-safe.  Designed for single-threaded streaming or batch use.
    """

    input_kind: ClassVar[str] = "tick"
    # True when the bar's notional/volume mapping is a proxy (e.g. minute
    # bars use close × volume instead of exact trade notional).
    approximate: ClassVar[bool] = False
    # Recommended warm-up duration advertised in the registry spec (metadata
    # only; the runtime warmup is the ``warmup_bars``/``warmup_minutes`` param).
    recommended_warmup_minutes: ClassVar[int] = 0
    # Canonical registry key (e.g. ``"minute_volume"``), set at registration.
    # The emitted ``bar_type`` label can be shorter (e.g. ``"volume"``); this
    # attribute is the unambiguous identity.
    registry_name: ClassVar[str | None] = None

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
        warmup_minutes: int | None = None,
        backend: Literal["python", "numba"] = "python",
        on_bar: Callable[[Bar], None] | None = None,
        on_threshold_update: Callable[[float], None] | None = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
        session_policy: Literal["close", "carry", "discard"] = "close",
        session_estimator: Literal["persist", "reset"] = "persist",
    ) -> None:
        if warmup_bars < 0:
            raise ValueError(f"warmup_bars must be non-negative, got {warmup_bars}")
        if warmup_minutes is not None and warmup_minutes < 0:
            raise ValueError(f"warmup_minutes must be non-negative, got {warmup_minutes}")
        if warmup_bars > 0 and warmup_minutes:
            raise ValueError("warmup_bars and warmup_minutes are mutually exclusive; set only one.")
        if backend not in ("python", "numba"):
            raise ValueError(f"backend must be 'python' or 'numba', got {backend!r}")
        if session_policy not in ("close", "carry", "discard"):
            raise ValueError(
                f"session_policy must be 'close', 'carry', or 'discard', got {session_policy!r}"
            )
        if session_estimator not in ("persist", "reset"):
            raise ValueError(
                f"session_estimator must be 'persist' or 'reset', got {session_estimator!r}"
            )

        self._accumulator = accumulator
        self._accumulator.configure_volume(multiplier, volume_type)
        self._threshold_estimator = threshold_estimator
        self._calendar = calendar if calendar is not None else ContinuousCalendar()
        self._schema = schema
        self._watermark_key = watermark
        self._stream_id = stream_id
        self._strict_ordering = strict_ordering
        self._warmup_bars = warmup_bars
        self._warmup_minutes = warmup_minutes
        self._warmup_start_ts: int | None = None
        self._backend = backend
        self._on_bar = on_bar
        self._on_threshold_update = on_threshold_update
        self._session_policy = session_policy
        self._session_estimator = session_estimator

        # Internal state
        self._bars_emitted: int = 0
        self._in_session: bool = True  # True until first boundary tick lands
        self._pending_bars: collections.deque[Bar] = collections.deque()
        self._last_watermark: int | None = None  # dedup watermark of last accepted tick
        self._last_timestamp: int | None = None  # timestamp of last accepted tick (ordering)
        self._session_id: str | None = None  # calendar session of the last in-session input
        self._last_session_end: int | None = None  # its close instant (Unix ms)

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
        if self._warmup_minutes and self._warmup_start_ts is None:
            self._warmup_start_ts = item.timestamp

        # 2. Session boundary detection
        is_boundary = self._calendar.is_session_boundary(item.timestamp)
        # A transition is a boundary->session or session->boundary flip.
        if is_boundary == self._in_session:
            if self._accumulator._has_tick and self._session_policy != "carry":
                if self._session_policy == "close":
                    # Force-close the in-progress bar as a partial bar.  Its
                    # close_ts is the session boundary, not this input's ts.
                    self._emit_bar(close_ts=self._session_transition_ts(item, is_boundary))
                else:  # "discard"
                    self._discard_bar()
            # A session gap never contributes a price return: reset carried
            # price state under every policy (a "carry" bar stays open but
            # skips the gap return).
            self._accumulator.on_session_boundary()
            if self._session_estimator == "reset":
                self._threshold_estimator.reset()
            self._in_session = not is_boundary

        # 3. Add input to accumulator
        self._ingest_input(item)
        self._last_watermark = wm
        self._last_timestamp = item.timestamp
        if not is_boundary:
            session_id = self._calendar.session_id(item.timestamp)
            if session_id is not None:
                self._session_id = session_id
            session_end = self._calendar.session_end(item.timestamp)
            if session_end is not None:
                self._last_session_end = session_end

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

            if self._past_warmup(item.timestamp) and self._should_return(bar):
                self._pending_bars.append(bar)

        # 6. Return one bar from queue, or None
        if self._pending_bars:
            return self._pending_bars.popleft()
        return None

    def update_many(self, ticks_df: pd.DataFrame, *, validate: bool = False) -> pd.DataFrame:
        """Stream every row of *ticks_df* through :meth:`update`.

        Equivalent to calling :meth:`update` once per row: the same schema
        mapping, watermark dedup, strict-ordering checks, callbacks, warmup,
        and session handling apply.  Returns the bars closed during the call
        (queued pending bars from a previous call are included first).

        ``backend`` is ignored here — streaming always uses the Python path.

        Parameters
        ----------
        ticks_df : pd.DataFrame
            Ticks (or minute candles) with user-mapped column names.
        validate : bool, default False
            Run :func:`~flowbars.bars.output.validate_bar_frame` on the
            returned frame (raises :class:`BarContractError` on violation).

        Returns
        -------
        pd.DataFrame
            Bars with the standard output schema (empty frame if none closed).
        """
        if self._schema is None:
            self._schema = self._default_schema()

        arrays = self._extract_batch_arrays(ticks_df)
        timestamps = arrays[0]
        wm_arr = self._resolve_watermark_array(ticks_df, timestamps)

        bars: list[Bar] = []
        while self._pending_bars:
            bars.append(self._pending_bars.popleft())

        for i in range(len(timestamps)):
            bar = self.update(self._make_input(arrays, i, wm_arr))
            if bar is not None:
                bars.append(bar)

        while self._pending_bars:
            bars.append(self._pending_bars.popleft())

        return self._maybe_validate(self._frame(bars), validate)

    def batch(self, ticks_df: pd.DataFrame, *, validate: bool = False) -> pd.DataFrame:
        """Feed all input rows from a DataFrame, return completed bars.

        Parameters
        ----------
        ticks_df : pd.DataFrame
            Ticks (or minute candles) with user-mapped column names (see
            :class:`SchemaMapping` / :class:`MinuteSchemaMapping`).
        validate : bool, default False
            Run :func:`~flowbars.bars.output.validate_bar_frame` on the
            returned frame (raises :class:`BarContractError` on violation).

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

        # ── duration warmup anchor (numba bypasses update()) ────────────
        if self._warmup_minutes and self._warmup_start_ts is None and len(timestamps) > 0:
            self._warmup_start_ts = int(timestamps[0])

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
            return self._maybe_validate(result, validate)

        # ── Python path ─────────────────────────────────────────────────
        return self._maybe_validate(self._batch_python(arrays, wm_arr), validate)

    def _maybe_validate(self, frame: pd.DataFrame, validate: bool) -> pd.DataFrame:
        """Validate *frame* against the output contract when requested."""
        if validate:
            from flowbars.bars.output import validate_bar_frame

            validate_bar_frame(frame)
        return frame

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
                    multiplier=self.multiplier,
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
                    multiplier=self.multiplier,
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
                    multiplier=self.multiplier,
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
        from flowbars.bars.numba_backend import _COL_CLOSE_TS, _bar_data_to_columns

        self._bars_emitted += len(bar_data)

        if self._on_bar is not None:
            for row in bar_data:
                self._on_bar(_row_to_bar(row, bar_type))

        if self._warmup_bars > 0 and len(bar_data) > 0:
            bar_data = bar_data[self._warmup_bars :]
        elif self._warmup_minutes and self._warmup_start_ts is not None and len(bar_data) > 0:
            # A bar closes on an input at `close_ts`, so the streaming
            # `_past_warmup(item.timestamp)` rule maps onto this mask.
            bar_data = bar_data[bar_data[:, _COL_CLOSE_TS] >= self._warmup_end_ts()]

        if len(bar_data) > 0:
            keep = np.array([self._should_return(_row_to_bar(row, bar_type)) for row in bar_data])
            bar_data = bar_data[keep]

        if len(bar_data) == 0:
            return self._frame([])

        cols = _bar_data_to_columns(bar_data)
        cols["bar_type"] = bar_type  # type: ignore[assignment]
        cols["partial"] = False  # type: ignore[assignment]
        frame = pd.DataFrame(cols)
        frame.attrs.update(self._frame_metadata())
        return frame

    def _frame_metadata(self) -> dict[str, Any]:
        """Frame-level metadata identifying the series that produced it."""
        return {
            "bar_type": self._accumulator._bar_type,
            "registry_name": self.registry_name,
            "params_hash": _params_hash(self.get_params()),
            "input_kind": self.input_kind,
            "engine_version": _ENGINE_VERSION,
            "state_schema_version": _STATE_SCHEMA_VERSION,
            "spec_version": _OUTPUT_CONTRACT_VERSION,
            "multiplier": self.multiplier,
            "volume_type": self.volume_type,
            "approximate": self.approximate,
        }

    def _frame(self, bars: list[Bar]) -> pd.DataFrame:
        """Build the output frame with series metadata attached."""
        frame = _bars_to_frame(bars)
        frame.attrs.update(self._frame_metadata())
        return frame

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

        return self._frame(bars)

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

    def _extra_params(self) -> dict[str, Any]:  # noqa: B027
        """Subclass-specific configuration merged into :meth:`get_params`."""
        return {}

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

    @property
    def warmup_done(self) -> bool:
        """``True`` once no warmup is configured or the warmup window passed.

        Bar-count warmup is done after ``warmup_bars`` + 1 bars have closed
        (the extra bar is the first returned one).  Duration warmup is done
        once the last accepted input reaches ``warmup_minutes`` minutes past
        the first accepted input.
        """
        if self._warmup_minutes:
            if self._warmup_start_ts is None or self._last_timestamp is None:
                return False
            return self._last_timestamp >= self._warmup_end_ts()
        return self._warmup_bars == 0 or self._bars_emitted > self._warmup_bars

    @property
    def multiplier(self) -> float:
        """Contract multiplier scaling notional (``dollar_value``)."""
        return self._accumulator._multiplier

    @property
    def volume_type(self) -> str:
        """What the ``volume`` column means (e.g. ``"contracts"``)."""
        return self._accumulator._volume_type

    # ── state persistence ───────────────────────────────────────────────

    def get_params(self) -> dict[str, Any]:
        """Configuration that defines this bar stream.

        Params are hashed into the serialized state so a resume against a
        constructor with a different configuration fails loudly instead of
        silently changing bar semantics.  Runtime values (partial-bar OHLCV,
        EWMA terms) live in the state, not here.
        """
        return {
            "input_kind": self.input_kind,
            "bar_type": self._accumulator._bar_type,
            "session_policy": self._session_policy,
            "session_estimator": self._session_estimator,
            "accumulator": self._accumulator.get_params(),
            "threshold_estimator": self._threshold_estimator.get_params(),
            **self._extra_params(),
        }

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the constructor's current state.

        Includes the bar-defining params and their hash, the engine version,
        accumulator/estimator state, and constructor-level bookkeeping.
        """
        params = self.get_params()
        state = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "engine_version": _ENGINE_VERSION,
            "input_kind": self.input_kind,
            "stream_id": self._stream_id,
            "bar_type": self._accumulator._bar_type,
            "registry_name": self.registry_name,
            "backend": self._backend,
            "watermark_key": self._watermark_key,
            "warmup_bars": self._warmup_bars,
            "warmup_minutes": self._warmup_minutes,
            "warmup_start_ts": self._warmup_start_ts,
            "strict_ordering": self._strict_ordering,
            "session_policy": self._session_policy,
            "session_estimator": self._session_estimator,
            "session_id": self._session_id,
            "last_session_end": self._last_session_end,
            "calendar": self._calendar.to_dict(),
            "calendar_fingerprint": self._calendar.fingerprint(),
            "schema_mapping": None if self._schema is None else self._schema.to_dict(),
            "schema_fingerprint": None if self._schema is None else self._schema.fingerprint(),
            "params": params,
            "params_hash": _params_hash(params),
            "last_watermark": self._last_watermark,
            "last_timestamp": self._last_timestamp,
            "accumulator": self._accumulator.get_state(),
            "threshold_estimator": self._threshold_estimator.get_state(),
            "bars_emitted": self._bars_emitted,
            "in_session": self._in_session,
        }
        state.update(self._extra_state())
        return state

    def _verify_saved_identity(self, state: dict[str, Any]) -> None:
        """Reject state written against a different registry key, calendar, or schema."""
        saved_registry = state.get("registry_name")
        if (
            saved_registry is not None
            and self.registry_name is not None
            and saved_registry != self.registry_name
        ):
            raise StateValidationError(
                f"registry_name mismatch: state has {saved_registry!r}, "
                f"constructor has {self.registry_name!r}"
            )

        saved_calendar_fp = state.get("calendar_fingerprint")
        if saved_calendar_fp is not None:
            calendar_fp = self._calendar.fingerprint()
            if saved_calendar_fp != calendar_fp:
                raise StateValidationError(
                    f"calendar mismatch: state was built with {state.get('calendar')!r} "
                    f"(fingerprint {saved_calendar_fp}), constructor has "
                    f"{self._calendar.to_dict()!r} (fingerprint {calendar_fp}). "
                    "Pass the same calendar or use verify_identity=False."
                )

        saved_schema_fp = state.get("schema_fingerprint")
        if saved_schema_fp is not None and self._schema is not None:
            schema_fp = self._schema.fingerprint()
            if saved_schema_fp != schema_fp:
                raise StateValidationError(
                    f"schema mapping mismatch: state was built with "
                    f"{state.get('schema_mapping')!r}, constructor has "
                    f"{self._schema.to_dict()!r}. Pass the same schema or use "
                    "verify_identity=False."
                )

    def load_state(self, state: dict[str, Any], *, verify_identity: bool = True) -> None:
        """Restore constructor state from a saved state dict.

        Validates stream identity and schema version before applying state.

        Parameters
        ----------
        state : dict
            State previously returned by :meth:`get_state`.
        verify_identity : bool, default True
            Also require the calendar and schema column mapping to match the
            ones recorded in the state.  Pass ``False`` to opt out (e.g. when
            intentionally replaying state under a different input mapping).

        Raises
        ------
        StateValidationError
            If *stream_id*, *bar_type*, *registry_name*, *schema_version*,
            calendar, or schema mapping is incompatible.
        """
        # Version check
        version = state.get("schema_version", 0)
        if version < 1:
            raise StateValidationError(
                f"State schema version {version} is not supported. "
                "Rebuild the state from the source input. Minimum required version is 1."
            )
        if version > _STATE_SCHEMA_VERSION:
            raise StateValidationError(
                f"State schema version {version} was written by a newer version "
                f"of flowbars (this build supports up to {_STATE_SCHEMA_VERSION}). "
                "Upgrade the library or rebuild the state from the source input."
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

        if verify_identity:
            self._verify_saved_identity(state)

        saved_engine = state.get("engine_version")
        if (
            saved_engine is not None
            and str(saved_engine).split(".")[0] != _ENGINE_VERSION.split(".")[0]
        ):
            raise StateValidationError(
                f"State was written by engine version {saved_engine}, which is "
                f"incompatible with engine version {_ENGINE_VERSION}. "
                "Rebuild the state from the source input."
            )

        if version >= 2:
            saved_params = state.get("params")
            saved_hash = state.get("params_hash")
            if saved_params is None or saved_hash is None:
                raise StateValidationError(
                    "State schema v2 is missing 'params'/'params_hash'. "
                    "Rebuild the state from the source input."
                )
            if _params_hash(saved_params) != saved_hash:
                raise StateValidationError(
                    "State params_hash does not match its params — the state is "
                    "corrupt. Rebuild it from the source input."
                )
            current_params = self.get_params()
            if _params_hash(current_params) != saved_hash:
                raise StateValidationError(
                    f"params mismatch: state was built with {saved_params!r}, "
                    f"constructor has {current_params!r}. Rebuild the state from "
                    "the source input or resume with a matching constructor."
                )

        # Apply state
        self._accumulator.load_state(state["accumulator"])
        self._threshold_estimator.load_state(state["threshold_estimator"])
        self._bars_emitted = state.get("bars_emitted", 0)
        self._in_session = state.get("in_session", True)
        self._session_id = state.get("session_id", self._session_id)
        self._last_session_end = state.get("last_session_end", self._last_session_end)
        self._last_watermark = state["last_watermark"]
        self._last_timestamp = state.get("last_timestamp")
        if version >= 2:
            self._warmup_bars = int(state.get("warmup_bars", self._warmup_bars))
            self._warmup_minutes = state.get("warmup_minutes", self._warmup_minutes)
            self._warmup_start_ts = state.get("warmup_start_ts", self._warmup_start_ts)
            self._strict_ordering = bool(state.get("strict_ordering", self._strict_ordering))
            self._session_policy = state.get("session_policy", self._session_policy)
            self._session_estimator = state.get("session_estimator", self._session_estimator)
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
        *,
        verify_identity: bool = True,
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
            Must match the calendar recorded in the state (the default
            :class:`ContinuousCalendar` is used when omitted).
        schema : SchemaMapping or MinuteSchemaMapping, optional
            Must match the schema recorded in the state when provided.
        verify_identity : bool, default True
            Require the saved registry key, calendar, and schema mapping to
            match.  Pass ``False`` to resume under a different configuration.

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
        inst._strict_ordering = False  # restored by load_state (v2+)
        inst._warmup_bars = 0  # restored by load_state (v2+)
        inst._warmup_minutes = state.get("warmup_minutes")  # restored by load_state (v2+)
        inst._warmup_start_ts = state.get("warmup_start_ts")
        # Session settings are part of the params identity, so seed them from
        # the state before load_state validates the params hash.
        inst._session_policy = state.get("session_policy", "close")
        inst._session_estimator = state.get("session_estimator", "persist")
        inst._backend = backend
        inst._on_bar = None
        inst._on_threshold_update = None
        inst._bars_emitted = 0
        inst._in_session = True
        inst._pending_bars = collections.deque()
        inst._last_watermark = None
        inst._last_timestamp = None
        inst._session_id = state.get("session_id")
        inst._last_session_end = state.get("last_session_end")
        # Volume semantics live in the accumulator state; restore them before
        # params validation so the identity check sees the right multiplier.
        acc_state = state.get("accumulator") or {}
        inst._accumulator.configure_volume(
            acc_state.get("multiplier", 1.0),
            acc_state.get("volume_type", "base"),
        )
        inst._initial_acc_state = copy.deepcopy(accumulator.get_state())
        inst._initial_est_state = copy.deepcopy(threshold_estimator.get_state())
        # Subclass params (e.g. min_run_length) are not passed through this
        # path — seed them from the state before validating params identity.
        inst._load_extra_state(state)
        inst.load_state(state, verify_identity=verify_identity)
        return inst

    # ── internal helpers ────────────────────────────────────────────────

    def _warmup_end_ts(self) -> int:
        """Timestamp at which duration warmup ends (requires a start ts)."""
        assert self._warmup_minutes is not None and self._warmup_start_ts is not None
        return self._warmup_start_ts + self._warmup_minutes * 60_000

    def _past_warmup(self, close_ts: int) -> bool:
        """Return ``True`` if a bar closing at *close_ts* is past warmup."""
        if self._warmup_minutes:
            if self._warmup_start_ts is None:
                return True
            return close_ts >= self._warmup_end_ts()
        return self._bars_emitted > self._warmup_bars

    def _should_return(self, bar: Bar) -> bool:  # noqa: B027
        """Return ``True`` if *bar* should be returned to the caller.

        Override in subclasses to filter bars (e.g. ``min_run_length`` for
        run bars).  The default always returns ``True``.
        """
        return True

    def _session_transition_ts(self, item: TickInfo | MinuteInfo, is_boundary: bool) -> int:
        """The instant a session-boundary force-close logically happens.

        Leaving a session: that session's close.  Entering a session: the
        open.  Falls back to the calendar's ``next_session_open`` when the
        calendar cannot describe the boundary.
        """
        ref = self._last_timestamp if self._last_timestamp is not None else item.timestamp
        if is_boundary:
            session_end = self._calendar.session_end(ref)
            if session_end is not None:
                return session_end
        return self._calendar.next_session_open(ref)

    def _emit_bar(self, close_ts: int | None = None) -> None:
        """Force-close the current bar and queue it for return.

        Does NOT call ``on_bar_close`` on the estimator — session-boundary
        closures are artificial and should not influence the adaptive
        threshold.  Price-state resets are applied by the caller after this
        returns (see :meth:`update`).

        ``close_ts`` overrides the accumulator's close timestamp; session
        force-closes pass the session boundary so a partial bar's
        ``close_ts`` is the instant its information became available.
        """
        threshold = self._threshold_estimator.current_threshold
        bar = self._accumulator.close(threshold)
        bar.partial = True
        if close_ts is not None:
            bar.close_ts = max(close_ts, bar.open_ts)
        self._bars_emitted += 1

        if self._on_bar is not None:
            self._on_bar(bar)

        # Warmup check + subclass filter (e.g. min_run_length)
        if self._past_warmup(bar.close_ts) and self._should_return(bar):
            self._pending_bars.append(bar)

    def _discard_bar(self) -> None:
        """Drop the in-progress bar at a session boundary (``"discard"`` policy).

        No bar is emitted, no callbacks fire, and ``bars_emitted`` does not
        advance; scalar accumulators are zeroed so no partial metric or
        overflow crosses the boundary.
        """
        self._accumulator.discard()
