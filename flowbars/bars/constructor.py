"""Base bar constructor — shared infrastructure for all bar types.

The ``BaseBarConstructor`` orchestrates an accumulator, threshold estimator,
and trading calendar to produce bars from a streaming tick feed.  It is
abstracted over the specific bar type — subclasses in ``standard/`` and
``information/`` provide the concrete accumulator + estimator wiring.
"""

from __future__ import annotations

import collections
from typing import Any, Callable

import numpy as np
import pandas as pd

from flowbars.bars.accumulators import BaseAccumulator
from flowbars.calendars import ContinuousCalendar, TradingCalendar
from flowbars.core import Bar, StateValidationError, TickInfo
from flowbars.schema import SchemaMapping
from flowbars.thresholds import ThresholdEstimator
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
        on_bar: Callable[[Bar], None] | None = None,
        on_threshold_update: Callable[[float], None] | None = None,
    ) -> None:
        if warmup_bars < 0:
            raise ValueError(f"warmup_bars must be non-negative, got {warmup_bars}")

        self._accumulator = accumulator
        self._threshold_estimator = threshold_estimator
        self._calendar = calendar if calendar is not None else ContinuousCalendar()
        self._schema = schema
        self._stream_id = stream_id
        self._warmup_bars = warmup_bars
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
            raise ValueError(
                "batch() requires a SchemaMapping. Set schema= when constructing "
                "the bar constructor, or feed ticks manually via update()."
            )

        timestamps, prices, volumes, sides = self._schema.extract_arrays(ticks_df)

        # Derive signs if not supplied
        sides = resolve_tick_signs(prices, sides)

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
        inst = cls(
            accumulator=accumulator,
            threshold_estimator=threshold_estimator,
            calendar=calendar,
            schema=schema,
            stream_id=stream_id,
            warmup_bars=warmup_bars,
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
