"""Time bars — close at fixed calendar-time intervals.

Time bars sample the market at regular intervals (e.g., every 5 minutes).
The threshold estimator is unused — closure is driven entirely by the
``TimeAccumulator`` boundary-crossing logic.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowbars.bars.accumulators import TimeAccumulator
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.registry import BarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.schema import SchemaMapping
from flowbars.thresholds import StaticThresholdEstimator


class TimeBarConstructor(BaseBarConstructor):
    """Bar constructor for time bars.

    A time bar closes when a tick's timestamp crosses the next interval
    boundary.  Intervals are anchored to either round-clock boundaries
    (``"clock"``) or the first tick's timestamp (``"first_tick"``).

    Parameters
    ----------
    interval_ms : int
        Bar interval in milliseconds (e.g. 300000 for 5-minute bars).
    anchor : str, default ``"clock"``
        ``"clock"`` — bars aligned to round UTC boundaries.
        ``"first_tick"`` — bars aligned relative to the first tick.
    calendar : TradingCalendar, optional
    schema : SchemaMapping, optional
    watermark : str or None, default ``"timestamp"``
        Dedup key for idempotent resume (``"timestamp"``, a column name, or
        ``None`` to disable).
    strict_ordering : bool, default False
        If True, raise ``TickDataError`` when a tick arrives with a timestamp
        earlier than the previous tick's (out-of-order input).
    stream_id : str, default ``"default"``
    on_bar : callable or None
    on_threshold_update : callable or None
    """

    def __init__(
        self,
        interval_ms: int,
        anchor: str = "clock",
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        on_bar: Any = None,
        on_threshold_update: Any = None,
    ) -> None:
        accumulator = TimeAccumulator(
            bar_type="time",
            interval_ms=interval_ms,
            anchor=anchor,
        )
        # The threshold value is unused by TimeAccumulator — closure is
        # driven by timestamp boundary-crossing, not by a numeric threshold.
        estimator = StaticThresholdEstimator(threshold=0.0)

        super().__init__(
            accumulator=accumulator,
            threshold_estimator=estimator,
            calendar=calendar,
            schema=schema,
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
            warmup_bars=0,
            on_bar=on_bar,
            on_threshold_update=on_threshold_update,
        )

    @classmethod
    def from_state(  # type: ignore[override]
        cls,
        state: dict[str, Any],
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
    ) -> TimeBarConstructor:
        """Reconstruct a ``TimeBarConstructor`` from a saved state dict."""
        acc_state = state["accumulator"]
        acc = TimeAccumulator(
            bar_type="time",
            interval_ms=acc_state["interval_ms"],
            anchor=acc_state.get("anchor", "clock"),
        )
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        inst: TimeBarConstructor = BaseBarConstructor.from_state(state, acc, est, calendar, schema)  # type: ignore[assignment]
        return inst


def compute_time_bars(
    ticks_df: pd.DataFrame,
    interval_ms: int,
    anchor: str = "clock",
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Build time bars from a DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pd.DataFrame
        Tick-level data with user-mapped columns.
    interval_ms : int
        Bar interval in milliseconds.
    anchor : str, default ``"clock"``
        ``"clock"`` or ``"first_tick"``.
    schema : SchemaMapping, optional
        Defaults to implicit ``{ts, px, vol}`` mapping (auto-detection
        not yet implemented).
    watermark : str or None, default ``"timestamp"``
        Dedup key for idempotent resume (``"timestamp"``, a column name, or
        ``None`` to disable).
    strict_ordering : bool, default False
        If True, raise ``TickDataError`` when a tick arrives with a timestamp
        earlier than the previous tick's (out-of-order input).
    calendar : TradingCalendar, optional

    Returns
    -------
    pd.DataFrame
        Completed bars as a DataFrame.
    """
    ctor = TimeBarConstructor(
        interval_ms=interval_ms,
        anchor=anchor,
        calendar=calendar,
        schema=schema,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return ctor.batch(ticks_df)


# Register the batch function separately
BarRegistry.register("time", TimeBarConstructor, batch_fn=compute_time_bars)
