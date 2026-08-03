"""Tick bars — close after a fixed number of ticks."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowbars.bars.accumulators import TickAccumulator
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.registry import BarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.schema import SchemaMapping
from flowbars.thresholds import StaticThresholdEstimator


class TickBarConstructor(BaseBarConstructor):
    """Bar constructor for tick bars.

    A tick bar samples every *threshold* ticks.

    Parameters
    ----------
    threshold : int
        Number of ticks per bar.  Must be positive.
    calendar : TradingCalendar, optional
    schema : SchemaMapping, optional
    stream_id : str, default ``"default"``
    on_bar : callable or None
    on_threshold_update : callable or None
    """

    def __init__(
        self,
        threshold: int,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        stream_id: str = "default",
        on_bar: Any = None,
        on_threshold_update: Any = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        accumulator = TickAccumulator(bar_type="tick")
        estimator = StaticThresholdEstimator(threshold=float(threshold))

        super().__init__(
            accumulator=accumulator,
            threshold_estimator=estimator,
            calendar=calendar,
            schema=schema,
            stream_id=stream_id,
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
    ) -> TickBarConstructor:
        """Reconstruct a ``TickBarConstructor`` from a saved state dict."""
        acc = TickAccumulator(bar_type="tick")
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        inst: TickBarConstructor = BaseBarConstructor.from_state(state, acc, est, calendar, schema)  # type: ignore[assignment]
        return inst


def compute_tick_bars(
    ticks_df: pd.DataFrame,
    threshold: int,
    schema: SchemaMapping | None = None,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Build tick bars from a DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pd.DataFrame
        Tick-level data with user-mapped columns.
    threshold : int
        Number of ticks per bar.
    schema : SchemaMapping, optional
    calendar : TradingCalendar, optional

    Returns
    -------
    pd.DataFrame
        Completed bars as a DataFrame.
    """
    ctor = TickBarConstructor(
        threshold=threshold,
        calendar=calendar,
        schema=schema,
    )
    return ctor.batch(ticks_df)


BarRegistry.register("tick", TickBarConstructor, batch_fn=compute_tick_bars)
