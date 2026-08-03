"""Dollar bars — close when cumulative notional value crosses a threshold."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowbars.bars.accumulators import DollarAccumulator
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.registry import BarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.schema import SchemaMapping
from flowbars.thresholds import StaticThresholdEstimator


class DollarBarConstructor(BaseBarConstructor):
    """Bar constructor for dollar bars.

    A dollar bar closes when the cumulative notional value (price × volume)
    reaches or exceeds *threshold*.  Excess rolls into the next bar.

    Parameters
    ----------
    threshold : float
        Notional-value threshold per bar (e.g. 1_000_000.0 for $1M bars).
        Must be positive.
    calendar : TradingCalendar, optional
    schema : SchemaMapping, optional
    stream_id : str, default ``"default"``
    on_bar : callable or None
    on_threshold_update : callable or None
    """

    def __init__(
        self,
        threshold: float,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        stream_id: str = "default",
        on_bar: Any = None,
        on_threshold_update: Any = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        accumulator = DollarAccumulator(bar_type="dollar")
        estimator = StaticThresholdEstimator(threshold=threshold)

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
    ) -> DollarBarConstructor:
        """Reconstruct a ``DollarBarConstructor`` from a saved state dict."""
        acc = DollarAccumulator(bar_type="dollar")
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        inst: DollarBarConstructor = BaseBarConstructor.from_state(
            state, acc, est, calendar, schema
        )  # type: ignore[assignment]
        return inst


def compute_dollar_bars(
    ticks_df: pd.DataFrame,
    threshold: float,
    schema: SchemaMapping | None = None,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Build dollar bars from a DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pd.DataFrame
        Tick-level data with user-mapped columns.
    threshold : float
        Notional-value threshold per bar (e.g. 1_000_000.0 for $1M bars).
    schema : SchemaMapping, optional
    calendar : TradingCalendar, optional

    Returns
    -------
    pd.DataFrame
        Completed bars as a DataFrame.
    """
    ctor = DollarBarConstructor(
        threshold=threshold,
        calendar=calendar,
        schema=schema,
    )
    return ctor.batch(ticks_df)


BarRegistry.register("dollar", DollarBarConstructor, batch_fn=compute_dollar_bars)
