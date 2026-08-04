"""Imbalance bars (volume) — close when absolute signed volume-imbalance crosses an adaptive threshold.

Same two-component EWMA formula as tick-imbalance bars, but the imbalance
is weighted by trade volume instead of tick count.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowbars.bars.accumulators import ImbalanceAccumulator
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.registry import BarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.schema import SchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator


class ImbalanceVolumeBarConstructor(BaseBarConstructor):
    """Bar constructor for volume-imbalance bars.

    A volume-imbalance bar closes when the absolute signed volume-imbalance
    reaches or exceeds an **adaptive** threshold estimated by an EWMA.

    Parameters
    ----------
    span : float, default 20.0
        EWMA span for the two-component threshold formula.
    halflife : float or None, default None
        EWMA halflife.  Takes precedence over *span* when given.
    initial_ewa_t : float, default 1.0
        Initial seed for :math:`E[T]` (expected volume per bar).
    initial_ewa_proportion : float, default 0.5
        Initial seed for :math:`E[\\theta]` (expected imbalance proportion).
    warmup_bars : int, default 0
        Number of initial bars to discard while the EWMA converges.
    calendar : TradingCalendar, optional
    schema : SchemaMapping, optional
    stream_id : str, default ``"default"``
    on_bar : callable or None
    on_threshold_update : callable or None
    """

    def __init__(
        self,
        span: float = 20.0,
        halflife: float | None = None,
        initial_ewa_t: float = 1.0,
        initial_ewa_proportion: float = 0.5,
        warmup_bars: int = 0,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        stream_id: str = "default",
        on_bar: Any = None,
        on_threshold_update: Any = None,
    ) -> None:
        accumulator = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        estimator = EWMAThresholdEstimator(
            bar_family="imbalance",
            span=span,
            halflife=halflife,
            initial_ewa_t=initial_ewa_t,
            initial_ewa_proportion=initial_ewa_proportion,
        )

        super().__init__(
            accumulator=accumulator,
            threshold_estimator=estimator,
            calendar=calendar,
            schema=schema,
            stream_id=stream_id,
            warmup_bars=warmup_bars,
            on_bar=on_bar,
            on_threshold_update=on_threshold_update,
        )

    @classmethod
    def from_state(  # type: ignore[override]
        cls,
        state: dict[str, Any],
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
    ) -> ImbalanceVolumeBarConstructor:
        """Reconstruct an ``ImbalanceVolumeBarConstructor`` from a saved state dict."""
        acc_state = state["accumulator"]
        acc = ImbalanceAccumulator(
            bar_type=state["bar_type"],
            metric=acc_state["metric"],
        )
        est = EWMAThresholdEstimator.from_state(state["threshold_estimator"])
        inst: ImbalanceVolumeBarConstructor = BaseBarConstructor.from_state(
            state, acc, est, calendar, schema
        )  # type: ignore[assignment]
        return inst


def compute_imbalance_volume_bars(
    ticks_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    schema: SchemaMapping | None = None,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Build volume-imbalance bars from a DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pd.DataFrame
        Tick-level data with user-mapped columns.
    span : float, default 20.0
        EWMA span for the adaptive threshold.
    halflife : float or None, default None
        EWMA halflife (takes precedence over *span*).
    initial_ewa_t : float, default 1.0
        Initial seed for expected volume per bar.
    initial_ewa_proportion : float, default 0.5
        Initial seed for expected imbalance proportion.
    warmup_bars : int, default 0
        Number of initial bars to discard.
    schema : SchemaMapping, optional
    calendar : TradingCalendar, optional

    Returns
    -------
    pd.DataFrame
        Completed bars as a DataFrame.
    """
    ctor = ImbalanceVolumeBarConstructor(
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        calendar=calendar,
        schema=schema,
    )
    return ctor.batch(ticks_df)


BarRegistry.register(
    "imbalance_volume", ImbalanceVolumeBarConstructor, batch_fn=compute_imbalance_volume_bars
)
