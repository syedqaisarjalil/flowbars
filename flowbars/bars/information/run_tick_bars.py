"""Run bars (tick) — close when the cumulative same-sign tick-run crosses an adaptive threshold.

Uses the two-component EWMA formula from AFML §2.4:

.. math::

    T_n = E[T]_n \\times \\max(E[P^+]_n,\\, 1 - E[P^+]_n)

where :math:`E[T]_n` is the expected run length per bar and
:math:`E[P^+]_n` is the expected buy-run proportion.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowbars.bars.accumulators import RunAccumulator
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.registry import BarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.core import Bar
from flowbars.schema import SchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator


class RunTickBarConstructor(BaseBarConstructor):
    """Bar constructor for tick-run bars.

    A tick-run bar closes when the cumulative same-sign tick-run total
    reaches or exceeds an **adaptive** threshold estimated by an EWMA.

    Parameters
    ----------
    span : float, default 20.0
        EWMA span for the two-component threshold formula.
    halflife : float or None, default None
        EWMA halflife.  Takes precedence over *span* when given.
    initial_ewa_t : float, default 1.0
        Initial seed for :math:`E[T]` (expected run length per bar).
    initial_ewa_proportion : float, default 0.5
        Initial seed for :math:`E[P^+]` (expected buy-run proportion).
    warmup_bars : int, default 0
        Number of initial bars to discard while the EWMA converges.
    min_run_length : int, default 0
        Minimum number of ticks a run bar must contain to be returned.
        Bars with fewer ticks are discarded (they still update the EWMA).
        Default 0 — no filtering.
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
        span: float = 20.0,
        halflife: float | None = None,
        initial_ewa_t: float = 1.0,
        initial_ewa_proportion: float = 0.5,
        warmup_bars: int = 0,
        min_run_length: int = 0,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        on_bar: Any = None,
        on_threshold_update: Any = None,
    ) -> None:
        if min_run_length < 0:
            raise ValueError(f"min_run_length must be non-negative, got {min_run_length}")
        self._min_run_length = min_run_length

        accumulator = RunAccumulator(bar_type="run_tick", metric="tick")
        estimator = EWMAThresholdEstimator(
            bar_family="run",
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
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
            warmup_bars=warmup_bars,
            on_bar=on_bar,
            on_threshold_update=on_threshold_update,
        )

    def _should_return(self, bar: Bar) -> bool:
        """Filter bars below ``min_run_length`` ticks."""
        return bar.num_ticks >= self._min_run_length

    @classmethod
    def from_state(  # type: ignore[override]
        cls,
        state: dict[str, Any],
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
    ) -> RunTickBarConstructor:
        """Reconstruct a ``RunTickBarConstructor`` from a saved state dict."""
        acc_state = state["accumulator"]
        acc = RunAccumulator(
            bar_type=state["bar_type"],
            metric=acc_state["metric"],
        )
        est = EWMAThresholdEstimator.from_state(state["threshold_estimator"])
        inst: RunTickBarConstructor = BaseBarConstructor.from_state(
            state, acc, est, calendar, schema
        )  # type: ignore[assignment]
        return inst


def compute_run_tick_bars(
    ticks_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    min_run_length: int = 0,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Build tick-run bars from a DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pd.DataFrame
        Tick-level data with user-mapped columns.
    span : float, default 20.0
        EWMA span for the adaptive threshold.
    halflife : float or None, default None
        EWMA halflife (takes precedence over *span*).
    initial_ewa_t : float, default 1.0
        Initial seed for expected run length per bar.
    initial_ewa_proportion : float, default 0.5
        Initial seed for expected buy-run proportion.
    warmup_bars : int, default 0
        Number of initial bars to discard.
    min_run_length : int, default 0
        Minimum ticks per bar to return.
    schema : SchemaMapping, optional
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
    ctor = RunTickBarConstructor(
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        min_run_length=min_run_length,
        calendar=calendar,
        schema=schema,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return ctor.batch(ticks_df)


BarRegistry.register("run_tick", RunTickBarConstructor, batch_fn=compute_run_tick_bars)
