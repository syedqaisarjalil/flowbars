"""Volume bars — close when cumulative traded volume crosses a threshold."""

from __future__ import annotations

from typing import Any

import pandas as pd

from flowbars.bars.accumulators import VolumeAccumulator
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.registry import BarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.schema import SchemaMapping
from flowbars.thresholds import StaticThresholdEstimator


class VolumeBarConstructor(BaseBarConstructor):
    """Bar constructor for volume bars.

    A volume bar closes when the cumulative traded volume reaches or
    exceeds *threshold*.  Excess volume rolls into the next bar.

    Parameters
    ----------
    threshold : float
        Volume threshold per bar (e.g. 1000.0 for 1000 shares/contracts).
        Must be positive.
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
        threshold: float,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        on_bar: Any = None,
        on_threshold_update: Any = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")

        accumulator = VolumeAccumulator(bar_type="volume")
        estimator = StaticThresholdEstimator(threshold=threshold)

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
    ) -> VolumeBarConstructor:
        """Reconstruct a ``VolumeBarConstructor`` from a saved state dict."""
        acc = VolumeAccumulator(bar_type="volume")
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        inst: VolumeBarConstructor = BaseBarConstructor.from_state(
            state, acc, est, calendar, schema
        )  # type: ignore[assignment]
        return inst


def compute_volume_bars(
    ticks_df: pd.DataFrame,
    threshold: float,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pd.DataFrame:
    """Build volume bars from a DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pd.DataFrame
        Tick-level data with user-mapped columns.
    threshold : float
        Volume threshold per bar.
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
    ctor = VolumeBarConstructor(
        threshold=threshold,
        calendar=calendar,
        schema=schema,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return ctor.batch(ticks_df)


BarRegistry.register("volume", VolumeBarConstructor, batch_fn=compute_volume_bars)
