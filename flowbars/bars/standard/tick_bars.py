"""Tick bars — close after a fixed number of ticks."""

from __future__ import annotations

from typing import Any, Literal

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
    watermark : str or None, default ``"timestamp"``
        Dedup key for idempotent resume (``"timestamp"``, a column name, or
        ``None`` to disable).
    strict_ordering : bool, default False
        If True, raise ``TickDataError`` when a tick arrives with a timestamp
        earlier than the previous tick's (out-of-order input).
    warmup_bars : int, default 0
        Number of initial bars to discard (not returned).  Warmup bars still
        fire ``on_bar`` and update the estimator.
    backend : str, default ``"python"``
        ``"python"`` or ``"numba"``.  The numba backend accelerates
        :meth:`batch` only; ``update`` always uses Python.
    stream_id : str, default ``"default"``
    on_bar : callable or None
    on_threshold_update : callable or None
    """

    def __init__(
        self,
        threshold: int,
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        warmup_bars: int = 0,
        warmup_minutes: int | None = None,
        backend: Literal["python", "numba"] = "python",
        on_bar: Any = None,
        on_threshold_update: Any = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
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
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
            warmup_bars=warmup_bars,
            warmup_minutes=warmup_minutes,
            backend=backend,
            on_bar=on_bar,
            on_threshold_update=on_threshold_update,
            multiplier=multiplier,
            volume_type=volume_type,
        )

    @classmethod
    def from_state(  # type: ignore[override]
        cls,
        state: dict[str, Any],
        calendar: TradingCalendar | None = None,
        schema: SchemaMapping | None = None,
        *,
        verify_identity: bool = True,
    ) -> TickBarConstructor:
        """Reconstruct a ``TickBarConstructor`` from a saved state dict."""
        acc = TickAccumulator(bar_type="tick")
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        inst: TickBarConstructor = super().from_state(
            state, acc, est, calendar, schema, verify_identity=verify_identity
        )  # type: ignore[assignment]
        return inst


def compute_tick_bars(
    ticks_df: pd.DataFrame,
    threshold: int,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
    *,
    warmup_bars: int = 0,
    warmup_minutes: int | None = None,
    backend: Literal["python", "numba"] = "python",
    stream_id: str = "default",
    on_bar: Any = None,
    on_threshold_update: Any = None,
    multiplier: float = 1.0,
    volume_type: str = "base",
) -> pd.DataFrame:
    """Build tick bars from a DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pd.DataFrame
        Tick-level data with user-mapped columns.
    threshold : int
        Number of ticks per bar.
    schema : SchemaMapping, optional
    watermark : str or None, default ``"timestamp"``
        Dedup key for idempotent resume (``"timestamp"``, a column name, or
        ``None`` to disable).
    strict_ordering : bool, default False
        If True, raise ``TickDataError`` when a tick arrives with a timestamp
        earlier than the previous tick's (out-of-order input).
    calendar : TradingCalendar, optional
    warmup_bars : int, default 0
        Number of initial bars to discard.
    backend : str, default ``"python"``
        ``"python"`` or ``"numba"``.
    stream_id : str, default ``"default"``
    on_bar : callable or None
    on_threshold_update : callable or None

    Returns
    -------
    pd.DataFrame
        Completed bars as a DataFrame.
    """
    ctor = TickBarConstructor(
        threshold=threshold,
        calendar=calendar,
        schema=schema,
        watermark=watermark,
        stream_id=stream_id,
        strict_ordering=strict_ordering,
        warmup_bars=warmup_bars,
        warmup_minutes=warmup_minutes,
        backend=backend,
        on_bar=on_bar,
        on_threshold_update=on_threshold_update,
        multiplier=multiplier,
        volume_type=volume_type,
    )
    return ctor.batch(ticks_df)


BarRegistry.register("tick", TickBarConstructor, batch_fn=compute_tick_bars)
