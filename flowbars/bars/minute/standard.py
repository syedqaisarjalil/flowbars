"""Standard minute bars: volume and dollar."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from flowbars.bars.minute.accumulators import (
    MinuteDollarAccumulator,
    MinuteVolumeAccumulator,
)
from flowbars.bars.minute.engine import BaseMinuteBarConstructor
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.schema import MinuteSchemaMapping
from flowbars.thresholds import StaticThresholdEstimator


class MinuteVolumeBarConstructor(BaseMinuteBarConstructor):
    """Minute volume bars: close when cumulative minute volume crosses a threshold.

    Excess volume (whole-minute attribution) rolls into the next bar.
    """

    def __init__(
        self,
        threshold: float,
        calendar: TradingCalendar | None = None,
        schema: MinuteSchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        warmup_bars: int = 0,
        warmup_minutes: int | None = None,
        backend: str = "python",
        on_bar: Any = None,
        on_threshold_update: Any = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
        session_policy: Literal["close", "carry", "discard"] = "close",
        session_estimator: Literal["persist", "reset"] = "persist",
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        super().__init__(
            accumulator=MinuteVolumeAccumulator(bar_type="volume"),
            threshold_estimator=StaticThresholdEstimator(threshold=threshold),
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
            session_policy=session_policy,
            session_estimator=session_estimator,
        )

    @classmethod
    def from_state(  # type: ignore[override]
        cls,
        state: dict[str, Any],
        calendar: TradingCalendar | None = None,
        schema: MinuteSchemaMapping | None = None,
        *,
        verify_identity: bool = True,
    ) -> MinuteVolumeBarConstructor:
        """Reconstruct a ``MinuteVolumeBarConstructor`` from a saved state dict."""
        acc = MinuteVolumeAccumulator(bar_type="volume")
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(
            state, acc, est, calendar, schema, verify_identity=verify_identity
        )  # type: ignore[return-value]


class MinuteDollarBarConstructor(BaseMinuteBarConstructor):
    """Minute dollar bars: close when cumulative ``close × volume`` crosses a threshold."""

    # close × volume is a per-minute notional proxy, not exact tick notional.
    approximate = True

    def __init__(
        self,
        threshold: float,
        calendar: TradingCalendar | None = None,
        schema: MinuteSchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        warmup_bars: int = 0,
        warmup_minutes: int | None = None,
        backend: str = "python",
        on_bar: Any = None,
        on_threshold_update: Any = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
        session_policy: Literal["close", "carry", "discard"] = "close",
        session_estimator: Literal["persist", "reset"] = "persist",
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        super().__init__(
            accumulator=MinuteDollarAccumulator(bar_type="dollar"),
            threshold_estimator=StaticThresholdEstimator(threshold=threshold),
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
            session_policy=session_policy,
            session_estimator=session_estimator,
        )

    @classmethod
    def from_state(  # type: ignore[override]
        cls,
        state: dict[str, Any],
        calendar: TradingCalendar | None = None,
        schema: MinuteSchemaMapping | None = None,
        *,
        verify_identity: bool = True,
    ) -> MinuteDollarBarConstructor:
        """Reconstruct a ``MinuteDollarBarConstructor`` from a saved state dict."""
        acc = MinuteDollarAccumulator(bar_type="dollar")
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(
            state, acc, est, calendar, schema, verify_identity=verify_identity
        )  # type: ignore[return-value]


def compute_volume_bars_from_minutes(
    minutes_df: pd.DataFrame,
    threshold: float,
    schema: MinuteSchemaMapping | None = None,
    *,
    warmup_bars: int = 0,
    warmup_minutes: int | None = None,
    backend: str = "python",
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
    stream_id: str = "default",
    on_bar: Any = None,
    on_threshold_update: Any = None,
    multiplier: float = 1.0,
    volume_type: str = "base",
    session_policy: Literal["close", "carry", "discard"] = "close",
    session_estimator: Literal["persist", "reset"] = "persist",
) -> pd.DataFrame:
    """Build volume bars from a DataFrame of minute OHLCV candles.

    Parameters
    ----------
    minutes_df : pd.DataFrame
        Minute candles with columns mapped via *schema*.
    threshold : float
        Volume threshold per bar. Must be positive.
    schema : MinuteSchemaMapping, optional
        Defaults to standard column names.
    warmup_bars : int, default 0
        Number of initial bars to discard.
    backend : str, default ``"python"``
        ``"python"`` or ``"numba"``.
    watermark : str or None, default ``"timestamp"``
        Dedup key for idempotent resume.
    strict_ordering : bool, default False
        Raise when timestamps are non-monotonic.
    calendar : TradingCalendar, optional
    stream_id : str, default ``"default"``
    on_bar : callable or None
    on_threshold_update : callable or None

    Returns
    -------
    pd.DataFrame
        Completed bars with the standard output schema. ``num_ticks`` counts
        minutes (not trades).
    """
    ctor = MinuteVolumeBarConstructor(
        threshold,
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
        session_policy=session_policy,
        session_estimator=session_estimator,
    )
    return ctor.batch(minutes_df)


def compute_dollar_bars_from_minutes(
    minutes_df: pd.DataFrame,
    threshold: float,
    schema: MinuteSchemaMapping | None = None,
    *,
    warmup_bars: int = 0,
    warmup_minutes: int | None = None,
    backend: str = "python",
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
    stream_id: str = "default",
    on_bar: Any = None,
    on_threshold_update: Any = None,
    multiplier: float = 1.0,
    volume_type: str = "base",
    session_policy: Literal["close", "carry", "discard"] = "close",
    session_estimator: Literal["persist", "reset"] = "persist",
) -> pd.DataFrame:
    """Build dollar (notional) bars from a DataFrame of minute OHLCV candles.

    The notional metric uses ``close × volume`` per minute as the price proxy.
    Parameters match :func:`compute_volume_bars_from_minutes`.
    """
    ctor = MinuteDollarBarConstructor(
        threshold,
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
        session_policy=session_policy,
        session_estimator=session_estimator,
    )
    return ctor.batch(minutes_df)


MinuteBarRegistry.register(
    "minute_volume", MinuteVolumeBarConstructor, batch_fn=compute_volume_bars_from_minutes
)
MinuteBarRegistry.register(
    "minute_dollar", MinuteDollarBarConstructor, batch_fn=compute_dollar_bars_from_minutes
)
