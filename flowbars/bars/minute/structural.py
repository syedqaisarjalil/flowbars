"""Structural minute bars: volatility, relative range, renko, hybrid.

These bar types exist only on the minute path.  They use static thresholds
(calibrate with :class:`~flowbars.thresholds.MinuteCalibrationHelper`).
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from flowbars.bars.minute.accumulators import (
    MinuteHybridAccumulator,
    MinuteRangeAccumulator,
    MinuteRenkoAccumulator,
    MinuteVolatilityAccumulator,
)
from flowbars.bars.minute.engine import BaseMinuteBarConstructor
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.schema import MinuteSchemaMapping
from flowbars.thresholds import StaticThresholdEstimator


class MinuteVolatilityBarConstructor(BaseMinuteBarConstructor):
    """Minute volatility bars: cumulative absolute close-to-close returns."""

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
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        super().__init__(
            accumulator=MinuteVolatilityAccumulator(bar_type="volatility"),
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
    ) -> MinuteVolatilityBarConstructor:
        acc = MinuteVolatilityAccumulator(bar_type=state["bar_type"])
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(state, acc, est, calendar, schema)  # type: ignore[return-value]


class MinuteRangeBarConstructor(BaseMinuteBarConstructor):
    """Minute relative-range bars: cumulative ``(high − low) / close``."""

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
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        super().__init__(
            accumulator=MinuteRangeAccumulator(bar_type="range"),
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
    ) -> MinuteRangeBarConstructor:
        acc = MinuteRangeAccumulator(bar_type=state["bar_type"])
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(state, acc, est, calendar, schema)  # type: ignore[return-value]


class MinuteRenkoBarConstructor(BaseMinuteBarConstructor):
    """Minute renko-displacement bars: ``|close − anchor| / anchor``."""

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
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        super().__init__(
            accumulator=MinuteRenkoAccumulator(bar_type="renko"),
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
    ) -> MinuteRenkoBarConstructor:
        acc = MinuteRenkoAccumulator(bar_type=state["bar_type"])
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(state, acc, est, calendar, schema)  # type: ignore[return-value]


class MinuteHybridBarConstructor(BaseMinuteBarConstructor):
    """Minute hybrid bars: close when both dollar and volatility targets are met.

    The engine's static threshold carries the dollar target for API symmetry;
    both targets are held on the accumulator and persisted in its state.
    """

    approximate = True

    def __init__(
        self,
        dollar_threshold: float,
        volatility_threshold: float,
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
        if dollar_threshold < 0:
            raise ValueError(f"dollar_threshold must be non-negative, got {dollar_threshold}")
        if volatility_threshold < 0:
            raise ValueError(
                f"volatility_threshold must be non-negative, got {volatility_threshold}"
            )
        super().__init__(
            accumulator=MinuteHybridAccumulator(
                bar_type="hybrid",
                dollar_threshold=dollar_threshold,
                volatility_threshold=volatility_threshold,
            ),
            threshold_estimator=StaticThresholdEstimator(threshold=dollar_threshold),
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
    ) -> MinuteHybridBarConstructor:
        acc_state = state["accumulator"]
        acc = MinuteHybridAccumulator(
            bar_type=state["bar_type"],
            dollar_threshold=acc_state.get("dollar_threshold", 0.0),
            volatility_threshold=acc_state.get("volatility_threshold", 0.0),
        )
        est = StaticThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(state, acc, est, calendar, schema)  # type: ignore[return-value]


def compute_volatility_bars_from_minutes(
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
    """Build volatility bars from minute OHLCV candles.

    The per-minute metric is ``|close − prev_close| / prev_close`` (0 for the
    first minute of the stream or when ``prev_close == 0``).
    """
    ctor = MinuteVolatilityBarConstructor(
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


def compute_range_bars_from_minutes(
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
    """Build relative-range bars from minute OHLCV candles.

    The per-minute metric is ``(high − low) / close`` (0 when ``close == 0``).
    """
    ctor = MinuteRangeBarConstructor(
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


def compute_renko_bars_from_minutes(
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
    """Build renko-displacement bars from minute OHLCV candles.

    The anchor is the open of the bar's first minute; displacement
    ``|close − anchor| / anchor`` is not additive, so no overflow rolls.
    """
    ctor = MinuteRenkoBarConstructor(
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


def compute_hybrid_bars_from_minutes(
    minutes_df: pd.DataFrame,
    dollar_threshold: float,
    volatility_threshold: float,
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
    """Build hybrid bars from minute OHLCV candles.

    Closes when cumulative notional (``close × volume``) and cumulative
    volatility both meet their targets; both scalar excesses roll.
    """
    ctor = MinuteHybridBarConstructor(
        dollar_threshold,
        volatility_threshold,
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
    "minute_volatility",
    MinuteVolatilityBarConstructor,
    batch_fn=compute_volatility_bars_from_minutes,
)
MinuteBarRegistry.register(
    "minute_range",
    MinuteRangeBarConstructor,
    batch_fn=compute_range_bars_from_minutes,
)
MinuteBarRegistry.register(
    "minute_renko",
    MinuteRenkoBarConstructor,
    batch_fn=compute_renko_bars_from_minutes,
)
MinuteBarRegistry.register(
    "minute_hybrid",
    MinuteHybridBarConstructor,
    batch_fn=compute_hybrid_bars_from_minutes,
)
