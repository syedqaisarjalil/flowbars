"""Information-driven minute bars: imbalance and run (volume, dollar variants).

Thresholds are the two-component EWMA from AFML §2.4, updated at bar close
from whole-minute bar statistics.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from flowbars.bars.minute.accumulators import (
    MinuteImbalanceAccumulator,
    MinuteRunAccumulator,
)
from flowbars.bars.minute.engine import BaseMinuteBarConstructor
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.calendars import TradingCalendar
from flowbars.core import Bar
from flowbars.schema import MinuteSchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator


class _MinuteEWMABarConstructor(BaseMinuteBarConstructor):
    """Shared constructor for minute EWMA-threshold bars."""

    _bar_family: str = "imbalance"

    def __init__(
        self,
        accumulator: Any,
        span: float,
        halflife: float | None,
        initial_ewa_t: float,
        initial_ewa_proportion: float,
        warmup_bars: int,
        warmup_minutes: int | None,
        calendar: TradingCalendar | None,
        schema: MinuteSchemaMapping | None,
        watermark: str | None,
        stream_id: str,
        strict_ordering: bool,
        backend: str,
        on_bar: Any,
        on_threshold_update: Any,
        multiplier: float,
        volume_type: str,
        session_policy: Literal["close", "carry", "discard"],
        session_estimator: Literal["persist", "reset"],
    ) -> None:
        estimator = EWMAThresholdEstimator(
            bar_family=self._bar_family,
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
            warmup_minutes=warmup_minutes,
            backend=backend,
            on_bar=on_bar,
            on_threshold_update=on_threshold_update,
            multiplier=multiplier,
            volume_type=volume_type,
            session_policy=session_policy,
            session_estimator=session_estimator,
        )


class MinuteImbalanceVolumeBarConstructor(_MinuteEWMABarConstructor):
    """Minute volume-imbalance bars (``sign × volume`` per minute)."""

    _bar_family = "imbalance"

    def __init__(
        self,
        span: float = 20.0,
        halflife: float | None = None,
        initial_ewa_t: float = 1.0,
        initial_ewa_proportion: float = 0.5,
        warmup_bars: int = 0,
        warmup_minutes: int | None = None,
        calendar: TradingCalendar | None = None,
        schema: MinuteSchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        backend: str = "python",
        on_bar: Any = None,
        on_threshold_update: Any = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
        session_policy: Literal["close", "carry", "discard"] = "close",
        session_estimator: Literal["persist", "reset"] = "persist",
    ) -> None:
        super().__init__(
            accumulator=MinuteImbalanceAccumulator(bar_type="imbalance_volume", metric="volume"),
            span=span,
            halflife=halflife,
            initial_ewa_t=initial_ewa_t,
            initial_ewa_proportion=initial_ewa_proportion,
            warmup_bars=warmup_bars,
            warmup_minutes=warmup_minutes,
            calendar=calendar,
            schema=schema,
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
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
    ) -> MinuteImbalanceVolumeBarConstructor:
        acc_state = state["accumulator"]
        acc = MinuteImbalanceAccumulator(bar_type=state["bar_type"], metric=acc_state["metric"])
        est = EWMAThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(
            state, acc, est, calendar, schema, verify_identity=verify_identity
        )  # type: ignore[return-value]


class MinuteImbalanceDollarBarConstructor(_MinuteEWMABarConstructor):
    """Minute dollar-imbalance bars (``sign × close × volume`` per minute)."""

    _bar_family = "imbalance"
    approximate = True

    def __init__(
        self,
        span: float = 20.0,
        halflife: float | None = None,
        initial_ewa_t: float = 1.0,
        initial_ewa_proportion: float = 0.5,
        warmup_bars: int = 0,
        warmup_minutes: int | None = None,
        calendar: TradingCalendar | None = None,
        schema: MinuteSchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        backend: str = "python",
        on_bar: Any = None,
        on_threshold_update: Any = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
        session_policy: Literal["close", "carry", "discard"] = "close",
        session_estimator: Literal["persist", "reset"] = "persist",
    ) -> None:
        super().__init__(
            accumulator=MinuteImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar"),
            span=span,
            halflife=halflife,
            initial_ewa_t=initial_ewa_t,
            initial_ewa_proportion=initial_ewa_proportion,
            warmup_bars=warmup_bars,
            warmup_minutes=warmup_minutes,
            calendar=calendar,
            schema=schema,
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
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
    ) -> MinuteImbalanceDollarBarConstructor:
        acc_state = state["accumulator"]
        acc = MinuteImbalanceAccumulator(bar_type=state["bar_type"], metric=acc_state["metric"])
        est = EWMAThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(
            state, acc, est, calendar, schema, verify_identity=verify_identity
        )  # type: ignore[return-value]


class _MinuteRunBarConstructor(_MinuteEWMABarConstructor):
    """Shared constructor for minute run bars with ``min_run_length`` filtering."""

    _bar_family = "run"

    def __init__(
        self,
        accumulator: Any,
        span: float,
        halflife: float | None,
        initial_ewa_t: float,
        initial_ewa_proportion: float,
        warmup_bars: int,
        warmup_minutes: int | None,
        min_run_length: int,
        calendar: TradingCalendar | None,
        schema: MinuteSchemaMapping | None,
        watermark: str | None,
        stream_id: str,
        strict_ordering: bool,
        backend: str,
        on_bar: Any,
        on_threshold_update: Any,
        multiplier: float,
        volume_type: str,
        session_policy: Literal["close", "carry", "discard"],
        session_estimator: Literal["persist", "reset"],
    ) -> None:
        if min_run_length < 0:
            raise ValueError(f"min_run_length must be non-negative, got {min_run_length}")
        self._min_run_length = min_run_length
        super().__init__(
            accumulator=accumulator,
            span=span,
            halflife=halflife,
            initial_ewa_t=initial_ewa_t,
            initial_ewa_proportion=initial_ewa_proportion,
            warmup_bars=warmup_bars,
            warmup_minutes=warmup_minutes,
            calendar=calendar,
            schema=schema,
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
            backend=backend,
            on_bar=on_bar,
            on_threshold_update=on_threshold_update,
            multiplier=multiplier,
            volume_type=volume_type,
            session_policy=session_policy,
            session_estimator=session_estimator,
        )

    def _should_return(self, bar: Bar) -> bool:
        """Filter bars below ``min_run_length`` minutes."""
        return bar.num_ticks >= self._min_run_length

    def _extra_params(self) -> dict[str, Any]:
        return {"min_run_length": self._min_run_length}

    def _extra_state(self) -> dict[str, Any]:
        state = super()._extra_state()
        state["min_run_length"] = self._min_run_length
        return state

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        super()._load_extra_state(state)
        self._min_run_length = int(state.get("min_run_length", 0))


class MinuteRunVolumeBarConstructor(_MinuteRunBarConstructor):
    """Minute volume-run bars."""

    def __init__(
        self,
        span: float = 20.0,
        halflife: float | None = None,
        initial_ewa_t: float = 1.0,
        initial_ewa_proportion: float = 0.5,
        warmup_bars: int = 0,
        warmup_minutes: int | None = None,
        min_run_length: int = 0,
        calendar: TradingCalendar | None = None,
        schema: MinuteSchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        backend: str = "python",
        on_bar: Any = None,
        on_threshold_update: Any = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
        session_policy: Literal["close", "carry", "discard"] = "close",
        session_estimator: Literal["persist", "reset"] = "persist",
    ) -> None:
        super().__init__(
            accumulator=MinuteRunAccumulator(bar_type="run_volume", metric="volume"),
            span=span,
            halflife=halflife,
            initial_ewa_t=initial_ewa_t,
            initial_ewa_proportion=initial_ewa_proportion,
            warmup_bars=warmup_bars,
            warmup_minutes=warmup_minutes,
            min_run_length=min_run_length,
            calendar=calendar,
            schema=schema,
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
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
    ) -> MinuteRunVolumeBarConstructor:
        acc_state = state["accumulator"]
        acc = MinuteRunAccumulator(bar_type=state["bar_type"], metric=acc_state["metric"])
        est = EWMAThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(
            state, acc, est, calendar, schema, verify_identity=verify_identity
        )  # type: ignore[return-value]


class MinuteRunDollarBarConstructor(_MinuteRunBarConstructor):
    """Minute dollar-run bars."""

    approximate = True

    def __init__(
        self,
        span: float = 20.0,
        halflife: float | None = None,
        initial_ewa_t: float = 1.0,
        initial_ewa_proportion: float = 0.5,
        warmup_bars: int = 0,
        warmup_minutes: int | None = None,
        min_run_length: int = 0,
        calendar: TradingCalendar | None = None,
        schema: MinuteSchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        backend: str = "python",
        on_bar: Any = None,
        on_threshold_update: Any = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
        session_policy: Literal["close", "carry", "discard"] = "close",
        session_estimator: Literal["persist", "reset"] = "persist",
    ) -> None:
        super().__init__(
            accumulator=MinuteRunAccumulator(bar_type="run_dollar", metric="dollar"),
            span=span,
            halflife=halflife,
            initial_ewa_t=initial_ewa_t,
            initial_ewa_proportion=initial_ewa_proportion,
            warmup_bars=warmup_bars,
            warmup_minutes=warmup_minutes,
            min_run_length=min_run_length,
            calendar=calendar,
            schema=schema,
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
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
    ) -> MinuteRunDollarBarConstructor:
        acc_state = state["accumulator"]
        acc = MinuteRunAccumulator(bar_type=state["bar_type"], metric=acc_state["metric"])
        est = EWMAThresholdEstimator.from_state(state["threshold_estimator"])
        return super().from_state(
            state, acc, est, calendar, schema, verify_identity=verify_identity
        )  # type: ignore[return-value]


def compute_imbalance_volume_bars_from_minutes(
    minutes_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    warmup_minutes: int | None = None,
    schema: MinuteSchemaMapping | None = None,
    *,
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
    """Build volume-imbalance bars from minute OHLCV candles."""
    ctor = MinuteImbalanceVolumeBarConstructor(
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        warmup_minutes=warmup_minutes,
        calendar=calendar,
        schema=schema,
        watermark=watermark,
        stream_id=stream_id,
        strict_ordering=strict_ordering,
        backend=backend,
        on_bar=on_bar,
        on_threshold_update=on_threshold_update,
        multiplier=multiplier,
        volume_type=volume_type,
        session_policy=session_policy,
        session_estimator=session_estimator,
    )
    return ctor.batch(minutes_df)


def compute_imbalance_dollar_bars_from_minutes(
    minutes_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    warmup_minutes: int | None = None,
    schema: MinuteSchemaMapping | None = None,
    *,
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
    """Build dollar-imbalance bars from minute OHLCV candles."""
    ctor = MinuteImbalanceDollarBarConstructor(
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        warmup_minutes=warmup_minutes,
        calendar=calendar,
        schema=schema,
        watermark=watermark,
        stream_id=stream_id,
        strict_ordering=strict_ordering,
        backend=backend,
        on_bar=on_bar,
        on_threshold_update=on_threshold_update,
        multiplier=multiplier,
        volume_type=volume_type,
        session_policy=session_policy,
        session_estimator=session_estimator,
    )
    return ctor.batch(minutes_df)


def compute_run_volume_bars_from_minutes(
    minutes_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    warmup_minutes: int | None = None,
    min_run_length: int = 0,
    schema: MinuteSchemaMapping | None = None,
    *,
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
    """Build volume-run bars from minute OHLCV candles.

    ``min_run_length`` is a minimum number of **minutes** per returned bar.
    """
    ctor = MinuteRunVolumeBarConstructor(
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        warmup_minutes=warmup_minutes,
        min_run_length=min_run_length,
        calendar=calendar,
        schema=schema,
        watermark=watermark,
        stream_id=stream_id,
        strict_ordering=strict_ordering,
        backend=backend,
        on_bar=on_bar,
        on_threshold_update=on_threshold_update,
        multiplier=multiplier,
        volume_type=volume_type,
        session_policy=session_policy,
        session_estimator=session_estimator,
    )
    return ctor.batch(minutes_df)


def compute_run_dollar_bars_from_minutes(
    minutes_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    warmup_minutes: int | None = None,
    min_run_length: int = 0,
    schema: MinuteSchemaMapping | None = None,
    *,
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
    """Build dollar-run bars from minute OHLCV candles."""
    ctor = MinuteRunDollarBarConstructor(
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        warmup_minutes=warmup_minutes,
        min_run_length=min_run_length,
        calendar=calendar,
        schema=schema,
        watermark=watermark,
        stream_id=stream_id,
        strict_ordering=strict_ordering,
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
    "minute_imbalance_volume",
    MinuteImbalanceVolumeBarConstructor,
    batch_fn=compute_imbalance_volume_bars_from_minutes,
)
MinuteBarRegistry.register(
    "minute_imbalance_dollar",
    MinuteImbalanceDollarBarConstructor,
    batch_fn=compute_imbalance_dollar_bars_from_minutes,
)
MinuteBarRegistry.register(
    "minute_run_volume",
    MinuteRunVolumeBarConstructor,
    batch_fn=compute_run_volume_bars_from_minutes,
)
MinuteBarRegistry.register(
    "minute_run_dollar",
    MinuteRunDollarBarConstructor,
    batch_fn=compute_run_dollar_bars_from_minutes,
)
