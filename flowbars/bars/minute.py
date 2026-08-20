"""Minute-OHLCV → bars path.

Builds non-time bars (volume, dollar, and the volume/dollar variants of
imbalance and run) from minute OHLCV candles instead of raw ticks.

This is a **separate input path** from the tick constructors — same output bar
schema, coarser (minute-granularity) input. It is batch-only for the first cut:
no streaming, resume, calendars, or numba backend.

Design decisions are specified in the repo's ``SPEC.md`` under
"Minute-OHLCV → bars path — design". Key points:

* **Direct OHLCV accumulation** (not pseudo-tick resampling).
* **Whole-minute attribution** — the crossing minute is consumed whole; only
  the scalar metric excess rolls into the next bar.
* **Minute tick rule** for signs — ``sign(close_i − close_{i−1})``, first
  minute ``NaN`` (reused from :func:`flowbars.tick_rule.derive_tick_sign`).
* ``num_ticks`` counts **minutes**; ``dollar_value``/``vwap`` use
  ``close × volume`` as the notional proxy.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from flowbars.bars.accumulators import (
    BaseAccumulator,
    DollarAccumulator,
    ImbalanceAccumulator,
    RunAccumulator,
    VolumeAccumulator,
    _minute_metric,
)
from flowbars.core import Bar, MinuteInfo
from flowbars.schema import MinuteSchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator, StaticThresholdEstimator, ThresholdEstimator
from flowbars.tick_rule import resolve_tick_signs

# ═══════════════════════════════════════════════════════════════════════════════
# Minute accumulators — mirror the tick accumulators but ingest OHLCV minutes
# ═══════════════════════════════════════════════════════════════════════════════


class MinuteVolumeAccumulator(VolumeAccumulator):
    """Volume accumulator for minute input (metric = minute volume)."""

    def add_minute(self, minute: MinuteInfo) -> None:
        super().add_minute(minute)
        self._cum_volume += minute.volume


class MinuteDollarAccumulator(DollarAccumulator):
    """Dollar accumulator for minute input (metric = close × volume)."""

    def add_minute(self, minute: MinuteInfo) -> None:
        super().add_minute(minute)
        self._cum_dollar += minute.close * minute.volume


class MinuteImbalanceAccumulator(ImbalanceAccumulator):
    """Signed-imbalance accumulator for minute input (metric = volume or dollar)."""

    def add_minute(self, minute: MinuteInfo) -> None:
        super().add_minute(minute)
        if minute.side is not None and not math.isnan(minute.side):
            self._signed_imbalance += minute.side * _minute_metric(minute, self._metric)


class MinuteRunAccumulator(RunAccumulator):
    """Same-sign run accumulator for minute input (metric = volume or dollar)."""

    def add_minute(self, minute: MinuteInfo) -> None:
        super().add_minute(minute)
        side = minute.side if minute.side is not None else np.nan
        self._accumulate_run(side, _minute_metric(minute, self._metric))


# ═══════════════════════════════════════════════════════════════════════════════
# Shared batch loop + output conversion
# ═══════════════════════════════════════════════════════════════════════════════

_BAR_COLUMNS = [
    "bar_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_value",
    "vwap",
    "num_ticks",
    "open_ts",
    "close_ts",
    "bar_type",
]

_DEFAULT_MINUTE_SCHEMA = MinuteSchemaMapping(
    {
        "timestamp": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
)


def _bars_to_dataframe(bars: list[Bar]) -> pd.DataFrame:
    """Convert completed :class:`Bar` objects to the standard output DataFrame."""
    if not bars:
        return pd.DataFrame(columns=_BAR_COLUMNS)
    return pd.DataFrame(
        [
            {
                "bar_id": b.bar_id,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "dollar_value": b.dollar_value,
                "vwap": b.vwap,
                "num_ticks": b.num_ticks,
                "open_ts": b.open_ts,
                "close_ts": b.close_ts,
                "bar_type": b.bar_type,
            }
            for b in bars
        ]
    )


def _build_minute_bars(
    minutes_df: pd.DataFrame,
    schema: MinuteSchemaMapping,
    accumulator: BaseAccumulator,
    estimator: ThresholdEstimator,
    warmup_bars: int = 0,
    min_run_length: int = 0,
) -> pd.DataFrame:
    """Run the minute bar-construction loop and return the bar DataFrame."""
    timestamps, opens, highs, lows, closes, volumes, sides = schema.extract_arrays(minutes_df)

    # Minute tick rule: derive signs from closes (or use supplied side column).
    signs = resolve_tick_signs(closes, sides)

    bars: list[Bar] = []
    bars_emitted = 0
    n = len(timestamps)
    for i in range(n):
        side = float(signs[i]) if not np.isnan(signs[i]) else None
        accumulator.add_minute(
            MinuteInfo(
                timestamp=int(timestamps[i]),
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                volume=float(volumes[i]),
                side=side,
            )
        )

        threshold = estimator.current_threshold
        if accumulator.should_close(threshold):
            t_stat, proportion_stat = accumulator.get_close_stats()
            bar = accumulator.close(threshold)
            estimator.on_bar_close(t_stat, proportion_stat)
            bars_emitted += 1
            if bars_emitted > warmup_bars and bar.num_ticks >= min_run_length:
                bars.append(bar)

    return _bars_to_dataframe(bars)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — batch functions
# ═══════════════════════════════════════════════════════════════════════════════


def compute_volume_bars_from_minutes(
    minutes_df: pd.DataFrame,
    threshold: float,
    schema: MinuteSchemaMapping | None = None,
) -> pd.DataFrame:
    """Build volume bars from a DataFrame of minute OHLCV candles.

    Parameters
    ----------
    minutes_df : pd.DataFrame
        Minute candles with columns mapped via *schema* (``timestamp``,
        ``open``, ``high``, ``low``, ``close``, ``volume``).
    threshold : float
        Volume threshold per bar. Must be positive.
    schema : MinuteSchemaMapping, optional
        Defaults to standard column names (``timestamp``/``open``/``high``/
        ``low``/``close``/``volume``).

    Returns
    -------
    pd.DataFrame
        Completed bars with the standard output schema. ``num_ticks`` counts
        minutes (not trades).
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    accumulator = MinuteVolumeAccumulator()
    estimator = StaticThresholdEstimator(threshold=threshold)
    return _build_minute_bars(minutes_df, schema or _DEFAULT_MINUTE_SCHEMA, accumulator, estimator)


def compute_dollar_bars_from_minutes(
    minutes_df: pd.DataFrame,
    threshold: float,
    schema: MinuteSchemaMapping | None = None,
) -> pd.DataFrame:
    """Build dollar (notional) bars from a DataFrame of minute OHLCV candles.

    The notional metric uses ``close × volume`` per minute as the price proxy.

    Parameters
    ----------
    minutes_df : pd.DataFrame
        Minute candles with columns mapped via *schema*.
    threshold : float
        Notional-value threshold per bar. Must be positive.
    schema : MinuteSchemaMapping, optional

    Returns
    -------
    pd.DataFrame
        Completed bars with the standard output schema.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    accumulator = MinuteDollarAccumulator()
    estimator = StaticThresholdEstimator(threshold=threshold)
    return _build_minute_bars(minutes_df, schema or _DEFAULT_MINUTE_SCHEMA, accumulator, estimator)


def compute_imbalance_volume_bars_from_minutes(
    minutes_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    schema: MinuteSchemaMapping | None = None,
) -> pd.DataFrame:
    """Build volume-imbalance bars from a DataFrame of minute OHLCV candles.

    Parameters
    ----------
    minutes_df : pd.DataFrame
        Minute candles with columns mapped via *schema*.
    span : float, default 20.0
        EWMA span for the two-component adaptive threshold.
    halflife : float or None, default None
        EWMA halflife (takes precedence over *span*).
    initial_ewa_t : float, default 1.0
        Initial seed for expected volume per bar.
    initial_ewa_proportion : float, default 0.5
        Initial seed for expected imbalance proportion.
    warmup_bars : int, default 0
        Number of initial bars to discard while the EWMA converges.
    schema : MinuteSchemaMapping, optional

    Returns
    -------
    pd.DataFrame
        Completed bars with the standard output schema.
    """
    accumulator = MinuteImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
    estimator = EWMAThresholdEstimator(
        bar_family="imbalance",
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
    )
    return _build_minute_bars(
        minutes_df, schema or _DEFAULT_MINUTE_SCHEMA, accumulator, estimator, warmup_bars=warmup_bars
    )


def compute_imbalance_dollar_bars_from_minutes(
    minutes_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    schema: MinuteSchemaMapping | None = None,
) -> pd.DataFrame:
    """Build dollar-imbalance bars from a DataFrame of minute OHLCV candles.

    Parameters match :func:`compute_imbalance_volume_bars_from_minutes`; the
    imbalance metric is ``sign × close × volume`` per minute.
    """
    accumulator = MinuteImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
    estimator = EWMAThresholdEstimator(
        bar_family="imbalance",
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
    )
    return _build_minute_bars(
        minutes_df, schema or _DEFAULT_MINUTE_SCHEMA, accumulator, estimator, warmup_bars=warmup_bars
    )


def compute_run_volume_bars_from_minutes(
    minutes_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    min_run_length: int = 0,
    schema: MinuteSchemaMapping | None = None,
) -> pd.DataFrame:
    """Build volume-run bars from a DataFrame of minute OHLCV candles.

    Parameters
    ----------
    minutes_df : pd.DataFrame
        Minute candles with columns mapped via *schema*.
    span : float, default 20.0
        EWMA span for the two-component adaptive threshold.
    halflife : float or None, default None
        EWMA halflife (takes precedence over *span*).
    initial_ewa_t : float, default 1.0
        Initial seed for expected run volume per bar.
    initial_ewa_proportion : float, default 0.5
        Initial seed for expected buy-run proportion.
    warmup_bars : int, default 0
        Number of initial bars to discard.
    min_run_length : int, default 0
        Minimum number of **minutes** a run bar must contain to be returned.
        Default 0 — no filtering.
    schema : MinuteSchemaMapping, optional

    Returns
    -------
    pd.DataFrame
        Completed bars with the standard output schema.
    """
    if min_run_length < 0:
        raise ValueError(f"min_run_length must be non-negative, got {min_run_length}")
    accumulator = MinuteRunAccumulator(bar_type="run_volume", metric="volume")
    estimator = EWMAThresholdEstimator(
        bar_family="run",
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
    )
    return _build_minute_bars(
        minutes_df,
        schema or _DEFAULT_MINUTE_SCHEMA,
        accumulator,
        estimator,
        warmup_bars=warmup_bars,
        min_run_length=min_run_length,
    )


def compute_run_dollar_bars_from_minutes(
    minutes_df: pd.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    min_run_length: int = 0,
    schema: MinuteSchemaMapping | None = None,
) -> pd.DataFrame:
    """Build dollar-run bars from a DataFrame of minute OHLCV candles.

    Parameters match :func:`compute_run_volume_bars_from_minutes`; the run
    metric is ``sign × close × volume`` per minute.
    """
    if min_run_length < 0:
        raise ValueError(f"min_run_length must be non-negative, got {min_run_length}")
    accumulator = MinuteRunAccumulator(bar_type="run_dollar", metric="dollar")
    estimator = EWMAThresholdEstimator(
        bar_family="run",
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
    )
    return _build_minute_bars(
        minutes_df,
        schema or _DEFAULT_MINUTE_SCHEMA,
        accumulator,
        estimator,
        warmup_bars=warmup_bars,
        min_run_length=min_run_length,
    )
