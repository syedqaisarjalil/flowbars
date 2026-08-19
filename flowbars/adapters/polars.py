"""Polars adapter — mirror functions for every ``compute_*_bars()``.

These functions accept ``polars.DataFrame`` as input and return
``polars.DataFrame`` as output.  Internally they convert
polars → pandas, delegate to the core functions, then convert
the result back to polars.

Polars uses Apache Arrow under the hood, so ``to_pandas()`` /
``from_pandas()`` are near-zero-copy when pyarrow is installed.

Batch-only — no streaming equivalent for polars.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl  # type: ignore

# Import pandas compute functions with _pd_ prefix to avoid
# shadowing the polars functions defined in this module.
from flowbars.bars.information import (
    compute_imbalance_dollar_bars as _pd_compute_imbalance_dollar_bars,
)
from flowbars.bars.information import (
    compute_imbalance_tick_bars as _pd_compute_imbalance_tick_bars,
)
from flowbars.bars.information import (
    compute_imbalance_volume_bars as _pd_compute_imbalance_volume_bars,
)
from flowbars.bars.information import (
    compute_run_dollar_bars as _pd_compute_run_dollar_bars,
)
from flowbars.bars.information import (
    compute_run_tick_bars as _pd_compute_run_tick_bars,
)
from flowbars.bars.information import (
    compute_run_volume_bars as _pd_compute_run_volume_bars,
)
from flowbars.bars.standard import (
    compute_dollar_bars as _pd_compute_dollar_bars,
)
from flowbars.bars.standard import (
    compute_tick_bars as _pd_compute_tick_bars,
)
from flowbars.bars.standard import (
    compute_time_bars as _pd_compute_time_bars,
)
from flowbars.bars.standard import (
    compute_volume_bars as _pd_compute_volume_bars,
)
from flowbars.calendars import TradingCalendar
from flowbars.schema import SchemaMapping

if TYPE_CHECKING:
    pass

__all__ = [
    "compute_dollar_bars",
    "compute_imbalance_dollar_bars",
    "compute_imbalance_tick_bars",
    "compute_imbalance_volume_bars",
    "compute_run_dollar_bars",
    "compute_run_tick_bars",
    "compute_run_volume_bars",
    "compute_tick_bars",
    "compute_time_bars",
    "compute_volume_bars",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Standard bars
# ═══════════════════════════════════════════════════════════════════════════════


def compute_tick_bars(
    ticks_df: pl.DataFrame,
    threshold: int,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Build tick bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
        Tick-level data with user-mapped columns.
    threshold : int
        Number of ticks per bar.  Must be positive.
    schema : SchemaMapping, optional
        Maps user column names to internal keys.
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_tick_bars(
        pd_ticks,
        threshold=threshold,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


def compute_volume_bars(
    ticks_df: pl.DataFrame,
    threshold: float,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Build volume bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
        Tick-level data with user-mapped columns.
    threshold : float
        Volume threshold per bar.
    schema : SchemaMapping, optional
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_volume_bars(
        pd_ticks,
        threshold=threshold,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


def compute_dollar_bars(
    ticks_df: pl.DataFrame,
    threshold: float,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Build dollar bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
        Tick-level data with user-mapped columns.
    threshold : float
        Notional-value threshold per bar (e.g. 1_000_000.0 for $1M bars).
    schema : SchemaMapping, optional
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_dollar_bars(
        pd_ticks,
        threshold=threshold,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


def compute_time_bars(
    ticks_df: pl.DataFrame,
    interval_ms: int,
    anchor: str = "clock",
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Build time bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
        Tick-level data with user-mapped columns.
    interval_ms : int
        Bar interval in milliseconds.
    anchor : str, default ``"clock"``
        ``"clock"`` or ``"first_tick"``.
    schema : SchemaMapping, optional
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_time_bars(
        pd_ticks,
        interval_ms=interval_ms,
        anchor=anchor,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


# ═══════════════════════════════════════════════════════════════════════════════
# Information-driven bars — imbalance
# ═══════════════════════════════════════════════════════════════════════════════


def compute_imbalance_tick_bars(
    ticks_df: pl.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Build tick-imbalance bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
        Tick-level data with user-mapped columns.
    span : float, default 20.0
        EWMA span for the adaptive threshold.
    halflife : float or None, default None
        EWMA halflife (takes precedence over *span*).
    initial_ewa_t : float, default 1.0
        Initial seed for expected tick count per bar.
    initial_ewa_proportion : float, default 0.5
        Initial seed for expected imbalance proportion.
    warmup_bars : int, default 0
        Number of initial bars to discard.
    schema : SchemaMapping, optional
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_imbalance_tick_bars(
        pd_ticks,
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


def compute_imbalance_volume_bars(
    ticks_df: pl.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Build volume-imbalance bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
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
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_imbalance_volume_bars(
        pd_ticks,
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


def compute_imbalance_dollar_bars(
    ticks_df: pl.DataFrame,
    span: float = 20.0,
    halflife: float | None = None,
    initial_ewa_t: float = 1.0,
    initial_ewa_proportion: float = 0.5,
    warmup_bars: int = 0,
    schema: SchemaMapping | None = None,
    watermark: str | None = "timestamp",
    strict_ordering: bool = False,
    calendar: TradingCalendar | None = None,
) -> pl.DataFrame:
    """Build dollar-imbalance bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
        Tick-level data with user-mapped columns.
    span : float, default 20.0
        EWMA span for the adaptive threshold.
    halflife : float or None, default None
        EWMA halflife (takes precedence over *span*).
    initial_ewa_t : float, default 1.0
        Initial seed for expected dollar value per bar.
    initial_ewa_proportion : float, default 0.5
        Initial seed for expected imbalance proportion.
    warmup_bars : int, default 0
        Number of initial bars to discard.
    schema : SchemaMapping, optional
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_imbalance_dollar_bars(
        pd_ticks,
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


# ═══════════════════════════════════════════════════════════════════════════════
# Information-driven bars — run
# ═══════════════════════════════════════════════════════════════════════════════


def compute_run_tick_bars(
    ticks_df: pl.DataFrame,
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
) -> pl.DataFrame:
    """Build tick-run bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
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
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_run_tick_bars(
        pd_ticks,
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        min_run_length=min_run_length,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


def compute_run_volume_bars(
    ticks_df: pl.DataFrame,
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
) -> pl.DataFrame:
    """Build volume-run bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
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
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_run_volume_bars(
        pd_ticks,
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        min_run_length=min_run_length,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)


def compute_run_dollar_bars(
    ticks_df: pl.DataFrame,
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
) -> pl.DataFrame:
    """Build dollar-run bars from a Polars DataFrame of ticks.

    Parameters
    ----------
    ticks_df : pl.DataFrame
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
    calendar : TradingCalendar, optional

    Returns
    -------
    pl.DataFrame
        Completed bars.
    """
    pd_ticks = ticks_df.to_pandas()
    pd_bars = _pd_compute_run_dollar_bars(
        pd_ticks,
        span=span,
        halflife=halflife,
        initial_ewa_t=initial_ewa_t,
        initial_ewa_proportion=initial_ewa_proportion,
        warmup_bars=warmup_bars,
        min_run_length=min_run_length,
        schema=schema,
        calendar=calendar,
        watermark=watermark,
        strict_ordering=strict_ordering,
    )
    return pl.from_pandas(pd_bars)
