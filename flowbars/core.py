"""Core types and shared infrastructure for bar construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TickInfo:
    """A single tick/trade, normalized to the internal representation.

    This is what flows through the bar construction pipeline regardless of
    whether the input was pandas, polars, or a plain dict.
    """

    timestamp: int  # Unix ms (UTC)
    price: float
    volume: float
    side: float | None = None  # +1.0 (buy), −1.0 (sell), NaN, or None if not yet derived
    watermark: int | None = None  # optional monotonic dedup key; None → use timestamp


@dataclass
class MinuteInfo:
    """A single minute OHLCV candle, normalized to the internal representation.

    Used by the minute-OHLCV → bars path (``flowbars.bars.minute``). Unlike
    :class:`TickInfo` (a single trade print), a minute already carries its own
    open/high/low/close, so the bar accumulator merges those directly instead
    of deriving them from a single price.
    """

    timestamp: int  # Unix ms (UTC) — the minute's open time
    open: float
    high: float
    low: float
    close: float
    volume: float
    side: float | None = None  # +1.0, −1.0, or None if not derived (first minute)


@dataclass
class Bar:
    """A completed bar — the output of bar construction.

    Fields match the output bar schema defined in the spec.
    """

    bar_id: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    dollar_value: float
    vwap: float
    num_ticks: int
    open_ts: int
    close_ts: int
    bar_type: str


class FlowbarsError(Exception):
    """Base exception for all flowbars errors."""


class SchemaError(FlowbarsError):
    """Schema mapping is missing required keys or has invalid values."""


class ThresholdError(FlowbarsError):
    """Threshold is invalid (zero, negative, or otherwise unusable)."""


class StateValidationError(FlowbarsError):
    """Saved state is invalid, mismatched, or from an incompatible version."""


class TickDataError(FlowbarsError):
    """A tick contains invalid data (NaN, Inf, negative price/volume, etc.)."""
