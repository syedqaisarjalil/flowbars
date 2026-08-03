"""Core types and shared infrastructure for bar construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TickInfo:
    """A single tick/trade, normalized to the internal representation.

    This is what flows through the bar construction pipeline regardless of
    whether the input was pandas, polars, or a plain dict.
    """

    timestamp: int  # Unix ms (UTC)
    price: float
    volume: float
    side: Optional[int] = None  # +1 (buy), −1 (sell), or None if not yet derived


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
