"""Core types and shared infrastructure for bar construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

# What the ``volume`` column means.  Metadata only — it never changes the
# arithmetic, only how downstream consumers interpret the column.
VOLUME_TYPES: tuple[str, ...] = ("base", "quote", "shares", "contracts", "tick")


def validate_volume_semantics(multiplier: float, volume_type: str) -> tuple[float, str]:
    """Validate and normalize a contract multiplier and volume-unit label.

    ``multiplier`` converts traded units to notional value
    (``dollar_value = Σ price × volume × multiplier``); it must be a
    positive finite number.  ``volume_type`` labels what the ``volume``
    column means (``base``, ``quote``, ``shares``, ``contracts``, or
    ``tick``).
    """
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
        raise ValueError(f"multiplier must be a number, got {multiplier!r}")
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"multiplier must be positive and finite, got {multiplier}")
    if volume_type not in VOLUME_TYPES:
        raise ValueError(f"volume_type must be one of {VOLUME_TYPES}, got {volume_type!r}")
    return float(multiplier), volume_type


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
    watermark: int | None = None  # optional monotonic dedup key; None → use timestamp


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
    partial: bool = False  # True for bars force-closed at a session end


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


class BarContractError(FlowbarsError):
    """A bar frame violates the public output contract."""
