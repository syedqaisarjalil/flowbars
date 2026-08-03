"""flowbars — production-grade bar construction for financial ML.

Standard bars: time, tick, volume, dollar.
Information-driven bars: imbalance, run (tick, volume, dollar variants).
"""

__version__ = "0.1.0"

# Public API — batch convenience functions (populated as bar types are registered)
from flowbars.bars.registry import BarRegistry

# Re-export the bar quality report
from flowbars.quality import bar_quality_report
from flowbars.schema import SchemaMapping
from flowbars.calendars import ContinuousCalendar, SessionCalendar, TradingCalendar
from flowbars.thresholds import (
    EWMAThresholdEstimator,
    StaticCalibrationHelper,
    StaticThresholdEstimator,
    ThresholdEstimator,
)

__all__ = [
    "BarRegistry",
    "ContinuousCalendar",
    "EWMAThresholdEstimator",
    "SchemaMapping",
    "SessionCalendar",
    "StaticCalibrationHelper",
    "StaticThresholdEstimator",
    "ThresholdEstimator",
    "TradingCalendar",
    "bar_quality_report",
]
