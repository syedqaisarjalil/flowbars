"""flowbars — production-grade bar construction for financial ML.

Standard bars: time, tick, volume, dollar.
Information-driven bars: imbalance, run (tick, volume, dollar variants).
"""

__version__ = "0.1.0"

# Public API — batch convenience functions (populated as bar types are registered)
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.information import (  # noqa: F401 — triggers @register_bar side-effects
    ImbalanceDollarBarConstructor,
    ImbalanceTickBarConstructor,
    ImbalanceVolumeBarConstructor,
    RunDollarBarConstructor,
    RunTickBarConstructor,
    RunVolumeBarConstructor,
    compute_imbalance_dollar_bars,
    compute_imbalance_tick_bars,
    compute_imbalance_volume_bars,
    compute_run_dollar_bars,
    compute_run_tick_bars,
    compute_run_volume_bars,
)
from flowbars.bars.registry import BarRegistry
from flowbars.bars.standard import (  # noqa: F401 — triggers @register_bar side-effects
    DollarBarConstructor,
    TickBarConstructor,
    TimeBarConstructor,
    VolumeBarConstructor,
    compute_dollar_bars,
    compute_tick_bars,
    compute_time_bars,
    compute_volume_bars,
)
from flowbars.calendars import (
    ContinuousCalendar,
    ExchangeCalendar,
    SessionCalendar,
    TradingCalendar,
    WeekdayCalendar,
)

# Re-export the bar quality report
from flowbars.quality import bar_quality_report

# Sample data loader
from flowbars.sample_data import load_sample_data
from flowbars.schema import MinuteSchemaMapping, SchemaMapping
from flowbars.thresholds import (
    EWMAThresholdEstimator,
    StaticCalibrationHelper,
    StaticThresholdEstimator,
    ThresholdEstimator,
)

__all__ = [
    "BarRegistry",
    "BaseBarConstructor",
    "ContinuousCalendar",
    "DollarBarConstructor",
    "ExchangeCalendar",
    "EWMAThresholdEstimator",
    "ImbalanceDollarBarConstructor",
    "ImbalanceTickBarConstructor",
    "ImbalanceVolumeBarConstructor",
    "MinuteSchemaMapping",
    "RunDollarBarConstructor",
    "RunTickBarConstructor",
    "RunVolumeBarConstructor",
    "SchemaMapping",
    "SessionCalendar",
    "StaticCalibrationHelper",
    "StaticThresholdEstimator",
    "ThresholdEstimator",
    "TickBarConstructor",
    "TimeBarConstructor",
    "TradingCalendar",
    "VolumeBarConstructor",
    "WeekdayCalendar",
    "bar_quality_report",
    "load_sample_data",
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
