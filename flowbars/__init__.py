"""flowbars — production-grade bar construction for financial ML.

Standard bars: time, tick, volume, dollar.
Information-driven bars: imbalance, run (tick, volume, dollar variants).
"""

# Public API — batch convenience functions (populated as bar types are registered)
from flowbars._version import __version__  # noqa: F401
from flowbars.bars.constructor import BaseBarConstructor, params_hash
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
from flowbars.bars.minute import (  # noqa: F401 — triggers minute registration
    BaseMinuteBarConstructor,
    MinuteBarRegistry,
    MinuteDollarAccumulator,
    MinuteDollarBarConstructor,
    MinuteHybridAccumulator,
    MinuteHybridBarConstructor,
    MinuteImbalanceAccumulator,
    MinuteImbalanceDollarBarConstructor,
    MinuteImbalanceVolumeBarConstructor,
    MinuteRangeAccumulator,
    MinuteRangeBarConstructor,
    MinuteRenkoAccumulator,
    MinuteRenkoBarConstructor,
    MinuteRunAccumulator,
    MinuteRunDollarBarConstructor,
    MinuteRunVolumeBarConstructor,
    MinuteVolatilityAccumulator,
    MinuteVolatilityBarConstructor,
    MinuteVolumeAccumulator,
    MinuteVolumeBarConstructor,
    compute_dollar_bars_from_minutes,
    compute_hybrid_bars_from_minutes,
    compute_imbalance_dollar_bars_from_minutes,
    compute_imbalance_volume_bars_from_minutes,
    compute_range_bars_from_minutes,
    compute_renko_bars_from_minutes,
    compute_run_dollar_bars_from_minutes,
    compute_run_volume_bars_from_minutes,
    compute_volatility_bars_from_minutes,
    compute_volume_bars_from_minutes,
)
from flowbars.bars.output import validate_bar_frame
from flowbars.bars.registry import BarRegistry
from flowbars.bars.spec import BarTypeSpec
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
from flowbars.core import BarContractError

# Re-export the bar quality report
from flowbars.quality import bar_quality_report

# Sample data loader
from flowbars.sample_data import load_sample_data
from flowbars.schema import MinuteSchemaMapping, SchemaMapping
from flowbars.thresholds import (
    EWMAThresholdEstimator,
    MinuteCalibrationHelper,
    StaticCalibrationHelper,
    StaticThresholdEstimator,
    ThresholdEstimator,
)

__all__ = [
    "BarContractError",
    "BarRegistry",
    "BarTypeSpec",
    "BaseBarConstructor",
    "BaseMinuteBarConstructor",
    "ContinuousCalendar",
    "DollarBarConstructor",
    "ExchangeCalendar",
    "EWMAThresholdEstimator",
    "ImbalanceDollarBarConstructor",
    "ImbalanceTickBarConstructor",
    "ImbalanceVolumeBarConstructor",
    "MinuteBarRegistry",
    "MinuteCalibrationHelper",
    "MinuteDollarAccumulator",
    "MinuteDollarBarConstructor",
    "MinuteHybridAccumulator",
    "MinuteHybridBarConstructor",
    "MinuteImbalanceAccumulator",
    "MinuteImbalanceDollarBarConstructor",
    "MinuteImbalanceVolumeBarConstructor",
    "MinuteRangeAccumulator",
    "MinuteRangeBarConstructor",
    "MinuteRenkoAccumulator",
    "MinuteRenkoBarConstructor",
    "MinuteRunAccumulator",
    "MinuteRunDollarBarConstructor",
    "MinuteRunVolumeBarConstructor",
    "MinuteSchemaMapping",
    "MinuteVolatilityAccumulator",
    "MinuteVolatilityBarConstructor",
    "MinuteVolumeAccumulator",
    "MinuteVolumeBarConstructor",
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
    "params_hash",
    "validate_bar_frame",
    "compute_dollar_bars",
    "compute_dollar_bars_from_minutes",
    "compute_hybrid_bars_from_minutes",
    "compute_imbalance_dollar_bars",
    "compute_imbalance_dollar_bars_from_minutes",
    "compute_imbalance_tick_bars",
    "compute_imbalance_volume_bars",
    "compute_imbalance_volume_bars_from_minutes",
    "compute_range_bars_from_minutes",
    "compute_renko_bars_from_minutes",
    "compute_run_dollar_bars",
    "compute_run_dollar_bars_from_minutes",
    "compute_run_tick_bars",
    "compute_run_volume_bars",
    "compute_run_volume_bars_from_minutes",
    "compute_tick_bars",
    "compute_time_bars",
    "compute_volatility_bars_from_minutes",
    "compute_volume_bars",
    "compute_volume_bars_from_minutes",
]
