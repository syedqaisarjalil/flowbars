"""Minute-OHLCV → bars path.

Builds non-time bars from minute OHLCV candles instead of raw ticks.  Same
output bar schema and the same orchestration engine as the tick path (batch,
streaming ``update()``, state save/resume, callbacks, warmup, watermark dedup,
strict ordering, calendars), plus a numba batch backend.

Design decisions are specified in the repo's ``SPEC.md`` under
"Minute-OHLCV → bars path — design".  Key points:

* **Direct OHLCV accumulation** (not pseudo-tick resampling).
* **Whole-minute attribution** — the crossing minute is consumed whole; only
  the scalar metric excess rolls into the next bar (renko has no overflow).
* **Minute tick rule** for signs — ``sign(close_i − close_{i−1})``, first
  minute ``NaN`` (reused from :func:`flowbars.tick_rule.resolve_tick_signs`).
* ``num_ticks`` counts **minutes**; ``dollar_value``/``vwap`` use
  ``close × volume`` as the notional proxy.

Bar types: volume, dollar, imbalance/run (volume, dollar), plus the
minute-only volatility, relative-range, renko-displacement, and hybrid types.
"""

from __future__ import annotations

from flowbars.bars.minute.accumulators import (
    MinuteDollarAccumulator,
    MinuteHybridAccumulator,
    MinuteImbalanceAccumulator,
    MinuteRangeAccumulator,
    MinuteRenkoAccumulator,
    MinuteRunAccumulator,
    MinuteVolatilityAccumulator,
    MinuteVolumeAccumulator,
)
from flowbars.bars.minute.engine import BaseMinuteBarConstructor

# Registration side-effects: importing these modules registers every type.
from flowbars.bars.minute.information import (
    MinuteImbalanceDollarBarConstructor,
    MinuteImbalanceVolumeBarConstructor,
    MinuteRunDollarBarConstructor,
    MinuteRunVolumeBarConstructor,
    compute_imbalance_dollar_bars_from_minutes,
    compute_imbalance_volume_bars_from_minutes,
    compute_run_dollar_bars_from_minutes,
    compute_run_volume_bars_from_minutes,
)
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.bars.minute.standard import (
    MinuteDollarBarConstructor,
    MinuteVolumeBarConstructor,
    compute_dollar_bars_from_minutes,
    compute_volume_bars_from_minutes,
)
from flowbars.bars.minute.structural import (
    MinuteHybridBarConstructor,
    MinuteRangeBarConstructor,
    MinuteRenkoBarConstructor,
    MinuteVolatilityBarConstructor,
    compute_hybrid_bars_from_minutes,
    compute_range_bars_from_minutes,
    compute_renko_bars_from_minutes,
    compute_volatility_bars_from_minutes,
)
from flowbars.thresholds import MinuteCalibrationHelper

__all__ = [
    "BaseMinuteBarConstructor",
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
    "MinuteVolatilityAccumulator",
    "MinuteVolatilityBarConstructor",
    "MinuteVolumeAccumulator",
    "MinuteVolumeBarConstructor",
    "compute_dollar_bars_from_minutes",
    "compute_hybrid_bars_from_minutes",
    "compute_imbalance_dollar_bars_from_minutes",
    "compute_imbalance_volume_bars_from_minutes",
    "compute_range_bars_from_minutes",
    "compute_renko_bars_from_minutes",
    "compute_run_dollar_bars_from_minutes",
    "compute_run_volume_bars_from_minutes",
    "compute_volatility_bars_from_minutes",
    "compute_volume_bars_from_minutes",
]
