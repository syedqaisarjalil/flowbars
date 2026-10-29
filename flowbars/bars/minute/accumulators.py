"""Accumulators for the minute-OHLCV → bars path.

The first four mirror the tick accumulators but ingest whole-minute candles.
The last four implement the minute-only structural bar types (volatility,
relative range, renko displacement, hybrid).  All state values are
JSON-serializable; prices carried across bars use ``None`` (never NaN).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from flowbars.bars.accumulators import (
    BaseAccumulator,
    DollarAccumulator,
    ImbalanceAccumulator,
    RunAccumulator,
    VolumeAccumulator,
    _minute_metric,
)
from flowbars.core import Bar, MinuteInfo


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
# Structural (minute-only) accumulators
# ═══════════════════════════════════════════════════════════════════════════════


class MinuteVolatilityAccumulator(BaseAccumulator):
    """Closes when cumulative absolute close-to-close returns cross a threshold.

    Per-minute metric: ``|close − prev_close| / prev_close`` (0 for the first
    minute of the stream or when ``prev_close == 0``).  ``prev_close`` carries
    across bars; a session boundary resets it so an overnight gap contributes
    nothing.  Scalar overflow rolls into the next bar.
    """

    def __init__(self, bar_type: str = "volatility") -> None:
        super().__init__(bar_type)
        self._cum_metric = 0.0
        self._prev_close: float | None = None

    def add_minute(self, minute: MinuteInfo) -> None:
        super().add_minute(minute)
        if self._prev_close is None or self._prev_close == 0.0:
            metric = 0.0
        else:
            metric = abs(minute.close - self._prev_close) / self._prev_close
        self._cum_metric += metric
        self._prev_close = minute.close

    def should_close(self, threshold: float) -> bool:
        return self._cum_metric >= threshold

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        self._cum_metric = max(0.0, self._cum_metric - threshold)
        return bar

    def get_close_stats(self) -> tuple[float, float]:
        return (self._cum_metric, 0.0)

    def on_session_boundary(self) -> None:
        self._prev_close = None

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["cum_metric"] = self._cum_metric
        state["prev_close"] = self._prev_close
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._cum_metric = state.get("cum_metric", 0.0)
        self._prev_close = state.get("prev_close")


class MinuteRangeAccumulator(BaseAccumulator):
    """Closes when cumulative relative range crosses a threshold.

    Per-minute metric: ``(high − low) / close`` (0 when ``close == 0``).
    Scalar overflow rolls into the next bar.
    """

    def __init__(self, bar_type: str = "range") -> None:
        super().__init__(bar_type)
        self._cum_metric = 0.0

    def add_minute(self, minute: MinuteInfo) -> None:
        super().add_minute(minute)
        if minute.close == 0.0:
            metric = 0.0
        else:
            metric = (minute.high - minute.low) / minute.close
        self._cum_metric += metric

    def should_close(self, threshold: float) -> bool:
        return self._cum_metric >= threshold

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        self._cum_metric = max(0.0, self._cum_metric - threshold)
        return bar

    def get_close_stats(self) -> tuple[float, float]:
        return (self._cum_metric, 0.0)

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["cum_metric"] = self._cum_metric
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._cum_metric = state.get("cum_metric", 0.0)


class MinuteRenkoAccumulator(BaseAccumulator):
    """Closes when price displacement from the bar's anchor crosses a threshold.

    The anchor is the **open of the bar's first minute**, and that first
    minute is evaluated against it (so a one-minute renko bar is possible).
    Per-minute displacement: ``|close − anchor| / anchor`` (0 when the anchor
    is non-positive).  Displacement is not additive, so there is **no
    overflow** — the anchor resets on close.
    """

    def __init__(self, bar_type: str = "renko") -> None:
        super().__init__(bar_type)
        self._anchor: float | None = None
        self._displacement = 0.0

    def add_minute(self, minute: MinuteInfo) -> None:
        super().add_minute(minute)
        if self._anchor is None:
            self._anchor = minute.open
        if self._anchor <= 0.0:
            self._displacement = 0.0
        else:
            self._displacement = abs(minute.close - self._anchor) / self._anchor

    def should_close(self, threshold: float) -> bool:
        return self._displacement >= threshold

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        self._anchor = None
        self._displacement = 0.0
        return bar

    def get_close_stats(self) -> tuple[float, float]:
        return (self._displacement, 0.0)

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["anchor"] = self._anchor
        state["displacement"] = self._displacement
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._anchor = state.get("anchor")
        self._displacement = state.get("displacement", 0.0)


class MinuteHybridAccumulator(BaseAccumulator):
    """Closes when **both** cumulative notional and cumulative volatility targets are met.

    Two running metrics: ``Σ close × volume`` and the volatility metric
    ``Σ |close − prev_close| / prev_close`` (0 for the first minute or when
    ``prev_close == 0``).  Both thresholds live on the accumulator.  At close,
    both scalar excesses roll into the next bar (the scalar-metric rule
    applied twice).  A session boundary resets ``prev_close``.
    """

    def __init__(
        self,
        bar_type: str = "hybrid",
        dollar_threshold: float = 0.0,
        volatility_threshold: float = 0.0,
    ) -> None:
        super().__init__(bar_type)
        if dollar_threshold < 0:
            raise ValueError(f"dollar_threshold must be non-negative, got {dollar_threshold}")
        if volatility_threshold < 0:
            raise ValueError(
                f"volatility_threshold must be non-negative, got {volatility_threshold}"
            )
        self._dollar_threshold = dollar_threshold
        self._volatility_threshold = volatility_threshold
        self._cum_dollar = 0.0
        self._cum_vol = 0.0
        self._prev_close: float | None = None

    def add_minute(self, minute: MinuteInfo) -> None:
        super().add_minute(minute)
        self._cum_dollar += minute.close * minute.volume
        if self._prev_close is None or self._prev_close == 0.0:
            metric = 0.0
        else:
            metric = abs(minute.close - self._prev_close) / self._prev_close
        self._cum_vol += metric
        self._prev_close = minute.close

    def should_close(self, threshold: float) -> bool:
        # ``threshold`` (the engine's static threshold) is the dollar target
        # for API symmetry; both targets are read from the accumulator.
        return (
            self._cum_dollar >= self._dollar_threshold
            and self._cum_vol >= self._volatility_threshold
        )

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        self._cum_dollar = max(0.0, self._cum_dollar - self._dollar_threshold)
        self._cum_vol = max(0.0, self._cum_vol - self._volatility_threshold)
        return bar

    def get_close_stats(self) -> tuple[float, float]:
        return (self._cum_dollar, 0.0)

    def on_session_boundary(self) -> None:
        self._prev_close = None

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["cum_dollar"] = self._cum_dollar
        state["cum_vol"] = self._cum_vol
        state["prev_close"] = self._prev_close
        state["dollar_threshold"] = self._dollar_threshold
        state["volatility_threshold"] = self._volatility_threshold
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._cum_dollar = state.get("cum_dollar", 0.0)
        self._cum_vol = state.get("cum_vol", 0.0)
        self._prev_close = state.get("prev_close")
        self._dollar_threshold = state.get("dollar_threshold", 0.0)
        self._volatility_threshold = state.get("volatility_threshold", 0.0)
