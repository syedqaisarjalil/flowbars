"""Bar accumulators — track running statistics and decide when to close a bar.

Each accumulator type embodies one bar-sampling logic (tick count, volume,
dollar value, time boundary, signed imbalance, or run of same-sign ticks).
The base class handles shared OHLCV tracking, partial-bar access, and state
persistence.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from flowbars.core import Bar, TickInfo


class BaseAccumulator(ABC):
    """Shared OHLCV tracking for all bar accumulators.

    Subclasses define the closing criterion (via ``should_close``),
    overflow handling (via ``close``), and any type-specific statistics
    (via ``_on_tick``).

    Parameters
    ----------
    bar_type : str
        The bar type label (e.g. ``"tick"``, ``"dollar"``, ``"imbalance_tick"``).
    """

    def __init__(self, bar_type: str) -> None:
        self._bar_type = bar_type
        self._bar_id = 0
        self._has_tick = False
        # OHLCV state
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume = 0.0
        self._dollar_value = 0.0
        self._num_ticks = 0
        self._open_ts = 0
        self._close_ts = 0

    # ── tick ingestion ──────────────────────────────────────────────────

    def add_tick(self, tick: TickInfo) -> None:
        """Update running OHLCV statistics with a new tick.

        Subclasses that override this must call ``super().add_tick(tick)``
        or manually update the OHLCV fields.
        """
        if not self._has_tick:
            self._open = tick.price
            self._high = tick.price
            self._low = tick.price
            self._open_ts = tick.timestamp
            self._has_tick = True
        else:
            if tick.price > self._high:
                self._high = tick.price
            if tick.price < self._low:
                self._low = tick.price

        self._close = tick.price
        self._volume += tick.volume
        self._dollar_value += tick.price * tick.volume
        self._num_ticks += 1
        self._close_ts = tick.timestamp

    # ── partial bar access ──────────────────────────────────────────────

    @property
    def current_bar(self) -> Bar | None:
        """The in-progress bar, or ``None`` if no ticks have been added."""
        if not self._has_tick:
            return None
        vwap = self._dollar_value / self._volume if self._volume > 0.0 else 0.0
        return Bar(
            bar_id=self._bar_id,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            dollar_value=self._dollar_value,
            vwap=vwap,
            num_ticks=self._num_ticks,
            open_ts=self._open_ts,
            close_ts=self._close_ts,
            bar_type=self._bar_type,
        )

    # ── closing interface ───────────────────────────────────────────────

    @abstractmethod
    def should_close(self, threshold: float) -> bool:
        """Return ``True`` if the running total meets or exceeds *threshold*."""
        ...

    @abstractmethod
    def close(self, threshold: float) -> Bar:
        """Emit the completed bar and reset state, with overflow rollover.

        The caller must have verified that ``should_close(threshold)``
        is ``True`` before calling this method.
        """
        ...

    # ── internal helpers for subclasses ─────────────────────────────────

    def _emit_and_reset(self) -> Bar:
        """Emit the current bar, increment ``bar_id``, and reset OHLCV fields.

        Returns
        -------
        Bar
            The completed bar (before overflow handling — subclasses must
            apply their own overflow logic after this call).
        """
        bar = self.current_bar
        assert bar is not None, "Cannot emit bar without ticks"
        self._bar_id += 1
        self._has_tick = False
        self._open = 0.0
        self._high = 0.0
        self._low = 0.0
        self._close = 0.0
        self._volume = 0.0
        self._dollar_value = 0.0
        self._num_ticks = 0
        self._open_ts = 0
        self._close_ts = 0
        return bar

    # ── bar-close statistics ────────────────────────────────────────────

    def get_close_stats(self) -> tuple[float, float]:
        """Return ``(t_stat, proportion_stat)`` for the bar about to be closed.

        Called by the bar constructor **before** ``close()`` so the
        pre-reset state is still available.  The default returns zeros —
        standard accumulators don't need meaningful close stats because
        their threshold estimator's ``on_bar_close`` is a no-op.

        Subclasses that are used with adaptive (EWMA) thresholds override
        this to return real values.
        """
        return (0.0, 0.0)

    # ── state persistence ───────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the accumulator's current state."""
        return {
            "bar_id": self._bar_id,
            "has_tick": self._has_tick,
            "open": self._open,
            "high": self._high if self._has_tick else 0.0,
            "low": self._low if self._has_tick else 0.0,
            "close": self._close,
            "volume": self._volume,
            "dollar_value": self._dollar_value,
            "num_ticks": self._num_ticks,
            "open_ts": self._open_ts,
            "close_ts": self._close_ts,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore accumulator state from a saved state dict."""
        self._bar_id = state["bar_id"]
        self._has_tick = state["has_tick"]
        self._open = state["open"]
        self._high = state["high"]
        self._low = state["low"]
        self._close = state["close"]
        self._volume = state["volume"]
        self._dollar_value = state["dollar_value"]
        self._num_ticks = state["num_ticks"]
        self._open_ts = state["open_ts"]
        self._close_ts = state["close_ts"]


# ═══════════════════════════════════════════════════════════════════════════════
# Standard-bar accumulators
# ═══════════════════════════════════════════════════════════════════════════════


class TickAccumulator(BaseAccumulator):
    """Closes a bar after a fixed number of ticks.

    Parameters
    ----------
    bar_type : str
        Bar type label (typically ``"tick"``).
    """

    def __init__(self, bar_type: str = "tick") -> None:
        super().__init__(bar_type)
        self._cum_ticks = 0.0

    def add_tick(self, tick: TickInfo) -> None:
        super().add_tick(tick)
        self._cum_ticks += 1.0

    def should_close(self, threshold: float) -> bool:
        return self._cum_ticks >= threshold

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        excess = self._cum_ticks - threshold
        self._cum_ticks = max(0.0, excess)
        return bar

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["cum_ticks"] = self._cum_ticks
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._cum_ticks = state.get("cum_ticks", 0.0)


class VolumeAccumulator(BaseAccumulator):
    """Closes a bar when cumulative traded volume meets or exceeds a threshold.

    Parameters
    ----------
    bar_type : str
        Bar type label (typically ``"volume"``).
    """

    def __init__(self, bar_type: str = "volume") -> None:
        super().__init__(bar_type)
        self._cum_volume = 0.0

    def add_tick(self, tick: TickInfo) -> None:
        super().add_tick(tick)
        self._cum_volume += tick.volume

    def should_close(self, threshold: float) -> bool:
        return self._cum_volume >= threshold

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        excess = self._cum_volume - threshold
        self._cum_volume = excess
        return bar

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["cum_volume"] = self._cum_volume
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._cum_volume = state.get("cum_volume", 0.0)


class DollarAccumulator(BaseAccumulator):
    """Closes a bar when cumulative notional value meets or exceeds a threshold.

    Parameters
    ----------
    bar_type : str
        Bar type label (typically ``"dollar"``).
    """

    def __init__(self, bar_type: str = "dollar") -> None:
        super().__init__(bar_type)
        self._cum_dollar = 0.0

    def add_tick(self, tick: TickInfo) -> None:
        super().add_tick(tick)
        self._cum_dollar += tick.price * tick.volume

    def should_close(self, threshold: float) -> bool:
        return self._cum_dollar >= threshold

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        excess = self._cum_dollar - threshold
        self._cum_dollar = excess
        return bar

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["cum_dollar"] = self._cum_dollar
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._cum_dollar = state.get("cum_dollar", 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Time accumulator
# ═══════════════════════════════════════════════════════════════════════════════


class TimeAccumulator(BaseAccumulator):
    """Closes a bar when a tick's timestamp crosses a time boundary.

    Parameters
    ----------
    bar_type : str
        Bar type label (typically ``"time"``).
    interval_ms : int
        Bar interval in milliseconds (e.g. 300000 for 5-minute bars).
    anchor : str
        ``"clock"`` — bars aligned to round UTC boundaries (e.g. 14:00:00).
        ``"first_tick"`` — bars aligned relative to the first tick's timestamp.
    """

    def __init__(
        self,
        bar_type: str = "time",
        interval_ms: int = 60000,
        anchor: str = "clock",
    ) -> None:
        super().__init__(bar_type)
        if interval_ms <= 0:
            raise ValueError(f"interval_ms must be positive, got {interval_ms}")
        if anchor not in ("clock", "first_tick"):
            raise ValueError(f"anchor must be 'clock' or 'first_tick', got {anchor!r}")
        self._interval_ms = interval_ms
        self._anchor = anchor
        # _next_boundary_ms: the timestamp (ms) at which the current bar closes
        self._next_boundary_ms: int | None = None

    def add_tick(self, tick: TickInfo) -> None:
        super().add_tick(tick)
        if self._next_boundary_ms is None:
            self._next_boundary_ms = self._compute_first_boundary(tick.timestamp)

    def should_close(self, threshold: float) -> bool:
        # threshold unused — time bars close on boundary crossing, not on a
        # numeric threshold (the "threshold" is the interval duration)
        if self._next_boundary_ms is None:
            return False
        return self._close_ts >= self._next_boundary_ms

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        # Advance to the next boundary
        assert self._next_boundary_ms is not None
        self._next_boundary_ms += self._interval_ms
        return bar

    def _compute_first_boundary(self, first_ts: int) -> int:
        """Compute the first closing boundary timestamp in milliseconds."""
        if self._anchor == "first_tick":
            return first_ts + self._interval_ms
        # clock anchor: round up to the next interval boundary
        # e.g. interval=300000 (5 min), ts=14:03:17 → next boundary = 14:05:00
        remainder = first_ts % self._interval_ms
        if remainder == 0:
            return first_ts + self._interval_ms
        return first_ts - remainder + self._interval_ms

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["next_boundary_ms"] = self._next_boundary_ms
        state["interval_ms"] = self._interval_ms
        state["anchor"] = self._anchor
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._next_boundary_ms = state.get("next_boundary_ms")
        self._interval_ms = state["interval_ms"]
        self._anchor = state["anchor"]


# ═══════════════════════════════════════════════════════════════════════════════
# Information-driven accumulators
# ═══════════════════════════════════════════════════════════════════════════════


class ImbalanceAccumulator(BaseAccumulator):
    """Closes a bar when the absolute signed imbalance crosses a threshold.

    The imbalance metric can be tick count, volume, or dollar value,
    weighted by trade sign (tick rule or supplied side).

    Parameters
    ----------
    bar_type : str
        Bar type label (e.g. ``"imbalance_tick"``).
    metric : str
        What to weight the sign by: ``"tick"``, ``"volume"``, or ``"dollar"``.
    """

    # Map metric names to extractor functions on TickInfo
    _METRIC_FNS: dict[str, Any] = {}  # populated below

    def __init__(self, bar_type: str = "imbalance_tick", metric: str = "tick") -> None:
        super().__init__(bar_type)
        if metric not in ("tick", "volume", "dollar"):
            raise ValueError(f"metric must be 'tick', 'volume', or 'dollar', got {metric!r}")
        self._metric = metric
        self._signed_imbalance = 0.0

    def add_tick(self, tick: TickInfo) -> None:
        super().add_tick(tick)
        # First-tick exclusion: if side is NaN, imbalance contribution is 0
        if tick.side is not None and not math.isnan(tick.side):
            value = _imbalance_metric(tick, self._metric)
            self._signed_imbalance += tick.side * value

    def should_close(self, threshold: float) -> bool:
        return abs(self._signed_imbalance) >= threshold

    def get_close_stats(self) -> tuple[float, float]:
        """Return (t_stat, θ) where θ = signed_imbalance / t_stat."""
        if self._metric == "tick":
            t_stat = float(self._num_ticks)
        elif self._metric == "volume":
            t_stat = self._volume
        else:  # dollar
            t_stat = self._dollar_value
        proportion = self._signed_imbalance / t_stat if t_stat > 0.0 else 0.0
        return (t_stat, proportion)

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        # Overflow: signed excess carries into next bar
        sign = 1.0 if self._signed_imbalance >= 0.0 else -1.0
        excess = abs(self._signed_imbalance) - threshold
        self._signed_imbalance = sign * max(0.0, excess)
        return bar

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["signed_imbalance"] = self._signed_imbalance
        state["metric"] = self._metric
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._signed_imbalance = state.get("signed_imbalance", 0.0)
        self._metric = state.get("metric", "tick")


def _imbalance_metric(tick: TickInfo, metric: str) -> float:
    """Extract the imbalance weight from a tick."""
    if metric == "tick":
        return 1.0
    elif metric == "volume":
        return tick.volume
    else:  # "dollar"
        return tick.price * tick.volume


class RunAccumulator(BaseAccumulator):
    """Closes a bar when the cumulative same-sign run statistic crosses a threshold.

    A *run* is a sequence of consecutive same-sign ticks.  When the direction
    changes, the current run is banked and a new one begins.  The bar closes
    when the total banked + current-run statistic reaches the threshold.

    Parameters
    ----------
    bar_type : str
        Bar type label (e.g. ``"run_tick"``).
    metric : str
        What to accumulate per run: ``"tick"``, ``"volume"``, or ``"dollar"``.
    """

    def __init__(self, bar_type: str = "run_tick", metric: str = "tick") -> None:
        super().__init__(bar_type)
        if metric not in ("tick", "volume", "dollar"):
            raise ValueError(f"metric must be 'tick', 'volume', or 'dollar', got {metric!r}")
        self._metric = metric
        self._first_tick = True  # True only before the very first tick (survives close())
        self._banked = 0.0  # cumulative from completed runs in this bar
        self._run_sign = np.nan  # NaN = no run started yet
        self._run_cum = 0.0  # cumulative within the current run
        self._buy_cum = 0.0  # total buy-side run metric in this bar
        self._sell_cum = 0.0  # total sell-side run metric in this bar

    @staticmethod
    def _same_direction(a: float, b: float) -> bool:
        """Two sides belong to the same run if either is NaN, or both are equal.

        NaN acts as a wildcard — it matches any direction.  This implements
        the spec's rule that the first tick (NaN side) is retroactively
        included in the first run whose direction is determined by tick 2.
        """
        if np.isnan(a) or np.isnan(b):
            return True
        return a == b

    def add_tick(self, tick: TickInfo) -> None:
        super().add_tick(tick)
        metric_value = _imbalance_metric(tick, self._metric)
        side = tick.side if tick.side is not None else np.nan

        if self._first_tick:
            # Very first tick of the stream: start the first run.
            self._first_tick = False
            self._run_sign = side  # may be NaN
            self._run_cum = metric_value
            return

        if self._same_direction(self._run_sign, side):
            # Same direction — continue the run
            self._run_cum += metric_value
            if np.isnan(self._run_sign) and not np.isnan(side):
                # Retroactively assign direction to a NaN-led run
                self._run_sign = side
        else:
            # Direction change: bank current run, start a new one
            self._banked += self._run_cum
            # Track buy vs sell for P⁺ computation
            if self._run_sign > 0.0:
                self._buy_cum += self._run_cum
            elif self._run_sign < 0.0:
                self._sell_cum += self._run_cum
            # else: _run_sign is NaN — run has no direction yet, don't count
            self._run_sign = side
            self._run_cum = metric_value

    @property
    def _total(self) -> float:
        """Total run statistic accumulated in this bar (banked + current)."""
        return self._banked + self._run_cum

    def should_close(self, threshold: float) -> bool:
        return self._total >= threshold

    def get_close_stats(self) -> tuple[float, float]:
        """Return (t_stat, P⁺) where P⁺ = buy_cum / (buy_cum + sell_cum).

        The current (un-banked) run is included in the proportion computation
        via the direction and cum tracked separately from buy_cum/sell_cum.
        """
        # Include the current (un-banked) run in the buy/sell totals
        cur_buy = self._buy_cum
        cur_sell = self._sell_cum
        if self._run_sign > 0.0:
            cur_buy += self._run_cum
        elif self._run_sign < 0.0:
            cur_sell += self._run_cum
        # else: run_sign is NaN — no direction, exclude from both

        total = cur_buy + cur_sell
        proportion = cur_buy / total if total > 0.0 else 0.5  # neutral prior
        return (self._total, proportion)

    def close(self, threshold: float) -> Bar:
        bar = self._emit_and_reset()
        # Overflow: the excess from total carries into the next bar.
        # The excess belongs to the current run (the one that crossed).
        excess = self._total - threshold
        self._banked = 0.0
        self._run_cum = max(0.0, excess)
        self._buy_cum = 0.0
        self._sell_cum = 0.0
        # Keep _run_sign — the overflow run continues into the next bar
        return bar

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["metric"] = self._metric
        state["first_tick"] = self._first_tick
        state["banked"] = self._banked
        state["run_sign"] = self._run_sign
        state["run_cum"] = self._run_cum
        state["buy_cum"] = self._buy_cum
        state["sell_cum"] = self._sell_cum
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        super().load_state(state)
        self._metric = state.get("metric", "tick")
        self._first_tick = state.get("first_tick", True)
        self._banked = state.get("banked", 0.0)
        self._run_sign = state.get("run_sign", np.nan)
        self._run_cum = state.get("run_cum", 0.0)
        self._buy_cum = state.get("buy_cum", 0.0)
        self._sell_cum = state.get("sell_cum", 0.0)
