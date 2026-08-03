"""Threshold estimators for bar construction.

Fixed thresholds for standard bars, adaptive two-component EWMA thresholds
per AFML Chapter 2 for information-driven bars, and a calibration helper to
seed both from historical data.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from flowbars.core import ThresholdError, TickInfo

# ═══════════════════════════════════════════════════════════════════════════════
# ABC
# ═══════════════════════════════════════════════════════════════════════════════


class ThresholdEstimator(ABC):
    """Abstract base class for threshold estimators.

    Every bar constructor holds a threshold estimator.  The constructor feeds
    every tick via :meth:`update` and, after emitting a bar, notifies the
    estimator via :meth:`on_bar_close` so adaptive estimators can recompute
    the threshold from bar-level statistics.
    """

    @abstractmethod
    def update(self, tick: TickInfo) -> None:
        """Called on every tick.  Subclasses may accumulate statistics."""
        ...

    @property
    @abstractmethod
    def current_threshold(self) -> float:
        """The threshold value to use for the bar currently being built."""
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the estimator's current state."""
        ...

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Restore estimator state from a saved state dict."""
        ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict[str, Any]) -> ThresholdEstimator:
        """Create a new estimator instance from a saved state dict."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset to the initial state (as if no ticks have been seen)."""
        ...

    def on_bar_close(self, t_stat: float, proportion_stat: float) -> None:  # noqa: B027
        """Called after each bar closes.  Default no-op.

        Adaptive estimators (EWMA) override this to update their running
        statistics from bar-level aggregates.

        Parameters
        ----------
        t_stat : float
            Total bar statistic: tick count, total volume, or total dollar
            value of the bar that just closed.
        proportion_stat : float
            For **imbalance** bars: signed imbalance proportion
            :math:`\\theta \\in [-1, 1]`
            (``signed_imbalance / t_stat``).
            For **run** bars: buy-run proportion
            :math:`P^+ \\in [0, 1]`
            (``buy_run_total / total_run_stat``).
        """
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Static (fixed) threshold
# ═══════════════════════════════════════════════════════════════════════════════


class StaticThresholdEstimator(ThresholdEstimator):
    """A fixed threshold that never changes.

    This is the estimator used for standard bars (tick, volume, dollar, time)
    where the user supplies a known threshold upfront.

    Parameters
    ----------
    threshold : float
        The fixed threshold value.  Must be non-negative.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        if threshold < 0:
            raise ThresholdError(f"Threshold must be non-negative, got {threshold}")
        self._threshold = threshold
        self._initial_threshold = threshold

    # ── ThresholdEstimator interface ──────────────────────────────────────

    def update(self, tick: TickInfo) -> None:
        """No-op — the threshold never changes."""
        pass

    @property
    def current_threshold(self) -> float:
        return self._threshold

    def get_state(self) -> dict[str, Any]:
        return {
            "threshold": self._threshold,
            "initial_threshold": self._initial_threshold,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._threshold = state["threshold"]
        self._initial_threshold = state.get("initial_threshold", self._threshold)

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> StaticThresholdEstimator:
        inst = cls(threshold=state["threshold"])
        inst._initial_threshold = state.get("initial_threshold", state["threshold"])
        return inst

    def reset(self) -> None:
        self._threshold = self._initial_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive two-component EWMA (AFML §2.4)
# ═══════════════════════════════════════════════════════════════════════════════


class EWMAThresholdEstimator(ThresholdEstimator):
    """Adaptive threshold via the two-component EWMA formula from AFML.

    Two EWMA terms are tracked per bar-close:

    * :math:`E[T]_n` — the expected bar statistic (tick count, volume, or
      dollar value).
    * :math:`E[\\theta]_n` or :math:`E[P^+]_n` — the expected imbalance
      proportion or buy-run proportion.

    The threshold is computed as:

    * **Imbalance bars:**
      :math:`T_n = E[T]_n \\times |E[\\theta]_n|`
    * **Run bars:**
      :math:`T_n = E[T]_n \\times \\max(E[P^+]_n,\\, 1 - E[P^+]_n)`

    Parameters
    ----------
    bar_family : str
        ``"imbalance"`` or ``"run"`` — controls the multiplier formula.
    span : float
        EWMA span.  Decay factor :math:`\\alpha = 2 / (\\text{span} + 1)`.
        Default 20.
    halflife : float or None
        EWMA halflife.  Takes precedence over *span* when given.
        :math:`\\alpha = 1 - \\exp(-\\ln 2 / \\text{halflife})`.
    initial_ewa_t : float
        Initial value for :math:`E[T]`.  Default 1.0 (one unit per bar).
    initial_ewa_proportion : float
        Initial value for :math:`E[\\theta]` or :math:`E[P^+]`.
        Default 0.5 (balanced market prior).
    """

    def __init__(
        self,
        bar_family: str = "imbalance",
        span: float = 20.0,
        halflife: float | None = None,
        initial_ewa_t: float = 1.0,
        initial_ewa_proportion: float = 0.5,
    ) -> None:
        if bar_family not in ("imbalance", "run"):
            raise ValueError(f"bar_family must be 'imbalance' or 'run', got {bar_family!r}")
        if span <= 0:
            raise ValueError(f"span must be positive, got {span}")
        if halflife is not None and halflife <= 0:
            raise ValueError(f"halflife must be positive, got {halflife}")
        if initial_ewa_t < 0:
            raise ThresholdError(f"initial_ewa_t must be non-negative, got {initial_ewa_t}")

        self._bar_family = bar_family
        self._span = span
        self._halflife = halflife
        self._initial_ewa_t = initial_ewa_t
        self._initial_ewa_proportion = initial_ewa_proportion

        # Compute decay factor
        if halflife is not None:
            self._alpha = 1.0 - math.exp(-math.log(2) / halflife)
        else:
            self._alpha = 2.0 / (span + 1.0)

        # Running EWMA state
        self._ewa_t = initial_ewa_t
        self._ewa_proportion = initial_ewa_proportion
        self._n_updates = 0

    # ── ThresholdEstimator interface ──────────────────────────────────────

    def update(self, tick: TickInfo) -> None:
        """No-op — the EWMA updates only at bar boundaries."""
        pass

    @property
    def current_threshold(self) -> float:
        """The current threshold computed from the two EWMA terms.

        Returns 0.0 when :math:`E[T] = 0` (no bars observed + zero initial
        seed).
        """
        if self._bar_family == "imbalance":
            return self._ewa_t * abs(self._ewa_proportion)
        else:  # run
            return self._ewa_t * max(self._ewa_proportion, 1.0 - self._ewa_proportion)

    def on_bar_close(self, t_stat: float, proportion_stat: float) -> None:
        """Update the two EWMA terms with bar-level statistics.

        Parameters
        ----------
        t_stat : float
            Total bar statistic for the bar that just closed.
        proportion_stat : float
            For imbalance: signed imbalance proportion :math:`\\theta`.
            For run: buy-run proportion :math:`P^+`.
        """
        if t_stat < 0:
            raise ValueError(f"t_stat must be non-negative, got {t_stat}")

        self._ewa_t = self._alpha * t_stat + (1.0 - self._alpha) * self._ewa_t
        self._ewa_proportion = (
            self._alpha * proportion_stat + (1.0 - self._alpha) * self._ewa_proportion
        )
        self._n_updates += 1

    def get_state(self) -> dict[str, Any]:
        return {
            "bar_family": self._bar_family,
            "span": self._span,
            "halflife": self._halflife,
            "alpha": self._alpha,
            "initial_ewa_t": self._initial_ewa_t,
            "initial_ewa_proportion": self._initial_ewa_proportion,
            "ewa_t": self._ewa_t,
            "ewa_proportion": self._ewa_proportion,
            "n_updates": self._n_updates,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._bar_family = state["bar_family"]
        self._span = state["span"]
        self._halflife = state["halflife"]
        self._alpha = state["alpha"]
        self._initial_ewa_t = state.get("initial_ewa_t", 1.0)
        self._initial_ewa_proportion = state.get("initial_ewa_proportion", 0.5)
        self._ewa_t = state["ewa_t"]
        self._ewa_proportion = state["ewa_proportion"]
        self._n_updates = state["n_updates"]

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> EWMAThresholdEstimator:
        inst = cls(
            bar_family=state["bar_family"],
            span=state["span"],
            halflife=state["halflife"],
            initial_ewa_t=state.get("initial_ewa_t", 1.0),
            initial_ewa_proportion=state.get("initial_ewa_proportion", 0.5),
        )
        inst._alpha = state["alpha"]
        inst._ewa_t = state["ewa_t"]
        inst._ewa_proportion = state["ewa_proportion"]
        inst._n_updates = state["n_updates"]
        return inst

    def reset(self) -> None:
        self._ewa_t = self._initial_ewa_t
        self._ewa_proportion = self._initial_ewa_proportion
        self._n_updates = 0

    # ── read-only helpers ─────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """The EWMA decay factor."""
        return self._alpha

    @property
    def span(self) -> float:
        """The EWMA span parameter."""
        return self._span

    @property
    def halflife(self) -> float | None:
        """The EWMA halflife parameter, or ``None`` if span was used."""
        return self._halflife

    @property
    def n_updates(self) -> int:
        """Number of times ``on_bar_close`` has been called."""
        return self._n_updates

    @property
    def ewa_t(self) -> float:
        """Current value of :math:`E[T]`."""
        return self._ewa_t

    @property
    def ewa_proportion(self) -> float:
        """Current value of :math:`E[\\theta]` or :math:`E[P^+]`."""
        return self._ewa_proportion


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration helper
# ═══════════════════════════════════════════════════════════════════════════════


class StaticCalibrationHelper:
    """Estimate threshold values from historical tick data.

    Two use-cases:

    1. **Fixed threshold** — given a target number of bars per day, estimate
       the threshold that achieves it.  Used with
       :class:`StaticThresholdEstimator`.
    2. **EWMA seeds** — run a fixed-threshold simulation through historical
       data, collect per-bar statistics, and warm an EWMA to produce
       sensible initial seeds for :class:`EWMAThresholdEstimator`.

    All methods are static; the class is not instantiated.
    """

    @staticmethod
    def estimate_fixed_threshold(
        ticks: list[TickInfo],
        bar_type: str,
        target_bars_per_day: float,
    ) -> float:
        """Estimate a fixed threshold to produce approximately *target_bars_per_day*.

        For standard bars the computation is exact (total metric / target).
        For information-driven bars a single-pass simulation is used, and the
        result is an approximation.

        Parameters
        ----------
        ticks : list of TickInfo
            Historical tick data.
        bar_type : str
            One of ``"tick"``, ``"volume"``, ``"dollar"``, ``"time"``,
            ``"imbalance_tick"``, ``"imbalance_volume"``, ``"imbalance_dollar"``,
            ``"run_tick"``, ``"run_volume"``, ``"run_dollar"``.
        target_bars_per_day : float
            Desired number of bars per trading day.

        Returns
        -------
        float
            The estimated fixed threshold.

        Raises
        ------
        ValueError
            If *ticks* is empty, *target_bars_per_day* is not positive, or
            *bar_type* is unrecognised.
        """
        if not ticks:
            raise ValueError("Cannot estimate threshold from empty tick list")
        if target_bars_per_day <= 0:
            raise ValueError(f"target_bars_per_day must be positive, got {target_bars_per_day}")

        # Standard bars: exact division
        if bar_type == "tick":
            return len(ticks) / target_bars_per_day
        elif bar_type == "volume":
            total_vol = sum(t.volume for t in ticks)
            return total_vol / target_bars_per_day
        elif bar_type == "dollar":
            total_dollar = sum(t.price * t.volume for t in ticks)
            return total_dollar / target_bars_per_day
        elif bar_type == "time":
            raise ValueError(
                "Time bars use an interval (ms), not a threshold. Use interval_ms directly."
            )
        elif bar_type.startswith("imbalance_"):
            return StaticCalibrationHelper._simulate_imbalance_threshold(
                ticks, bar_type, target_bars_per_day
            )
        elif bar_type.startswith("run_"):
            return StaticCalibrationHelper._simulate_run_threshold(
                ticks, bar_type, target_bars_per_day
            )
        else:
            raise ValueError(f"Unknown bar type: {bar_type!r}")

    @staticmethod
    def estimate_ewma_seeds(
        ticks: list[TickInfo],
        bar_family: str,
        metric: str = "tick",
        span: float = 20.0,
        target_bars: int = 50,
    ) -> dict[str, float]:
        """Estimate initial EWMA seeds from historical tick data.

        Simulates bars through *ticks* using a fixed threshold chosen to
        produce ~*target_bars* bars.  Per-bar statistics are collected and an
        EWMA is warmed forward through them; the final EWMA values are
        returned as seeds.

        Parameters
        ----------
        ticks : list of TickInfo
            Historical tick data.
        bar_family : str
            ``"imbalance"`` or ``"run"``.
        metric : str
            ``"tick"``, ``"volume"``, or ``"dollar"``.
        span : float
            EWMA span used for warming.
        target_bars : int
            Approximate number of bars to simulate for seed estimation.

        Returns
        -------
        dict
            Keys ``"initial_ewa_t"`` and ``"initial_ewa_proportion"``,
            suitable for passing to :class:`EWMAThresholdEstimator`.
        """
        if not ticks:
            raise ValueError("Cannot estimate seeds from empty tick list")
        if bar_family not in ("imbalance", "run"):
            raise ValueError(f"bar_family must be 'imbalance' or 'run', got {bar_family!r}")
        if metric not in ("tick", "volume", "dollar"):
            raise ValueError(f"metric must be 'tick', 'volume', or 'dollar', got {metric!r}")

        bar_type = f"{bar_family}_{metric}"

        # Simulate with a threshold targeting ~target_bars bars
        threshold = StaticCalibrationHelper.estimate_fixed_threshold(
            ticks, bar_type, float(target_bars)
        )

        if bar_family == "imbalance":
            t_stats, proportion_stats = _simulate_imbalance_bars(ticks, metric, threshold)
        else:
            t_stats, proportion_stats = _simulate_run_bars(ticks, metric, threshold)

        if not t_stats:
            # All ticks fell into a single bar; use that as the seed
            return {
                "initial_ewa_t": float(_total_metric(ticks, metric) if ticks else 1.0),
                "initial_ewa_proportion": proportion_stats[0] if proportion_stats else 0.5,
            }

        # Warm the EWMA forward through collected bar stats
        alpha = 2.0 / (span + 1.0)
        ewa_t = t_stats[0]
        ewa_proportion = proportion_stats[0]
        for t, p in zip(t_stats[1:], proportion_stats[1:]):
            ewa_t = alpha * t + (1.0 - alpha) * ewa_t
            ewa_proportion = alpha * p + (1.0 - alpha) * ewa_proportion

        return {
            "initial_ewa_t": ewa_t,
            "initial_ewa_proportion": ewa_proportion,
        }

    # ── internal simulation helpers ───────────────────────────────────────

    @staticmethod
    def _simulate_imbalance_threshold(
        ticks: list[TickInfo],
        bar_type: str,
        target_bars_per_day: float,
    ) -> float:
        """Estimate threshold for imbalance bars via simulation."""
        metric = _metric_from_bar_type(bar_type)
        total = _total_metric(ticks, metric)

        # Initial guess: same as standard bar formula
        guess = total / target_bars_per_day

        # Single-pass simulation with this guess
        t_stats, _ = _simulate_imbalance_bars(ticks, metric, guess)

        n_bars = len(t_stats)
        if n_bars == 0:
            # Threshold too large — all ticks in one bar; use total_metric
            return total

        # Adjust proportionally
        if n_bars > 0:
            return guess * n_bars / target_bars_per_day

        return guess

    @staticmethod
    def _simulate_run_threshold(
        ticks: list[TickInfo],
        bar_type: str,
        target_bars_per_day: float,
    ) -> float:
        """Estimate threshold for run bars via simulation."""
        metric = _metric_from_bar_type(bar_type)
        total = _total_metric(ticks, metric)

        guess = total / target_bars_per_day
        t_stats, _ = _simulate_run_bars(ticks, metric, guess)

        n_bars = len(t_stats)
        if n_bars == 0:
            return total
        return guess * n_bars / target_bars_per_day


# ═══════════════════════════════════════════════════════════════════════════════
# Internal simulation helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _metric_from_bar_type(bar_type: str) -> str:
    """Extract the metric name from a bar type string."""
    if bar_type.startswith("imbalance_"):
        return bar_type[len("imbalance_") :]
    if bar_type.startswith("run_"):
        return bar_type[len("run_") :]
    return bar_type


def _total_metric(ticks: list[TickInfo], metric: str) -> float:
    """Total metric value across all ticks."""
    if metric == "tick":
        return float(len(ticks))
    elif metric == "volume":
        return sum(t.volume for t in ticks)
    else:  # dollar
        return sum(t.price * t.volume for t in ticks)


def _metric_value(tick: TickInfo, metric: str) -> float:
    """Extract the metric value from a single tick."""
    if metric == "tick":
        return 1.0
    elif metric == "volume":
        return tick.volume
    else:  # dollar
        return tick.price * tick.volume


def _simulate_imbalance_bars(
    ticks: list[TickInfo],
    metric: str,
    threshold: float,
) -> tuple[list[float], list[float]]:
    """Simulate imbalance-bar construction, returning per-bar (T, θ) pairs.

    Returns
    -------
    t_stats : list of float
        Total metric in each bar.
    proportion_stats : list of float
        Signed imbalance proportion θ = signed_imbalance / t_stat for each bar.
    """
    t_stats: list[float] = []
    proportion_stats: list[float] = []

    signed_imbalance = 0.0
    total_metric = 0.0

    for tick in ticks:
        metric_val = _metric_value(tick, metric)
        side = tick.side if tick.side is not None else np.nan

        total_metric += metric_val
        if side is not None and not math.isnan(side):
            signed_imbalance += side * metric_val

        if abs(signed_imbalance) >= threshold:
            t_stats.append(total_metric)
            proportion = signed_imbalance / total_metric if total_metric > 0.0 else 0.0
            proportion_stats.append(proportion)

            # Overflow: signed excess carries into next bar
            sign = 1.0 if signed_imbalance >= 0.0 else -1.0
            excess = abs(signed_imbalance) - threshold
            signed_imbalance = sign * max(0.0, excess)
            total_metric = 0.0

    return t_stats, proportion_stats


def _simulate_run_bars(
    ticks: list[TickInfo],
    metric: str,
    threshold: float,
) -> tuple[list[float], list[float]]:
    """Simulate run-bar construction, returning per-bar (T, P⁺) pairs.

    Returns
    -------
    t_stats : list of float
        Total run statistic in each bar.
    proportion_stats : list of float
        Buy-run proportion P⁺ = buy_run_total / total_run for each bar.
    """
    t_stats: list[float] = []
    proportion_stats: list[float] = []

    first_tick = True
    banked = 0.0
    run_sign = np.nan
    run_cum = 0.0
    buy_run_total = 0.0
    sell_run_total = 0.0

    for tick in ticks:
        metric_val = _metric_value(tick, metric)
        side = tick.side if tick.side is not None else np.nan

        if first_tick:
            first_tick = False
            run_sign = side
            run_cum = metric_val
        elif _same_direction(run_sign, side):
            run_cum += metric_val
            if np.isnan(run_sign) and not np.isnan(side):
                run_sign = side
        else:
            # Bank the completed run
            banked += run_cum
            if run_sign > 0.0:
                buy_run_total += run_cum
            elif run_sign < 0.0:
                sell_run_total += run_cum
            run_sign = side
            run_cum = metric_val

        total = banked + run_cum
        if total >= threshold:
            # Include current run in buy/sell tally for the proportion
            cur_buy = buy_run_total
            cur_sell = sell_run_total
            if run_sign > 0.0:
                cur_buy += run_cum
            elif run_sign < 0.0:
                cur_sell += run_cum

            total_buysell = cur_buy + cur_sell
            if total_buysell > 0.0:
                proportion = cur_buy / total_buysell
            else:
                proportion = 0.5  # neutral

            t_stats.append(banked + run_cum)
            proportion_stats.append(proportion)

            # Overflow: excess carries in current run's direction
            excess = (banked + run_cum) - threshold
            banked = 0.0
            run_cum = max(0.0, excess)
            buy_run_total = 0.0
            sell_run_total = 0.0
            if run_cum == 0.0:
                run_sign = np.nan

    return t_stats, proportion_stats


def _same_direction(a: float, b: float) -> bool:
    """Two sides belong to the same run if either is NaN or both equal."""
    if np.isnan(a) or np.isnan(b):
        return True
    return a == b
