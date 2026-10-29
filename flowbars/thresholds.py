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

from flowbars.core import MinuteInfo, ThresholdError, TickInfo

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
    def update(self, tick: TickInfo | MinuteInfo) -> None:
        """Called on every tick/minute.  Subclasses may accumulate statistics."""
        ...

    @property
    @abstractmethod
    def current_threshold(self) -> float:
        """The threshold value to use for the bar currently being built."""
        ...

    def get_params(self) -> dict[str, Any]:
        """Return the configuration that defines this estimator.

        Params exclude mutable runtime values (seeds, accumulated EWMA terms)
        and are hashed into constructor state for resume validation.
        """
        return {}

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

    def update(self, tick: TickInfo | MinuteInfo) -> None:
        """No-op — the threshold never changes."""
        pass

    @property
    def current_threshold(self) -> float:
        return self._threshold

    def get_params(self) -> dict[str, Any]:
        return {"threshold": self._threshold}

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

    def update(self, tick: TickInfo | MinuteInfo) -> None:
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

    def get_params(self) -> dict[str, Any]:
        # Initial seeds are excluded: the saved state supersedes them on
        # resume, so a differing seed does not change bar semantics.
        return {
            "bar_family": self._bar_family,
            "span": self._span,
            "halflife": self._halflife,
        }

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
        *,
        multiplier: float = 1.0,
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
        multiplier : float
            Contract multiplier applied to notional metrics (``"dollar"`` and
            ``"*_dollar"``).  Must match the constructor's ``multiplier``.

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

        target_total = target_bars_per_day * _observed_days([t.timestamp for t in ticks])

        # Standard bars: exact division
        if bar_type == "tick":
            return len(ticks) / target_total
        elif bar_type == "volume":
            total_vol = sum(t.volume for t in ticks)
            return total_vol / target_total
        elif bar_type == "dollar":
            total_dollar = sum(t.price * t.volume for t in ticks) * multiplier
            return total_dollar / target_total
        elif bar_type == "time":
            raise ValueError(
                "Time bars use an interval (ms), not a threshold. Use interval_ms directly."
            )
        elif bar_type.startswith("imbalance_"):
            return StaticCalibrationHelper._simulate_imbalance_threshold(
                ticks, bar_type, target_total, multiplier
            )
        elif bar_type.startswith("run_"):
            return StaticCalibrationHelper._simulate_run_threshold(
                ticks, bar_type, target_total, multiplier
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
        *,
        multiplier: float = 1.0,
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
        multiplier : float
            Contract multiplier applied to the ``"dollar"`` metric.  Must
            match the constructor's ``multiplier``.

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
            ticks, bar_type, float(target_bars), multiplier=multiplier
        )

        if bar_family == "imbalance":
            t_stats, proportion_stats = _simulate_imbalance_bars(
                ticks, metric, threshold, multiplier
            )
        else:
            t_stats, proportion_stats = _simulate_run_bars(ticks, metric, threshold, multiplier)

        if not t_stats:
            # All ticks fell into a single bar; use that as the seed
            return {
                "initial_ewa_t": float(_total_metric(ticks, metric, multiplier) if ticks else 1.0),
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
        multiplier: float = 1.0,
    ) -> float:
        """Estimate threshold for imbalance bars via simulation."""
        metric = _metric_from_bar_type(bar_type)
        total = _total_metric(ticks, metric, multiplier)

        # Initial guess: same as standard bar formula
        guess = total / target_bars_per_day

        # Single-pass simulation with this guess
        t_stats, _ = _simulate_imbalance_bars(ticks, metric, guess, multiplier)

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
        multiplier: float = 1.0,
    ) -> float:
        """Estimate threshold for run bars via simulation."""
        metric = _metric_from_bar_type(bar_type)
        total = _total_metric(ticks, metric, multiplier)

        guess = total / target_bars_per_day
        t_stats, _ = _simulate_run_bars(ticks, metric, guess, multiplier)

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


def _total_metric(ticks: list[TickInfo], metric: str, multiplier: float = 1.0) -> float:
    """Total metric value across all ticks."""
    if metric == "tick":
        return float(len(ticks))
    elif metric == "volume":
        return sum(t.volume for t in ticks)
    else:  # dollar
        return sum(t.price * t.volume for t in ticks) * multiplier


def _metric_value(tick: TickInfo, metric: str, multiplier: float = 1.0) -> float:
    """Extract the metric value from a single tick."""
    if metric == "tick":
        return 1.0
    elif metric == "volume":
        return tick.volume
    else:  # dollar
        return tick.price * tick.volume * multiplier


def _simulate_imbalance_bars(
    ticks: list[TickInfo],
    metric: str,
    threshold: float,
    multiplier: float = 1.0,
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
        metric_val = _metric_value(tick, metric, multiplier)
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
    multiplier: float = 1.0,
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
        metric_val = _metric_value(tick, metric, multiplier)
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


def _observed_days(timestamps: list[int], day_length_ms: int = 86_400_000) -> int:
    """Count distinct UTC day buckets covered by *timestamps*."""
    return len({int(ts) // day_length_ms for ts in timestamps})


# ═══════════════════════════════════════════════════════════════════════════════
# Minute calibration
# ═══════════════════════════════════════════════════════════════════════════════


class MinuteCalibrationHelper:
    """Deterministic threshold calibration from historical minute candles.

    All methods are static and side-effect free: repeated calls on the same
    input return identical values (no RNG, fixed iteration counts).

    Targets are expressed as bars per UTC day and scaled by the number of
    distinct UTC day buckets observed in the sample.
    """

    @staticmethod
    def estimate_volume_threshold(
        minutes: list[MinuteInfo],
        target_bars_per_day: float,
        *,
        day_length_ms: int = 86_400_000,
    ) -> float:
        """Volume threshold from the total minute volume."""
        target_total = _minute_target_total(minutes, target_bars_per_day, day_length_ms)
        return sum(m.volume for m in minutes) / target_total

    @staticmethod
    def estimate_dollar_threshold(
        minutes: list[MinuteInfo],
        target_bars_per_day: float,
        *,
        multiplier: float = 1.0,
        day_length_ms: int = 86_400_000,
    ) -> float:
        """Dollar threshold from the total ``close × volume`` notional.

        ``multiplier`` scales the notional and must match the constructor's.
        """
        target_total = _minute_target_total(minutes, target_bars_per_day, day_length_ms)
        return sum(m.close * m.volume for m in minutes) * multiplier / target_total

    @staticmethod
    def estimate_volatility_threshold(
        minutes: list[MinuteInfo],
        target_bars_per_day: float,
        *,
        day_length_ms: int = 86_400_000,
    ) -> float:
        """Volatility threshold from cumulative absolute close-to-close returns."""
        target_total = _minute_target_total(minutes, target_bars_per_day, day_length_ms)
        return sum(_minute_volatility_metrics(minutes)) / target_total

    @staticmethod
    def estimate_range_threshold(
        minutes: list[MinuteInfo],
        target_bars_per_day: float,
        *,
        day_length_ms: int = 86_400_000,
    ) -> float:
        """Relative-range threshold from cumulative ``(high − low) / close``."""
        target_total = _minute_target_total(minutes, target_bars_per_day, day_length_ms)
        total = 0.0
        for m in minutes:
            total += (m.high - m.low) / m.close if m.close != 0.0 else 0.0
        return total / target_total

    @staticmethod
    def estimate_renko_threshold(
        minutes: list[MinuteInfo],
        target_bars_per_day: float,
        *,
        max_iter: int = 80,
        day_length_ms: int = 86_400_000,
    ) -> float:
        """Renko-displacement threshold by deterministic bisection.

        Returns the largest threshold whose simulated bar count meets the
        target; on ties between the final bracket endpoints, their midpoint.
        """
        target_total = _minute_target_total(minutes, target_bars_per_day, day_length_ms)

        def count(threshold: float) -> int:
            return _simulate_renko_bar_count(minutes, threshold)

        max_disp = 0.0
        for m in minutes:
            if m.open > 0.0:
                max_disp = max(max_disp, abs(m.close - m.open) / m.open)
        return _bisect_threshold(count, target_total, max_iter, hi_start=max_disp or 1.0)

    @staticmethod
    def estimate_hybrid_thresholds(
        minutes: list[MinuteInfo],
        target_bars_per_day: float,
        *,
        dollar_threshold: float | None = None,
        volatility_threshold: float | None = None,
        multiplier: float = 1.0,
        max_iter: int = 80,
        day_length_ms: int = 86_400_000,
    ) -> tuple[float, float]:
        """Estimate the ``(dollar, volatility)`` hybrid threshold pair.

        With one threshold supplied, it is held fixed and the other is
        bisected.  With neither supplied, a single scalar ``c`` scales both
        mean per-minute metrics (a documented one-parameter ray) and is
        bisected.

        ``multiplier`` scales the dollar leg and must match the
        constructor's; a supplied ``dollar_threshold`` is returned unchanged.
        """
        target_total = _minute_target_total(minutes, target_bars_per_day, day_length_ms)

        if dollar_threshold is not None and volatility_threshold is not None:
            if dollar_threshold < 0 or volatility_threshold < 0:
                raise ValueError("Hybrid thresholds must be non-negative")
            return float(dollar_threshold), float(volatility_threshold)

        if dollar_threshold is not None:
            if dollar_threshold < 0:
                raise ValueError("dollar_threshold must be non-negative")

            def count_vol(vol_threshold: float) -> int:
                return _simulate_hybrid_bar_count(
                    minutes, dollar_threshold, vol_threshold, multiplier
                )

            return (
                float(dollar_threshold),
                _bisect_threshold(count_vol, target_total, max_iter),
            )

        if volatility_threshold is not None:
            if volatility_threshold < 0:
                raise ValueError("volatility_threshold must be non-negative")

            def count_dollar(dollar_thr: float) -> int:
                return _simulate_hybrid_bar_count(
                    minutes, dollar_thr, volatility_threshold, multiplier
                )

            return (
                _bisect_threshold(count_dollar, target_total, max_iter),
                float(volatility_threshold),
            )

        n = len(minutes)
        mean_dollar = sum(m.close * m.volume for m in minutes) * multiplier / n
        mean_vol = sum(_minute_volatility_metrics(minutes)) / n

        def count_ray(c: float) -> int:
            return _simulate_hybrid_bar_count(minutes, c * mean_dollar, c * mean_vol, multiplier)

        c = _bisect_threshold(count_ray, target_total, max_iter)
        return (float(c * mean_dollar), float(c * mean_vol))

    @staticmethod
    def estimate_ewma_seeds(
        minutes: list[MinuteInfo],
        bar_family: str,
        metric: str = "volume",
        span: float = 20.0,
        target_bars: int = 50,
        *,
        multiplier: float = 1.0,
    ) -> dict[str, float]:
        """Estimate EWMA seeds for minute imbalance/run bars.

        Simulates fixed-threshold minute bars, collects per-bar statistics,
        and warms an EWMA forward to produce the seeds.  ``multiplier`` scales
        the ``"dollar"`` metric and must match the constructor's.
        """
        return _minute_ewma_seeds(minutes, bar_family, metric, span, target_bars, multiplier)


# ── minute calibration internals ────────────────────────────────────────────


def _minute_target_total(
    minutes: list[MinuteInfo],
    target_bars_per_day: float,
    day_length_ms: int,
) -> float:
    """Validate minute input and return the absolute bar target."""
    if not minutes:
        raise ValueError("Cannot calibrate from an empty minute list")
    if target_bars_per_day <= 0:
        raise ValueError(f"target_bars_per_day must be positive, got {target_bars_per_day}")
    return target_bars_per_day * _observed_days([m.timestamp for m in minutes], day_length_ms)


def _minute_volatility_metrics(minutes: list[MinuteInfo]) -> list[float]:
    """Per-minute absolute close-to-close returns (first minute = 0)."""
    metrics: list[float] = []
    prev_close: float | None = None
    for m in minutes:
        if prev_close is None or prev_close == 0.0:
            metrics.append(0.0)
        else:
            metrics.append(abs(m.close - prev_close) / prev_close)
        prev_close = m.close
    return metrics


def _simulate_renko_bar_count(minutes: list[MinuteInfo], threshold: float) -> int:
    """Count renko bars produced by *threshold* (anchor resets on each close)."""
    anchor: float | None = None
    count = 0
    for m in minutes:
        if anchor is None:
            anchor = m.open
        displacement = abs(m.close - anchor) / anchor if anchor > 0.0 else 0.0
        if displacement >= threshold:
            count += 1
            anchor = None
    return count


def _simulate_hybrid_bar_count(
    minutes: list[MinuteInfo],
    dollar_threshold: float,
    volatility_threshold: float,
    multiplier: float = 1.0,
) -> int:
    """Count hybrid bars produced by the given threshold pair."""
    cum_dollar = 0.0
    cum_vol = 0.0
    prev_close: float | None = None
    count = 0
    for m in minutes:
        cum_dollar += m.close * m.volume * multiplier
        if prev_close is None or prev_close == 0.0:
            metric = 0.0
        else:
            metric = abs(m.close - prev_close) / prev_close
        cum_vol += metric
        prev_close = m.close
        if cum_dollar >= dollar_threshold and cum_vol >= volatility_threshold:
            count += 1
            cum_dollar = max(0.0, cum_dollar - dollar_threshold)
            cum_vol = max(0.0, cum_vol - volatility_threshold)
    return count


def _bisect_threshold(
    count_fn: Any,
    target: float,
    max_iter: int,
    hi_start: float = 1.0,
) -> float:
    """Find the largest threshold whose simulated count is >= *target*.

    ``count_fn`` must be non-increasing in its threshold argument.  Returns
    ``0.0`` when even a zero threshold cannot reach the target.
    """
    if count_fn(0.0) < target:
        return 0.0

    lo = 0.0
    hi = hi_start if hi_start > 0.0 else 1.0
    for _ in range(60):
        if count_fn(hi) <= target:
            break
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if count_fn(mid) >= target:
            lo = mid
        else:
            hi = mid

    if count_fn(lo) == target and count_fn(hi) == target:
        return 0.5 * (lo + hi)
    return lo


def _minute_metric_value(m: MinuteInfo, metric: str, multiplier: float = 1.0) -> float:
    """Imbalance/run weight of one minute candle."""
    if metric == "tick":
        raise ValueError("metric='tick' is not available for minute input")
    if metric == "volume":
        return m.volume
    if metric == "dollar":
        return m.close * m.volume * multiplier
    raise ValueError(f"metric must be 'volume' or 'dollar', got {metric!r}")


def _minute_signs(minutes: list[MinuteInfo]) -> list[float]:
    """Derive the minute tick rule: sign(close_i − close_{i−1}), first NaN."""
    from flowbars.tick_rule import derive_tick_sign

    prices = np.asarray([m.close for m in minutes], dtype=np.float64)
    return [float(sign) for sign in derive_tick_sign(prices)]


def _minute_ewma_seeds(
    minutes: list[MinuteInfo],
    bar_family: str,
    metric: str,
    span: float,
    target_bars: int,
    multiplier: float = 1.0,
) -> dict[str, float]:
    """Shared implementation of :meth:`MinuteCalibrationHelper.estimate_ewma_seeds`."""
    if not minutes:
        raise ValueError("Cannot estimate seeds from empty minute list")
    if bar_family not in ("imbalance", "run"):
        raise ValueError(f"bar_family must be 'imbalance' or 'run', got {bar_family!r}")
    if metric not in ("volume", "dollar"):
        raise ValueError(f"metric must be 'volume' or 'dollar', got {metric!r}")
    if target_bars <= 0:
        raise ValueError(f"target_bars must be positive, got {target_bars}")

    signs = _minute_signs(minutes)
    total = sum(_minute_metric_value(m, metric, multiplier) for m in minutes)
    threshold = total / target_bars

    if bar_family == "imbalance":
        t_stats, proportion_stats = _simulate_minute_imbalance_bars(
            minutes, metric, threshold, signs, multiplier
        )
    else:
        t_stats, proportion_stats = _simulate_minute_run_bars(
            minutes, metric, threshold, signs, multiplier
        )

    if not t_stats:
        return {
            "initial_ewa_t": float(total if total > 0.0 else 1.0),
            "initial_ewa_proportion": proportion_stats[0] if proportion_stats else 0.5,
        }

    alpha = 2.0 / (span + 1.0)
    ewa_t = t_stats[0]
    ewa_proportion = proportion_stats[0]
    for t, p in zip(t_stats[1:], proportion_stats[1:]):
        ewa_t = alpha * t + (1.0 - alpha) * ewa_t
        ewa_proportion = alpha * p + (1.0 - alpha) * ewa_proportion

    return {"initial_ewa_t": float(ewa_t), "initial_ewa_proportion": float(ewa_proportion)}


def _simulate_minute_imbalance_bars(
    minutes: list[MinuteInfo],
    metric: str,
    threshold: float,
    signs: list[float],
    multiplier: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Simulate minute imbalance bars, returning per-bar ``(T, θ)``."""
    t_stats: list[float] = []
    proportion_stats: list[float] = []
    signed_imbalance = 0.0
    total_metric = 0.0

    for m, side in zip(minutes, signs):
        metric_val = _minute_metric_value(m, metric, multiplier)
        total_metric += metric_val
        if not math.isnan(side):
            signed_imbalance += side * metric_val

        if abs(signed_imbalance) >= threshold:
            t_stats.append(total_metric)
            proportion_stats.append(signed_imbalance / total_metric if total_metric > 0.0 else 0.0)
            sign = 1.0 if signed_imbalance >= 0.0 else -1.0
            excess = abs(signed_imbalance) - threshold
            signed_imbalance = sign * max(0.0, excess)
            total_metric = 0.0

    return t_stats, proportion_stats


def _simulate_minute_run_bars(
    minutes: list[MinuteInfo],
    metric: str,
    threshold: float,
    signs: list[float],
    multiplier: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Simulate minute run bars, returning per-bar ``(T, P⁺)``."""
    t_stats: list[float] = []
    proportion_stats: list[float] = []

    first_minute = True
    banked = 0.0
    run_sign = np.nan
    run_cum = 0.0
    buy_run_total = 0.0
    sell_run_total = 0.0

    for m, side in zip(minutes, signs):
        metric_val = _minute_metric_value(m, metric, multiplier)

        if first_minute:
            first_minute = False
            run_sign = side
            run_cum = metric_val
        elif _same_direction(run_sign, side):
            run_cum += metric_val
            if np.isnan(run_sign) and not np.isnan(side):
                run_sign = side
        else:
            banked += run_cum
            if run_sign > 0.0:
                buy_run_total += run_cum
            elif run_sign < 0.0:
                sell_run_total += run_cum
            run_sign = side
            run_cum = metric_val

        total = banked + run_cum
        if total >= threshold:
            cur_buy = buy_run_total
            cur_sell = sell_run_total
            if run_sign > 0.0:
                cur_buy += run_cum
            elif run_sign < 0.0:
                cur_sell += run_cum
            total_buysell = cur_buy + cur_sell
            proportion = cur_buy / total_buysell if total_buysell > 0.0 else 0.5

            t_stats.append(total)
            proportion_stats.append(proportion)

            excess = total - threshold
            banked = 0.0
            buy_run_total = 0.0
            sell_run_total = 0.0
            run_cum = max(0.0, excess)
            if run_cum == 0.0:
                run_sign = np.nan

    return t_stats, proportion_stats
