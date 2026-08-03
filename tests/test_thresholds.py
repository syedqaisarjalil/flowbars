"""Tests for threshold estimators — Phase 3."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flowbars.core import ThresholdError, TickInfo
from flowbars.thresholds import (
    EWMAThresholdEstimator,
    StaticCalibrationHelper,
    StaticThresholdEstimator,
    ThresholdEstimator,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def tick(ts: int, price: float, volume: float, side: float | None = None) -> TickInfo:
    """Shorthand to create a TickInfo with optional side."""
    return TickInfo(timestamp=ts, price=price, volume=volume, side=side)


# ═══════════════════════════════════════════════════════════════════════════════
# ThresholdEstimator ABC
# ═══════════════════════════════════════════════════════════════════════════════


class TestThresholdEstimatorABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            ThresholdEstimator()  # type: ignore[abstract]

    def test_must_implement_all_abstracts(self) -> None:
        """Concrete subclasses must implement all abstract methods."""

        class Incomplete(ThresholdEstimator):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_from_state_is_classmethod(self) -> None:
        """from_state is defined as a classmethod on the ABC."""
        assert isinstance(ThresholdEstimator.__dict__["from_state"], classmethod)

    def test_on_bar_close_default_noop(self) -> None:
        """Default on_bar_close is a no-op (doesn't raise)."""
        est = StaticThresholdEstimator(5.0)
        # Should not raise
        est.on_bar_close(100.0, 0.02)


# ═══════════════════════════════════════════════════════════════════════════════
# StaticThresholdEstimator
# ═══════════════════════════════════════════════════════════════════════════════


class TestStaticThresholdEstimator:
    def test_returns_fixed_threshold(self) -> None:
        est = StaticThresholdEstimator(10.0)
        assert est.current_threshold == 10.0

    def test_default_threshold_is_zero(self) -> None:
        est = StaticThresholdEstimator()
        assert est.current_threshold == 0.0

    def test_update_is_noop(self) -> None:
        est = StaticThresholdEstimator(5.0)
        est.update(tick(1000, 100.0, 1.0))
        est.update(tick(2000, 101.0, 2.0))
        assert est.current_threshold == 5.0

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ThresholdError):
            StaticThresholdEstimator(-1.0)

    def test_get_state(self) -> None:
        est = StaticThresholdEstimator(7.5)
        state = est.get_state()
        assert state["threshold"] == 7.5
        assert state["initial_threshold"] == 7.5

    def test_load_state(self) -> None:
        est = StaticThresholdEstimator(1.0)
        est.load_state({"threshold": 3.0, "initial_threshold": 5.0})
        assert est.current_threshold == 3.0
        est.reset()
        assert est.current_threshold == 5.0  # reset to loaded initial

    def test_from_state(self) -> None:
        est = StaticThresholdEstimator.from_state({"threshold": 4.0, "initial_threshold": 10.0})
        assert est.current_threshold == 4.0
        est.reset()
        assert est.current_threshold == 10.0

    def test_from_state_missing_initial(self) -> None:
        """from_state with missing initial_threshold uses threshold as initial."""
        est = StaticThresholdEstimator.from_state({"threshold": 4.0})
        assert est.current_threshold == 4.0
        est.reset()
        assert est.current_threshold == 4.0

    def test_reset(self) -> None:
        est = StaticThresholdEstimator(5.0)
        est.load_state({"threshold": 99.0, "initial_threshold": 5.0})
        assert est.current_threshold == 99.0
        est.reset()
        assert est.current_threshold == 5.0

    def test_on_bar_close_noop(self) -> None:
        """Static estimator ignores bar-close events."""
        est = StaticThresholdEstimator(5.0)
        est.on_bar_close(100.0, 0.5)
        assert est.current_threshold == 5.0


# ═══════════════════════════════════════════════════════════════════════════════
# EWMAThresholdEstimator
# ═══════════════════════════════════════════════════════════════════════════════


class TestEWMAThresholdEstimator:
    # ── construction ─────────────────────────────────────────────────

    def test_default_construction(self) -> None:
        est = EWMAThresholdEstimator()
        assert est.alpha == pytest.approx(2.0 / 21.0)
        assert est.n_updates == 0
        # initial threshold: 1.0 * |0.5| = 0.5
        assert est.current_threshold == 0.5

    def test_invalid_bar_family_raises(self) -> None:
        with pytest.raises(ValueError):
            EWMAThresholdEstimator(bar_family="volume")

    def test_negative_span_raises(self) -> None:
        with pytest.raises(ValueError):
            EWMAThresholdEstimator(span=-5.0)

    def test_zero_span_raises(self) -> None:
        with pytest.raises(ValueError):
            EWMAThresholdEstimator(span=0.0)

    def test_negative_halflife_raises(self) -> None:
        with pytest.raises(ValueError):
            EWMAThresholdEstimator(halflife=-1.0)

    def test_negative_initial_ewa_t_raises(self) -> None:
        with pytest.raises(ThresholdError):
            EWMAThresholdEstimator(initial_ewa_t=-1.0)

    # ── span vs halflife ─────────────────────────────────────────────

    def test_halflife_takes_precedence(self) -> None:
        est = EWMAThresholdEstimator(span=20.0, halflife=10.0)
        expected_alpha = 1.0 - math.exp(-math.log(2) / 10.0)
        assert est.alpha == pytest.approx(expected_alpha)

    def test_span_and_halflife_equivalence(self) -> None:
        """Halflife=3 gives α equivalent to span ~5.4."""
        est_h = EWMAThresholdEstimator(halflife=3.0)
        alpha_h = est_h.alpha
        # Convert alpha back to equivalent span
        equiv_span = 2.0 / alpha_h - 1.0
        est_s = EWMAThresholdEstimator(span=equiv_span)
        assert est_h.alpha == pytest.approx(est_s.alpha)

    # ── initial threshold (before any bar closes) ────────────────────

    def test_initial_threshold_imbalance(self) -> None:
        est = EWMAThresholdEstimator(
            bar_family="imbalance",
            initial_ewa_t=100.0,
            initial_ewa_proportion=0.5,
        )
        # T = 100 * |0.5| = 50
        assert est.current_threshold == 50.0

    def test_initial_threshold_run(self) -> None:
        est = EWMAThresholdEstimator(
            bar_family="run",
            initial_ewa_t=100.0,
            initial_ewa_proportion=0.5,
        )
        # T = 100 * max(0.5, 0.5) = 50
        assert est.current_threshold == 50.0

    def test_initial_threshold_run_asymmetric(self) -> None:
        """max(0.8, 0.2) = 0.8 → T = 100 * 0.8 = 80."""
        est = EWMAThresholdEstimator(
            bar_family="run",
            initial_ewa_t=100.0,
            initial_ewa_proportion=0.8,
        )
        assert est.current_threshold == 80.0

    def test_initial_threshold_zero_ewa_t(self) -> None:
        """E[T]=0 → threshold=0 regardless of proportion."""
        est = EWMAThresholdEstimator(initial_ewa_t=0.0)
        assert est.current_threshold == 0.0

    # ── update is no-op ──────────────────────────────────────────────

    def test_update_is_noop(self) -> None:
        est = EWMAThresholdEstimator(initial_ewa_t=100.0, initial_ewa_proportion=0.5)
        est.update(tick(1000, 100.0, 1.0))
        est.update(tick(2000, 101.0, 2.0))
        assert est.current_threshold == 50.0
        assert est.n_updates == 0

    # ── imbalance bar: hand-computed EWMA sequence ───────────────────

    def test_imbalance_ewma_sequence(self) -> None:
        """Hand-computed EWMA values for span=20 (α=2/21).

        Initial: E[T]=1.0, E[θ]=0.5, threshold=0.5
        Bar 1: t=50, θ=0.02
          E[T] = (2/21)*50 + (19/21)*1.0 = 100/21 + 19/21 = 119/21 ≈ 5.6667
          E[θ] = (2/21)*0.02 + (19/21)*0.5 = 0.04/21 + 9.5/21 = 9.54/21 ≈ 0.4543
          T_n  = 119/21 * |9.54/21| ≈ 2.5743
        Bar 2: t=30, θ=-0.03
          E[T] = (2/21)*30 + (19/21)*(119/21) = 60/21 + 2261/441
               = (1260+2261)/441 = 3521/441 ≈ 7.9841
          E[θ] = (2/21)*(-0.03) + (19/21)*(9.54/21)
               = -0.06/21 + 181.26/441 = (-1.26+181.26)/441 = 180/441 ≈ 0.4082
          T_n  = (3521/441) * (180/441) ≈ 3.2588
        Bar 3: t=40, θ=0.01
          E[T] = (2/21)*40 + (19/21)*(3521/441) = 80/21 + 66899/9261
               = (35280+66899)/9261 = 102179/9261 ≈ 11.0333
          E[θ] = (2/21)*0.01 + (19/21)*(180/441) = 0.02/21 + 3420/9261
               = (8.82+3420)/9261 = 3428.82/9261 ≈ 0.3702
          T_n  ≈ 11.0333 * 0.3702 ≈ 4.0853
        """
        est = EWMAThresholdEstimator(bar_family="imbalance", span=20.0)

        # Initial
        assert est.current_threshold == 0.5

        # Bar 1
        est.on_bar_close(t_stat=50.0, proportion_stat=0.02)
        assert est.ewa_t == pytest.approx(119.0 / 21.0)
        assert est.ewa_proportion == pytest.approx(9.54 / 21.0)
        assert est.current_threshold == pytest.approx((119.0 / 21.0) * abs(9.54 / 21.0))
        assert est.n_updates == 1

        # Bar 2
        est.on_bar_close(t_stat=30.0, proportion_stat=-0.03)
        assert est.ewa_t == pytest.approx(3521.0 / 441.0)
        assert est.ewa_proportion == pytest.approx(180.0 / 441.0)
        assert est.n_updates == 2

        # Bar 3
        est.on_bar_close(t_stat=40.0, proportion_stat=0.01)
        assert est.n_updates == 3
        assert est.current_threshold > 4.0  # ~4.085

    # ── run bar: hand-computed EWMA sequence ─────────────────────────

    def test_run_ewma_sequence(self) -> None:
        """Hand-computed EWMA values for run bars with span=20 (α=2/21).

        Initial: E[T]=1.0, E[P⁺]=0.5, threshold=0.5
        Bar 1: t=50, P⁺=0.6
          E[T] = (2/21)*50 + (19/21)*1.0 = 119/21 ≈ 5.6667
          E[P⁺] = (2/21)*0.6 + (19/21)*0.5 = 1.2/21 + 9.5/21 = 10.7/21 ≈ 0.5095
          T_n = (119/21) * max(10.7/21, 10.3/21) = (119/21)*(10.7/21) ≈ 2.8873
        Bar 2: t=30, P⁺=0.3
          E[T] = (2/21)*30 + (19/21)*(119/21) = 3521/441 ≈ 7.9841
          E[P⁺] = (2/21)*0.3 + (19/21)*(10.7/21) = 0.6/21 + 203.3/441
                = (12.6+203.3)/441 = 215.9/441 ≈ 0.4896
          T_n = (3521/441) * max(215.9/441, 225.1/441) ≈ 7.9841 * 0.5104 ≈ 4.075
        Bar 3: t=40, P⁺=0.8
          E[T] ≈ 11.0333
          E[P⁺] = (2/21)*0.8 + (19/21)*(215.9/441)
                = 1.6/21 + 4102.1/9261 = ... ≈ 0.5191
          T_n ≈ 5.7274
        """
        est = EWMAThresholdEstimator(bar_family="run", span=20.0)

        # Initial
        assert est.current_threshold == 0.5

        # Bar 1
        est.on_bar_close(t_stat=50.0, proportion_stat=0.6)
        assert est.ewa_t == pytest.approx(119.0 / 21.0)
        assert est.ewa_proportion == pytest.approx(10.7 / 21.0)
        assert est.n_updates == 1

        # Bar 2
        est.on_bar_close(t_stat=30.0, proportion_stat=0.3)
        assert est.ewa_t == pytest.approx(3521.0 / 441.0)
        assert est.ewa_proportion == pytest.approx(215.9 / 441.0)
        assert est.n_updates == 2

        # Bar 3
        est.on_bar_close(t_stat=40.0, proportion_stat=0.8)
        assert est.n_updates == 3
        assert est.current_threshold > 5.0  # ~5.727

    # ── run: P⁺ < 0.5 → multiplier uses (1-P⁺) ──────────────────────

    def test_run_multiplier_flips_at_point_five(self) -> None:
        """When P⁺ < 0.5, max(P⁺, 1-P⁺) = 1-P⁺."""
        est = EWMAThresholdEstimator(
            bar_family="run",
            initial_ewa_t=100.0,
            initial_ewa_proportion=0.3,
        )
        # max(0.3, 0.7) = 0.7 → T = 100 * 0.7 = 70
        assert est.current_threshold == 70.0

    # ── negative t_stat raises ───────────────────────────────────────

    def test_negative_t_stat_raises(self) -> None:
        est = EWMAThresholdEstimator()
        with pytest.raises(ValueError):
            est.on_bar_close(t_stat=-1.0, proportion_stat=0.5)

    # ── convergence: many updates → alpha-weighted ───────────────────

    def test_convergence_with_many_updates(self) -> None:
        """After many identical updates, EWMA converges to that value."""
        est = EWMAThresholdEstimator(
            bar_family="imbalance",
            span=5.0,
            initial_ewa_t=1.0,
            initial_ewa_proportion=0.5,
        )
        # α = 2/(5+1) = 2/6 = 1/3
        # After many updates of (100, 0.02), should converge to those values
        for _ in range(50):
            est.on_bar_close(t_stat=100.0, proportion_stat=0.02)

        assert est.ewa_t == pytest.approx(100.0, rel=1e-6)
        assert est.ewa_proportion == pytest.approx(0.02, rel=1e-6)
        # threshold = 100 * |0.02| = 2.0
        assert est.current_threshold == pytest.approx(2.0, rel=1e-6)

    # ── negative proportion for imbalance ────────────────────────────

    def test_imbalance_negative_proportion_uses_abs(self) -> None:
        """Negative E[θ] → threshold uses |E[θ]|."""
        est = EWMAThresholdEstimator(
            bar_family="imbalance",
            initial_ewa_t=100.0,
            initial_ewa_proportion=-0.3,
        )
        # |E[θ]| = 0.3, T = 100 * 0.3 = 30
        assert est.current_threshold == 30.0

    def test_imbalance_proportion_can_be_negative(self) -> None:
        """on_bar_close accepts negative proportion for imbalance (sell-heavy)."""
        est = EWMAThresholdEstimator(bar_family="imbalance")
        # Should not raise
        est.on_bar_close(t_stat=50.0, proportion_stat=-0.5)
        assert est.current_threshold >= 0.0  # threshold is always non-negative


# ═══════════════════════════════════════════════════════════════════════════════
# State persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    def test_static_state_round_trip(self) -> None:
        est = StaticThresholdEstimator(7.0)
        state = est.get_state()

        est2 = StaticThresholdEstimator.from_state(state)
        assert est2.current_threshold == 7.0

        est2.reset()
        assert est2.current_threshold == 7.0

    def test_static_load_state(self) -> None:
        est = StaticThresholdEstimator(1.0)
        est.load_state({"threshold": 9.0, "initial_threshold": 9.0})
        assert est.current_threshold == 9.0

    def test_ewma_state_round_trip(self) -> None:
        est = EWMAThresholdEstimator(
            bar_family="imbalance",
            span=10.0,
            initial_ewa_t=50.0,
            initial_ewa_proportion=0.1,
        )
        est.on_bar_close(t_stat=100.0, proportion_stat=0.05)
        est.on_bar_close(t_stat=80.0, proportion_stat=0.03)

        state = est.get_state()
        assert state["bar_family"] == "imbalance"
        assert state["span"] == 10.0
        assert state["n_updates"] == 2

        est2 = EWMAThresholdEstimator.from_state(state)
        assert est2.current_threshold == est.current_threshold
        assert est2.ewa_t == pytest.approx(est.ewa_t)
        assert est2.ewa_proportion == pytest.approx(est.ewa_proportion)
        assert est2.n_updates == 2

    def test_ewma_load_state(self) -> None:
        est = EWMAThresholdEstimator(span=5.0)
        est.on_bar_close(t_stat=10.0, proportion_stat=0.5)

        state = est.get_state()

        est2 = EWMAThresholdEstimator(span=99.0)  # different config
        est2.load_state(state)
        assert est2.current_threshold == est.current_threshold
        assert est2.span == 5.0  # restored from state
        assert est2.n_updates == 1

    def test_ewma_state_after_reset(self) -> None:
        """After reset, state reflects initial values."""
        est = EWMAThresholdEstimator(
            bar_family="run",
            span=10.0,
            initial_ewa_t=100.0,
            initial_ewa_proportion=0.6,
        )
        est.on_bar_close(t_stat=200.0, proportion_stat=0.7)
        est.on_bar_close(t_stat=150.0, proportion_stat=0.4)
        est.reset()

        assert est.n_updates == 0
        assert est.ewa_t == 100.0
        assert est.ewa_proportion == 0.6
        assert est.current_threshold == pytest.approx(100.0 * max(0.6, 0.4))

    def test_empty_ewma_state_round_trip(self) -> None:
        """Round-trip state before any updates."""
        est = EWMAThresholdEstimator(
            bar_family="run",
            halflife=5.0,
            initial_ewa_t=200.0,
            initial_ewa_proportion=0.5,
        )
        state = est.get_state()

        est2 = EWMAThresholdEstimator.from_state(state)
        assert est2.current_threshold == est.current_threshold
        assert est2.n_updates == 0
        assert est2.halflife == 5.0

    def test_state_backward_compat_missing_optionals(self) -> None:
        """load_state handles missing optional keys with defaults."""
        est = EWMAThresholdEstimator()
        est.load_state(
            {
                "bar_family": "imbalance",
                "span": 20.0,
                "halflife": None,
                "alpha": 2.0 / 21.0,
                "ewa_t": 50.0,
                "ewa_proportion": 0.02,
                "n_updates": 3,
                # initial_ewa_t and initial_ewa_proportion missing
            }
        )
        assert est.current_threshold == pytest.approx(50.0 * 0.02)
        # Reset should still work (uses defaults then)
        est.reset()


# ═══════════════════════════════════════════════════════════════════════════════
# StaticCalibrationHelper
# ═══════════════════════════════════════════════════════════════════════════════


class TestStaticCalibrationHelper:
    # ── fixed threshold: standard bars ───────────────────────────────

    def test_tick_fixed_threshold(self) -> None:
        """100 ticks, target 10 bars/day → threshold = 10 ticks/bar."""
        ticks = [tick(i, 100.0, 1.0) for i in range(100)]
        result = StaticCalibrationHelper.estimate_fixed_threshold(ticks, "tick", 10.0)
        assert result == 10.0

    def test_volume_fixed_threshold(self) -> None:
        """Total vol=1000, target 5 bars/day → threshold = 200."""
        ticks = [tick(i, 100.0, 10.0) for i in range(100)]  # total vol = 1000
        result = StaticCalibrationHelper.estimate_fixed_threshold(ticks, "volume", 5.0)
        assert result == 200.0

    def test_dollar_fixed_threshold(self) -> None:
        """10 ticks at $100×10 = $10000, target 2 bars/day → threshold = $5000."""
        ticks = [tick(i, 100.0, 10.0) for i in range(10)]
        result = StaticCalibrationHelper.estimate_fixed_threshold(ticks, "dollar", 2.0)
        assert result == 5000.0

    def test_time_bar_raises(self) -> None:
        ticks = [tick(1000, 100.0, 1.0)]
        with pytest.raises(ValueError, match="Time bars"):
            StaticCalibrationHelper.estimate_fixed_threshold(ticks, "time", 10.0)

    # ── fixed threshold: info-driven bars (simulation) ───────────────

    def test_imbalance_tick_fixed_threshold(self) -> None:
        """Pure buy stream → threshold close to total_ticks/target."""
        rng = np.random.default_rng(42)
        ticks = [tick(i, 100.0 + rng.normal(0, 0.01), 1.0, side=1.0) for i in range(1000)]
        result = StaticCalibrationHelper.estimate_fixed_threshold(ticks, "imbalance_tick", 50.0)
        # All buys → |θ|=num_ticks in each bar, so threshold ≈ 1000/50 = 20
        assert result > 0
        # Should be in rough ballpark
        assert 15.0 <= result <= 25.0

    def test_run_tick_fixed_threshold(self) -> None:
        """Alternating buy/sell → threshold close to total/target."""
        ticks = [tick(i, 100.0, 1.0, side=1.0 if i % 2 == 0 else -1.0) for i in range(1000)]
        result = StaticCalibrationHelper.estimate_fixed_threshold(ticks, "run_tick", 50.0)
        # Each run is 1 tick, bars close at threshold ticks
        # expected ≈ 1000/50 = 20
        assert result > 0
        assert 15.0 <= result <= 25.0

    # ── edge cases ───────────────────────────────────────────────────

    def test_empty_ticks_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            StaticCalibrationHelper.estimate_fixed_threshold([], "tick", 10.0)

    def test_non_positive_target_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            StaticCalibrationHelper.estimate_fixed_threshold([tick(1000, 100.0, 1.0)], "tick", 0.0)

    def test_unknown_bar_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            StaticCalibrationHelper.estimate_fixed_threshold(
                [tick(1000, 100.0, 1.0)], "quantum", 10.0
            )

    # ── EWMA seeds ───────────────────────────────────────────────────

    def test_ewma_seeds_imbalance_tick(self) -> None:
        """Estimate seeds from a buy-heavy tick stream."""
        ticks = [
            tick(i, 100.0, 1.0, side=1.0 if i % 3 != 0 else -1.0) for i in range(300)
        ]  # 2/3 buys, 1/3 sells
        seeds = StaticCalibrationHelper.estimate_ewma_seeds(
            ticks, bar_family="imbalance", metric="tick", target_bars=10
        )
        assert "initial_ewa_t" in seeds
        assert "initial_ewa_proportion" in seeds
        assert seeds["initial_ewa_t"] > 0
        # Imbalance proportion should be positive (buy-heavy)
        assert seeds["initial_ewa_proportion"] > 0

    def test_ewma_seeds_run_tick(self) -> None:
        """Estimate seeds from alternating tick stream."""
        ticks = [tick(i, 100.0, 1.0, side=1.0 if i % 2 == 0 else -1.0) for i in range(300)]
        seeds = StaticCalibrationHelper.estimate_ewma_seeds(
            ticks, bar_family="run", metric="tick", target_bars=10
        )
        assert seeds["initial_ewa_t"] > 0
        # Alternating → roughly balanced runs, P⁺ ≈ 0.5
        assert 0.4 <= seeds["initial_ewa_proportion"] <= 0.6

    def test_ewma_seeds_volume_metric(self) -> None:
        """Estimate seeds with volume-weighted metric."""
        rng = np.random.default_rng(42)
        ticks = [
            tick(i, 100.0, float(rng.uniform(1, 100)), side=1.0 if rng.random() > 0.4 else -1.0)
            for i in range(500)
        ]
        seeds = StaticCalibrationHelper.estimate_ewma_seeds(
            ticks, bar_family="imbalance", metric="volume", target_bars=20
        )
        assert seeds["initial_ewa_t"] > 0
        assert -1.0 <= seeds["initial_ewa_proportion"] <= 1.0

    def test_ewma_seeds_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            StaticCalibrationHelper.estimate_ewma_seeds([], bar_family="imbalance", metric="tick")

    def test_ewma_seeds_invalid_bar_family_raises(self) -> None:
        with pytest.raises(ValueError, match="bar_family"):
            StaticCalibrationHelper.estimate_ewma_seeds(
                [tick(1000, 100.0, 1.0)], bar_family="dollar", metric="tick"
            )

    def test_ewma_seeds_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            StaticCalibrationHelper.estimate_ewma_seeds(
                [tick(1000, 100.0, 1.0)], bar_family="imbalance", metric="shares"
            )

    def test_ewma_seeds_single_tick(self) -> None:
        """Single tick → seeds from that one bar."""
        seeds = StaticCalibrationHelper.estimate_ewma_seeds(
            [tick(1000, 100.0, 10.0, side=1.0)],
            bar_family="imbalance",
            metric="tick",
            target_bars=1,
        )
        assert seeds["initial_ewa_t"] > 0
        assert seeds["initial_ewa_proportion"] is not None
