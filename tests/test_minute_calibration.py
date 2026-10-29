"""Deterministic calibration tests for ``MinuteCalibrationHelper``.

Hand-computed thresholds for the closed-form estimators, bisection
reproducibility and target proximity for renko/hybrid, EWMA seed sanity,
multi-day normalization, and validation errors.
"""

from __future__ import annotations

import pytest

from flowbars.core import MinuteInfo
from flowbars.thresholds import MinuteCalibrationHelper

DAY_MS = 86_400_000


def _mi(
    ts: int,
    close: float,
    volume: float = 1.0,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> MinuteInfo:
    o = close if open_ is None else open_
    return MinuteInfo(
        timestamp=ts,
        open=o,
        high=o if high is None else high,
        low=o if low is None else low,
        close=close,
        volume=volume,
    )


def _minute_series(volumes: list[float], closes: list[float]) -> list[MinuteInfo]:
    return [
        _mi(i * 60_000, close, volume=vol) for i, (vol, close) in enumerate(zip(volumes, closes))
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Closed-form estimators
# ═══════════════════════════════════════════════════════════════════════════════


class TestClosedForm:
    def test_volume_threshold(self) -> None:
        minutes = _minute_series([1.0, 2.0, 3.0, 4.0], [100.0] * 4)
        assert MinuteCalibrationHelper.estimate_volume_threshold(minutes, 2.0) == pytest.approx(5.0)

    def test_dollar_threshold(self) -> None:
        minutes = _minute_series([1.0, 2.0, 3.0, 4.0], [100.0, 110.0, 120.0, 130.0])
        # 100 + 220 + 360 + 520 = 1200
        assert MinuteCalibrationHelper.estimate_dollar_threshold(minutes, 3.0) == pytest.approx(
            400.0
        )

    def test_volatility_threshold(self) -> None:
        minutes = _minute_series([1.0] * 4, [100.0, 110.0, 121.0, 100.0])
        expected = (0.0 + 0.1 + 0.1 + 21.0 / 121.0) / 2.0
        assert MinuteCalibrationHelper.estimate_volatility_threshold(minutes, 2.0) == pytest.approx(
            expected
        )

    def test_volatility_zero_previous_close(self) -> None:
        minutes = _minute_series([1.0] * 3, [0.0, 10.0, 11.0])
        # first minute and the minute after a zero close contribute 0
        expected = (0.0 + 0.0 + 1.0 / 10.0) / 2.0
        assert MinuteCalibrationHelper.estimate_volatility_threshold(minutes, 2.0) == pytest.approx(
            expected
        )

    def test_range_threshold(self) -> None:
        minutes = [
            _mi(0, 100.0, high=101.0, low=99.0),
            _mi(60_000, 110.0, high=111.0, low=109.0),
            _mi(120_000, 121.0, high=122.0, low=120.0),
            _mi(180_000, 100.0, high=101.0, low=99.0),
        ]
        per_minute = 2.0 / 100.0 + 2.0 / 110.0 + 2.0 / 121.0 + 2.0 / 100.0
        assert MinuteCalibrationHelper.estimate_range_threshold(minutes, 2.0) == pytest.approx(
            per_minute / 2.0
        )

    def test_range_zero_close_contributes_nothing(self) -> None:
        minutes = [_mi(0, 0.0, high=1.0, low=0.0), _mi(60_000, 0.0, high=1.0, low=0.0)]
        assert MinuteCalibrationHelper.estimate_range_threshold(minutes, 2.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Renko bisection
# ═══════════════════════════════════════════════════════════════════════════════


def _renko_blocks(blocks: int = 3, steps: int = 9) -> list[MinuteInfo]:
    """Upward runs of ``steps`` minutes; each block can produce one bar."""
    minutes = []
    for b in range(blocks):
        for j in range(steps):
            minutes.append(_mi((b * steps + j) * 60_000, 100.0 + 2.0 * j, open_=100.0))
    return minutes


class TestRenko:
    def test_matches_simulated_target(self) -> None:
        minutes = _renko_blocks()
        t = MinuteCalibrationHelper.estimate_renko_threshold(minutes, 3.0)
        # Each 9-minute block tops out at a 16% displacement from its anchor.
        assert t == pytest.approx(0.16, abs=1e-6)

        # Simulated count at the returned threshold hits the target.
        from flowbars.thresholds import _simulate_renko_bar_count

        assert _simulate_renko_bar_count(minutes, t) == 3

    def test_reproducible(self) -> None:
        minutes = _renko_blocks()
        first = MinuteCalibrationHelper.estimate_renko_threshold(minutes, 2.0)
        second = MinuteCalibrationHelper.estimate_renko_threshold(minutes, 2.0)
        assert first == second

    def test_target_unreachable_returns_zero(self) -> None:
        minutes = [_mi(i * 60_000, 100.0, open_=100.0) for i in range(5)]
        assert MinuteCalibrationHelper.estimate_renko_threshold(minutes, 2.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid bisection
# ═══════════════════════════════════════════════════════════════════════════════


def _hybrid_minutes() -> list[MinuteInfo]:
    return [
        _mi(0, 100.0, volume=10.0),
        _mi(60_000, 101.0, volume=10.0),
        _mi(120_000, 100.0, volume=10.0),
        _mi(180_000, 101.0, volume=10.0),
    ]


class TestHybrid:
    def test_fixed_dollar_bisects_volatility(self) -> None:
        minutes = _hybrid_minutes()
        dollar, vol = MinuteCalibrationHelper.estimate_hybrid_thresholds(
            minutes, 1.0, dollar_threshold=2_000.0
        )
        assert dollar == 2_000.0
        # First bar closes once cumulative volatility reaches this value.
        expected = 0.0 + 0.01 + 1.0 / 101.0 + 0.01
        assert vol == pytest.approx(expected, abs=1e-6)

    def test_fixed_volatility_bisects_dollar(self) -> None:
        minutes = _hybrid_minutes()
        dollar, vol = MinuteCalibrationHelper.estimate_hybrid_thresholds(
            minutes, 1.0, volatility_threshold=0.0
        )
        # Volatility is always met; the dollar target alone sets the bar size.
        assert vol == 0.0
        assert dollar >= 2_000.0

        from flowbars.thresholds import _simulate_hybrid_bar_count

        assert _simulate_hybrid_bar_count(minutes, dollar, vol) == 1

    def test_both_supplied_passthrough(self) -> None:
        minutes = _hybrid_minutes()
        assert MinuteCalibrationHelper.estimate_hybrid_thresholds(
            minutes, 1.0, dollar_threshold=123.0, volatility_threshold=0.004
        ) == (123.0, 0.004)

    def test_ray_uses_one_scalar(self) -> None:
        minutes = _hybrid_minutes()
        dollar, vol = MinuteCalibrationHelper.estimate_hybrid_thresholds(minutes, 1.0)
        mean_dollar = sum(m.close * m.volume for m in minutes) / len(minutes)
        mean_vol = (0.0 + 0.01 + 1.0 / 101.0 + 0.01) / len(minutes)
        assert dollar / mean_dollar == pytest.approx(vol / mean_vol, rel=1e-9)
        assert dollar > 0.0 and vol > 0.0

    def test_reproducible(self) -> None:
        minutes = _hybrid_minutes()
        first = MinuteCalibrationHelper.estimate_hybrid_thresholds(minutes, 1.0)
        second = MinuteCalibrationHelper.estimate_hybrid_thresholds(minutes, 1.0)
        assert first == second


# ═══════════════════════════════════════════════════════════════════════════════
# EWMA seeds
# ═══════════════════════════════════════════════════════════════════════════════


class TestEWMASeeds:
    @pytest.mark.parametrize("family", ["imbalance", "run"])
    @pytest.mark.parametrize("metric", ["volume", "dollar"])
    def test_keys_and_positivity(self, family: str, metric: str) -> None:
        closes = [100.0 + (i % 5) for i in range(60)]
        minutes = _minute_series([float(1 + i % 7) for i in range(60)], closes)
        seeds = MinuteCalibrationHelper.estimate_ewma_seeds(
            minutes, family, metric=metric, span=10.0, target_bars=10
        )
        assert set(seeds) == {"initial_ewa_t", "initial_ewa_proportion"}
        assert seeds["initial_ewa_t"] > 0.0
        assert 0.0 <= seeds["initial_ewa_proportion"] <= 1.0

    def test_reproducible(self) -> None:
        closes = [100.0 + (i % 5) for i in range(60)]
        minutes = _minute_series([1.0] * 60, closes)
        first = MinuteCalibrationHelper.estimate_ewma_seeds(minutes, "run", target_bars=10)
        second = MinuteCalibrationHelper.estimate_ewma_seeds(minutes, "run", target_bars=10)
        assert first == second

    def test_rejects_tick_metric(self) -> None:
        minutes = _minute_series([1.0] * 5, [100.0] * 5)
        with pytest.raises(ValueError, match="tick"):
            MinuteCalibrationHelper.estimate_ewma_seeds(minutes, "run", metric="tick")

    def test_rejects_unknown_family(self) -> None:
        minutes = _minute_series([1.0] * 5, [100.0] * 5)
        with pytest.raises(ValueError, match="bar_family"):
            MinuteCalibrationHelper.estimate_ewma_seeds(minutes, "time")


# ═══════════════════════════════════════════════════════════════════════════════
# Day normalization and validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestDayNormalization:
    def test_two_identical_days_scale_target(self) -> None:
        day = _minute_series([1.0, 2.0, 3.0, 4.0], [100.0, 110.0, 120.0, 130.0])
        two_days = day + [
            MinuteInfo(
                timestamp=m.timestamp + DAY_MS,
                open=m.open,
                high=m.high,
                low=m.low,
                close=m.close,
                volume=m.volume,
            )
            for m in day
        ]
        single = MinuteCalibrationHelper.estimate_volume_threshold(day, 2.0)
        doubled = MinuteCalibrationHelper.estimate_volume_threshold(two_days, 2.0)
        assert doubled == pytest.approx(single)

    def test_custom_day_length(self) -> None:
        day = _minute_series([1.0, 2.0], [100.0, 101.0])
        # Two 12-hour buckets observed with a 12h day length.
        long = day + [_mi(12 * 3_600_000, 102.0, volume=1.0)]
        same_bucket = MinuteCalibrationHelper.estimate_volume_threshold(long, 2.0)
        split_bucket = MinuteCalibrationHelper.estimate_volume_threshold(
            long, 2.0, day_length_ms=12 * 3_600_000
        )
        assert split_bucket == pytest.approx(same_bucket / 2.0)


class TestValidation:
    @pytest.mark.parametrize(
        "method",
        [
            "estimate_volume_threshold",
            "estimate_dollar_threshold",
            "estimate_volatility_threshold",
            "estimate_range_threshold",
            "estimate_renko_threshold",
            "estimate_hybrid_thresholds",
        ],
    )
    def test_empty_minutes_raise(self, method: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            getattr(MinuteCalibrationHelper, method)([], 1.0)

    @pytest.mark.parametrize("target", [0.0, -1.0])
    def test_non_positive_target_raises(self, target: float) -> None:
        minutes = _minute_series([1.0], [100.0])
        with pytest.raises(ValueError, match="positive"):
            MinuteCalibrationHelper.estimate_volume_threshold(minutes, target)

    def test_negative_hybrid_threshold_raises(self) -> None:
        minutes = _minute_series([1.0], [100.0])
        with pytest.raises(ValueError, match="non-negative"):
            MinuteCalibrationHelper.estimate_hybrid_thresholds(minutes, 1.0, dollar_threshold=-1.0)
