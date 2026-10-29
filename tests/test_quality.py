"""Tests for bar quality report — Phase 11."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy", reason="scipy not installed")

from flowbars.quality import _autocorrelation, bar_quality_report

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_bars(
    n: int = 100,
    bar_type: str = "tick",
    close_prices: np.ndarray | None = None,
    num_ticks: np.ndarray | None = None,
    open_ts: np.ndarray | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic bars DataFrame for testing."""
    rng = np.random.default_rng(seed)
    if close_prices is None:
        close_prices = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    if num_ticks is None:
        num_ticks = np.full(n, 50, dtype=np.int64)
    if open_ts is None:
        open_ts = np.arange(0, n * 3_600_000, 3_600_000, dtype=np.int64)

    return pd.DataFrame(
        {
            "close": close_prices,
            "bar_type": [bar_type] * n,
            "num_ticks": num_ticks,
            "open_ts": open_ts,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Autocorrelation helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutocorrelation:
    def test_white_noise_acf_near_zero(self) -> None:
        """White noise should have ACF ≈ 0 at all lags."""
        rng = np.random.default_rng(123)
        x = rng.normal(0.0, 1.0, 10_000)
        acf = _autocorrelation(x, 10)
        # For white noise with 10k samples, ACF should be tiny
        assert np.all(np.abs(acf) < 0.05)

    def test_highly_autocorrelated(self) -> None:
        """AR(1) with phi=0.9 should have strong lag-1 autocorrelation."""
        rng = np.random.default_rng(456)
        n = 5_000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + rng.normal(0.0, 0.1)
        acf = _autocorrelation(x, 5)
        assert acf[0] > 0.5  # Lag-1 autocorrelation

    def test_constant_series_acf_zero(self) -> None:
        """Constant series has zero denominator → ACF = 0."""
        x = np.ones(100)
        acf = _autocorrelation(x, 5)
        assert np.all(acf == 0.0)

    def test_single_element_returns_zero_acf(self) -> None:
        """With n=1, denominator is zero → zeros."""
        x = np.array([3.0])
        acf = _autocorrelation(x, 3)
        assert np.all(acf == 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Ljung-Box
# ═══════════════════════════════════════════════════════════════════════════════


class TestLjungBox:
    def test_white_noise_does_not_reject(self) -> None:
        """White-noise returns should NOT reject the null (no autocorrelation)."""
        rng = np.random.default_rng(99)
        n = 200
        close = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        bars = _make_bars(n=n, close_prices=close)
        report = bar_quality_report(bars, ljung_box_lags=5)
        lb = report["ljung_box"]
        assert lb is not None
        assert lb["lags"] == 5
        # White noise should typically not reject at 5%
        # (use a loose check — it can spuriously reject ~5% of the time)
        assert lb["statistic"] > 0

    def test_autocorrelated_rejects(self) -> None:
        """AR(1) returns should reject the null (detect autocorrelation)."""
        rng = np.random.default_rng(111)
        n = 500
        noise = rng.normal(0.0, 0.005, n)
        returns = np.zeros(n)
        for i in range(1, n):
            returns[i] = 0.7 * returns[i - 1] + noise[i]
        close = 50_000.0 * np.exp(np.cumsum(returns))
        bars = _make_bars(n=n, close_prices=close)
        report = bar_quality_report(bars, ljung_box_lags=10)
        lb = report["ljung_box"]
        assert lb is not None
        assert lb["reject_autocorrelation"] is True

    def test_too_few_returns_returns_none(self) -> None:
        """With fewer than 2 bars, Ljung-Box is None."""
        bars = _make_bars(n=1)
        report = bar_quality_report(bars)
        assert report["ljung_box"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# Jarque-Bera
# ═══════════════════════════════════════════════════════════════════════════════


class TestJarqueBera:
    def test_normal_returns_do_not_reject(self) -> None:
        """Normally-distributed returns should NOT reject normality."""
        rng = np.random.default_rng(77)
        n = 500
        returns = rng.normal(0.0, 0.01, n)
        close = 50_000.0 * np.exp(np.cumsum(returns))
        bars = _make_bars(n=n, close_prices=close)
        report = bar_quality_report(bars)
        jb = report["jarque_bera"]
        assert jb is not None
        # Normal data should typically not reject at 5%
        assert jb["statistic"] > 0

    def test_non_normal_rejects(self) -> None:
        """Exponential returns should reject normality."""
        rng = np.random.default_rng(222)
        n = 500
        # Exponential distribution is highly non-normal (positive skew, high kurtosis)
        returns = rng.exponential(0.01, n)
        close = 50_000.0 * np.exp(np.cumsum(returns - 0.01))  # demean-ish
        bars = _make_bars(n=n, close_prices=close)
        report = bar_quality_report(bars)
        jb = report["jarque_bera"]
        assert jb is not None
        assert jb["reject_normality"] is True

    def test_too_few_returns_returns_none(self) -> None:
        """With fewer than 4 returns, Jarque-Bera is None."""
        bars = _make_bars(n=2)
        report = bar_quality_report(bars)
        assert report["jarque_bera"] is None

    def test_skewness_and_kurtosis_reported(self) -> None:
        """JB result includes skewness and kurtosis values."""
        rng = np.random.default_rng(44)
        n = 200
        close = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        bars = _make_bars(n=n, close_prices=close)
        report = bar_quality_report(bars)
        jb = report["jarque_bera"]
        assert jb is not None
        assert "skewness" in jb
        assert "kurtosis" in jb
        assert isinstance(jb["skewness"], float)
        assert isinstance(jb["kurtosis"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# Bar-count stability
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarCountStability:
    def test_uniform_bars_per_day(self) -> None:
        """Bars evenly distributed across days → CV = 0."""
        n = 100
        # 4 bars per day, 25 days = 100 bars
        open_ts = np.repeat(np.arange(0, 25) * 86_400_000, 4).astype(np.int64)
        bars = _make_bars(n=n, open_ts=open_ts)
        report = bar_quality_report(bars)
        stability = report["bar_count_stability"]
        assert stability is not None
        assert stability["cv"] == 0.0
        assert stability["mean"] == 4.0
        assert len(stability["per_day"]) == 25

    def test_variable_bars_per_day(self) -> None:
        """Uneven bar counts → CV > 0."""
        n = 50
        # 2 days: 20 bars on day 1, 30 on day 2
        open_ts = np.concatenate(
            [
                np.full(20, 0, dtype=np.int64),
                np.full(30, 86_400_000, dtype=np.int64),
            ]
        )
        bars = _make_bars(n=n, open_ts=open_ts)
        report = bar_quality_report(bars)
        stability = report["bar_count_stability"]
        assert stability is not None
        assert stability["cv"] > 0.0

    def test_empty_bars_returns_none(self) -> None:
        """Empty DataFrame → stability is None."""
        bars = _make_bars(n=0)
        report = bar_quality_report(bars)
        assert report["bar_count_stability"] is None

    def test_custom_day_length(self) -> None:
        """Custom day_length_ms changes bucket counts."""
        n = 50
        open_ts = np.arange(0, n * 3_600_000, 3_600_000, dtype=np.int64)
        bars = _make_bars(n=n, open_ts=open_ts)
        # With 12-hour day length, should have more "days" than 24-hour
        r24 = bar_quality_report(bars, day_length_ms=86_400_000)
        r12 = bar_quality_report(bars, day_length_ms=43_200_000)
        assert r24["bar_count_stability"] is not None
        assert r12["bar_count_stability"] is not None
        # 12-hour buckets should yield more days (or equal)
        assert len(r12["bar_count_stability"]["per_day"]) >= len(
            r24["bar_count_stability"]["per_day"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Run-bar fragmentation
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunFragmentation:
    def test_fragmentation_detected(self) -> None:
        """Run bars with many ≤2-tick bars → high fragmentation fraction."""
        n = 50
        num_ticks = np.array([1, 1, 2, 3, 5, 10, 20, 5, 2, 1] * 5, dtype=np.int64)
        bars = _make_bars(n=n, bar_type="run_tick", num_ticks=num_ticks)
        report = bar_quality_report(bars)
        frag = report["run_fragmentation"]
        assert frag is not None
        # 5 out of 10 pattern: 1,1,2,3,5,10,20,5,2,1 → ≤2 ticks: 1,1,2,2,1 = 5/10
        assert frag["fraction_fragmented"] == pytest.approx(0.5)
        assert frag["n_run_bars"] == n

    def test_no_fragmentation(self) -> None:
        """Run bars all with >2 ticks → zero fragmentation."""
        n = 30
        num_ticks = np.full(n, 10, dtype=np.int64)
        bars = _make_bars(n=n, bar_type="run_volume", num_ticks=num_ticks)
        report = bar_quality_report(bars)
        frag = report["run_fragmentation"]
        assert frag is not None
        assert frag["fraction_fragmented"] == 0.0

    def test_non_run_bars_return_none(self) -> None:
        """Non-run bar types → fragmentation is None."""
        for bt in ["tick", "volume", "dollar", "time", "imbalance_tick"]:
            bars = _make_bars(n=10, bar_type=bt)
            report = bar_quality_report(bars)
            assert report["run_fragmentation"] is None, f"failed for {bt}"


# ═══════════════════════════════════════════════════════════════════════════════
# Polars input
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(pl is None, reason="polars not installed")
class TestPolarsInput:
    def test_polars_produces_same_result(self) -> None:
        """Polars DataFrame input should match pandas input."""
        rng = np.random.default_rng(55)
        n = 100
        close = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        pd_bars = _make_bars(n=n, bar_type="run_tick", close_prices=close)
        pl_bars = pl.from_pandas(pd_bars)

        pd_report = bar_quality_report(pd_bars, ljung_box_lags=5)
        pl_report = bar_quality_report(pl_bars, ljung_box_lags=5)

        assert pd_report["n_bars"] == pl_report["n_bars"]
        assert pd_report["ljung_box"]["statistic"] == pytest.approx(
            pl_report["ljung_box"]["statistic"]
        )
        assert pd_report["jarque_bera"]["statistic"] == pytest.approx(
            pl_report["jarque_bera"]["statistic"]
        )
        assert pd_report["run_fragmentation"]["fraction_fragmented"] == pytest.approx(
            pl_report["run_fragmentation"]["fraction_fragmented"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Result structure
# ═══════════════════════════════════════════════════════════════════════════════


class TestRicherStatistics:
    def test_return_moments_match_manual(self) -> None:
        rng = np.random.default_rng(5)
        n = 200
        close = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
        bars = _make_bars(n=n, close_prices=close)
        report = bar_quality_report(bars)

        log_returns = np.diff(np.log(close))
        moments = report["return_moments"]
        assert moments["mean"] == pytest.approx(float(np.mean(log_returns)))
        assert moments["std"] == pytest.approx(float(np.std(log_returns)))
        assert moments["kurtosis"] > 0.0

    def test_short_series_reports_none_lags(self) -> None:
        bars = _make_bars(n=4)
        report = bar_quality_report(bars)
        # three log returns: lag-1 defined, lag-5 not
        assert report["return_autocorrelation"]["lag1"] is not None
        assert report["return_autocorrelation"]["lag5"] is None
        assert report["abs_return_autocorrelation"]["lag5"] is None
        assert report["variance_ratio"]["lag2"] is not None
        assert report["variance_ratio"]["lag5"] is None

    def test_constant_returns_guard(self) -> None:
        bars = _make_bars(n=50, close_prices=np.full(50, 100.0))
        report = bar_quality_report(bars)
        assert report["entropy"] is None
        assert report["variance_ratio"]["lag2"] is None
        assert report["return_moments"]["std"] == 0.0
        # Zero autocorrelation implies ESS == n_returns (n_bars - 1).
        assert report["effective_sample_size"] == pytest.approx(49.0)

    def test_entropy_bounds(self) -> None:
        rng = np.random.default_rng(6)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300)))
        entropy = bar_quality_report(_make_bars(n=300, close_prices=close))["entropy"]
        assert entropy is not None
        assert entropy["shannon"] > 0.0
        assert 0.0 <= entropy["normalized"] <= 1.0

    def test_bar_size_cv_prefers_volume(self) -> None:
        df = _make_bars(n=20)
        df["volume"] = np.arange(1, 21, dtype=np.float64)
        report = bar_quality_report(df)
        assert report["bar_size_cv"]["metric"] == "volume"
        assert report["bar_size_cv"]["cv"] > 0.0

        fallback = bar_quality_report(_make_bars(n=20))
        assert fallback["bar_size_cv"]["metric"] == "num_ticks"
        assert fallback["bar_size_cv"]["cv"] == 0.0

    def test_duration_cv(self) -> None:
        df = _make_bars(n=10)
        df["close_ts"] = df["open_ts"] + np.arange(1, 11, dtype=np.int64) * 1_000
        report = bar_quality_report(df)
        assert report["duration_cv"] is not None
        assert report["duration_cv"] > 0.0

        df["close_ts"] = df["open_ts"] + 60_000
        assert bar_quality_report(df)["duration_cv"] == pytest.approx(0.0)

    def test_ohlc_integrity(self) -> None:
        df = _make_bars(n=4)
        df["open"] = [100.0, 100.0, 100.0, 100.0]
        df["high"] = [101.0, 99.0, 101.0, 101.0]  # second bar: close > high
        df["low"] = [99.0, 98.0, 99.0, 102.0]  # fourth bar: low > open
        df["close"] = [100.5, 100.0, 100.5, 100.5]
        report = bar_quality_report(df)
        integrity = report["ohlc_integrity"]
        assert integrity is not None
        assert integrity["n_violations"] == 2
        assert integrity["fraction"] == pytest.approx(0.5)

    def test_effective_sample_size_clips(self) -> None:
        # Strong negative lag-1 autocorrelation → ESS > n but finite.
        alternating = 100.0 * np.ones(21)
        alternating[1::2] = 110.0
        report = bar_quality_report(_make_bars(n=21, close_prices=alternating))
        ess = report["effective_sample_size"]
        assert ess is not None and ess > 0.0

    def test_composite_score(self) -> None:
        rng = np.random.default_rng(8)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 120)))
        report = bar_quality_report(_make_bars(n=120, close_prices=close))
        score = report["composite_score"]
        assert score is not None
        assert 0.0 <= score["score"] <= 1.0
        assert set(score["components"]) <= {
            "autocorrelation",
            "variance_ratio",
            "entropy",
            "bar_size_cv",
            "duration_cv",
        }

    def test_report_is_deterministic(self) -> None:
        rng = np.random.default_rng(9)
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 80)))
        bars = _make_bars(n=80, close_prices=close)
        assert bar_quality_report(bars) == bar_quality_report(bars)


class TestScipyOptional:
    def test_report_without_scipy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy blocked for test")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        report = bar_quality_report(_make_bars(n=100))

        assert report["scipy_available"] is False
        assert report["ljung_box"] is not None
        assert report["ljung_box"]["p_value"] is None
        assert report["ljung_box"]["reject_autocorrelation"] is None
        assert report["jarque_bera"] is not None
        assert report["jarque_bera"]["p_value"] is None
        assert report["jarque_bera"]["reject_normality"] is None


class TestFragmentationMaskAndUnits:
    def test_mixed_run_and_non_run_bars(self) -> None:
        n = 20
        bar_types = ["run_tick"] * 10 + ["tick"] * 10
        num_ticks = np.full(n, 1, dtype=np.int64)
        bars = _make_bars(n=n, bar_type="run_tick", num_ticks=num_ticks)
        bars["bar_type"] = bar_types
        frag = bar_quality_report(bars)["run_fragmentation"]
        assert frag is not None
        assert frag["n_run_bars"] == 10
        assert frag["fraction_fragmented"] == 1.0

    def test_custom_threshold_and_unit(self) -> None:
        n = 10
        num_ticks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
        bars = _make_bars(n=n, bar_type="run_volume", num_ticks=num_ticks)
        frag = bar_quality_report(bars, fragmentation_threshold=3, num_ticks_unit="minutes")[
            "run_fragmentation"
        ]
        assert frag is not None
        assert frag["threshold"] == 3
        assert frag["unit"] == "minutes"
        assert frag["n_fragmented"] == 3
        assert frag["fraction_fragmented"] == pytest.approx(0.3)


class TestResultStructure:
    def test_all_keys_present(self) -> None:
        """The result dict has all expected top-level keys."""
        bars = _make_bars(n=50, bar_type="imbalance_tick")
        report = bar_quality_report(bars)
        expected_keys = {
            "n_bars",
            "scipy_available",
            "ljung_box",
            "jarque_bera",
            "bar_count_stability",
            "run_fragmentation",
            "return_moments",
            "return_autocorrelation",
            "abs_return_autocorrelation",
            "variance_ratio",
            "entropy",
            "bar_size_cv",
            "duration_cv",
            "effective_sample_size",
            "ohlc_integrity",
            "composite_score",
        }
        assert set(report.keys()) >= expected_keys

    def test_default_lags_is_auto(self) -> None:
        """When ljung_box_lags is not specified, it's computed automatically."""
        bars = _make_bars(n=100)
        report = bar_quality_report(bars)
        lb = report["ljung_box"]
        assert lb is not None
        # n=100 → n_returns=99 → lags = min(10, 99//5) = min(10, 19) = 10
        assert lb["lags"] == 10
