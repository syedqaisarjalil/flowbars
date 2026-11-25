"""Regression tests for the imbalance-bar θ invariant, ``|θ| <= 1``.

A closed imbalance bar carries its signed overflow into the next bar while the
metric denominator resets.  θ must therefore be computed from the bar's own
signed flow only.  If the carried overflow leaks into the numerator, ``|θ|``
can exceed 1 and poisons ``E[θ]``; the EWMA threshold then decays toward zero
and bars freeze.

These tests pin the fix on every code path that computes θ: the Python
accumulator, the tick and minute numba EWMA kernels, and both calibration
simulators — plus the ``t_stat == 0`` guard in ``EWMAThresholdEstimator``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowbars.bars.accumulators import ImbalanceAccumulator
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.minute import (
    MinuteImbalanceDollarBarConstructor,
    MinuteImbalanceVolumeBarConstructor,
)
from flowbars.bars.minute.accumulators import MinuteImbalanceAccumulator
from flowbars.bars.minute.engine import BaseMinuteBarConstructor
from flowbars.bars.numba_backend import _NUMBA_AVAILABLE
from flowbars.calendars import SessionCalendar
from flowbars.core import MinuteInfo, TickInfo
from flowbars.schema import MinuteSchemaMapping, SchemaMapping
from flowbars.thresholds import (
    EWMAThresholdEstimator,
    MinuteCalibrationHelper,
    StaticCalibrationHelper,
    StaticThresholdEstimator,
    _simulate_imbalance_bars,
    _simulate_minute_imbalance_bars,
)
from flowbars.tick_rule import resolve_tick_signs

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

TICK_SCHEMA = SchemaMapping({"timestamp": "ts", "price": "px", "volume": "vol", "side": "side"})
MINUTE_SCHEMA = MinuteSchemaMapping(
    {
        "timestamp": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
)

# 2024-01-02 14:30:00 UTC and 2024-01-02 21:00:00 UTC
SESSION_START_MS = 1_704_205_800_000
DAY_MS = 86_400_000

requires_numba = pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="numba not installed")


def tick(ts: int, price: float, volume: float, side: float | None = None) -> TickInfo:
    return TickInfo(timestamp=ts, price=price, volume=volume, side=side)


def minute(ts: int, close: float, volume: float, side: float | None = None) -> MinuteInfo:
    return MinuteInfo(
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        side=side,
    )


class _RecordingEWMA(EWMAThresholdEstimator):
    """EWMA estimator that records every close statistic it receives."""

    def __init__(self, **kwargs: float) -> None:
        super().__init__(bar_family="imbalance", **kwargs)
        self.stats: list[tuple[float, float]] = []
        self.ewa_proportions: list[float] = []

    def on_bar_close(self, t_stat: float, proportion_stat: float) -> None:
        self.stats.append((t_stat, proportion_stat))
        super().on_bar_close(t_stat, proportion_stat)
        self.ewa_proportions.append(self.ewa_proportion)


def _bars_equal(df1: pd.DataFrame, df2: pd.DataFrame, float_tol: float = 1e-12) -> bool:
    if list(df1.columns) != list(df2.columns) or len(df1) != len(df2):
        return False
    for col in df1.columns:
        if col == "bar_type":
            if not (df1[col] == df2[col]).all():
                return False
        elif col in ("bar_id", "num_ticks", "open_ts", "close_ts"):
            if not (df1[col].astype(np.int64) == df2[col].astype(np.int64)).all():
                return False
        else:
            a = np.asarray(df1[col], dtype=np.float64)
            b = np.asarray(df2[col], dtype=np.float64)
            if not np.allclose(a, b, rtol=float_tol, atol=1e-14):
                return False
    return True


def _tick_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = np.clip(100.0 + np.cumsum(rng.normal(0.0, 0.05, n)), 50.0, 200.0)
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype=np.int64) * 10,
            "px": prices,
            "vol": rng.exponential(5.0, n) + 0.1,
            "side": resolve_tick_signs(prices, None),
        }
    )


def _minute_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, n)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype=np.int64) * 60_000,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.001,
            "low": np.minimum(opens, closes) * 0.999,
            "close": closes,
            "volume": rng.integers(1, 31, n).astype(np.float64),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Python accumulator — θ is bounded by 1
# ═══════════════════════════════════════════════════════════════════════════════


class TestThetaInvariant:
    def test_positive_overflow_proportion_is_one(self) -> None:
        """Overflow carry must not leak into the next bar's θ.

        Bar 1 closes at 150 volume against a 100 threshold, carrying +50.
        Bar 2 then adds 1 + 49 volume: numerator 100, denominator 50.  The
        pre-fix code computed 100/50 = 2.0; θ must be 1.0.
        """
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        acc.add_tick(tick(1000, 100.0, 150.0, side=1.0))
        assert acc.should_close(100.0)

        t_stat, theta = acc.get_close_stats()
        assert t_stat == 150.0
        assert theta == pytest.approx(1.0)
        acc.close(100.0)  # carries +50

        acc.add_tick(tick(2000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(3000, 100.0, 49.0, side=1.0))
        assert acc.should_close(100.0)

        t_stat, theta = acc.get_close_stats()
        assert t_stat == 50.0
        assert theta == pytest.approx(1.0)
        assert abs(theta) <= 1.0

    def test_negative_overflow_proportion_is_minus_one(self) -> None:
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        acc.add_tick(tick(1000, 100.0, 150.0, side=-1.0))
        assert acc.should_close(100.0)
        acc.close(100.0)  # carries -50

        acc.add_tick(tick(2000, 100.0, 1.0, side=-1.0))
        acc.add_tick(tick(3000, 100.0, 49.0, side=-1.0))

        t_stat, theta = acc.get_close_stats()
        assert t_stat == 50.0
        assert theta == pytest.approx(-1.0)
        assert abs(theta) <= 1.0

    @pytest.mark.parametrize("signed", [1e9, -1e9])
    def test_theta_is_clamped(self, signed: float) -> None:
        """Defensive clamp covers any non-rational state (e.g. resumed data)."""
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc._signed_imbalance = signed
        acc._bar_start_signed = 0.0

        _, theta = acc.get_close_stats()
        assert theta == (1.0 if signed > 0 else -1.0)

    def test_zero_metric_bar_returns_zero_theta(self) -> None:
        """A zero-volume closing tick has no proportion information."""
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        acc.add_tick(tick(1000, 100.0, 150.0, side=1.0))
        acc.close(40.0)  # carries +110, already above the next threshold

        acc.add_tick(tick(2000, 100.0, 0.0, side=1.0))
        assert acc.should_close(40.0)

        t_stat, theta = acc.get_close_stats()
        assert t_stat == 0.0
        assert theta == 0.0

    def test_naive_side_tick_still_counts_in_denominator(self) -> None:
        """NaN-side ticks add to t_stat but not to θ's numerator."""
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        acc.add_tick(tick(1000, 100.0, 100.0, side=np.nan))
        acc.add_tick(tick(2000, 100.0, 100.0, side=1.0))

        t_stat, theta = acc.get_close_stats()
        assert t_stat == 200.0
        assert theta == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# EWMA estimator — zero-metric bars are ignored
# ═══════════════════════════════════════════════════════════════════════════════


class TestZeroMetricEWMAUpdate:
    def test_zero_t_stat_does_not_move_ewma(self) -> None:
        est = EWMAThresholdEstimator(
            bar_family="imbalance",
            span=5.0,
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
        )
        est.on_bar_close(12.0, 0.8)
        before = (est.ewa_t, est.ewa_proportion, est.n_updates)

        est.on_bar_close(0.0, 0.0)

        assert (est.ewa_t, est.ewa_proportion, est.n_updates) == before

    def test_negative_t_stat_still_rejected(self) -> None:
        est = EWMAThresholdEstimator(bar_family="imbalance", span=5.0)
        with pytest.raises(ValueError, match="non-negative"):
            est.on_bar_close(-1.0, 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# State persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestBarStartSignedState:
    def test_roundtrip_preserves_carry_baseline(self) -> None:
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        acc.add_tick(tick(1000, 100.0, 150.0, side=1.0))
        acc.close(100.0)  # carries +50

        state = acc.get_state()
        assert state["signed_imbalance"] == 50.0
        assert state["bar_start_signed"] == 50.0

        restored = ImbalanceAccumulator(metric="volume")
        restored.load_state(state)
        restored.add_tick(tick(2000, 100.0, 50.0, side=1.0))

        assert restored.get_close_stats() == (50.0, pytest.approx(1.0))

    def test_discard_resets_carry_baseline(self) -> None:
        acc = ImbalanceAccumulator(metric="volume")
        acc.add_tick(tick(1000, 100.0, 150.0, side=1.0))
        acc.close(100.0)  # carries +50
        acc.add_tick(tick(2000, 100.0, 10.0, side=1.0))

        acc.discard()

        state = acc.get_state()
        assert state["signed_imbalance"] == 0.0
        assert state["bar_start_signed"] == 0.0

    def test_old_state_without_field_defaults_to_zero(self) -> None:
        acc = ImbalanceAccumulator(metric="volume")
        acc.add_tick(tick(1000, 100.0, 10.0, side=1.0))

        state = acc.get_state()
        del state["bar_start_signed"]

        restored = ImbalanceAccumulator(metric="volume")
        restored.load_state(state)
        assert restored.get_state()["bar_start_signed"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming paths
# ═══════════════════════════════════════════════════════════════════════════════


def _mixed_session_ticks(start_ms: int, seed: int) -> list[TickInfo]:
    rng = np.random.default_rng(seed)
    out: list[TickInfo] = []
    ts = start_ms
    for i in range(40):  # alternating small-volume flow
        side = 1.0 if (i // 5) % 2 == 0 else -1.0
        out.append(tick(ts, 100.0 + 0.01 * i, float(rng.integers(1, 6)), side))
        ts += 1000
    out.append(tick(ts, 101.0, 500.0, side=1.0))  # overflow spike
    ts += 1000
    for _ in range(6):  # zero-volume stretch follows the carried overflow
        out.append(tick(ts, 101.0, 0.0, side=1.0))
        ts += 1000
    for i in range(40):  # recovery
        side = -1.0 if (i // 4) % 2 == 0 else 1.0
        out.append(tick(ts, 100.5 + 0.01 * i, float(rng.integers(1, 8)), side))
        ts += 1000
    return out


def _mixed_session_minutes(start_ms: int, seed: int) -> list[MinuteInfo]:
    rng = np.random.default_rng(seed)
    out: list[MinuteInfo] = []
    ts = start_ms
    close = 100.0
    for i in range(40):
        close += 0.01 * (1 if (i // 5) % 2 == 0 else -1)
        side = 1.0 if (i // 5) % 2 == 0 else -1.0
        out.append(minute(ts, close, float(rng.integers(1, 6)), side))
        ts += 60_000
    close += 5.0
    out.append(minute(ts, close, 500.0, side=1.0))  # overflow spike
    ts += 60_000
    for _ in range(6):  # zero-volume stretch
        out.append(minute(ts, close, 0.0, side=1.0))
        ts += 60_000
    for i in range(40):
        close += 0.01 * (1 if (i // 4) % 2 == 0 else -1)
        side = -1.0 if (i // 4) % 2 == 0 else 1.0
        out.append(minute(ts, close, float(rng.integers(1, 8)), side))
        ts += 60_000
    return out


def _assert_stream_invariants(est: _RecordingEWMA, bars: list[object]) -> None:
    assert est.stats, "no bars closed during the stream"
    for t_stat, theta in est.stats:
        assert t_stat >= 0.0
        assert np.isfinite(theta)
        assert abs(theta) <= 1.0 + 1e-12
    for proportion in est.ewa_proportions:
        assert np.isfinite(proportion)
        assert -1.0 - 1e-12 <= proportion <= 1.0 + 1e-12
    assert bars
    assert est.current_threshold <= est.ewa_t + 1e-9
    assert np.isfinite(est.current_threshold)


class TestTickStreaming:
    def test_zero_metric_bar_skips_ewma_update(self) -> None:
        """A carried overflow closes zero-volume bars without touching E[θ]."""
        est = _RecordingEWMA(span=4.0, initial_ewa_t=100.0, initial_ewa_proportion=0.5)
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        ctor = BaseBarConstructor(acc, est, schema=TICK_SCHEMA, backend="python")

        ctor.update(tick(1000, 100.0, 500.0, side=1.0))
        ctor.update(tick(2000, 100.0, 0.0, side=1.0))
        ctor.update(tick(3000, 100.0, 0.0, side=1.0))
        ctor.update(tick(4000, 100.0, 0.0, side=1.0))  # below threshold, no close

        assert len(est.stats) == 3
        assert est.stats[0] == (500.0, pytest.approx(1.0))
        assert est.stats[1] == (0.0, 0.0)
        assert est.stats[2] == (0.0, 0.0)
        # Only the first close moved the estimate.
        assert est.n_updates == 1
        assert est.ewa_t == pytest.approx(260.0)
        assert est.ewa_proportion == pytest.approx(0.7)

    def test_multi_session_stream_keeps_theta_bounded(self) -> None:
        calendar = SessionCalendar(14, 30, 21, 0)
        est = _RecordingEWMA(span=4.0, initial_ewa_t=100.0, initial_ewa_proportion=0.5)
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        ctor = BaseBarConstructor(acc, est, calendar=calendar, schema=TICK_SCHEMA)

        bars: list[object] = []
        for t in _mixed_session_ticks(SESSION_START_MS, seed=1):
            bar = ctor.update(t)
            if bar is not None:
                bars.append(bar)
        bars_after_session_one = len(bars)
        assert bars_after_session_one > 0

        # Session two must keep emitting bars, not freeze at the boundary.
        for t in _mixed_session_ticks(SESSION_START_MS + DAY_MS, seed=2):
            bar = ctor.update(t)
            if bar is not None:
                bars.append(bar)

        assert len(bars) > bars_after_session_one
        _assert_stream_invariants(est, bars)


class TestMinuteStreaming:
    def test_multi_session_stream_keeps_theta_bounded(self) -> None:
        calendar = SessionCalendar(14, 30, 21, 0)
        est = _RecordingEWMA(span=4.0, initial_ewa_t=50.0, initial_ewa_proportion=0.5)
        acc = MinuteImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        ctor = BaseMinuteBarConstructor(acc, est, calendar=calendar, schema=MINUTE_SCHEMA)

        bars: list[object] = []
        for m in _mixed_session_minutes(SESSION_START_MS, seed=3):
            bar = ctor.update(m)
            if bar is not None:
                bars.append(bar)
        bars_after_session_one = len(bars)
        assert bars_after_session_one > 0

        for m in _mixed_session_minutes(SESSION_START_MS + DAY_MS, seed=4):
            bar = ctor.update(m)
            if bar is not None:
                bars.append(bar)

        assert len(bars) > bars_after_session_one
        _assert_stream_invariants(est, bars)


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration simulators share the invariant
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalibrationSimulators:
    def test_simulate_imbalance_bars_excludes_carry(self) -> None:
        ticks = [
            tick(0, 100.0, 150.0, side=1.0),
            tick(1000, 100.0, 50.0, side=1.0),
        ]
        t_stats, proportions = _simulate_imbalance_bars(ticks, "volume", 100.0)

        assert t_stats == [150.0, 50.0]
        assert proportions == pytest.approx([1.0, 1.0])  # second was 2.0 pre-fix

    def test_simulate_minute_imbalance_bars_excludes_carry(self) -> None:
        minutes = [
            minute(0, 100.0, 150.0),
            minute(60_000, 100.0, 50.0),
        ]
        t_stats, proportions = _simulate_minute_imbalance_bars(minutes, "volume", 100.0, [1.0, 1.0])

        assert t_stats == [150.0, 50.0]
        assert proportions == pytest.approx([1.0, 1.0])

    def test_static_calibration_seed_is_bounded(self) -> None:
        volumes = [150.0, 50.0, 150.0, 50.0]
        ticks = [tick(i * 1000, 100.0, v, side=1.0) for i, v in enumerate(volumes)]

        seeds = StaticCalibrationHelper.estimate_ewma_seeds(
            ticks, "imbalance", metric="volume", span=4.0, target_bars=4
        )

        assert seeds["initial_ewa_proportion"] == pytest.approx(1.0)
        assert abs(seeds["initial_ewa_proportion"]) <= 1.0

    def test_minute_calibration_seed_is_bounded(self) -> None:
        volumes = [150.0, 50.0, 150.0, 50.0]
        minutes = [minute(i * 60_000, 100.0 + i, v) for i, v in enumerate(volumes)]

        seeds = MinuteCalibrationHelper.estimate_ewma_seeds(
            minutes, "imbalance", metric="volume", span=4.0, target_bars=4
        )

        # Bars: [350] θ=200/350 then [50] θ=1.0.  Pre-fix the second was 3.0.
        expected = 0.4 * 1.0 + 0.6 * (200.0 / 350.0)
        assert seeds["initial_ewa_proportion"] == pytest.approx(expected)
        assert abs(seeds["initial_ewa_proportion"]) <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Backend parity — multiplier must not split Python and numba
# ═══════════════════════════════════════════════════════════════════════════════


@requires_numba
class TestTickBackendParity:
    @pytest.mark.parametrize("metric", ["tick", "volume", "dollar"])
    @pytest.mark.parametrize("multiplier", [1.0, 3.0])
    def test_ewma_imbalance_parity(self, metric: str, multiplier: float) -> None:
        df = _tick_frame(5000, seed=7)
        bar_type = f"imbalance_{metric}"
        initial_ewa_t = {"tick": 50.0, "volume": 200.0, "dollar": 100_000.0}[metric]

        def build(backend: str) -> BaseBarConstructor:
            acc = ImbalanceAccumulator(bar_type=bar_type, metric=metric)
            est = EWMAThresholdEstimator(
                bar_family="imbalance",
                span=10.0,
                initial_ewa_t=initial_ewa_t,
                initial_ewa_proportion=0.4,
            )
            return BaseBarConstructor(
                acc,
                est,
                schema=TICK_SCHEMA,
                backend=backend,
                multiplier=multiplier,
            )

        py_bars = build("python").batch(df)
        nb_bars = build("numba").batch(df)

        assert len(py_bars) == len(nb_bars) > 0
        assert _bars_equal(py_bars, nb_bars)

    @pytest.mark.parametrize("multiplier", [1.0, 3.0])
    def test_static_imbalance_dollar_parity(self, multiplier: float) -> None:
        df = _tick_frame(5000, seed=8)

        def build(backend: str) -> BaseBarConstructor:
            acc = ImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
            est = StaticThresholdEstimator(threshold=30_000.0 * multiplier)
            return BaseBarConstructor(
                acc,
                est,
                schema=TICK_SCHEMA,
                backend=backend,
                multiplier=multiplier,
            )

        py_bars = build("python").batch(df)
        nb_bars = build("numba").batch(df)

        assert len(py_bars) == len(nb_bars) > 0
        assert _bars_equal(py_bars, nb_bars)


@requires_numba
class TestMinuteBackendParity:
    @pytest.mark.parametrize(
        ("kind", "initial_ewa_t"),
        [("volume", 200.0), ("dollar", 45_000.0)],
    )
    @pytest.mark.parametrize("multiplier", [1.0, 3.0])
    def test_ewma_imbalance_parity(
        self, kind: str, initial_ewa_t: float, multiplier: float
    ) -> None:
        df = _minute_frame(240, seed=9)

        def build(backend: str) -> BaseMinuteBarConstructor:
            cls = (
                MinuteImbalanceVolumeBarConstructor
                if kind == "volume"
                else MinuteImbalanceDollarBarConstructor
            )
            return cls(
                span=10.0,
                initial_ewa_t=initial_ewa_t,
                initial_ewa_proportion=0.4,
                schema=MINUTE_SCHEMA,
                backend=backend,
                multiplier=multiplier,
            )

        py_bars = build("python").batch(df)
        nb_bars = build("numba").batch(df)

        assert len(py_bars) == len(nb_bars) > 0
        assert _bars_equal(py_bars, nb_bars)
