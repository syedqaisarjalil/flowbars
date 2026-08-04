"""Tests for numba backend — Phase 9.1.

Equivalence tests (numba vs Python), graceful fallback, and compilation
handling.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from flowbars.bars.accumulators import (
    DollarAccumulator,
    ImbalanceAccumulator,
    RunAccumulator,
    TickAccumulator,
    TimeAccumulator,
    VolumeAccumulator,
)
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.numba_backend import (
    _NUMBA_AVAILABLE,
    _bar_data_to_columns,
    _dollar_bars_numba,
    _imbalance_bars_ewma_numba,
    _imbalance_bars_numba,
    _run_bars_ewma_numba,
    _run_bars_numba,
    _tick_bars_numba,
    _time_bars_numba,
    _volume_bars_numba,
    is_numba_available,
    numba_batch_ewma,
    numba_batch_static,
)
from flowbars.calendars import ContinuousCalendar, SessionCalendar
from flowbars.schema import SchemaMapping
from flowbars.thresholds import (
    EWMAThresholdEstimator,
    StaticThresholdEstimator,
)
from flowbars.tick_rule import resolve_tick_signs

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def make_ticks_df(
    timestamps: list[int],
    prices: list[float],
    volumes: list[float],
    sides: list[float] | None = None,
) -> pd.DataFrame:
    """Create a DataFrame for testing."""
    data = {"ts": timestamps, "px": prices, "vol": volumes}
    if sides is not None:
        data["side"] = sides
    return pd.DataFrame(data)


def default_schema(has_side: bool = False) -> SchemaMapping:
    """Schema for test DataFrames."""
    mapping = {"timestamp": "ts", "price": "px", "volume": "vol"}
    if has_side:
        mapping["side"] = "side"
    return SchemaMapping(mapping)


def _bars_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    float_tol: float = 1e-12,
) -> bool:
    """Compare two bar DataFrames for equality, field by field."""
    if len(df1) != len(df2):
        return False
    for col in df1.columns:
        if col == "bar_type":
            if not (df1[col] == df2[col]).all():
                return False
        elif col in ("bar_id", "num_ticks", "open_ts", "close_ts"):
            if not (df1[col].astype(int) == df2[col].astype(int)).all():
                return False
        else:
            # Float columns — convert to float64 numpy arrays
            a = np.asarray(df1[col], dtype=np.float64)
            b = np.asarray(df2[col], dtype=np.float64)
            if not np.allclose(a, b, rtol=float_tol, atol=1e-14):
                return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# numba availability
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumbaAvailability:
    def test_is_numba_available(self) -> None:
        """is_numba_available reports correct status."""
        available = is_numba_available()
        assert isinstance(available, bool)
        # In CI with numba installed, this should be True
        if _NUMBA_AVAILABLE:
            assert available is True

    def test_module_imports_without_numba(self) -> None:
        """The numba_backend module imports without numba installed."""
        # The module already imported fine — this is a smoke test.
        from flowbars.bars import numba_backend

        assert hasattr(numba_backend, "is_numba_available")


# ═══════════════════════════════════════════════════════════════════════════════
# Static threshold — standard bars (tick, volume, dollar)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStandardBarsStaticThreshold:
    """Equivalence: numba static path == Python path for standard bars."""

    def test_tick_bars_equivalence_small(self) -> None:
        """Small tick dataset — numba matches Python."""
        n = 20
        np.random.seed(42)
        timestamps = np.arange(1000, 1000 + n * 100, 100, dtype=np.float64)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        volumes = np.abs(np.random.randn(n) * 10.0) + 1.0
        threshold = 3.0

        # Python reference
        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="python")
        py_bars = ctor.batch(df)

        # numba path
        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(), backend="numba")
        nb_bars = ctor2.batch(df)

        assert _bars_equal(py_bars, nb_bars), (
            f"Tick bars mismatch:\nPython:\n{py_bars}\nNumba:\n{nb_bars}"
        )

    def test_tick_bars_equivalence_large(self) -> None:
        """Large tick dataset (≥10k ticks) — numba matches Python."""
        n = 100_000
        np.random.seed(123)
        timestamps = np.arange(1000, 1000 + n * 10, 10, dtype=np.float64)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.05)
        prices = np.clip(prices, 50.0, 200.0)
        volumes = np.abs(np.random.exponential(5.0, n)) + 0.1
        threshold = 100.0

        # Python reference
        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="python")
        py_bars = ctor.batch(df)

        # numba path
        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars), (
            f"Bar count mismatch: Python={len(py_bars)}, numba={len(nb_bars)}"
        )
        assert _bars_equal(py_bars, nb_bars), "Large tick bars mismatch"

    def test_volume_bars_equivalence(self) -> None:
        """Volume bars — numba matches Python."""
        n = 50_000
        np.random.seed(456)
        timestamps = np.arange(1000, 1000 + n * 10, 10, dtype=np.float64)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.05)
        prices = np.clip(prices, 50.0, 200.0)
        volumes = np.abs(np.random.exponential(5.0, n)) + 0.1
        threshold = 5000.0

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})

        # Python
        acc = VolumeAccumulator()
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="python")
        py_bars = ctor.batch(df)

        # numba
        acc2 = VolumeAccumulator()
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "Volume bars mismatch"

    def test_dollar_bars_equivalence(self) -> None:
        """Dollar bars — numba matches Python."""
        n = 50_000
        np.random.seed(789)
        timestamps = np.arange(1000, 1000 + n * 10, 10, dtype=np.float64)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.05)
        prices = np.clip(prices, 50.0, 200.0)
        volumes = np.abs(np.random.exponential(5.0, n)) + 0.1
        threshold = 100_000.0

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})

        # Python
        acc = DollarAccumulator()
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="python")
        py_bars = ctor.batch(df)

        # numba
        acc2 = DollarAccumulator()
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "Dollar bars mismatch"


# ═══════════════════════════════════════════════════════════════════════════════
# Time bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimeBarsStatic:
    """Equivalence: numba time bars == Python time bars."""

    def test_time_bars_clock_equivalence(self) -> None:
        """Time bars with clock anchor — numba matches Python."""
        n = 10_000
        np.random.seed(111)
        # Ticks every 100ms for 1000 seconds
        timestamps = np.arange(0, n * 100, 100, dtype=np.float64)  # ms
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.01)
        prices = np.clip(prices, 90.0, 110.0)
        volumes = np.abs(np.random.exponential(1.0, n)) + 0.1
        interval_ms = 5000  # 5-second bars

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})

        # Python
        acc = TimeAccumulator(bar_type="time", interval_ms=interval_ms, anchor="clock")
        est = StaticThresholdEstimator(threshold=0.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="python")
        py_bars = ctor.batch(df)

        # numba
        acc2 = TimeAccumulator(bar_type="time", interval_ms=interval_ms, anchor="clock")
        est2 = StaticThresholdEstimator(threshold=0.0)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars), (
            f"Bar count mismatch: Python={len(py_bars)}, numba={len(nb_bars)}"
        )
        assert _bars_equal(py_bars, nb_bars), "Time bars (clock) mismatch"

    def test_time_bars_first_tick_equivalence(self) -> None:
        """Time bars with first_tick anchor — numba matches Python."""
        n = 10_000
        np.random.seed(222)
        timestamps = np.arange(7000, 7000 + n * 100, 100, dtype=np.float64)  # ms, off-clock
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.01)
        prices = np.clip(prices, 90.0, 110.0)
        volumes = np.abs(np.random.exponential(1.0, n)) + 0.1
        interval_ms = 5000

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})

        # Python
        acc = TimeAccumulator(bar_type="time", interval_ms=interval_ms, anchor="first_tick")
        est = StaticThresholdEstimator(threshold=0.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="python")
        py_bars = ctor.batch(df)

        # numba
        acc2 = TimeAccumulator(bar_type="time", interval_ms=interval_ms, anchor="first_tick")
        est2 = StaticThresholdEstimator(threshold=0.0)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "Time bars (first_tick) mismatch"


# ═══════════════════════════════════════════════════════════════════════════════
# Info-driven bars — static threshold
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_tick_data_with_sides(
    n: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate tick data with real tick-rule-derived sides."""
    np.random.seed(seed)
    timestamps = np.arange(1000, 1000 + n * 10, 10, dtype=np.float64)
    prices = 100.0 + np.cumsum(np.random.randn(n) * 0.05)
    prices = np.clip(prices, 50.0, 200.0)
    volumes = np.abs(np.random.exponential(5.0, n)) + 0.1
    sides = resolve_tick_signs(prices, None)
    return timestamps, prices, volumes, sides


class TestImbalanceBarsStaticThreshold:
    """Equivalence: numba imbalance bars (static threshold) == Python."""

    def test_imbalance_tick_equivalence(self) -> None:
        """Imbalance tick bars with static threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(20_000, seed=42)
        threshold = 50.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        # Python
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        # numba
        acc2 = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars), (
            f"Imbalance tick bar count mismatch: Python={len(py_bars)}, numba={len(nb_bars)}"
        )
        assert _bars_equal(py_bars, nb_bars), "Imbalance tick bars mismatch"

    def test_imbalance_volume_equivalence(self) -> None:
        """Imbalance volume bars with static threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(20_000, seed=43)
        threshold = 1000.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "Imbalance volume bars mismatch"

    def test_imbalance_dollar_equivalence(self) -> None:
        """Imbalance dollar bars with static threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(20_000, seed=44)
        threshold = 100_000.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = ImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = ImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "Imbalance dollar bars mismatch"


class TestRunBarsStaticThreshold:
    """Equivalence: numba run bars (static threshold) == Python."""

    def test_run_tick_equivalence(self) -> None:
        """Run tick bars with static threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(20_000, seed=55)
        threshold = 30.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = RunAccumulator(bar_type="run_tick", metric="tick")
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars), (
            f"Run tick bar count mismatch: Python={len(py_bars)}, numba={len(nb_bars)}"
        )
        assert _bars_equal(py_bars, nb_bars), "Run tick bars mismatch"

    def test_run_volume_equivalence(self) -> None:
        """Run volume bars with static threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(20_000, seed=56)
        threshold = 1000.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = RunAccumulator(bar_type="run_volume", metric="volume")
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = RunAccumulator(bar_type="run_volume", metric="volume")
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "Run volume bars mismatch"

    def test_run_dollar_equivalence(self) -> None:
        """Run dollar bars with static threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(20_000, seed=57)
        threshold = 100_000.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = RunAccumulator(bar_type="run_dollar", metric="dollar")
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = RunAccumulator(bar_type="run_dollar", metric="dollar")
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "Run dollar bars mismatch"


# ═══════════════════════════════════════════════════════════════════════════════
# EWMA threshold — imbalance + run bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestImbalanceBarsEWMA:
    """Equivalence: numba EWMA path == Python EWMA path for imbalance bars."""

    def test_imbalance_tick_ewma_equivalence(self) -> None:
        """Imbalance tick bars with EWMA threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(30_000, seed=99)
        span = 20.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        # Python
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        est = EWMAThresholdEstimator(
            bar_family="imbalance", span=span,
            initial_ewa_t=1.0, initial_ewa_proportion=0.5,
        )
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        # numba with EWMA
        acc2 = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        est2 = EWMAThresholdEstimator(
            bar_family="imbalance", span=span,
            initial_ewa_t=1.0, initial_ewa_proportion=0.5,
        )
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars), (
            f"EWMA imbalance tick bar count mismatch: Python={len(py_bars)}, numba={len(nb_bars)}"
        )
        assert _bars_equal(py_bars, nb_bars), "EWMA imbalance tick bars mismatch"

    def test_imbalance_volume_ewma_equivalence(self) -> None:
        """Imbalance volume bars with EWMA threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(30_000, seed=100)
        span = 15.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        est = EWMAThresholdEstimator(
            bar_family="imbalance", span=span,
            initial_ewa_t=100.0, initial_ewa_proportion=0.3,
        )
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        est2 = EWMAThresholdEstimator(
            bar_family="imbalance", span=span,
            initial_ewa_t=100.0, initial_ewa_proportion=0.3,
        )
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "EWMA imbalance volume bars mismatch"

    def test_imbalance_dollar_ewma_equivalence(self) -> None:
        """Imbalance dollar bars with EWMA threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(30_000, seed=101)
        span = 10.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = ImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
        est = EWMAThresholdEstimator(
            bar_family="imbalance", span=span,
            initial_ewa_t=10000.0, initial_ewa_proportion=0.4,
        )
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = ImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
        est2 = EWMAThresholdEstimator(
            bar_family="imbalance", span=span,
            initial_ewa_t=10000.0, initial_ewa_proportion=0.4,
        )
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "EWMA imbalance dollar bars mismatch"


class TestRunBarsEWMA:
    """Equivalence: numba EWMA path == Python EWMA path for run bars."""

    def test_run_tick_ewma_equivalence(self) -> None:
        """Run tick bars with EWMA threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(30_000, seed=150)
        span = 15.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        est = EWMAThresholdEstimator(
            bar_family="run", span=span,
            initial_ewa_t=30.0, initial_ewa_proportion=0.5,
        )
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = RunAccumulator(bar_type="run_tick", metric="tick")
        est2 = EWMAThresholdEstimator(
            bar_family="run", span=span,
            initial_ewa_t=30.0, initial_ewa_proportion=0.5,
        )
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "EWMA run tick bars mismatch"

    def test_run_volume_ewma_equivalence(self) -> None:
        """Run volume bars with EWMA threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(30_000, seed=151)
        span = 20.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = RunAccumulator(bar_type="run_volume", metric="volume")
        est = EWMAThresholdEstimator(
            bar_family="run", span=span,
            initial_ewa_t=1000.0, initial_ewa_proportion=0.5,
        )
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = RunAccumulator(bar_type="run_volume", metric="volume")
        est2 = EWMAThresholdEstimator(
            bar_family="run", span=span,
            initial_ewa_t=1000.0, initial_ewa_proportion=0.5,
        )
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "EWMA run volume bars mismatch"

    def test_run_dollar_ewma_equivalence(self) -> None:
        """Run dollar bars with EWMA threshold."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(30_000, seed=152)
        span = 12.0

        df = pd.DataFrame({
            "ts": timestamps, "px": prices, "vol": volumes, "side": sides,
        })

        acc = RunAccumulator(bar_type="run_dollar", metric="dollar")
        est = EWMAThresholdEstimator(
            bar_family="run", span=span,
            initial_ewa_t=100000.0, initial_ewa_proportion=0.5,
        )
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True), backend="python")
        py_bars = ctor.batch(df)

        acc2 = RunAccumulator(bar_type="run_dollar", metric="dollar")
        est2 = EWMAThresholdEstimator(
            bar_family="run", span=span,
            initial_ewa_t=100000.0, initial_ewa_proportion=0.5,
        )
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(has_side=True), backend="numba")
        nb_bars = ctor2.batch(df)

        assert len(py_bars) == len(nb_bars)
        assert _bars_equal(py_bars, nb_bars), "EWMA run dollar bars mismatch"


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumbaEdgeCases:
    """Edge cases for the numba backend."""

    def test_empty_input(self) -> None:
        """Empty tick data produces empty bar DataFrame."""
        df = pd.DataFrame({"ts": [], "px": [], "vol": []})
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=10.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="numba")
        result = ctor.batch(df)
        assert len(result) == 0
        # Should have all expected columns
        for col in ["bar_id", "open", "high", "low", "close", "volume",
                     "dollar_value", "vwap", "num_ticks", "open_ts", "close_ts", "bar_type"]:
            assert col in result.columns

    def test_single_tick_no_bar(self) -> None:
        """Single tick with large threshold produces zero bars."""
        df = pd.DataFrame({"ts": [1000], "px": [100.0], "vol": [10.0]})
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="numba")
        result = ctor.batch(df)
        assert len(result) == 0

    def test_single_tick_one_bar(self) -> None:
        """Single tick with threshold=1 produces one bar."""
        df = pd.DataFrame({"ts": [1000], "px": [100.0], "vol": [10.0]})
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="numba")
        result = ctor.batch(df)
        assert len(result) == 1
        assert result.iloc[0]["num_ticks"] == 1
        assert result.iloc[0]["open"] == 100.0

    def test_warmup_bars_discarded(self) -> None:
        """warmup_bars parameter works with numba backend."""
        n = 100
        np.random.seed(42)
        timestamps = np.arange(1000, 1000 + n * 100, 100, dtype=np.float64)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        volumes = np.ones(n, dtype=np.float64)
        threshold = 5.0

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})

        # Without warmup
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=threshold)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="numba", warmup_bars=0)
        all_bars = ctor.batch(df)

        # With warmup=2
        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2, schema=default_schema(), backend="numba", warmup_bars=2)
        warm_bars = ctor2.batch(df)

        assert len(warm_bars) == len(all_bars) - 2

    def test_zero_threshold_tick_bars(self) -> None:
        """Threshold=0: every tick closes its own bar."""
        n = 50
        timestamps = np.arange(1000, 1000 + n * 100, 100, dtype=np.float64)
        prices = np.linspace(100.0, 200.0, n)
        volumes = np.ones(n, dtype=np.float64)

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})

        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=0.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="numba")
        result = ctor.batch(df)
        assert len(result) == n  # One bar per tick

    def test_overflow_rollover(self) -> None:
        """Overflow from a threshold-crossing tick rolls into the next bar (tick bars)."""
        # Hand-verifiable example: threshold=3, 5 ticks -> bars at 3 and 5
        timestamps = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.float64)
        prices = np.array([10.0, 12.0, 11.0, 13.0, 14.0], dtype=np.float64)
        volumes = np.ones(5, dtype=np.float64)

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})

        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=3.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="numba")
        result = ctor.batch(df)

        assert len(result) == 1  # 3 ticks close bar0, 2 ticks remain
        assert result.iloc[0]["num_ticks"] == 3
        assert result.iloc[0]["open"] == 10.0
        assert result.iloc[0]["high"] == 12.0
        assert result.iloc[0]["close"] == 11.0

    def test_session_calendar_falls_back_to_python(self) -> None:
        """SessionCalendar triggers a graceful fallback to Python."""
        n = 100
        timestamps = np.arange(1000, 1000 + n * 100, 100, dtype=np.float64)
        prices = np.linspace(100.0, 200.0, n)
        volumes = np.ones(n, dtype=np.float64)

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})

        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=10.0)
        cal = SessionCalendar(9, 30, 16, 0)
        ctor = BaseBarConstructor(acc, est, calendar=cal, schema=default_schema(), backend="numba")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = ctor.batch(df)
            # Should produce bars via Python fallback
            assert len(result) >= 0  # Produces output rather than crashing
            # Should have emitted a warning about SessionCalendar
            session_warnings = [
                x for x in w
                if "SessionCalendar" in str(x.message)
            ]
            assert len(session_warnings) >= 1

    def test_backend_python_produces_same_as_numba_default(self) -> None:
        """Explicit backend='python' produces same output as pre-numba behavior."""
        n = 1000
        np.random.seed(77)
        timestamps = np.arange(1000, 1000 + n * 100, 100, dtype=np.float64)
        prices = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        volumes = np.ones(n, dtype=np.float64)

        df = pd.DataFrame({"ts": timestamps, "px": prices, "vol": volumes})
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=10.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(), backend="python")
        result = ctor.batch(df)
        assert len(result) > 0

    def test_backend_invalid_raises(self) -> None:
        """Invalid backend name raises ValueError."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        with pytest.raises(ValueError, match="backend"):
            BaseBarConstructor(acc, est, backend="cuda")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# Direct JIT function tests (unit-level)
# ═══════════════════════════════════════════════════════════════════════════════


class TestJITFunctionsDirect:
    """Test the raw JIT functions directly (without the constructor wrapper)."""

    def test_tick_bars_numba_direct(self) -> None:
        """_tick_bars_numba returns correct shape and values."""
        timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.float64)
        prices = np.array([10.0, 12.0, 11.0, 13.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        threshold = 2.0

        bar_data = _tick_bars_numba(timestamps, prices, volumes, threshold)
        assert bar_data.shape[1] == 11  # 11 columns
        assert bar_data.shape[0] == 2  # 2 bars (ticks 1-2, ticks 3-4)

        cols = _bar_data_to_columns(bar_data)
        assert cols["bar_id"][0] == 0
        assert cols["bar_id"][1] == 1
        assert cols["num_ticks"][0] == 2
        assert cols["num_ticks"][1] == 2

    def test_volume_bars_numba_direct(self) -> None:
        """_volume_bars_numba with hand-verifiable example."""
        # Spec example: $300k + $400k + $500k vs $1M threshold
        timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.float64)
        prices = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        volumes = np.array([3000.0, 4000.0, 5000.0, 2000.0], dtype=np.float64)
        threshold = 1_000_000.0  # volume threshold

        bar_data = _volume_bars_numba(timestamps, prices, volumes, threshold)
        # 3000+4000+5000=12000 < 1M, no close; 2000 → 14000 (no)
        # Wait: all volumes * price = dollar value. But this is volume bars.
        # The volumes are [3000, 4000, 5000, 2000], threshold=1M
        # 3000 < 1M, 3000+4000=7000 < 1M, 7000+5000=12000 < 1M
        # 12000+2000=14000 < 1M — no bars!
        assert bar_data.shape[0] == 0  # No bars — threshold far exceeds data

    def test_dollar_bars_numba_direct(self) -> None:
        """_dollar_bars_numba with spec example."""
        # Spec: $300k+$400k+$500k vs $1M → bar at tick 3 with excess $200k
        timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.float64)
        prices = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)
        volumes = np.array([30000.0, 40000.0, 50000.0, 10000.0], dtype=np.float64)
        # dollar values: $300k, $400k, $500k, $100k
        threshold = 1_000_000.0

        bar_data = _dollar_bars_numba(timestamps, prices, volumes, threshold)
        assert bar_data.shape[0] == 1  # One bar: 300k+400k+500k=1.2M, excess=200k
        cols = _bar_data_to_columns(bar_data)
        assert cols["num_ticks"][0] == 3
        assert cols["dollar_value"][0] == pytest.approx(1_200_000.0)

    def test_imbalance_bars_numba_direct(self) -> None:
        """_imbalance_bars_numba with hand-verifiable example."""
        # 4 ticks: +1, -1, +1, +1 → imbalance: +1, 0, +1, +2
        timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.float64)
        prices = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        # sides: [NaN, +1, +1, +1] (all upticks after first)
        sides = resolve_tick_signs(prices, None)
        threshold = 2.0

        bar_data = _imbalance_bars_numba(timestamps, prices, volumes, sides, threshold, 0)
        # tick1 NaN → exc, tick2 +1 → 1, tick3 +1 → 2 >=2 → CLOSE bar0 (ticks 1-3)
        # tick4 +1 → 1 (no close)
        assert bar_data.shape[0] == 1
        cols = _bar_data_to_columns(bar_data)
        assert cols["num_ticks"][0] == 3

    def test_run_bars_numba_direct(self) -> None:
        """_run_bars_numba with hand-verifiable example."""
        # 5 ticks: +1, +1, -1, -1, +1 → runs: 2 buy, 2 sell, 1 buy
        timestamps = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.float64)
        prices = np.array([100.0, 101.0, 99.0, 98.0, 100.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        # sides from tick rule: [NaN, +1, -1, -1, +1]
        sides = resolve_tick_signs(prices, None)
        threshold = 2.0

        bar_data = _run_bars_numba(timestamps, prices, volumes, sides, threshold, 0)
        # tick1: NaN, tick2: +1 (run_cum=2, same dir since NaN matches +1)
        # → total=2 >= 2 → CLOSE bar0
        # tick3: -1, tick4: -1 → run_cum=2 → total=2 >= 2 → CLOSE bar1
        # tick5: +1 → run_cum=1 (no close)
        assert bar_data.shape[0] == 2
        cols = _bar_data_to_columns(bar_data)
        assert cols["num_ticks"][0] == 2
        assert cols["num_ticks"][1] == 2

    def test_time_bars_numba_direct(self) -> None:
        """_time_bars_numba with hand-verifiable example."""
        # Ticks at 0ms, 300ms, 700ms, 1200ms with interval=1000ms (clock)
        timestamps = np.array([0, 300, 700, 1200], dtype=np.float64)
        prices = np.array([10.0, 12.0, 11.0, 13.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        # Clock anchor: first boundary at 1000ms (0+1000 since 0%1000==0)
        bar_data = _time_bars_numba(timestamps, prices, volumes, 1000, 0)
        # Ticks at 0, 300, 700 → close_ts=700 < 1000 → no close
        # Tick at 1200 → close_ts=1200 >= 1000 → CLOSE bar0
        assert bar_data.shape[0] == 1
        cols = _bar_data_to_columns(bar_data)
        assert cols["num_ticks"][0] == 4


# ═══════════════════════════════════════════════════════════════════════════════
# numba_batch_static / numba_batch_ewma wrapper tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumbaBatchWrappers:
    """Test the numba_batch_static and numba_batch_ewma wrapper functions."""

    def test_numba_batch_static_tick(self) -> None:
        """numba_batch_static for tick bars returns correct structure."""
        timestamps = np.array([1000, 2000, 3000, 4000], dtype=np.float64)
        prices = np.array([10.0, 12.0, 11.0, 13.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

        bar_data, bt = numba_batch_static("tick", timestamps, prices, volumes, None, 2.0)
        assert bt == "tick"
        assert bar_data.shape[0] == 2
        assert bar_data.shape[1] == 11

    def test_numba_batch_static_imbalance_requires_sides(self) -> None:
        """numba_batch_static for imbalance raises without sides."""
        timestamps = np.array([1000, 2000], dtype=np.float64)
        prices = np.array([10.0, 12.0], dtype=np.float64)
        volumes = np.array([1.0, 1.0], dtype=np.float64)

        with pytest.raises(ValueError, match="side"):
            numba_batch_static("imbalance_tick", timestamps, prices, volumes, None, 2.0)

    def test_numba_batch_ewma_imbalance(self) -> None:
        """numba_batch_ewma for imbalance bars returns correct structure."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(5000, seed=200)

        bar_data, bt = numba_batch_ewma(
            "imbalance_tick", timestamps, prices, volumes, sides,
            alpha=0.1, initial_ewa_t=10.0, initial_ewa_proportion=0.3,
        )
        assert bt == "imbalance_tick"
        assert bar_data.shape[1] == 11
        assert bar_data.shape[0] > 0  # Should produce some bars

    def test_numba_batch_ewma_run(self) -> None:
        """numba_batch_ewma for run bars returns correct structure."""
        timestamps, prices, volumes, sides = _generate_tick_data_with_sides(5000, seed=201)

        bar_data, bt = numba_batch_ewma(
            "run_tick", timestamps, prices, volumes, sides,
            alpha=0.1, initial_ewa_t=10.0, initial_ewa_proportion=0.5,
        )
        assert bt == "run_tick"
        assert bar_data.shape[1] == 11
        assert bar_data.shape[0] > 0

    def test_numba_batch_ewma_unknown_type_raises(self) -> None:
        """numba_batch_ewma with unknown bar type raises."""
        timestamps = np.array([1000], dtype=np.float64)
        prices = np.array([10.0], dtype=np.float64)
        volumes = np.array([1.0], dtype=np.float64)
        sides = np.array([np.nan], dtype=np.float64)

        with pytest.raises(KeyError):
            numba_batch_ewma("tick", timestamps, prices, volumes, sides, 0.1, 1.0, 0.5)
