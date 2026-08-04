"""Benchmark script for flowbars backends.

Run with::

    python -m flowbars.benchmarks

Measures bar-construction throughput for both the Python and numba
backends across all bar types.  Generates synthetic tick data
(≥100 000 ticks), warms up each backend before timing, and reports
speed-up honestly — compilation cost is listed separately.

Output is printed to stdout; results are intended for the README.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from flowbars.bars.accumulators import (
    DollarAccumulator,
    ImbalanceAccumulator,
    RunAccumulator,
    TickAccumulator,
    TimeAccumulator,
    VolumeAccumulator,
)
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.numba_backend import is_numba_available
from flowbars.schema import SchemaMapping
from flowbars.thresholds import (
    EWMAThresholdEstimator,
    StaticThresholdEstimator,
)
from flowbars.tick_rule import resolve_tick_signs

# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic data generation
# ═══════════════════════════════════════════════════════════════════════════════

_N_TICKS = 250_000


def _generate_ticks(n: int = _N_TICKS, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic tick data: random-walk prices, exponential volumes.

    Returns a DataFrame with columns ``timestamp``, ``price``, ``volume``,
    and a ``side`` column derived via the tick rule.
    """
    rng = np.random.default_rng(seed)
    timestamps = np.arange(0, n * 100, 100, dtype=np.int64)  # 100 ms apart
    returns = rng.normal(0.0, 0.0002, n)  # ~3.2% daily vol
    prices = 50_000.0 * np.exp(np.cumsum(returns))
    volumes = np.abs(rng.exponential(0.5, n)) + 0.01  # positive, realistic
    sides = resolve_tick_signs(prices, None)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════════════


def _time_batch(
    ctor: BaseBarConstructor,
    ticks_df: pd.DataFrame,
    n_warmup: int = 1,
    n_timing: int = 5,
) -> tuple[float, float]:
    """Time a batch bar-construction run.

    Parameters
    ----------
    ctor : BaseBarConstructor
        The bar constructor to benchmark.
    ticks_df : pd.DataFrame
        Tick data.
    n_warmup : int
        Number of warm-up runs (discarded before timing).
    n_timing : int
        Number of timed runs (averaged).

    Returns
    -------
    compile_time : float
        Time for the first call (includes JIT compilation for numba).
    mean_time : float
        Mean time of the *n_timing* measured runs.
    """
    # First call — may include numba compilation
    t0 = time.perf_counter()
    ctor.batch(ticks_df)
    compile_time = time.perf_counter() - t0

    # Warm-up calls (additional, if n_warmup > 1)
    for _ in range(n_warmup - 1):
        ctor.batch(ticks_df)

    # Timed runs
    times: list[float] = []
    for _ in range(n_timing):
        # Create a fresh constructor each time to avoid state carry-over
        t0 = time.perf_counter()
        _fresh_batch(ctor, ticks_df)
        times.append(time.perf_counter() - t0)

    mean_time = float(np.mean(times))
    return compile_time, mean_time


def _fresh_batch(ctor: BaseBarConstructor, ticks_df: pd.DataFrame) -> pd.DataFrame:
    """Run a batch on a fresh equivalent of *ctor*."""
    # Clone the constructor's configuration
    schema = SchemaMapping(
        {"timestamp": "timestamp", "price": "price", "volume": "volume", "side": "side"}
    )
    new_ctor = _clone_constructor(ctor, schema)
    return new_ctor.batch(ticks_df)


def _clone_constructor(ctor: BaseBarConstructor, schema: SchemaMapping) -> BaseBarConstructor:
    """Create an equivalent constructor from *ctor*'s configuration."""
    acc = ctor._accumulator
    est = ctor._threshold_estimator
    bar_type = acc._bar_type
    backend = ctor._backend

    # Re-create accumulator of the same type
    if bar_type == "tick":
        new_acc = TickAccumulator()
        new_est = StaticThresholdEstimator(threshold=est.current_threshold)
    elif bar_type == "volume":
        new_acc = VolumeAccumulator()
        new_est = StaticThresholdEstimator(threshold=est.current_threshold)
    elif bar_type == "dollar":
        new_acc = DollarAccumulator()
        new_est = StaticThresholdEstimator(threshold=est.current_threshold)
    elif bar_type == "time":
        interval_ms = getattr(acc, "_interval_ms", 60000)
        anchor = getattr(acc, "_anchor", "clock")
        new_acc = TimeAccumulator(interval_ms=interval_ms, anchor=anchor)
        new_est = StaticThresholdEstimator(threshold=0.0)
    elif bar_type.startswith("imbalance_"):
        metric = getattr(acc, "_metric", "tick")
        new_acc = ImbalanceAccumulator(bar_type=bar_type, metric=metric)
        if isinstance(est, EWMAThresholdEstimator):
            new_est = EWMAThresholdEstimator(
                bar_family="imbalance",
                span=est.span,
                halflife=est.halflife,
                initial_ewa_t=est.ewa_t,
                initial_ewa_proportion=est.ewa_proportion,
            )
        else:
            new_est = StaticThresholdEstimator(threshold=est.current_threshold)
    elif bar_type.startswith("run_"):
        metric = getattr(acc, "_metric", "tick")
        new_acc = RunAccumulator(bar_type=bar_type, metric=metric)
        if isinstance(est, EWMAThresholdEstimator):
            new_est = EWMAThresholdEstimator(
                bar_family="run",
                span=est.span,
                halflife=est.halflife,
                initial_ewa_t=est.ewa_t,
                initial_ewa_proportion=est.ewa_proportion,
            )
        else:
            new_est = StaticThresholdEstimator(threshold=est.current_threshold)
    else:
        raise ValueError(f"Unknown bar type: {bar_type}")

    return BaseBarConstructor(
        accumulator=new_acc,
        threshold_estimator=new_est,
        schema=schema,
        backend=backend,  # type: ignore[arg-type]
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Bar type configurations for benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

# Each entry: (label, accumulator_factory, estimator_factory)
BenchConfig = tuple[str, Any, Any, dict[str, Any]]


def _bench_configs() -> list[BenchConfig]:
    """Return the list of bar-type configurations to benchmark."""
    configs: list[BenchConfig] = []

    # Standard bars (static threshold)
    configs.append((
        "tick",
        lambda: TickAccumulator(),
        lambda: StaticThresholdEstimator(threshold=100.0),
        {},
    ))
    configs.append((
        "volume",
        lambda: VolumeAccumulator(),
        lambda: StaticThresholdEstimator(threshold=5000.0),
        {},
    ))
    configs.append((
        "dollar",
        lambda: DollarAccumulator(),
        lambda: StaticThresholdEstimator(threshold=250_000.0),
        {},
    ))
    configs.append((
        "time (5-min)",
        lambda: TimeAccumulator(interval_ms=300000),
        lambda: StaticThresholdEstimator(threshold=0.0),
        {"time_special": True},
    ))

    # Information-driven bars (static threshold — baseline)
    configs.append((
        "imbalance_tick (static)",
        lambda: ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick"),
        lambda: StaticThresholdEstimator(threshold=30.0),
        {},
    ))
    configs.append((
        "imbalance_volume (static)",
        lambda: ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume"),
        lambda: StaticThresholdEstimator(threshold=50.0),
        {},
    ))
    configs.append((
        "imbalance_dollar (static)",
        lambda: ImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar"),
        lambda: StaticThresholdEstimator(threshold=100_000.0),
        {},
    ))
    configs.append((
        "run_tick (static)",
        lambda: RunAccumulator(bar_type="run_tick", metric="tick"),
        lambda: StaticThresholdEstimator(threshold=30.0),
        {},
    ))
    configs.append((
        "run_volume (static)",
        lambda: RunAccumulator(bar_type="run_volume", metric="volume"),
        lambda: StaticThresholdEstimator(threshold=2000.0),
        {},
    ))
    configs.append((
        "run_dollar (static)",
        lambda: RunAccumulator(bar_type="run_dollar", metric="dollar"),
        lambda: StaticThresholdEstimator(threshold=100_000.0),
        {},
    ))

    # Information-driven bars (EWMA threshold)
    configs.append((
        "imbalance_tick (EWMA)",
        lambda: ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick"),
        lambda: EWMAThresholdEstimator(
            bar_family="imbalance", span=20.0,
            initial_ewa_t=5.0, initial_ewa_proportion=0.3,
        ),
        {},
    ))
    configs.append((
        "run_tick (EWMA)",
        lambda: RunAccumulator(bar_type="run_tick", metric="tick"),
        lambda: EWMAThresholdEstimator(
            bar_family="run", span=20.0,
            initial_ewa_t=5.0, initial_ewa_proportion=0.5,
        ),
        {},
    ))

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def run_benchmarks() -> None:
    """Run all benchmarks and print results."""
    has_numba = is_numba_available()

    print("=" * 78)
    print("flowbars benchmark")
    print(f"  Ticks:        {_N_TICKS:,}")
    print(f"  numba:        {'available' if has_numba else 'NOT AVAILABLE'}")
    print(f"  Warm-up:      1 dry run (excluded from timing)")
    print(f"  Timed runs:   5 (mean reported)")
    print("=" * 78)
    print()

    ticks_df = _generate_ticks(_N_TICKS)
    print(f"  Synthetic data: {len(ticks_df):,} ticks, "
          f"price range [{ticks_df['price'].min():.2f}, {ticks_df['price'].max():.2f}]")
    print()

    # Header
    header = f"{'Bar type':<30s} {'Backend':>8s}  {'Bars':>8s}  {'Compile (s)':>11s}  {'Mean (s)':>10s}  {'Speedup':>8s}"
    print(header)
    print("-" * len(header))

    configs = _bench_configs()
    py_baseline: dict[str, float] = {}  # label → mean python time

    for label, acc_fn, est_fn, opts in configs:
        schema = SchemaMapping(
            {"timestamp": "timestamp", "price": "price", "volume": "volume", "side": "side"}
        )

        # Python backend
        acc = acc_fn()
        est = est_fn()
        ctor_py = BaseBarConstructor(
            accumulator=acc,
            threshold_estimator=est,
            schema=schema,
            backend="python",
        )

        # Suppress SessionCalendar warnings (we use ContinuousCalendar by default)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compile_py, mean_py = _time_batch(ctor_py, ticks_df)
            py_baseline[label] = mean_py

        # Count bars from a single clean run
        ctor_count = _clone_constructor(ctor_py, schema)
        n_bars = len(ctor_count.batch(ticks_df))

        print(f"{label:<30s} {'python':>8s}  {n_bars:>8d}  {compile_py:>11.4f}  {mean_py:>10.4f}  {'1.00x':>8s}")

        # numba backend
        if has_numba:
            acc_nb = acc_fn()
            est_nb = est_fn()
            ctor_nb = BaseBarConstructor(
                accumulator=acc_nb,
                threshold_estimator=est_nb,
                schema=schema,
                backend="numba",
            )

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    compile_nb, mean_nb = _time_batch(ctor_nb, ticks_df)

                speedup = mean_py / mean_nb if mean_nb > 0.0 else float("inf")
                print(f"{'':30s} {'numba':>8s}  {n_bars:>8d}  {compile_nb:>11.4f}  {mean_nb:>10.4f}  {speedup:>7.2f}x")
            except Exception as e:
                print(f"{'':30s} {'numba':>8s}  {'--':>8s}  {'--':>11s}  {'--':>10s}  {'FAILED':>8s}")
                print(f"  [numba error: {e}]")

        print()

    print("=" * 78)
    print("Benchmark complete.  Speedup = Python mean / numba mean.")
    if not has_numba:
        print("Install numba to see the accelerated path:  pip install flowbars[numba]")
    print("=" * 78)


# Entry point for `python -m flowbars.benchmarks`
if __name__ == "__main__":
    run_benchmarks()
