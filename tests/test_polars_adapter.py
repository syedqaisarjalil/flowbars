# ruff: noqa: E402
"""Tests for Polars adapter — Phase 10.

Equivalence tests: for each bar type, generate synthetic data as both
pandas and polars DataFrames, run both paths, assert identical output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars", reason="polars not installed")

from flowbars.adapters.polars import (
    compute_dollar_bars,
    compute_imbalance_dollar_bars,
    compute_imbalance_tick_bars,
    compute_imbalance_volume_bars,
    compute_run_dollar_bars,
    compute_run_tick_bars,
    compute_run_volume_bars,
    compute_tick_bars,
    compute_time_bars,
    compute_volume_bars,
)
from flowbars.bars.information import (
    compute_imbalance_dollar_bars as pd_compute_imbalance_dollar_bars,
)
from flowbars.bars.information import (
    compute_imbalance_tick_bars as pd_compute_imbalance_tick_bars,
)
from flowbars.bars.information import (
    compute_imbalance_volume_bars as pd_compute_imbalance_volume_bars,
)
from flowbars.bars.information import (
    compute_run_dollar_bars as pd_compute_run_dollar_bars,
)
from flowbars.bars.information import (
    compute_run_tick_bars as pd_compute_run_tick_bars,
)
from flowbars.bars.information import (
    compute_run_volume_bars as pd_compute_run_volume_bars,
)
from flowbars.bars.standard import (
    compute_dollar_bars as pd_compute_dollar_bars,
)
from flowbars.bars.standard import (
    compute_tick_bars as pd_compute_tick_bars,
)
from flowbars.bars.standard import (
    compute_time_bars as pd_compute_time_bars,
)
from flowbars.bars.standard import (
    compute_volume_bars as pd_compute_volume_bars,
)
from flowbars.calendars import ContinuousCalendar
from flowbars.schema import MinuteSchemaMapping, SchemaMapping
from flowbars.tick_rule import resolve_tick_signs

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


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
            a = np.asarray(df1[col], dtype=np.float64)
            b = np.asarray(df2[col], dtype=np.float64)
            if not np.allclose(a, b, rtol=float_tol, atol=1e-14):
                return False
    return True


def _make_tick_data(
    n: int = 500,
    seed: int = 42,
    with_side: bool = True,
) -> pd.DataFrame:
    """Generate synthetic tick data with sides derived via tick rule."""
    rng = np.random.default_rng(seed)
    timestamps = np.arange(0, n * 100, 100, dtype=np.int64)
    returns = rng.normal(0.0, 0.0002, n)
    prices = 50_000.0 * np.exp(np.cumsum(returns))
    volumes = np.abs(rng.exponential(0.5, n)) + 0.01

    data: dict[str, np.ndarray] = {
        "dt": timestamps,
        "px": prices,
        "vol": volumes,
    }
    if with_side:
        data["side"] = resolve_tick_signs(prices, None)

    return pd.DataFrame(data)


def _default_schema(has_side: bool = False) -> SchemaMapping:
    """Schema for test DataFrames."""
    mapping = {"timestamp": "dt", "price": "px", "volume": "vol"}
    if has_side:
        mapping["side"] = "side"
    return SchemaMapping(mapping)


# ═══════════════════════════════════════════════════════════════════════════════
# Standard bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolarsStandardBars:
    """Equivalence tests for standard bars (tick, volume, dollar, time)."""

    def test_tick_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=500, with_side=False)
        schema = _default_schema(has_side=False)
        pd_result = pd_compute_tick_bars(pd_ticks, threshold=50, schema=schema)

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_tick_bars(pl_ticks, threshold=50, schema=schema)

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_volume_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=500, with_side=False)
        schema = _default_schema(has_side=False)
        pd_result = pd_compute_volume_bars(pd_ticks, threshold=20.0, schema=schema)

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_volume_bars(pl_ticks, threshold=20.0, schema=schema)

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_dollar_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=500, with_side=False)
        schema = _default_schema(has_side=False)
        pd_result = pd_compute_dollar_bars(pd_ticks, threshold=1_000_000.0, schema=schema)

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_dollar_bars(pl_ticks, threshold=1_000_000.0, schema=schema)

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_time_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=500, with_side=False)
        schema = _default_schema(has_side=False)
        pd_result = pd_compute_time_bars(pd_ticks, interval_ms=30_000, schema=schema)

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_time_bars(pl_ticks, interval_ms=30_000, schema=schema)

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_time_bars_first_tick_anchor(self) -> None:
        pd_ticks = _make_tick_data(n=500, with_side=False)
        schema = _default_schema(has_side=False)
        pd_result = pd_compute_time_bars(
            pd_ticks, interval_ms=30_000, anchor="first_tick", schema=schema
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_time_bars(
            pl_ticks, interval_ms=30_000, anchor="first_tick", schema=schema
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())


# ═══════════════════════════════════════════════════════════════════════════════
# Information-driven bars — imbalance
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolarsImbalanceBars:
    """Equivalence tests for imbalance bars (tick, volume, dollar)."""

    def test_imbalance_tick_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=1000, with_side=True)
        schema = _default_schema(has_side=True)
        pd_result = pd_compute_imbalance_tick_bars(
            pd_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_imbalance_tick_bars(
            pl_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_imbalance_volume_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=1000, with_side=True)
        schema = _default_schema(has_side=True)
        pd_result = pd_compute_imbalance_volume_bars(
            pd_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_imbalance_volume_bars(
            pl_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_imbalance_dollar_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=1000, with_side=True)
        schema = _default_schema(has_side=True)
        pd_result = pd_compute_imbalance_dollar_bars(
            pd_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_imbalance_dollar_bars(
            pl_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())


# ═══════════════════════════════════════════════════════════════════════════════
# Information-driven bars — run
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolarsRunBars:
    """Equivalence tests for run bars (tick, volume, dollar)."""

    def test_run_tick_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=1000, with_side=True)
        schema = _default_schema(has_side=True)
        pd_result = pd_compute_run_tick_bars(
            pd_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_run_tick_bars(
            pl_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_run_volume_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=1000, with_side=True)
        schema = _default_schema(has_side=True)
        pd_result = pd_compute_run_volume_bars(
            pd_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_run_volume_bars(
            pl_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_run_dollar_bars_equivalence(self) -> None:
        pd_ticks = _make_tick_data(n=1000, with_side=True)
        schema = _default_schema(has_side=True)
        pd_result = pd_compute_run_dollar_bars(
            pd_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_run_dollar_bars(
            pl_ticks,
            span=20.0,
            warmup_bars=1,
            schema=schema,
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolarsEdgeCases:
    """Edge-case and integration tests for the Polars adapter."""

    def test_empty_data_produces_empty_result(self) -> None:
        pd_ticks = _make_tick_data(n=0, with_side=False)
        schema = _default_schema(has_side=False)

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_tick_bars(pl_ticks, threshold=10, schema=schema)

        assert isinstance(pl_result, pl.DataFrame)
        assert len(pl_result) == 0

    def test_single_tick_produces_no_bar(self) -> None:
        """A single tick never crosses threshold, so no bars."""
        pd_ticks = _make_tick_data(n=1, with_side=False)
        schema = _default_schema(has_side=False)

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_tick_bars(pl_ticks, threshold=10, schema=schema)

        assert len(pl_result) == 0

    def test_calendar_passthrough(self) -> None:
        pd_ticks = _make_tick_data(n=500, with_side=False)
        schema = _default_schema(has_side=False)
        cal = ContinuousCalendar()
        pd_result = pd_compute_tick_bars(pd_ticks, threshold=50, schema=schema, calendar=cal)

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_tick_bars(pl_ticks, threshold=50, schema=schema, calendar=cal)

        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_run_bars_with_min_run_length(self) -> None:
        pd_ticks = _make_tick_data(n=1000, with_side=True)
        schema = _default_schema(has_side=True)
        pd_result = pd_compute_run_tick_bars(
            pd_ticks,
            span=20.0,
            warmup_bars=0,
            min_run_length=3,
            schema=schema,
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_run_tick_bars(
            pl_ticks,
            span=20.0,
            warmup_bars=0,
            min_run_length=3,
            schema=schema,
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())

    def test_large_dataset_equivalence(self) -> None:
        """Verify equivalence on a larger dataset (10k ticks)."""
        pd_ticks = _make_tick_data(n=10_000, with_side=True)
        schema = _default_schema(has_side=True)
        pd_result = pd_compute_imbalance_tick_bars(
            pd_ticks,
            span=20.0,
            warmup_bars=2,
            schema=schema,
        )

        pl_ticks = pl.from_pandas(pd_ticks)
        pl_result = compute_imbalance_tick_bars(
            pl_ticks,
            span=20.0,
            warmup_bars=2,
            schema=schema,
        )

        assert isinstance(pl_result, pl.DataFrame)
        assert _bars_equal(pd_result, pl_result.to_pandas())
        # Ensure we actually produced bars
        assert len(pd_result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Minute bars
# ═══════════════════════════════════════════════════════════════════════════════


def _make_minute_data(n: int = 240, seed: int = 11) -> pd.DataFrame:
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


def _minute_schema() -> MinuteSchemaMapping:
    return MinuteSchemaMapping(
        {
            "timestamp": "ts",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )


class TestPolarsMinuteBars:
    """Every minute mirror must match the pandas compute function exactly."""

    @pytest.mark.parametrize("kind", ["volume", "dollar"])
    def test_standard_minute_equivalence(self, kind: str) -> None:
        from flowbars.adapters import polars as pl_adapter
        from flowbars.bars.minute import standard as pd_standard

        if kind == "volume":
            pd_fn = pd_standard.compute_volume_bars_from_minutes
            pl_fn = pl_adapter.compute_volume_bars_from_minutes
            threshold = 60.0
        else:
            pd_fn = pd_standard.compute_dollar_bars_from_minutes
            pl_fn = pl_adapter.compute_dollar_bars_from_minutes
            threshold = 6_000.0

        df = _make_minute_data()
        schema = _minute_schema()
        pd_bars = pd_fn(df, threshold, schema)
        pl_bars = pl_fn(pl.from_pandas(df), threshold, schema)
        assert isinstance(pl_bars, pl.DataFrame)
        assert _bars_equal(pd_bars, pl_bars.to_pandas())
        assert len(pd_bars) > 0

    @pytest.mark.parametrize(
        "kind",
        ["imbalance_volume", "imbalance_dollar", "run_volume", "run_dollar"],
    )
    def test_information_minute_equivalence(self, kind: str) -> None:
        from flowbars.adapters import polars as pl_adapter
        from flowbars.bars.minute import information as pd_info

        kwargs = {
            "span": 10.0,
            "initial_ewa_t": 100.0 if kind == "imbalance_volume" else 1_000.0,
            "initial_ewa_proportion": 0.5,
        }
        if kind.startswith("run_"):
            kwargs["min_run_length"] = 2
        pd_fn = getattr(pd_info, f"compute_{kind}_bars_from_minutes")
        pl_fn = getattr(pl_adapter, f"compute_{kind}_bars_from_minutes")

        df = _make_minute_data()
        schema = _minute_schema()
        pd_bars = pd_fn(df, schema=schema, **kwargs)
        pl_bars = pl_fn(pl.from_pandas(df), schema=schema, **kwargs)
        assert isinstance(pl_bars, pl.DataFrame)
        assert _bars_equal(pd_bars, pl_bars.to_pandas())
        assert len(pd_bars) > 0

    @pytest.mark.parametrize("kind", ["volatility", "range", "renko"])
    def test_structural_minute_equivalence(self, kind: str) -> None:
        from flowbars.adapters import polars as pl_adapter
        from flowbars.bars.minute import structural as pd_structural

        threshold = {"volatility": 0.02, "range": 0.02, "renko": 0.01}[kind]
        pd_fn = getattr(pd_structural, f"compute_{kind}_bars_from_minutes")
        pl_fn = getattr(pl_adapter, f"compute_{kind}_bars_from_minutes")

        df = _make_minute_data()
        schema = _minute_schema()
        pd_bars = pd_fn(df, threshold, schema)
        pl_bars = pl_fn(pl.from_pandas(df), threshold, schema)
        assert isinstance(pl_bars, pl.DataFrame)
        assert _bars_equal(pd_bars, pl_bars.to_pandas())
        assert len(pd_bars) > 0

    def test_hybrid_minute_equivalence(self) -> None:
        from flowbars.adapters import polars as pl_adapter
        from flowbars.bars.minute import structural as pd_structural

        df = _make_minute_data()
        schema = _minute_schema()
        pd_bars = pd_structural.compute_hybrid_bars_from_minutes(df, 6_000.0, 0.02, schema)
        pl_bars = pl_adapter.compute_hybrid_bars_from_minutes(
            pl.from_pandas(df), 6_000.0, 0.02, schema
        )
        assert isinstance(pl_bars, pl.DataFrame)
        assert _bars_equal(pd_bars, pl_bars.to_pandas())
        assert len(pd_bars) > 0
