"""Tests for bundled sample data — Phase 12."""

from __future__ import annotations

import pandas as pd

from flowbars import load_sample_data
from flowbars.bars.information import compute_imbalance_tick_bars
from flowbars.bars.standard import compute_tick_bars
from flowbars.schema import SchemaMapping
from flowbars.tick_rule import resolve_tick_signs


class TestLoadSampleData:
    """Tests for ``load_sample_data()``."""

    def test_returns_dataframe(self) -> None:
        df = load_sample_data()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        df = load_sample_data()
        assert set(df.columns) == {"timestamp", "price", "volume"}

    def test_has_expected_row_count(self) -> None:
        df = load_sample_data()
        assert len(df) == 500

    def test_no_null_values(self) -> None:
        df = load_sample_data()
        assert not df.isnull().any().any()

    def test_timestamps_are_monotonic(self) -> None:
        df = load_sample_data()
        assert df["timestamp"].is_monotonic_increasing

    def test_prices_positive(self) -> None:
        df = load_sample_data()
        assert (df["price"] > 0).all()

    def test_volumes_positive(self) -> None:
        df = load_sample_data()
        assert (df["volume"] > 0).all()

    def test_deterministic(self) -> None:
        """Multiple calls return identical data."""
        df1 = load_sample_data()
        df2 = load_sample_data()
        pd.testing.assert_frame_equal(df1, df2)


class TestQuickstart:
    """The README quickstart example must run copy-paste without error."""

    def test_quickstart_tick_bars(self) -> None:
        """Build tick bars from sample data — the first quickstart example."""
        df = load_sample_data()
        df["side"] = resolve_tick_signs(df["price"].values, None)

        schema = SchemaMapping(
            {
                "timestamp": "timestamp",
                "price": "price",
                "volume": "volume",
                "side": "side",
            }
        )

        bars = compute_tick_bars(df, threshold=50, schema=schema)
        assert len(bars) > 0
        assert set(bars.columns) == {
            "bar_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "dollar_value",
            "vwap",
            "num_ticks",
            "open_ts",
            "close_ts",
            "bar_type",
        }
        # Bars should be tick bars
        assert (bars["bar_type"] == "tick").all()

    def test_quickstart_imbalance_bars(self) -> None:
        """Build imbalance bars from sample data."""
        df = load_sample_data()
        df["side"] = resolve_tick_signs(df["price"].values, None)

        schema = SchemaMapping(
            {
                "timestamp": "timestamp",
                "price": "price",
                "volume": "volume",
                "side": "side",
            }
        )

        bars = compute_imbalance_tick_bars(df, span=20.0, schema=schema)
        assert len(bars) > 0
        assert (bars["bar_type"] == "imbalance_tick").all()

    def test_quickstart_without_side_column(self) -> None:
        """Sample data works without a pre-supplied side column."""
        df = load_sample_data()

        schema = SchemaMapping(
            {
                "timestamp": "timestamp",
                "price": "price",
                "volume": "volume",
            }
        )

        bars = compute_tick_bars(df, threshold=100, schema=schema)
        assert len(bars) > 0
