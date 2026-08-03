"""Tests for schema mapping — Phase 0.1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowbars.core import SchemaError, TickDataError, TickInfo
from flowbars.schema import (
    SchemaMapping,
    validate_price,
    validate_price_array,
    validate_side,
    validate_side_array,
    validate_volume,
    validate_volume_array,
)

# ── Test data ────────────────────────────────────────────────────────────────

VALID_SCHEMA = {"timestamp": "ts", "price": "px", "volume": "qty"}
VALID_SCHEMA_WITH_SIDE = {
    "timestamp": "ts",
    "price": "px",
    "volume": "qty",
    "side": "direction",
}


def make_df(rows: list[dict]) -> pd.DataFrame:
    """Helper to create a DataFrame from a list of dicts."""
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SchemaMapping __init__ tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchemaMappingConstruction:
    def test_valid_minimal_schema(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        assert sm.timestamp_col == "ts"
        assert sm.price_col == "px"
        assert sm.volume_col == "qty"
        assert sm.side_col is None
        assert sm.has_side is False

    def test_valid_schema_with_side(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        assert sm.side_col == "direction"
        assert sm.has_side is True

    def test_missing_timestamp(self) -> None:
        with pytest.raises(SchemaError, match="timestamp"):
            SchemaMapping({"price": "px", "volume": "qty"})

    def test_missing_price(self) -> None:
        with pytest.raises(SchemaError, match="price"):
            SchemaMapping({"timestamp": "ts", "volume": "qty"})

    def test_missing_volume(self) -> None:
        with pytest.raises(SchemaError, match="volume"):
            SchemaMapping({"timestamp": "ts", "price": "px"})

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(SchemaError, match="Unknown schema keys"):
            SchemaMapping({"timestamp": "ts", "price": "px", "volume": "qty", "foo": "bar"})

    def test_empty_schema(self) -> None:
        with pytest.raises(SchemaError):
            SchemaMapping({})


# ═══════════════════════════════════════════════════════════════════════════════
# Column validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestColumnValidation:
    def test_all_columns_present(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        sm.validate_columns({"ts", "px", "qty", "extra_col"})

    def test_missing_column_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        with pytest.raises(SchemaError, match="'px'"):
            sm.validate_columns({"ts", "qty"})

    def test_missing_optional_side_column_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        with pytest.raises(SchemaError, match="'direction'"):
            sm.validate_columns({"ts", "px", "qty"})


# ═══════════════════════════════════════════════════════════════════════════════
# extract_arrays (batch path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractArrays:
    def test_basic_extraction(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df(
            [
                {"ts": 1000, "px": 100.0, "qty": 10.0},
                {"ts": 1001, "px": 101.0, "qty": 5.0},
                {"ts": 1002, "px": 99.0, "qty": 12.0},
            ]
        )
        timestamps, prices, volumes, sides = sm.extract_arrays(df)

        assert len(timestamps) == 3
        assert prices.dtype == np.float64
        assert volumes.dtype == np.float64
        assert sides is None
        np.testing.assert_array_equal(prices, np.array([100.0, 101.0, 99.0]))
        np.testing.assert_array_equal(volumes, np.array([10.0, 5.0, 12.0]))

    def test_with_side_column(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        df = make_df(
            [
                {"ts": 1000, "px": 100.0, "qty": 10.0, "direction": 1},
                {"ts": 1001, "px": 101.0, "qty": 5.0, "direction": -1},
            ]
        )
        _, _, _, sides = sm.extract_arrays(df)
        assert sides is not None
        np.testing.assert_array_equal(sides, np.array([1.0, -1.0]))

    def test_missing_column_in_dataframe(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df([{"ts": 1000, "qty": 10.0}])  # missing px
        with pytest.raises(SchemaError, match="'px'"):
            sm.extract_arrays(df)

    def test_nan_price_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df(
            [
                {"ts": 1000, "px": 100.0, "qty": 10.0},
                {"ts": 1001, "px": np.nan, "qty": 5.0},
            ]
        )
        with pytest.raises(TickDataError, match="Price at row 1"):
            sm.extract_arrays(df)

    def test_inf_price_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df([{"ts": 1000, "px": float("inf"), "qty": 10.0}])
        with pytest.raises(TickDataError, match="Price at row 0"):
            sm.extract_arrays(df)

    def test_negative_price_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df([{"ts": 1000, "px": -0.01, "qty": 10.0}])
        with pytest.raises(TickDataError, match="Price at row 0"):
            sm.extract_arrays(df)

    def test_negative_volume_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df([{"ts": 1000, "px": 100.0, "qty": -1.0}])
        with pytest.raises(TickDataError, match="Volume at row 0"):
            sm.extract_arrays(df)

    def test_nan_volume_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df([{"ts": 1000, "px": 100.0, "qty": np.nan}])
        with pytest.raises(TickDataError, match="Volume at row 0"):
            sm.extract_arrays(df)

    def test_zero_volume_accepted(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df([{"ts": 1000, "px": 100.0, "qty": 0.0}])
        _, _, volumes, _ = sm.extract_arrays(df)
        assert volumes[0] == 0.0

    def test_zero_price_accepted(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df([{"ts": 1000, "px": 0.0, "qty": 10.0}])
        _, prices, _, _ = sm.extract_arrays(df)
        assert prices[0] == 0.0

    def test_malformed_side_values(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        df = make_df(
            [
                {"ts": 1000, "px": 100.0, "qty": 10.0, "direction": 1},
                {"ts": 1001, "px": 101.0, "qty": 5.0, "direction": 0},  # bad
            ]
        )
        with pytest.raises(TickDataError, match="Side at row 1"):
            sm.extract_arrays(df)

    def test_side_nan_accepted(self) -> None:
        """NaN side = undetermined sign (first tick of a stream)."""
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        df = make_df([{"ts": 1000, "px": 100.0, "qty": 10.0, "direction": np.nan}])
        _, _, _, sides = sm.extract_arrays(df)
        assert sides is not None
        assert np.isnan(sides[0])

    def test_side_non_integer_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        df = make_df([{"ts": 1000, "px": 100.0, "qty": 10.0, "direction": 0.5}])
        with pytest.raises(TickDataError, match="Side at row 0"):
            sm.extract_arrays(df)

    def test_single_row_dataframe(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        df = make_df([{"ts": 1000, "px": 100.0, "qty": 10.0}])
        timestamps, prices, volumes, sides = sm.extract_arrays(df)
        assert len(prices) == 1

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame with correct columns — returns zero-length arrays."""
        sm = SchemaMapping(VALID_SCHEMA)
        df = pd.DataFrame({"ts": [], "px": [], "qty": []}).astype(
            {"ts": "int64", "px": "float64", "qty": "float64"}
        )
        timestamps, prices, volumes, sides = sm.extract_arrays(df)
        assert len(prices) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# normalize_tick (streaming path)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNormalizeTick:
    def test_basic_normalization(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        tick = sm.normalize_tick({"ts": 1000, "px": 100.0, "qty": 10.0})
        assert isinstance(tick, TickInfo)
        assert tick.timestamp == 1000
        assert tick.price == 100.0
        assert tick.volume == 10.0
        assert tick.side is None

    def test_with_side_column(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        tick = sm.normalize_tick({"ts": 1000, "px": 100.0, "qty": 10.0, "direction": 1})
        assert tick.side == 1

    def test_with_side_none(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        tick = sm.normalize_tick({"ts": 1000, "px": 100.0, "qty": 10.0, "direction": None})
        assert tick.side is None

    def test_missing_column_in_tick(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        with pytest.raises(SchemaError, match="'px'"):
            sm.normalize_tick({"ts": 1000, "qty": 10.0})

    def test_nan_price_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        with pytest.raises(TickDataError, match="Price"):
            sm.normalize_tick({"ts": 1000, "px": np.nan, "qty": 10.0})

    def test_negative_price_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        with pytest.raises(TickDataError, match="Price"):
            sm.normalize_tick({"ts": 1000, "px": -1.0, "qty": 10.0})

    def test_negative_volume_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        with pytest.raises(TickDataError, match="Volume"):
            sm.normalize_tick({"ts": 1000, "px": 100.0, "qty": -1.0})

    def test_inf_volume_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        with pytest.raises(TickDataError, match="Volume"):
            sm.normalize_tick({"ts": 1000, "px": 100.0, "qty": float("inf")})

    def test_malformed_side_raises(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA_WITH_SIDE)
        with pytest.raises(TickDataError, match="Side"):
            sm.normalize_tick({"ts": 1000, "px": 100.0, "qty": 10.0, "direction": 0})

    def test_zero_values_accepted(self) -> None:
        sm = SchemaMapping(VALID_SCHEMA)
        tick = sm.normalize_tick({"ts": 1000, "px": 0.0, "qty": 0.0})
        assert tick.price == 0.0
        assert tick.volume == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Value validation functions (unit tests for edge cases)
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidatePrice:
    def test_valid(self) -> None:
        validate_price(100.0)
        validate_price(0.0)
        validate_price(1e-8)

    def test_nan(self) -> None:
        with pytest.raises(TickDataError):
            validate_price(float("nan"))

    def test_inf(self) -> None:
        with pytest.raises(TickDataError):
            validate_price(float("inf"))

    def test_negative(self) -> None:
        with pytest.raises(TickDataError):
            validate_price(-0.01)


class TestValidateVolume:
    def test_valid(self) -> None:
        validate_volume(10.0)
        validate_volume(0.0)

    def test_nan(self) -> None:
        with pytest.raises(TickDataError):
            validate_volume(float("nan"))

    def test_negative(self) -> None:
        with pytest.raises(TickDataError):
            validate_volume(-1.0)


class TestValidateSide:
    def test_valid(self) -> None:
        validate_side(1.0)
        validate_side(-1.0)

    def test_nan_accepted(self) -> None:
        validate_side(float("nan"))

    def test_zero_raises(self) -> None:
        with pytest.raises(TickDataError):
            validate_side(0.0)

    def test_fractional_raises(self) -> None:
        with pytest.raises(TickDataError):
            validate_side(0.5)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(TickDataError):
            validate_side(2.0)


class TestValidateArraysAllClean:
    def test_large_clean_array(self) -> None:
        """Verify array validation doesn't raise on valid data."""
        prices = np.random.default_rng(42).uniform(0.01, 1000.0, size=10000)
        volumes = np.random.default_rng(42).uniform(0.0, 100.0, size=10000)
        validate_price_array(prices)  # should not raise
        validate_volume_array(volumes)  # should not raise
        # sides: alternating +1/-1
        sides = np.ones(10000)
        sides[1::2] = -1.0
        validate_side_array(sides)  # should not raise

    def test_empty_arrays(self) -> None:
        validate_price_array(np.array([]))
        validate_volume_array(np.array([]))
        validate_side_array(np.array([]))
