"""Tests for tick rule — Phase 1."""

from __future__ import annotations

import numpy as np
import pytest

from flowbars.tick_rule import derive_tick_sign, resolve_tick_signs

# ═══════════════════════════════════════════════════════════════════════════════
# derive_tick_sign
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeriveTickSign:
    def test_basic_five_tick_example(self) -> None:
        """The canonical 5-tick example from spec: up, flat, down, up."""
        prices = np.array([100.0, 101.0, 101.0, 99.0, 102.0], dtype=np.float64)
        signs = derive_tick_sign(prices)

        expected = np.array([np.nan, 1.0, 1.0, -1.0, 1.0], dtype=np.float64)
        # NaN != NaN, so compare element-wise
        assert np.isnan(signs[0])
        np.testing.assert_array_equal(signs[1:], expected[1:])

    def test_empty_array(self) -> None:
        signs = derive_tick_sign(np.array([], dtype=np.float64))
        assert len(signs) == 0

    def test_single_tick(self) -> None:
        signs = derive_tick_sign(np.array([100.0], dtype=np.float64))
        assert len(signs) == 1
        assert np.isnan(signs[0])

    def test_all_rising(self) -> None:
        prices = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
        signs = derive_tick_sign(prices)

        assert np.isnan(signs[0])
        np.testing.assert_array_equal(signs[1:], np.array([1.0, 1.0, 1.0]))

    def test_all_falling(self) -> None:
        prices = np.array([100.0, 99.0, 98.0, 97.0], dtype=np.float64)
        signs = derive_tick_sign(prices)

        assert np.isnan(signs[0])
        np.testing.assert_array_equal(signs[1:], np.array([-1.0, -1.0, -1.0]))

    def test_all_equal_prices(self) -> None:
        """All equal prices → all NaN (carry-forward from first NaN)."""
        prices = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        signs = derive_tick_sign(prices)

        for s in signs:
            assert np.isnan(s)

    def test_equal_prices_mid_stream(self) -> None:
        """Equal mid-stream carries forward the last known sign."""
        prices = np.array([100.0, 101.0, 101.0, 101.0, 99.0], dtype=np.float64)
        signs = derive_tick_sign(prices)

        assert np.isnan(signs[0])
        assert signs[1] == 1.0  # up
        assert signs[2] == 1.0  # equal, carry +1
        assert signs[3] == 1.0  # equal, carry +1
        assert signs[4] == -1.0  # down

    def test_equal_at_start(self) -> None:
        """First two prices equal → second tick carries NaN forward."""
        prices = np.array([100.0, 100.0, 101.0], dtype=np.float64)
        signs = derive_tick_sign(prices)

        assert np.isnan(signs[0])  # first
        assert np.isnan(signs[1])  # equal, carry NaN
        assert signs[2] == 1.0  # up

    def test_output_dtype(self) -> None:
        signs = derive_tick_sign(np.array([100.0, 101.0], dtype=np.float64))
        assert signs.dtype == np.float64

    def test_large_array(self) -> None:
        """Regression test — large random walk."""
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.normal(0, 0.1, size=50000))
        signs = derive_tick_sign(prices.astype(np.float64))

        assert len(signs) == 50000
        assert np.isnan(signs[0])
        # Every non-first tick should be +1 or -1 (no NaN after first in random walk)
        assert not np.any(np.isnan(signs[1:]))
        assert np.all((signs[1:] == 1.0) | (signs[1:] == -1.0))

    def test_float_precision_boundary(self) -> None:
        """Very small price differences should still be detected."""
        prices = np.array([100.0, 100.0 + 1e-12, 100.0 + 2e-12], dtype=np.float64)
        signs = derive_tick_sign(prices)

        assert np.isnan(signs[0])
        assert signs[1] == 1.0  # tiny increase → uptick
        assert signs[2] == 1.0  # tiny increase → uptick

    def test_alternating(self) -> None:
        """Alternating up/down should produce alternating signs."""
        prices = np.array([100.0, 101.0, 100.0, 101.0, 100.0], dtype=np.float64)
        signs = derive_tick_sign(prices)

        expected = np.array([np.nan, 1.0, -1.0, 1.0, -1.0], dtype=np.float64)
        assert np.isnan(signs[0])
        np.testing.assert_array_equal(signs[1:], expected[1:])

    def test_large_gap_down(self) -> None:
        """Large price gaps still produce correct signs."""
        prices = np.array([100.0, 50.0, 200.0], dtype=np.float64)
        signs = derive_tick_sign(prices)

        assert np.isnan(signs[0])
        assert signs[1] == -1.0
        assert signs[2] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# resolve_tick_signs
# ═══════════════════════════════════════════════════════════════════════════════


class TestResolveTickSigns:
    def test_uses_supplied_sides(self) -> None:
        prices = np.array([100.0, 101.0, 99.0], dtype=np.float64)
        supplied = np.array([-1.0, 1.0, -1.0], dtype=np.float64)
        result = resolve_tick_signs(prices, supplied)
        np.testing.assert_array_equal(result, supplied)

    def test_derives_when_none(self) -> None:
        prices = np.array([100.0, 101.0, 99.0], dtype=np.float64)
        result = resolve_tick_signs(prices, None)

        assert np.isnan(result[0])
        assert result[1] == 1.0
        assert result[2] == -1.0

    def test_supplied_sides_not_mutated(self) -> None:
        """resolve_tick_signs returns a copy, not a reference."""
        prices = np.array([100.0, 101.0], dtype=np.float64)
        supplied = np.array([1.0, -1.0], dtype=np.float64)
        result = resolve_tick_signs(prices, supplied)

        # Mutating result should not affect supplied
        result[0] = 999.0
        assert supplied[0] == 1.0

    def test_supplied_with_nan(self) -> None:
        """NaN in supplied sides is passed through (first-tick scenario)."""
        prices = np.array([100.0, 101.0, 99.0], dtype=np.float64)
        supplied = np.array([np.nan, 1.0, -1.0], dtype=np.float64)
        result = resolve_tick_signs(prices, supplied)

        assert np.isnan(result[0])
        assert result[1] == 1.0
        assert result[2] == -1.0

    def test_length_mismatch_raises(self) -> None:
        prices = np.array([100.0, 101.0], dtype=np.float64)
        supplied = np.array([1.0], dtype=np.float64)  # too short
        with pytest.raises(ValueError, match="length"):
            resolve_tick_signs(prices, supplied)

    def test_empty_arrays(self) -> None:
        prices = np.array([], dtype=np.float64)
        result = resolve_tick_signs(prices, None)
        assert len(result) == 0

        result = resolve_tick_signs(prices, np.array([], dtype=np.float64))
        assert len(result) == 0
