"""Tests for the minute-OHLCV → bars path (``flowbars.bars.minute``).

Covers the six feasible bar types (volume, dollar, imbalance_volume,
imbalance_dollar, run_volume, run_dollar), the whole-minute OHLC merge, scalar
overflow, the ``num_ticks`` = minutes proxy, the ``close × volume`` notional
proxy, and the ``MinuteSchemaMapping`` validation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from flowbars.bars.minute import (
    MinuteImbalanceAccumulator,
    MinuteRunAccumulator,
    compute_dollar_bars_from_minutes,
    compute_imbalance_dollar_bars_from_minutes,
    compute_imbalance_volume_bars_from_minutes,
    compute_run_dollar_bars_from_minutes,
    compute_run_volume_bars_from_minutes,
    compute_volume_bars_from_minutes,
)
from flowbars.core import MinuteInfo, SchemaError, TickDataError
from flowbars.schema import MinuteSchemaMapping
from flowbars.thresholds import StaticThresholdEstimator as StaticEst

# Standard column names + a schema that maps them.
BAR_COLUMNS = [
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
]

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


def _df(ts, open_, high, low, close, volume):
    return pd.DataFrame(
        {
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _minute(ts, open_, high, low, close, volume, side=None) -> MinuteInfo:
    return MinuteInfo(ts, open_, high, low, close, volume, side)


# ═══════════════════════════════════════════════════════════════════════════════
# Standard bars (volume, dollar)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMinuteVolumeBars:
    def test_ohlc_merge_and_close(self) -> None:
        """Whole-minute OHLC merge: open=first open, high/low=extrema, close=last."""
        df = _df(
            ts=[0, 60_000, 120_000, 180_000, 240_000],
            open_=[100, 101, 102, 103, 103],
            high=[101, 102, 103, 104, 103],
            low=[99, 100, 101, 102, 102],
            close=[101, 102, 103, 103, 102],
            volume=[3, 2, 4, 1, 5],
        )
        bars = compute_volume_bars_from_minutes(df, threshold=5, schema=MINUTE_SCHEMA)

        assert list(bars.columns) == BAR_COLUMNS
        assert len(bars) == 3

        row0 = bars.iloc[0]
        assert row0["bar_id"] == 0
        assert row0["open"] == 100.0  # minute 0 open
        assert row0["high"] == 102.0  # max(101, 102)
        assert row0["low"] == 99.0  # min(99, 100)
        assert row0["close"] == 102.0  # minute 1 close
        assert row0["volume"] == 5.0  # 3 + 2
        assert row0["num_ticks"] == 2  # two minutes
        assert row0["open_ts"] == 0
        assert row0["close_ts"] == 60_000
        assert row0["bar_type"] == "volume"

    def test_dollar_value_is_close_times_volume(self) -> None:
        """dollar_value uses close × volume per minute (notional proxy)."""
        df = _df([0, 60_000], [100, 101], [101, 102], [99, 100], [101, 102], [3, 2])
        bars = compute_volume_bars_from_minutes(df, threshold=5, schema=MINUTE_SCHEMA)
        assert len(bars) == 1
        assert bars.iloc[0]["dollar_value"] == pytest.approx(101 * 3 + 102 * 2)

    def test_vwap_proxy(self) -> None:
        """vwap = dollar_value / volume (close-weighted)."""
        df = _df([0, 60_000], [100, 101], [101, 102], [99, 100], [101, 102], [3, 2])
        bars = compute_volume_bars_from_minutes(df, threshold=5, schema=MINUTE_SCHEMA)
        assert bars.iloc[0]["vwap"] == pytest.approx((101 * 3 + 102 * 2) / 5.0)

    def test_overflow_rollover(self) -> None:
        """Excess volume rolls into the next bar's accumulator."""
        df = _df([0, 60_000], [100, 101], [101, 102], [99, 100], [101, 102], [7, 3])
        bars = compute_volume_bars_from_minutes(df, threshold=5, schema=MINUTE_SCHEMA)
        # minute 0 (vol 7) closes bar 0 with 2 overflow; minute 1 (vol 3) -> 5 closes bar 1
        assert len(bars) == 2
        assert bars.iloc[0]["volume"] == 7.0
        assert bars.iloc[1]["volume"] == 3.0

    def test_threshold_not_reached_zero_bars(self) -> None:
        df = _df([0, 60_000], [100, 101], [101, 102], [99, 100], [101, 102], [1, 2])
        bars = compute_volume_bars_from_minutes(df, threshold=100, schema=MINUTE_SCHEMA)
        assert len(bars) == 0
        assert list(bars.columns) == BAR_COLUMNS

    def test_num_ticks_counts_minutes(self) -> None:
        df = _df(
            [0, 60_000, 120_000, 180_000],
            [100, 101, 102, 103],
            [101, 102, 103, 104],
            [99, 100, 101, 102],
            [101, 102, 103, 103],
            [2, 2, 2, 2],
        )
        bars = compute_volume_bars_from_minutes(df, threshold=4, schema=MINUTE_SCHEMA)
        assert len(bars) == 2
        assert bars.iloc[0]["num_ticks"] == 2
        assert bars.iloc[1]["num_ticks"] == 2


class TestMinuteDollarBars:
    def test_metric_close_times_volume(self) -> None:
        df = _df(
            [0, 60_000, 120_000, 180_000],
            [100, 101, 102, 103],
            [101, 102, 103, 104],
            [99, 100, 101, 102],
            [101, 102, 103, 103],
            [3, 2, 4, 1],
        )
        bars = compute_dollar_bars_from_minutes(df, threshold=1000, schema=MINUTE_SCHEMA)
        assert len(bars) == 1
        row = bars.iloc[0]
        assert row["dollar_value"] == pytest.approx(101 * 3 + 102 * 2 + 103 * 4 + 103 * 1)
        assert row["volume"] == pytest.approx(10.0)
        assert row["vwap"] == pytest.approx(row["dollar_value"] / 10.0)
        assert row["bar_type"] == "dollar"


# ═══════════════════════════════════════════════════════════════════════════════
# Information-driven bars (imbalance, run) — metric mechanics at the accumulator
# level with a fixed threshold, so they are hand-verifiable.
# ═══════════════════════════════════════════════════════════════════════════════


class TestMinuteImbalanceAccumulator:
    def test_signed_volume_imbalance(self) -> None:
        acc = MinuteImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        est = StaticEst(threshold=5.0)

        acc.add_minute(_minute(0, 100, 101, 99, 101, 3, side=1.0))
        assert not acc.should_close(est.current_threshold)  # |+3| < 5

        acc.add_minute(_minute(1, 101, 102, 100, 102, 2, side=-1.0))
        assert not acc.should_close(est.current_threshold)  # |+1| < 5

        acc.add_minute(_minute(2, 102, 103, 101, 103, 6, side=1.0))
        assert acc.should_close(est.current_threshold)  # |+7| >= 5

        t_stat, prop = acc.get_close_stats()
        assert t_stat == pytest.approx(11.0)  # total volume
        assert prop == pytest.approx(7.0 / 11.0)  # signed / total

        bar = acc.close(est.current_threshold)
        assert bar.volume == pytest.approx(11.0)
        assert bar.num_ticks == 3

        # overflow: +2 carries; next sell 4 → −2 (no close)
        acc.add_minute(_minute(3, 103, 104, 102, 102, 4, side=-1.0))
        assert not acc.should_close(est.current_threshold)

    def test_nan_side_excluded_from_imbalance(self) -> None:
        acc = MinuteImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        est = StaticEst(threshold=10.0)
        # First minute NaN side: contributes to OHLCV/volume but not imbalance.
        acc.add_minute(_minute(0, 100, 101, 99, 101, 8, side=None))
        assert not acc.should_close(est.current_threshold)  # |0| < 10

        t_stat, prop = acc.get_close_stats()
        assert t_stat == pytest.approx(8.0)
        assert prop == pytest.approx(0.0)


class TestMinuteRunAccumulator:
    def test_run_banking(self) -> None:
        acc = MinuteRunAccumulator(bar_type="run_volume", metric="volume")
        est = StaticEst(threshold=10.0)

        acc.add_minute(_minute(0, 100, 101, 99, 101, 4, side=1.0))
        assert not acc.should_close(est.current_threshold)  # 4 < 10

        acc.add_minute(_minute(1, 101, 102, 100, 102, 3, side=1.0))
        assert not acc.should_close(est.current_threshold)  # 7 < 10

        acc.add_minute(_minute(2, 102, 103, 101, 103, 5, side=-1.0))
        assert acc.should_close(est.current_threshold)  # banked 7 + current 5 = 12

        t_stat, prop = acc.get_close_stats()
        assert t_stat == pytest.approx(12.0)
        assert prop == pytest.approx(7.0 / 12.0)  # P+ = buy / (buy + sell)

        bar = acc.close(est.current_threshold)
        assert bar.volume == pytest.approx(12.0)
        assert bar.num_ticks == 3


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end smoke: all six functions produce correctly-typed output
# ═══════════════════════════════════════════════════════════════════════════════


class TestMinuteEndToEnd:
    def _minutes(self) -> pd.DataFrame:
        return _df(
            [0, 60_000, 120_000, 180_000, 240_000, 300_000],
            [100, 101, 102, 103, 103, 102],
            [101, 102, 103, 104, 103, 102],
            [99, 100, 101, 102, 102, 101],
            [101, 102, 103, 103, 102, 101],
            [3, 2, 4, 1, 5, 2],
        )

    def test_all_functions_return_standard_schema(self) -> None:
        df = self._minutes()
        funcs = [
            lambda: compute_volume_bars_from_minutes(df, threshold=5, schema=MINUTE_SCHEMA),
            lambda: compute_dollar_bars_from_minutes(df, threshold=1000, schema=MINUTE_SCHEMA),
            lambda: compute_imbalance_volume_bars_from_minutes(df, schema=MINUTE_SCHEMA),
            lambda: compute_imbalance_dollar_bars_from_minutes(df, schema=MINUTE_SCHEMA),
            lambda: compute_run_volume_bars_from_minutes(df, schema=MINUTE_SCHEMA),
            lambda: compute_run_dollar_bars_from_minutes(df, schema=MINUTE_SCHEMA),
        ]
        for fn in funcs:
            bars = fn()
            assert list(bars.columns) == BAR_COLUMNS

    def test_bar_type_labels(self) -> None:
        df = self._minutes()
        assert set(compute_volume_bars_from_minutes(df, 5, MINUTE_SCHEMA)["bar_type"]) == {"volume"}
        assert set(compute_dollar_bars_from_minutes(df, 1000, MINUTE_SCHEMA)["bar_type"]) == {"dollar"}
        assert set(compute_imbalance_volume_bars_from_minutes(df, schema=MINUTE_SCHEMA)["bar_type"]) == {
            "imbalance_volume"
        }
        assert set(compute_imbalance_dollar_bars_from_minutes(df, schema=MINUTE_SCHEMA)["bar_type"]) == {
            "imbalance_dollar"
        }
        assert set(compute_run_volume_bars_from_minutes(df, schema=MINUTE_SCHEMA)["bar_type"]) == {
            "run_volume"
        }
        assert set(compute_run_dollar_bars_from_minutes(df, schema=MINUTE_SCHEMA)["bar_type"]) == {
            "run_dollar"
        }

    def test_empty_input(self) -> None:
        df = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        bars = compute_volume_bars_from_minutes(df, threshold=5, schema=MINUTE_SCHEMA)
        assert len(bars) == 0
        assert list(bars.columns) == BAR_COLUMNS


# ═══════════════════════════════════════════════════════════════════════════════
# MinuteSchemaMapping validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestMinuteSchema:
    def test_missing_required_key_raises(self) -> None:
        with pytest.raises(SchemaError, match="Missing required schema keys"):
            MinuteSchemaMapping({"timestamp": "ts", "open": "o", "high": "h", "low": "l", "close": "c"})

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(SchemaError, match="Unknown schema keys"):
            MinuteSchemaMapping(
                {
                    "timestamp": "ts",
                    "open": "o",
                    "high": "h",
                    "low": "l",
                    "close": "c",
                    "volume": "v",
                    "price": "p",  # tick-schema key, not minute
                }
            )

    def test_missing_column_raises(self) -> None:
        df = _df([0], [100], [101], [99], [101], [1]).drop(columns=["high"])
        with pytest.raises(SchemaError, match="not found in input"):
            compute_volume_bars_from_minutes(df, threshold=5, schema=MINUTE_SCHEMA)

    def test_ohlc_inconsistent_raises(self) -> None:
        # high below low
        df = _df([0], [100], [99], [101], [101], [1])
        with pytest.raises(TickDataError, match="OHLC inconsistent"):
            compute_volume_bars_from_minutes(df, threshold=5, schema=MINUTE_SCHEMA)

    def test_default_schema_uses_standard_names(self) -> None:
        df = pd.DataFrame(
            {
                "timestamp": [0, 60_000],
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [101, 102],
                "volume": [3, 2],
            }
        )
        bars = compute_volume_bars_from_minutes(df, threshold=5)  # no schema
        assert len(bars) == 1
