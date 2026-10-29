"""Hand-computed tests for the four minute-only structural bar types.

Volatility, relative range, renko displacement, and hybrid bars are tested at
the accumulator level (metric, rollover, state) and end-to-end through the
batch functions, including session-boundary ``prev_close`` resets.
"""

from __future__ import annotations

import datetime
import json

import pandas as pd
import pytest

from flowbars.bars.minute import (
    MinuteHybridAccumulator,
    MinuteHybridBarConstructor,
    MinuteRangeAccumulator,
    MinuteRangeBarConstructor,
    MinuteRenkoAccumulator,
    MinuteRenkoBarConstructor,
    MinuteVolatilityAccumulator,
    MinuteVolatilityBarConstructor,
    compute_hybrid_bars_from_minutes,
    compute_range_bars_from_minutes,
    compute_renko_bars_from_minutes,
    compute_volatility_bars_from_minutes,
)
from flowbars.calendars import SessionCalendar
from flowbars.core import MinuteInfo
from flowbars.schema import MinuteSchemaMapping
from flowbars.thresholds import StaticThresholdEstimator

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


def _mi(
    ts: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    side: float | None = None,
) -> MinuteInfo:
    return MinuteInfo(ts, open_, high, low, close, volume, side)


def _df(ts, open_, high, low, close, volume) -> pd.DataFrame:
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


def _weekday_ts(weekday: int, hour: int, minute: int) -> int:
    base = datetime.datetime(2024, 1, 15, hour, minute, tzinfo=datetime.timezone.utc)
    target = base + datetime.timedelta(days=weekday)
    return int(target.timestamp() * 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# Volatility
# ═══════════════════════════════════════════════════════════════════════════════


class TestVolatilityAccumulator:
    def test_metric_and_overflow(self) -> None:
        acc = MinuteVolatilityAccumulator()
        est = StaticThresholdEstimator(0.15)

        acc.add_minute(_mi(0, 100, 100, 100, 100, 2))  # first minute → metric 0
        assert not acc.should_close(est.current_threshold)

        acc.add_minute(_mi(60_000, 100, 110, 100, 110, 3))  # 0.10
        assert not acc.should_close(est.current_threshold)

        acc.add_minute(_mi(120_000, 110, 121, 110, 121, 5))  # +0.10 → 0.20
        assert acc.should_close(est.current_threshold)

        bar = acc.close(est.current_threshold)
        assert bar.num_ticks == 3
        assert bar.open == 100.0
        assert bar.close == 121.0
        assert bar.volume == 10.0
        assert acc._cum_metric == pytest.approx(0.05)  # 0.20 − 0.15

        # prev_close carries across bars (not reset by close)
        acc.add_minute(_mi(180_000, 121, 132, 121, 132, 1))
        assert acc._cum_metric == pytest.approx(0.05 + 11.0 / 121.0)
        assert acc._prev_close == 132.0

    def test_zero_prev_close(self) -> None:
        acc = MinuteVolatilityAccumulator()
        acc.add_minute(_mi(0, 0, 0, 0, 0, 1))
        acc.add_minute(_mi(1, 0, 100, 0, 100, 1))  # prev_close == 0 → metric 0
        assert acc._cum_metric == 0.0

    def test_session_boundary_resets_prev_close(self) -> None:
        acc = MinuteVolatilityAccumulator()
        acc.add_minute(_mi(0, 100, 100, 100, 100, 1))
        acc.add_minute(_mi(1, 100, 110, 100, 110, 1))
        assert acc._cum_metric == pytest.approx(0.10)

        acc.on_session_boundary()
        acc.add_minute(_mi(2, 110, 300, 110, 300, 1))  # overnight gap → 0
        assert acc._cum_metric == pytest.approx(0.10)
        assert acc._prev_close == 300.0

    def test_state_round_trip(self) -> None:
        acc = MinuteVolatilityAccumulator()
        acc.add_minute(_mi(0, 100, 100, 100, 100, 1))
        acc.add_minute(_mi(1, 100, 110, 100, 110, 1))
        state = json.loads(json.dumps(acc.get_state()))

        acc2 = MinuteVolatilityAccumulator()
        acc2.load_state(state)
        assert acc2._cum_metric == pytest.approx(acc._cum_metric)
        assert acc2._prev_close == acc._prev_close


class TestVolatilityBars:
    def test_hand_computed_bars(self) -> None:
        df = _df(
            [0, 60_000, 120_000, 180_000],
            [100, 100, 110, 121],
            [100, 110, 121, 121],
            [100, 100, 110, 100],
            [100, 110, 121, 100],
            [1, 1, 1, 1],
        )
        bars = compute_volatility_bars_from_minutes(df, threshold=0.15, schema=MINUTE_SCHEMA)
        # m0=0, m1=0.10, m2=0.10 → close bar 0 (3 minutes); roll 0.05.
        # m3 = 21/121 = 0.1736 → 0.2236 ≥ 0.15 → 1-minute bar 1.
        assert len(bars) == 2
        assert set(bars["bar_type"]) == {"volatility"}

        bar0 = bars.iloc[0]
        assert bar0["num_ticks"] == 3
        assert bar0["open"] == 100.0
        assert bar0["high"] == 121.0
        assert bar0["low"] == 100.0
        assert bar0["close"] == 121.0
        assert bar0["volume"] == 3.0
        assert bar0["open_ts"] == 0
        assert bar0["close_ts"] == 120_000  # crossing minute's open timestamp

        assert bars.iloc[1]["num_ticks"] == 1
        assert bars.iloc[1]["close"] == 100.0

    def test_zero_threshold_closes_every_minute(self) -> None:
        df = _df([0, 60_000, 120_000], [100] * 3, [101] * 3, [99] * 3, [100] * 3, [1] * 3)
        bars = compute_volatility_bars_from_minutes(df, threshold=0.0, schema=MINUTE_SCHEMA)
        assert len(bars) == 3
        assert list(bars["num_ticks"]) == [1, 1, 1]

    def test_session_gap_contributes_no_return(self) -> None:
        cal = SessionCalendar(9, 30, 16, 0)
        ctor = MinuteVolatilityBarConstructor(threshold=10.0, calendar=cal, schema=MINUTE_SCHEMA)

        assert ctor.update(_mi(_weekday_ts(0, 10, 0), 100, 100, 100, 100, 1)) is None

        # Outside session → force-close the in-session bar; new minute starts
        # a fresh bar (prev_close reset before ingestion).
        bar = ctor.update(_mi(_weekday_ts(0, 16, 30), 100, 150, 100, 150, 1))
        assert bar is not None
        assert bar.num_ticks == 1
        assert ctor._accumulator._cum_metric == 0.0

        # Next session: the after-hours bar is force-closed; the first
        # in-session minute must again contribute 0 (no fake overnight return).
        bar2 = ctor.update(_mi(_weekday_ts(1, 10, 0), 150, 200, 150, 200, 1))
        assert bar2 is not None
        assert bar2.num_ticks == 1
        assert ctor._accumulator._cum_metric == 0.0

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MinuteVolatilityBarConstructor(-0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Relative range
# ═══════════════════════════════════════════════════════════════════════════════


class TestRangeAccumulator:
    def test_metric_and_overflow(self) -> None:
        acc = MinuteRangeAccumulator()
        est = StaticThresholdEstimator(0.07)

        acc.add_minute(_mi(0, 100, 102, 99, 100, 1))  # 3/100 = 0.03
        acc.add_minute(_mi(1, 100, 105, 100, 103, 1))  # 5/103 ≈ 0.04854
        assert acc.should_close(est.current_threshold)

        acc.close(est.current_threshold)
        assert acc._cum_metric == pytest.approx(0.03 + 5.0 / 103.0 - 0.07)

    def test_zero_close(self) -> None:
        acc = MinuteRangeAccumulator()
        acc.add_minute(_mi(0, 0, 0, 0, 0, 1))
        assert acc._cum_metric == 0.0


class TestRangeBars:
    def test_hand_computed_bars(self) -> None:
        df = _df(
            [0, 60_000, 120_000],
            [100, 100, 103],
            [102, 105, 103],
            [99, 100, 103],
            [100, 103, 103],
            [1, 1, 1],
        )
        bars = compute_range_bars_from_minutes(df, threshold=0.07, schema=MINUTE_SCHEMA)
        assert len(bars) == 1
        bar = bars.iloc[0]
        assert bar["bar_type"] == "range"
        assert bar["num_ticks"] == 2
        assert bar["close"] == 103.0
        assert bar["close_ts"] == 60_000

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MinuteRangeBarConstructor(-0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Renko displacement
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenkoAccumulator:
    def test_anchor_is_first_open_and_no_overflow(self) -> None:
        acc = MinuteRenkoAccumulator()
        est = StaticThresholdEstimator(0.10)

        acc.add_minute(_mi(0, 100, 100, 100, 100, 1))  # anchor 100, disp 0
        assert not acc.should_close(est.current_threshold)

        acc.add_minute(_mi(60_000, 100, 112, 100, 112, 1))  # 0.12
        assert acc.should_close(est.current_threshold)

        bar = acc.close(est.current_threshold)
        assert bar.num_ticks == 2
        assert acc._anchor is None  # no overflow — anchor fully reset
        assert acc._displacement == 0.0

        acc.add_minute(_mi(120_000, 112, 115, 112, 115, 1))
        assert acc._anchor == 112.0
        assert acc._displacement == pytest.approx(3.0 / 112.0)

    def test_single_minute_bar(self) -> None:
        acc = MinuteRenkoAccumulator()
        est = StaticThresholdEstimator(0.10)
        acc.add_minute(_mi(0, 100, 112, 100, 112, 1))  # displacement 0.12
        assert acc.should_close(est.current_threshold)
        bar = acc.close(est.current_threshold)
        assert bar.num_ticks == 1

    def test_non_positive_anchor(self) -> None:
        acc = MinuteRenkoAccumulator()
        acc.add_minute(_mi(0, 0, 0, 0, 0, 1))
        assert acc._anchor == 0.0
        assert acc._displacement == 0.0

    def test_state_round_trip(self) -> None:
        acc = MinuteRenkoAccumulator()
        acc.add_minute(_mi(0, 100, 100, 100, 100, 1))
        acc.add_minute(_mi(1, 100, 105, 100, 105, 1))
        state = json.loads(json.dumps(acc.get_state()))
        acc2 = MinuteRenkoAccumulator()
        acc2.load_state(state)
        assert acc2._anchor == 100.0
        assert acc2._displacement == pytest.approx(0.05)


class TestRenkoBars:
    def test_hand_computed_bars(self) -> None:
        df = _df(
            [0, 60_000, 120_000, 180_000],
            [100, 105, 112, 115],
            [106, 113, 116, 116],
            [99, 104, 111, 99],
            [105, 112, 115, 100],
            [1, 1, 1, 1],
        )
        bars = compute_renko_bars_from_minutes(df, threshold=0.10, schema=MINUTE_SCHEMA)
        # m0 disp 0.05; m1 disp 0.12 → bar 0 (2 minutes).
        # m2 anchors at 112; m3 disp |100−112|/112 = 0.107 → bar 1 (2 minutes).
        assert len(bars) == 2
        assert set(bars["bar_type"]) == {"renko"}

        bar0 = bars.iloc[0]
        assert bar0["num_ticks"] == 2
        assert bar0["open"] == 100.0
        assert bar0["close"] == 112.0
        assert bar0["high"] == 113.0
        assert bar0["low"] == 99.0
        assert bar0["open_ts"] == 0
        assert bar0["close_ts"] == 60_000

        bar1 = bars.iloc[1]
        assert bar1["num_ticks"] == 2
        assert bar1["open"] == 112.0
        assert bar1["close"] == 100.0
        assert bar1["open_ts"] == 120_000
        assert bar1["close_ts"] == 180_000

    def test_zero_threshold_closes_every_minute(self) -> None:
        df = _df([0, 60_000], [100, 100], [101, 101], [99, 99], [100, 100], [1, 1])
        bars = compute_renko_bars_from_minutes(df, threshold=0.0, schema=MINUTE_SCHEMA)
        assert len(bars) == 2
        assert list(bars["num_ticks"]) == [1, 1]

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MinuteRenkoBarConstructor(-0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid
# ═══════════════════════════════════════════════════════════════════════════════


class TestHybridAccumulator:
    def test_requires_both_targets_and_rolls_both(self) -> None:
        acc = MinuteHybridAccumulator(
            bar_type="hybrid", dollar_threshold=250.0, volatility_threshold=0.15
        )

        acc.add_minute(_mi(0, 100, 100, 100, 100, 1))  # dollar 100, vol 0
        assert not acc.should_close(0.0)
        acc.add_minute(_mi(1, 100, 110, 100, 110, 1))  # dollar 210, vol 0.10
        assert not acc.should_close(0.0)
        acc.add_minute(_mi(2, 110, 121, 110, 121, 1))  # dollar 331, vol 0.20
        assert acc.should_close(0.0)

        bar = acc.close(0.0)
        assert bar.num_ticks == 3
        assert acc._cum_dollar == pytest.approx(81.0)
        assert acc._cum_vol == pytest.approx(0.05)

    def test_one_target_met_early_waits_for_the_other(self) -> None:
        acc = MinuteHybridAccumulator(
            bar_type="hybrid", dollar_threshold=50.0, volatility_threshold=0.15
        )
        acc.add_minute(_mi(0, 100, 100, 100, 100, 1))  # dollar met, vol 0
        assert not acc.should_close(0.0)
        acc.add_minute(_mi(1, 100, 110, 100, 110, 1))  # vol 0.10
        assert not acc.should_close(0.0)
        acc.add_minute(_mi(2, 110, 121, 110, 121, 1))  # vol 0.20 → both met
        assert acc.should_close(0.0)

    def test_session_boundary_resets_prev_close(self) -> None:
        acc = MinuteHybridAccumulator(
            bar_type="hybrid", dollar_threshold=10_000.0, volatility_threshold=10.0
        )
        acc.add_minute(_mi(0, 100, 100, 100, 100, 1))
        acc.add_minute(_mi(1, 100, 110, 100, 110, 1))
        assert acc._cum_vol == pytest.approx(0.10)
        acc.on_session_boundary()
        acc.add_minute(_mi(2, 110, 300, 110, 300, 1))
        assert acc._cum_vol == pytest.approx(0.10)

    def test_state_persists_thresholds(self) -> None:
        acc = MinuteHybridAccumulator(
            bar_type="hybrid", dollar_threshold=123.0, volatility_threshold=0.4
        )
        acc.add_minute(_mi(0, 100, 100, 100, 100, 1))
        state = json.loads(json.dumps(acc.get_state()))
        assert state["dollar_threshold"] == 123.0
        assert state["volatility_threshold"] == 0.4

        acc2 = MinuteHybridAccumulator(bar_type="hybrid")
        acc2.load_state(state)
        assert acc2._dollar_threshold == 123.0
        assert acc2._volatility_threshold == 0.4


class TestHybridBars:
    def test_hand_computed_bars(self) -> None:
        df = _df(
            [0, 60_000, 120_000, 180_000],
            [100, 100, 110, 121],
            [100, 110, 121, 121],
            [100, 100, 110, 100],
            [100, 110, 121, 100],
            [1, 1, 1, 1],
        )
        bars = compute_hybrid_bars_from_minutes(
            df, dollar_threshold=250.0, volatility_threshold=0.15, schema=MINUTE_SCHEMA
        )
        # m0: d=100, v=0; m1: d=210, v=0.10; m2: d=331, v=0.20 → close (3 min).
        # Rolls d=81, v=0.05; m3: d=181, v=0.2236 → dollar target not met.
        assert len(bars) == 1
        bar = bars.iloc[0]
        assert bar["bar_type"] == "hybrid"
        assert bar["num_ticks"] == 3
        assert bar["close"] == 121.0
        assert bar["volume"] == 3.0

    def test_zero_thresholds_close_every_minute(self) -> None:
        df = _df([0, 60_000], [100, 100], [101, 101], [99, 99], [100, 100], [1, 1])
        bars = compute_hybrid_bars_from_minutes(
            df, dollar_threshold=0.0, volatility_threshold=0.0, schema=MINUTE_SCHEMA
        )
        assert len(bars) == 2
        assert list(bars["num_ticks"]) == [1, 1]

    def test_negative_thresholds_raise(self) -> None:
        with pytest.raises(ValueError, match="dollar_threshold"):
            MinuteHybridBarConstructor(-1.0, 0.1)
        with pytest.raises(ValueError, match="volatility_threshold"):
            MinuteHybridBarConstructor(1.0, -0.1)
