"""Tests for standard bar constructors — Phase 6."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from flowbars.bars.standard.dollar_bars import (
    DollarBarConstructor,
    compute_dollar_bars,
)
from flowbars.bars.standard.tick_bars import TickBarConstructor, compute_tick_bars
from flowbars.bars.standard.time_bars import TimeBarConstructor, compute_time_bars
from flowbars.bars.standard.volume_bars import (
    VolumeBarConstructor,
    compute_volume_bars,
)
from flowbars.calendars import SessionCalendar
from flowbars.core import Bar, TickInfo
from flowbars.schema import SchemaMapping

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def tick(ts: int, price: float, volume: float, side: float | None = None) -> TickInfo:
    return TickInfo(timestamp=ts, price=price, volume=volume, side=side)


def make_ticks_df(
    timestamps: list[int],
    prices: list[float],
    volumes: list[float],
    sides: list[float] | None = None,
) -> pd.DataFrame:
    data = {"ts": timestamps, "px": prices, "vol": volumes}
    if sides is not None:
        data["side"] = sides
    return pd.DataFrame(data)


def default_schema(has_side: bool = False) -> SchemaMapping:
    mapping = {"timestamp": "ts", "price": "px", "volume": "vol"}
    if has_side:
        mapping["side"] = "side"
    return SchemaMapping(mapping)


def bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=[
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
        )
    return pd.DataFrame(
        [
            {
                "bar_id": b.bar_id,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "dollar_value": b.dollar_value,
                "vwap": b.vwap,
                "num_ticks": b.num_ticks,
                "open_ts": b.open_ts,
                "close_ts": b.close_ts,
                "bar_type": b.bar_type,
            }
            for b in bars
        ]
    )


def weekday_ts(weekday: int, hour: int, minute: int, second: int = 0) -> int:
    base = datetime.datetime(2024, 1, 15, hour, minute, second, tzinfo=datetime.timezone.utc)
    target = base + datetime.timedelta(days=weekday)
    return int(target.timestamp() * 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# Time bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimeBarConstructor:
    def test_clock_anchor_single_bar(self) -> None:
        """5-minute time bar: ticks at 0s and 30s, close at 5min boundary."""
        ctor = TimeBarConstructor(interval_ms=300_000, anchor="clock")
        assert ctor.update(tick(50_580_000, 100.0, 1.0)) is None  # 14:03:00
        assert ctor.update(tick(50_610_000, 101.0, 1.0)) is None  # 14:03:30
        bar = ctor.update(tick(50_700_000, 102.0, 1.0))  # 14:05:00 → crosses 14:05 boundary
        assert bar is not None
        assert bar.num_ticks == 3
        assert bar.bar_type == "time"

    def test_clock_anchor_multi_bar(self) -> None:
        """Two time bars: 1-min intervals, ticks at 0:30, 1:00, 1:15, 2:10."""
        ctor = TimeBarConstructor(interval_ms=60_000, anchor="clock")
        # Bar 0: ticks at 10:00:30 and 10:01:00 → closes at 10:01:00
        assert ctor.update(tick(36_030_000, 10.0, 1.0)) is None
        bar0 = ctor.update(tick(36_060_000, 11.0, 1.0))
        assert bar0 is not None
        assert bar0.num_ticks == 2
        assert bar0.bar_id == 0

        # Bar 1: ticks at 10:01:15 and 10:02:10 → closes at 10:02:00 boundary
        assert ctor.update(tick(36_075_000, 12.0, 1.0)) is None
        bar1 = ctor.update(tick(36_130_000, 13.0, 1.0))
        assert bar1 is not None
        assert bar1.num_ticks == 2
        assert bar1.bar_id == 1

    def test_first_tick_anchor(self) -> None:
        """Anchor='first_tick': bars measured from the first tick's timestamp."""
        ctor = TimeBarConstructor(interval_ms=60_000, anchor="first_tick")
        first_ts = 36_030_000  # 10:00:30
        assert ctor.update(tick(first_ts, 10.0, 1.0)) is None
        assert ctor.update(tick(first_ts + 30_000, 11.0, 1.0)) is None
        bar = ctor.update(tick(first_ts + 60_001, 12.0, 1.0))
        assert bar is not None
        assert bar.num_ticks == 3
        assert bar.open_ts == first_ts

    def test_batch_time_bars(self) -> None:
        """compute_time_bars() produces correct bars."""
        df = make_ticks_df(
            timestamps=[36_000_000, 36_030_000, 36_060_000, 36_090_000, 36_120_100],
            prices=[10.0, 11.0, 12.0, 13.0, 14.0],
            volumes=[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        result = compute_time_bars(df, interval_ms=60_000, schema=default_schema())
        assert len(result) >= 1

    def test_batch_time_bars_first_tick_anchor(self) -> None:
        """compute_time_bars() with first_tick anchor."""
        df = make_ticks_df(
            timestamps=[10_000, 40_000, 70_001, 100_000, 130_000],
            prices=[1.0, 2.0, 3.0, 4.0, 5.0],
            volumes=[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        result = compute_time_bars(
            df, interval_ms=30_000, anchor="first_tick", schema=default_schema()
        )
        # 10k→40k (bar0: 2 ticks), 70k (bar1: 1 tick), 100k (bar2: 1 tick), 130k (bar3: 1 tick)
        assert len(result) == 4
        assert result.iloc[0]["num_ticks"] == 2

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        timestamps = [36_000_000, 36_030_000, 36_060_000, 36_090_000, 36_120_000, 36_150_000]
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        volumes = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        df = make_ticks_df(timestamps, prices, volumes)
        batch_result = compute_time_bars(df, interval_ms=60_000, schema=default_schema())

        ctor = TimeBarConstructor(interval_ms=60_000)
        stream_bars = []
        for i in range(len(timestamps)):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working time bar constructor."""
        ctor = TimeBarConstructor(interval_ms=60_000, anchor="clock")
        ctor.update(tick(36_000_000, 10.0, 1.0))
        state = ctor.get_state()

        ctor2 = TimeBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.num_ticks == 1

    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeBarConstructor(interval_ms=-1000)

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeBarConstructor(interval_ms=0)

    def test_invalid_anchor_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeBarConstructor(interval_ms=60_000, anchor="midnight")

    def test_with_session_calendar(self) -> None:
        """Time bars respect session boundaries."""
        cal = SessionCalendar(9, 30, 16, 0)  # 09:30–16:00 UTC
        ctor = TimeBarConstructor(interval_ms=300_000, calendar=cal)

        # Monday 10:00 — inside session
        assert ctor.update(tick(weekday_ts(0, 10, 0), 100.0, 1.0)) is None
        assert ctor.current_bar is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Tick bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestTickBarConstructor:
    def test_basic_tick_bar(self) -> None:
        """Threshold=3: 3 ticks → 1 bar."""
        ctor = TickBarConstructor(threshold=3)
        assert ctor.update(tick(1000, 10.0, 1.0)) is None
        assert ctor.update(tick(2000, 11.0, 1.0)) is None
        bar = ctor.update(tick(3000, 12.0, 1.0))
        assert bar is not None
        assert bar.num_ticks == 3
        assert bar.bar_type == "tick"

    def test_multi_bar_sequence(self) -> None:
        """Threshold=2, 6 ticks → 3 bars."""
        ctor = TickBarConstructor(threshold=2)
        bars = []
        for i in range(6):
            bar = ctor.update(tick(1000 + i * 1000, 10.0 + i, 1.0))
            if bar is not None:
                bars.append(bar)
        assert len(bars) == 3
        assert [b.bar_id for b in bars] == [0, 1, 2]

    def test_batch_tick_bars(self) -> None:
        """compute_tick_bars() produces correct output."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000, 5000, 6000],
            prices=[10.0, 12.0, 11.0, 13.0, 14.0, 15.0],
            volumes=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        result = compute_tick_bars(df, threshold=2, schema=default_schema())
        assert len(result) == 3
        assert result.iloc[0]["bar_type"] == "tick"
        assert result.iloc[0]["num_ticks"] == 2

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        timestamps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        prices = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        volumes = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        df = make_ticks_df(timestamps, prices, volumes)
        batch_result = compute_tick_bars(df, threshold=3, schema=default_schema())

        ctor = TickBarConstructor(threshold=3)
        stream_bars = []
        for i in range(len(timestamps)):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working tick bar constructor."""
        ctor = TickBarConstructor(threshold=5)
        ctor.update(tick(1000, 10.0, 1.0))
        state = ctor.get_state()

        ctor2 = TickBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.num_ticks == 1

    def test_from_state_preserves_threshold(self) -> None:
        """from_state() correctly restores the threshold value."""
        ctor = TickBarConstructor(threshold=7)
        ctor.update(tick(1000, 10.0, 1.0))
        state = ctor.get_state()

        ctor2 = TickBarConstructor.from_state(state)
        # Add 6 more ticks → total 7 → should close
        for i in range(6):
            bar = ctor2.update(tick(2000 + i * 1000, 10.0 + i, 1.0))
        assert bar is not None  # the last one returned a bar

    def test_state_round_trip_interrupted(self) -> None:
        """Save/resume produces the same bars as uninterrupted run."""
        ctor1 = TickBarConstructor(threshold=2)
        uninterrupted_bars = []
        for i in range(6):
            bar = ctor1.update(tick(1000 + i * 1000, 10.0 + i, 1.0))
            if bar is not None:
                uninterrupted_bars.append(bar)
        uninterrupted_df = bars_to_df(uninterrupted_bars)

        # Save after 3 ticks
        ctor2 = TickBarConstructor(threshold=2)
        for i in range(3):
            ctor2.update(tick(1000 + i * 1000, 10.0 + i, 1.0))
        state = ctor2.get_state()

        # Resume
        ctor3 = TickBarConstructor.from_state(state)
        interrupted_bars = []
        for i in range(3, 6):
            bar = ctor3.update(tick(1000 + i * 1000, 10.0 + i, 1.0))
            if bar is not None:
                interrupted_bars.append(bar)
        interrupted_df = bars_to_df(interrupted_bars)

        # Compare — uninterrupted has all bars, interrupted has bars from ticks 3-5
        n = len(interrupted_df)
        uninterrupted_tail = uninterrupted_df.iloc[-n:].reset_index(drop=True)
        interrupted_clean = interrupted_df.reset_index(drop=True)
        for col in uninterrupted_tail.columns:
            if col == "bar_id":
                continue
            pd.testing.assert_series_equal(
                uninterrupted_tail[col].reset_index(drop=True),
                interrupted_clean[col].reset_index(drop=True),
                check_names=False,
            )

    def test_zero_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            TickBarConstructor(threshold=0)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            TickBarConstructor(threshold=-5)

    def test_with_session_calendar(self) -> None:
        """Tick bars respect session boundaries with a calendar."""
        cal = SessionCalendar(9, 30, 16, 0)
        ctor = TickBarConstructor(threshold=100, calendar=cal)  # large threshold
        ctor.update(tick(weekday_ts(0, 10, 0), 100.0, 1.0))
        assert ctor.current_bar is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Volume bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestVolumeBarConstructor:
    def test_basic_volume_bar(self) -> None:
        """Threshold=1000: 300+400+500=1200 → 1 bar, 200 overflow."""
        ctor = VolumeBarConstructor(threshold=1000.0)
        assert ctor.update(tick(1000, 100.0, 300.0)) is None
        assert ctor.update(tick(2000, 101.0, 400.0)) is None
        bar = ctor.update(tick(3000, 102.0, 500.0))
        assert bar is not None
        assert bar.volume == pytest.approx(1200.0)
        assert bar.bar_type == "volume"

    def test_overflow_volume_bar(self) -> None:
        """Excess volume from bar 0 rolls into bar 1."""
        ctor = VolumeBarConstructor(threshold=1000.0)
        ctor.update(tick(1000, 100.0, 300.0))
        ctor.update(tick(2000, 101.0, 400.0))
        bar0 = ctor.update(tick(3000, 102.0, 500.0))  # 300+400+500=1200 → excess 200
        assert bar0 is not None

        # Next bar starts with 200 overflow
        ctor.update(tick(4000, 103.0, 100.0))  # cum = 200+100 = 300
        ctor.update(tick(5000, 104.0, 600.0))  # cum = 300+600 = 900
        bar1 = ctor.update(tick(6000, 105.0, 200.0))  # cum = 900+200 = 1100
        assert bar1 is not None
        assert bar1.volume == pytest.approx(100.0 + 600.0 + 200.0)

    def test_batch_volume_bars(self) -> None:
        """compute_volume_bars() produces correct output."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000, 5000, 6000, 7000],
            prices=[100.0] * 7,
            volumes=[300.0, 400.0, 500.0, 100.0, 600.0, 200.0, 100.0],
        )
        result = compute_volume_bars(df, threshold=1000.0, schema=default_schema())
        assert len(result) == 2  # bar0: 300+400+500=1200, bar1: 100+600+200=900
        assert result.iloc[0]["volume"] == pytest.approx(1200.0)
        assert result.iloc[1]["volume"] == pytest.approx(900.0)

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        timestamps = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
        prices = [100.0] * 7
        volumes = [300.0, 400.0, 500.0, 100.0, 600.0, 200.0, 100.0]

        df = make_ticks_df(timestamps, prices, volumes)
        batch_result = compute_volume_bars(df, threshold=1000.0, schema=default_schema())

        ctor = VolumeBarConstructor(threshold=1000.0)
        stream_bars = []
        for i in range(len(timestamps)):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() preserves volume bar state."""
        ctor = VolumeBarConstructor(threshold=500.0)
        ctor.update(tick(1000, 100.0, 200.0))
        state = ctor.get_state()

        ctor2 = VolumeBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.volume == 200.0

    def test_state_with_overflow(self) -> None:
        """State captures overflow volume correctly."""
        ctor = VolumeBarConstructor(threshold=1000.0)
        ctor.update(tick(1000, 100.0, 600.0))
        ctor.update(tick(2000, 101.0, 600.0))  # 1200 → closes, excess 200

        state = ctor.get_state()
        assert state["accumulator"]["cum_volume"] == 200.0  # overflow

        ctor2 = VolumeBarConstructor.from_state(state)
        ctor2.update(tick(3000, 102.0, 100.0))  # 200 + 100 = 300
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.volume == 100.0  # only the new tick

    def test_zero_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            VolumeBarConstructor(threshold=0.0)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            VolumeBarConstructor(threshold=-100.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Dollar bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestDollarBarConstructor:
    def test_basic_dollar_bar(self) -> None:
        """Threshold=$1M: $300k+$400k+$500k=$1.2M → 1 bar, $200k overflow."""
        ctor = DollarBarConstructor(threshold=1_000_000.0)
        assert ctor.update(tick(1000, 100.0, 3000.0)) is None  # $300k
        assert ctor.update(tick(2000, 100.0, 4000.0)) is None  # $700k
        bar = ctor.update(tick(3000, 100.0, 5000.0))  # $1.2M
        assert bar is not None
        assert bar.dollar_value == pytest.approx(1_200_000.0)
        assert bar.bar_type == "dollar"

    def test_overflow_dollar_bar(self) -> None:
        """Excess notional from bar 0 rolls into bar 1."""
        ctor = DollarBarConstructor(threshold=1_000_000.0)
        ctor.update(tick(1000, 100.0, 3000.0))
        ctor.update(tick(2000, 100.0, 4000.0))
        ctor.update(tick(3000, 100.0, 5000.0))  # $1.2M, excess $200k

        # Bar 1: $200k overflow + $100k = $300k
        ctor.update(tick(4000, 100.0, 1000.0))
        assert ctor.current_bar is not None
        assert ctor.current_bar.dollar_value == pytest.approx(100_000.0)

    def test_batch_dollar_bars(self) -> None:
        """compute_dollar_bars() produces correct output."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000, 5000, 6000],
            prices=[100.0, 200.0, 50.0, 100.0, 100.0, 100.0],
            volumes=[3000.0, 2000.0, 4000.0, 3000.0, 4000.0, 5000.0],
        )
        # $300K + $400K + $200K + $300K = $1.2M → close bar0 (excess $200K)
        # $400K + $500K = $900K → close bar1 (below threshold, final bar drained)
        result = compute_dollar_bars(df, threshold=1_000_000.0, schema=default_schema())
        assert len(result) == 2

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        timestamps = [1000, 2000, 3000, 4000, 5000]
        prices = [100.0, 200.0, 50.0, 100.0, 100.0]
        volumes = [3000.0, 2000.0, 4000.0, 5000.0, 2000.0]

        df = make_ticks_df(timestamps, prices, volumes)
        batch_result = compute_dollar_bars(df, threshold=500_000.0, schema=default_schema())

        ctor = DollarBarConstructor(threshold=500_000.0)
        stream_bars = []
        for i in range(len(timestamps)):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() preserves dollar bar state."""
        ctor = DollarBarConstructor(threshold=1_000_000.0)
        ctor.update(tick(1000, 100.0, 2000.0))
        state = ctor.get_state()

        ctor2 = DollarBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.dollar_value == pytest.approx(200_000.0)

    def test_zero_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            DollarBarConstructor(threshold=0.0)

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            DollarBarConstructor(threshold=-1_000_000.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryIntegration:
    def test_all_four_types_registered(self) -> None:
        from flowbars.bars.registry import BarRegistry

        registered = BarRegistry.list()
        assert "tick" in registered
        assert "volume" in registered
        assert "dollar" in registered
        assert "time" in registered

    def test_get_constructor_for_each_type(self) -> None:
        from flowbars.bars.registry import BarRegistry

        assert BarRegistry.get_constructor("tick") is TickBarConstructor
        assert BarRegistry.get_constructor("volume") is VolumeBarConstructor
        assert BarRegistry.get_constructor("dollar") is DollarBarConstructor
        assert BarRegistry.get_constructor("time") is TimeBarConstructor

    def test_get_batch_function_for_each_type(self) -> None:
        from flowbars.bars.registry import BarRegistry

        assert BarRegistry.get_batch_function("tick") is compute_tick_bars
        assert BarRegistry.get_batch_function("volume") is compute_volume_bars
        assert BarRegistry.get_batch_function("dollar") is compute_dollar_bars
        assert BarRegistry.get_batch_function("time") is compute_time_bars
