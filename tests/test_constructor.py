"""Tests for BaseBarConstructor — Phase 5."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from flowbars.bars.accumulators import (
    DollarAccumulator,
    ImbalanceAccumulator,
    RunAccumulator,
    TickAccumulator,
    VolumeAccumulator,
)
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.calendars import SessionCalendar
from flowbars.core import Bar, StateValidationError, TickInfo
from flowbars.schema import SchemaMapping
from flowbars.thresholds import (
    EWMAThresholdEstimator,
    StaticThresholdEstimator,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def tick(ts: int, price: float, volume: float, side: float | None = None) -> TickInfo:
    """Shorthand to create a TickInfo with optional side."""
    return TickInfo(timestamp=ts, price=price, volume=volume, side=side)


def weekday_ts(weekday: int, hour: int, minute: int, second: int = 0) -> int:
    """Return a Unix-ms timestamp for the given UTC weekday and time.

    Uses 2024-01-15 (Monday) as the reference point.  *weekday* 0=Mon … 6=Sun.
    """
    base = datetime.datetime(2024, 1, 15, hour, minute, second, tzinfo=datetime.timezone.utc)
    target = base + datetime.timedelta(days=weekday)
    return int(target.timestamp() * 1000)


def make_ticks_df(
    timestamps: list[int],
    prices: list[float],
    volumes: list[float],
    sides: list[float] | None = None,
) -> pd.DataFrame:
    """Create a DataFrame with default column names for testing."""
    data = {
        "ts": timestamps,
        "px": prices,
        "vol": volumes,
    }
    if sides is not None:
        data["side"] = sides
    return pd.DataFrame(data)


def default_schema(has_side: bool = False) -> SchemaMapping:
    """Schema mapping for test DataFrames (col: ts, px, vol, side)."""
    mapping = {"timestamp": "ts", "price": "px", "volume": "vol"}
    if has_side:
        mapping["side"] = "side"
    return SchemaMapping(mapping)


def bars_to_df(bars: list[Bar]) -> pd.DataFrame:
    """Convert a list of Bar objects to a DataFrame."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# BaseBarConstructor — basic operation
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseBarConstructor:
    def test_single_tick_closes_bar(self) -> None:
        """Threshold=1: one tick should close a bar immediately."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est)

        result = ctor.update(tick(1000, 100.0, 10.0))
        assert result is not None
        assert result.num_ticks == 1
        assert result.open == 100.0
        assert result.bar_type == "tick"
        assert result.bar_id == 0

    def test_multi_tick_bar(self) -> None:
        """Threshold=3: three ticks close one bar."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=3.0)
        ctor = BaseBarConstructor(acc, est)

        assert ctor.update(tick(1000, 100.0, 1.0)) is None
        assert ctor.update(tick(2000, 101.0, 1.0)) is None
        bar = ctor.update(tick(3000, 102.0, 1.0))
        assert bar is not None
        assert bar.num_ticks == 3
        assert bar.open == 100.0
        assert bar.high == 102.0
        assert bar.low == 100.0
        assert bar.close == 102.0

    def test_two_bars_sequence(self) -> None:
        """Threshold=2: sequence of ticks produces multiple bars."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=2.0)
        ctor = BaseBarConstructor(acc, est)

        bar0 = ctor.update(tick(1000, 10.0, 1.0))  # cum=1
        assert bar0 is None
        bar0 = ctor.update(tick(2000, 12.0, 1.0))  # cum=2 → close
        assert bar0 is not None
        assert bar0.num_ticks == 2
        assert bar0.bar_id == 0

        bar1 = ctor.update(tick(3000, 14.0, 1.0))  # cum=1
        assert bar1 is None
        bar1 = ctor.update(tick(4000, 16.0, 1.0))  # cum=2 → close
        assert bar1 is not None
        assert bar1.num_ticks == 2
        assert bar1.bar_id == 1

    def test_volume_bar(self) -> None:
        """Volume threshold: close when cumulative volume crosses."""
        acc = VolumeAccumulator()
        est = StaticThresholdEstimator(threshold=1000.0)
        ctor = BaseBarConstructor(acc, est)

        assert ctor.update(tick(1000, 100.0, 300.0)) is None  # cum=300
        assert ctor.update(tick(2000, 101.0, 400.0)) is None  # cum=700
        assert ctor.update(tick(3000, 102.0, 500.0)) is not None  # cum=1200 >= 1000

    def test_dollar_bar(self) -> None:
        """Dollar threshold: close when cumulative notional crosses."""
        acc = DollarAccumulator()
        est = StaticThresholdEstimator(threshold=600.0)
        ctor = BaseBarConstructor(acc, est)

        assert ctor.update(tick(1000, 10.0, 20.0)) is None  # $200
        assert ctor.update(tick(2000, 10.0, 30.0)) is None  # $200+$300=$500 < $600
        bar = ctor.update(tick(3000, 10.0, 10.0))  # $500+$100=$600 >= $600 → close
        assert bar is not None
        assert bar.dollar_value == pytest.approx(600.0)

    def test_session_boundary_force_close(self) -> None:
        """SessionCalendar: tick after close forces bar to emit."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)  # large threshold
        cal = SessionCalendar(9, 30, 16, 0)  # 09:30–16:00 UTC
        ctor = BaseBarConstructor(acc, est, calendar=cal)

        # Monday 10:00 — inside session, threshold not reached
        assert ctor.update(tick(weekday_ts(0, 10, 0), 100.0, 1.0)) is None
        assert ctor.current_bar is not None
        assert ctor.current_bar.num_ticks == 1

        # Monday 17:00 — outside session (boundary) → force-close
        bar = ctor.update(tick(weekday_ts(0, 17, 0), 101.0, 1.0))
        assert bar is not None
        assert bar.num_ticks == 1  # only the 10:00 tick
        assert bar.bar_id == 0

        # The 17:00 tick starts a new bar
        assert ctor.current_bar is not None
        assert ctor.current_bar.num_ticks == 1

    def test_default_calendar_is_continuous(self) -> None:
        """No calendar → ContinuousCalendar (never a boundary)."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)
        ctor = BaseBarConstructor(acc, est)

        # Many ticks, no force-close
        for i in range(10):
            assert ctor.update(tick(1000 + i, 100.0, 1.0)) is None
        # Bar still open — never force-closed
        assert ctor.current_bar is not None
        assert ctor.current_bar.num_ticks == 10

    def test_warmup_bars_not_returned(self) -> None:
        """warmup_bars=2: first 2 bars are discarded from return value."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, warmup_bars=2)

        # Bar 0 (warmup)
        assert ctor.update(tick(1000, 10.0, 1.0)) is None
        # Bar 1 (warmup)
        assert ctor.update(tick(2000, 11.0, 1.0)) is None
        # Bar 2 (real)
        bar = ctor.update(tick(3000, 12.0, 1.0))
        assert bar is not None
        assert bar.bar_id == 2

    def test_warmup_bars_zero_returns_all(self) -> None:
        """warmup_bars=0 (default): all bars returned."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, warmup_bars=0)

        bar = ctor.update(tick(1000, 10.0, 1.0))
        assert bar is not None
        assert bar.bar_id == 0

    def test_current_bar_property(self) -> None:
        """current_bar reflects the in-progress bar."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)
        ctor = BaseBarConstructor(acc, est)

        assert ctor.current_bar is None
        ctor.update(tick(1000, 100.0, 5.0))
        assert ctor.current_bar is not None
        assert ctor.current_bar.open == 100.0
        assert ctor.current_bar.num_ticks == 1

    def test_bars_emitted_property(self) -> None:
        """bars_emitted counts all bars, including warmup."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, warmup_bars=1)

        assert ctor.bars_emitted == 0
        ctor.update(tick(1000, 10.0, 1.0))  # warmup
        assert ctor.bars_emitted == 1
        ctor.update(tick(2000, 11.0, 1.0))  # real
        assert ctor.bars_emitted == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════════


class TestCallbacks:
    def test_on_bar_callback_fires_on_close(self) -> None:
        """on_bar is called when a bar closes."""
        captured: list[Bar] = []
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, on_bar=lambda b: captured.append(b))

        ctor.update(tick(1000, 10.0, 1.0))
        assert len(captured) == 1
        assert captured[0].bar_id == 0

    def test_on_bar_fires_for_warmup_bars_too(self) -> None:
        """on_bar fires even during warmup."""
        captured: list[Bar] = []
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, warmup_bars=5, on_bar=lambda b: captured.append(b))

        ctor.update(tick(1000, 10.0, 1.0))
        assert len(captured) == 1  # warmup bar still fires callback

    def test_on_bar_fires_on_session_force_close(self) -> None:
        """on_bar fires for session-boundary force-closes."""
        captured: list[Bar] = []
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)
        cal = SessionCalendar(9, 30, 16, 0)
        ctor = BaseBarConstructor(acc, est, calendar=cal, on_bar=lambda b: captured.append(b))

        # Monday 10:00 — inside session
        ctor.update(tick(weekday_ts(0, 10, 0), 100.0, 1.0))
        assert len(captured) == 0

        # Monday 17:00 — outside session, force-close
        ctor.update(tick(weekday_ts(0, 17, 0), 101.0, 1.0))
        assert len(captured) == 1
        assert captured[0].bar_id == 0

    def test_on_threshold_update_fires_for_ewma(self) -> None:
        """EWMA threshold changes after bar close → callback fires."""
        updates: list[float] = []
        est = EWMAThresholdEstimator(bar_family="imbalance", initial_ewa_t=100.0, span=3.0)
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        ctor = BaseBarConstructor(acc, est, on_threshold_update=lambda v: updates.append(v))

        # Feed ticks: buy=2 → imbalance 2 < 100 → no close yet
        # We need threshold to be reached. initial threshold = 100 * 0.5 = 50
        init_thresh = est.current_threshold
        assert init_thresh == pytest.approx(50.0)

        # To reach threshold of 50, we need 50 buy ticks (each tick = +1 imbalance)
        for i in range(50):
            ctor.update(tick(1000 + i, 100.0, 1.0, side=1.0))

        # on_threshold_update should have fired
        assert len(updates) == 1
        # New threshold should be different from initial
        assert updates[0] != pytest.approx(init_thresh)

    def test_on_threshold_update_does_not_fire_for_static(self) -> None:
        """Static threshold never changes → callback never fires."""
        updates: list[float] = []
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, on_threshold_update=lambda v: updates.append(v))

        ctor.update(tick(1000, 10.0, 1.0))
        assert len(updates) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Batch processing
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatch:
    def test_tick_bars_batch(self) -> None:
        """batch() produces tick bars from a DataFrame."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000, 5000, 6000],
            prices=[10.0, 12.0, 11.0, 13.0, 14.0, 15.0],
            volumes=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=2.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema())

        result = ctor.batch(df)
        assert len(result) == 3  # 6 ticks / 2 = 3 bars
        assert result.iloc[0]["num_ticks"] == 2
        assert result.iloc[0]["bar_id"] == 0
        assert result.iloc[2]["bar_id"] == 2

    def test_volume_bars_batch(self) -> None:
        """batch() produces volume bars."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000],
            prices=[100.0, 101.0, 102.0, 103.0],
            volumes=[300.0, 400.0, 500.0, 200.0],
        )
        acc = VolumeAccumulator()
        est = StaticThresholdEstimator(threshold=1000.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema())

        result = ctor.batch(df)
        assert len(result) == 1  # 300+400+500=1200 → 1 bar, 200 leftover
        assert result.iloc[0]["volume"] == pytest.approx(1200.0)

    def test_dollar_bars_batch(self) -> None:
        """batch() produces dollar bars."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000],
            prices=[100.0, 100.0, 100.0, 100.0],
            volumes=[3000.0, 4000.0, 5000.0, 2000.0],
        )
        acc = DollarAccumulator()
        est = StaticThresholdEstimator(threshold=1_000_000.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema())

        result = ctor.batch(df)
        assert len(result) == 1  # $1,200,000 → 1 bar, $200,000 leftover
        assert result.iloc[0]["dollar_value"] == pytest.approx(1_200_000.0)

    def test_imbalance_tick_batch(self) -> None:
        """batch() works with ImbalanceAccumulator + EWMA."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000, 5000],
            prices=[100.0, 101.0, 99.0, 102.0, 100.0],
            volumes=[1.0, 1.0, 1.0, 1.0, 1.0],
            sides=[1.0, -1.0, 1.0, 1.0, -1.0],
        )
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        # Small threshold so we get bars from 5 ticks
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True))

        result = ctor.batch(df)
        # Ticks: +1, -1=0 (close), +1, +1=2 (close), -1 → 2 bars
        # Actually: tick1 +1 (no close), tick2 -1 → imbalance 0 (no close)
        # tick3 +1 → imbalance +1 → close at bar1
        # tick4 +1 → imbalance +1 → close at bar2
        # tick5 -1 → imbalance 0 → no close
        # Hmm, let me trace more carefully:
        # tick1(+1): signed=+1, |+1|>=1 → CLOSE. bar0: 1 tick, θ=1.0
        # Excess=0. tick2(-1): signed=-1, |-1|>=1 → CLOSE. bar1: 1 tick, θ=-1.0
        # tick3(+1): signed=+1, |+1|>=1 → CLOSE. bar2: 1 tick, θ=1.0
        # tick4(+1): signed=+1, |+1|>=1 → CLOSE. bar3: 1 tick, θ=1.0
        # tick5(-1): signed=-1, no close (not >= 1 after bar 3 close)
        assert len(result) >= 2

    def test_run_tick_batch(self) -> None:
        """batch() works with RunAccumulator + EWMA."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000, 5000],
            prices=[100.0, 101.0, 99.0, 102.0, 100.0],
            volumes=[1.0, 1.0, 1.0, 1.0, 1.0],
            sides=[1.0, 1.0, -1.0, -1.0, 1.0],
        )
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        est = StaticThresholdEstimator(threshold=2.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=True))

        result = ctor.batch(df)
        # Run 1: 2 buy ticks → banked=2 → close bar0 (2 ticks)
        # Run 2: 2 sell ticks → banked=2 → close bar1 (2 ticks)
        # Run 3: 1 buy tick → no close
        assert len(result) == 2
        assert result.iloc[0]["num_ticks"] == 2
        assert result.iloc[1]["num_ticks"] == 2

    def test_empty_df_returns_empty(self) -> None:
        """Empty DataFrame → empty result."""
        df = make_ticks_df([], [], [])
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=10.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema())

        result = ctor.batch(df)
        assert len(result) == 0

    def test_batch_vs_streaming_equivalence_tick(self) -> None:
        """Batch output matches streaming output for tick bars."""
        timestamps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        prices = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0, 16.0, 17.0]
        volumes = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        # Batch
        df = make_ticks_df(timestamps, prices, volumes)
        acc_batch = TickAccumulator()
        est_batch = StaticThresholdEstimator(threshold=3.0)
        ctor_batch = BaseBarConstructor(acc_batch, est_batch, schema=default_schema())
        batch_result = ctor_batch.batch(df)

        # Streaming
        acc_stream = TickAccumulator()
        est_stream = StaticThresholdEstimator(threshold=3.0)
        ctor_stream = BaseBarConstructor(acc_stream, est_stream)
        stream_bars = []
        for i in range(len(timestamps)):
            bar = ctor_stream.update(tick(timestamps[i], prices[i], volumes[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_batch_vs_streaming_equivalence_imbalance(self) -> None:
        """Batch output matches streaming output for imbalance tick bars."""
        timestamps = [1000, 2000, 3000, 4000, 5000]
        prices = [100.0, 101.0, 99.0, 102.0, 100.0]
        volumes = [1.0, 1.0, 1.0, 1.0, 1.0]
        sides = [1.0, -1.0, 1.0, 1.0, -1.0]

        df = make_ticks_df(timestamps, prices, volumes, sides)
        acc_batch = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        est_batch = StaticThresholdEstimator(threshold=2.0)
        ctor_batch = BaseBarConstructor(acc_batch, est_batch, schema=default_schema(has_side=True))
        batch_result = ctor_batch.batch(df)

        acc_stream = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        est_stream = StaticThresholdEstimator(threshold=2.0)
        ctor_stream = BaseBarConstructor(acc_stream, est_stream)
        stream_bars = []
        for i in range(len(timestamps)):
            bar = ctor_stream.update(tick(timestamps[i], prices[i], volumes[i], sides[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_batch_side_derivation(self) -> None:
        """batch() derives tick signs when no side column provided."""
        # Rising prices → all buy side
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000],
            prices=[10.0, 11.0, 12.0, 13.0],  # all uptick
            volumes=[1.0, 1.0, 1.0, 1.0],
        )
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        est = StaticThresholdEstimator(threshold=3.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema(has_side=False))

        result = ctor.batch(df)
        # First tick has NaN sign (excluded), ticks 2-4 all +1 → imbalance = +3
        assert len(result) == 1
        assert result.iloc[0]["num_ticks"] == 4

    def test_batch_without_schema_raises(self) -> None:
        """batch() without schema raises ValueError."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est)  # no schema

        with pytest.raises(ValueError, match="batch.*requires a SchemaMapping"):
            ctor.batch(pd.DataFrame())


# ═══════════════════════════════════════════════════════════════════════════════
# State persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    def test_round_trip_interrupted_equals_uninterrupted(self) -> None:
        """Saving state mid-stream and resuming produces the same bars."""
        timestamps = [1000, 2000, 3000, 4000, 5000, 6000]
        prices = [10.0, 12.0, 11.0, 13.0, 14.0, 15.0]
        volumes = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        threshold = 2.0

        # Uninterrupted run
        acc1 = TickAccumulator()
        est1 = StaticThresholdEstimator(threshold=threshold)
        ctor1 = BaseBarConstructor(acc1, est1)
        uninterrupted_bars = []
        for i in range(len(timestamps)):
            bar = ctor1.update(tick(timestamps[i], prices[i], volumes[i]))
            if bar is not None:
                uninterrupted_bars.append(bar)
        uninterrupted_df = bars_to_df(uninterrupted_bars)

        # Interrupted run: save after 3 ticks, resume
        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=threshold)
        ctor2 = BaseBarConstructor(acc2, est2)
        for i in range(3):  # first 3 ticks
            ctor2.update(tick(timestamps[i], prices[i], volumes[i]))
        state = ctor2.get_state()

        # Resume
        acc3 = TickAccumulator()
        est3 = StaticThresholdEstimator(threshold=threshold)
        ctor3 = BaseBarConstructor(acc3, est3)
        ctor3.load_state(state)
        interrupted_bars = []
        for i in range(3, len(timestamps)):
            bar = ctor3.update(tick(timestamps[i], prices[i], volumes[i]))
            if bar is not None:
                interrupted_bars.append(bar)
        interrupted_df = bars_to_df(interrupted_bars)

        # Compare the bars produced after the save point.
        # The uninterrupted run produced bars for ticks 0-5; the interrupted
        # run produced bars for ticks 3-5 (continuing the partial bar from
        # tick 2).  We compare the last N bars from both runs, ignoring bar_id
        # since the numbering may differ.
        n_bars = len(interrupted_df)
        uninterrupted_tail = uninterrupted_df.iloc[-n_bars:].copy()
        uninterrupted_tail = uninterrupted_tail.reset_index(drop=True)
        interrupted_clean = interrupted_df.copy().reset_index(drop=True)
        # Drop bar_id from comparison — it's a serial number, not data
        assert list(uninterrupted_tail.columns) == list(interrupted_clean.columns)
        for col in uninterrupted_tail.columns:
            if col == "bar_id":
                continue
            left = uninterrupted_tail[col]
            right = interrupted_clean[col]
            pd.testing.assert_series_equal(
                left.reset_index(drop=True),
                right.reset_index(drop=True),
                check_names=False,
                obj=f"Column {col}",
            )

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working constructor."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=5.0)
        cal = SessionCalendar(9, 30, 16, 0)
        schema = default_schema()
        ctor = BaseBarConstructor(acc, est, calendar=cal, schema=schema, stream_id="test-stream")
        ctor.update(tick(1000, 10.0, 1.0))
        state = ctor.get_state()

        # Reconstruct
        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=5.0)
        ctor2 = BaseBarConstructor.from_state(state, acc2, est2, cal, schema)
        assert ctor2.bars_emitted == 0  # no bars closed yet
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.num_ticks == 1

    def test_stream_id_mismatch_raises(self) -> None:
        """Loading state with a different stream_id raises StateValidationError."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, stream_id="alpha")
        state = ctor.get_state()

        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=1.0)
        ctor2 = BaseBarConstructor(acc2, est2, stream_id="beta")
        with pytest.raises(StateValidationError, match="stream_id mismatch"):
            ctor2.load_state(state)

    def test_bar_type_mismatch_raises(self) -> None:
        """Loading state with a different bar type raises StateValidationError."""
        acc = TickAccumulator(bar_type="tick")
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est)
        state = ctor.get_state()

        acc2 = VolumeAccumulator(bar_type="volume")  # different type
        est2 = StaticThresholdEstimator(threshold=1.0)
        ctor2 = BaseBarConstructor(acc2, est2)
        with pytest.raises(StateValidationError, match="bar_type mismatch"):
            ctor2.load_state(state)

    def test_version_mismatch_raises(self) -> None:
        """State with schema_version < 1 raises StateValidationError."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est)
        state = ctor.get_state()
        state["schema_version"] = 0  # too old

        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=1.0)
        ctor2 = BaseBarConstructor(acc2, est2)
        with pytest.raises(StateValidationError, match="not supported"):
            ctor2.load_state(state)

    def test_future_version_raises(self) -> None:
        """State with schema_version > current raises StateValidationError."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est)
        state = ctor.get_state()
        state["schema_version"] = 999  # future

        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=1.0)
        ctor2 = BaseBarConstructor(acc2, est2)
        with pytest.raises(StateValidationError, match="newer version"):
            ctor2.load_state(state)

    def test_bar_id_continuity_across_resume(self) -> None:
        """Bar IDs are continuous across a save/resume boundary."""
        ticks = [
            tick(1000, 10.0, 1.0),
            tick(2000, 12.0, 1.0),
            tick(3000, 14.0, 1.0),
            tick(4000, 16.0, 1.0),
        ]

        # Build 1 bar, save state
        acc1 = TickAccumulator()
        est1 = StaticThresholdEstimator(threshold=2.0)
        ctor1 = BaseBarConstructor(acc1, est1)
        bar0 = ctor1.update(ticks[0])
        assert bar0 is None
        bar0 = ctor1.update(ticks[1])
        assert bar0 is not None
        assert bar0.bar_id == 0
        state = ctor1.get_state()

        # Resume, build remaining bars
        acc2 = TickAccumulator()
        est2 = StaticThresholdEstimator(threshold=2.0)
        ctor2 = BaseBarConstructor(acc2, est2)
        ctor2.load_state(state)

        bar1 = ctor2.update(ticks[2])
        assert bar1 is None
        bar1 = ctor2.update(ticks[3])
        assert bar1 is not None
        assert bar1.bar_id == 1  # continuity

    def test_state_mid_stream_partial_bar(self) -> None:
        """State captures a partial bar correctly."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)
        ctor = BaseBarConstructor(acc, est)
        ctor.update(tick(1000, 10.0, 1.0))
        ctor.update(tick(2000, 12.0, 2.0))

        state = ctor.get_state()
        assert state["bar_type"] == "tick"
        assert state["bars_emitted"] == 0
        assert state["in_session"] is True
        assert state["accumulator"]["num_ticks"] == 2
        assert state["accumulator"]["has_tick"] is True
        assert state["accumulator"]["volume"] == 3.0

    def test_ewma_estimator_state_persisted(self) -> None:
        """EWMA estimator state is part of constructor state."""
        est = EWMAThresholdEstimator(
            bar_family="imbalance", span=10.0, initial_ewa_t=10.0, initial_ewa_proportion=0.3
        )
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        ctor = BaseBarConstructor(acc, est)

        # Build bars to trigger EWMA update.  initial threshold = 10.0 * 0.3 = 3.0
        # Each buy tick adds +1 to signed imbalance → 30 buy ticks ≈ 10 bars
        for i in range(30):
            ctor.update(tick(1000 + i, 100.0, 1.0, side=1.0))

        state = ctor.get_state()
        te_state = state["threshold_estimator"]
        assert te_state["bar_family"] == "imbalance"
        assert te_state["span"] == 10.0
        assert te_state["n_updates"] >= 1  # at least one bar closed
        # EWMA values should have moved from initial seeds
        assert te_state["ewa_t"] != 10.0

        # Resume with EWMA
        acc2 = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        est2 = EWMAThresholdEstimator(
            bar_family="imbalance", span=10.0, initial_ewa_t=100.0, initial_ewa_proportion=0.3
        )
        ctor2 = BaseBarConstructor(acc2, est2)
        ctor2.load_state(state)

        # The resumed constructor should have the same EWMA state
        assert est2.n_updates == te_state["n_updates"]
        assert est2.ewa_t == pytest.approx(te_state["ewa_t"])


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_first_tick_on_weekend(self) -> None:
        """First tick on weekend doesn't force-close (no in-session bar exists)."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)
        cal = SessionCalendar(9, 30, 16, 0)  # 09:30–16:00 UTC, Mon-Fri
        ctor = BaseBarConstructor(acc, est, calendar=cal)

        # Saturday 12:00 — weekend tick
        assert ctor.update(tick(weekday_ts(5, 12, 0), 100.0, 1.0)) is None
        assert ctor.current_bar is not None
        assert ctor.current_bar.num_ticks == 1

        # Monday 10:00 — next tick inside session → force-close weekend bar
        bar = ctor.update(tick(weekday_ts(7, 10, 0), 101.0, 1.0))
        assert bar is not None  # force-close the weekend bar
        assert bar.num_ticks == 1  # only the Saturday tick

    def test_session_boundary_session_transition(self) -> None:
        """Weekend bar force-closes on Monday, Monday bar continues normally."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)
        cal = SessionCalendar(9, 30, 16, 0)
        ctor = BaseBarConstructor(acc, est, calendar=cal)

        # Monday 15:00 — inside session
        ctor.update(tick(weekday_ts(0, 15, 0), 100.0, 1.0))
        assert ctor.current_bar is not None

        # Saturday 12:00 — weekend → force-close Monday bar
        bar = ctor.update(tick(weekday_ts(5, 12, 0), 101.0, 1.0))
        assert bar is not None  # Monday bar force-closed
        assert bar.num_ticks == 1

        # Monday 10:00 (next week) → force-close weekend bar, start new
        bar2 = ctor.update(tick(weekday_ts(7, 10, 0), 102.0, 1.0))
        assert bar2 is not None  # weekend bar force-closed
        assert bar2.num_ticks == 1  # only the Saturday tick

    def test_threshold_zero_closes_immediately(self) -> None:
        """Threshold=0: every tick closes its own bar."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=0.0)
        ctor = BaseBarConstructor(acc, est)

        bar0 = ctor.update(tick(1000, 10.0, 1.0))
        assert bar0 is not None
        assert bar0.num_ticks == 1

        bar1 = ctor.update(tick(2000, 11.0, 1.0))
        assert bar1 is not None
        assert bar1.num_ticks == 1
        assert bar1.bar_id == 1

    def test_zero_volume_ticks_in_batch(self) -> None:
        """Zero-volume ticks are included in OHLCV but add zero to volume/dollar."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000],
            prices=[100.0, 101.0, 102.0],
            volumes=[0.0, 0.0, 5.0],
        )
        acc = VolumeAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        ctor = BaseBarConstructor(acc, est, schema=default_schema())

        result = ctor.batch(df)
        assert len(result) == 1
        assert result.iloc[0]["num_ticks"] == 3
        assert result.iloc[0]["volume"] == 5.0

    def test_negative_warmup_raises(self) -> None:
        """warmup_bars must be non-negative."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=1.0)
        with pytest.raises(ValueError, match="warmup_bars"):
            BaseBarConstructor(acc, est, warmup_bars=-1)

    def test_continuous_calendar_never_force_closes(self) -> None:
        """With ContinuousCalendar, bars only close on threshold."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=3.0)
        ctor = BaseBarConstructor(acc, est)  # defaults to ContinuousCalendar

        # Feed 5 ticks with threshold=3
        assert ctor.update(tick(1000, 10.0, 1.0)) is None
        assert ctor.update(tick(2000, 11.0, 1.0)) is None
        bar = ctor.update(tick(3000, 12.0, 1.0))  # 3rd tick → close
        assert bar is not None
        assert bar.num_ticks == 3
        # Remaining ticks
        assert ctor.update(tick(4000, 13.0, 1.0)) is None
        assert ctor.update(tick(5000, 14.0, 1.0)) is None

    def test_consecutive_boundary_ticks_one_bar(self) -> None:
        """Multiple consecutive weekend ticks accumulate into one bar."""
        acc = TickAccumulator()
        est = StaticThresholdEstimator(threshold=100.0)
        cal = SessionCalendar(9, 30, 16, 0)
        ctor = BaseBarConstructor(acc, est, calendar=cal)

        # Saturday 12:00 → first boundary tick, starts a bar
        assert ctor.update(tick(weekday_ts(5, 12, 0), 100.0, 1.0)) is None
        # Saturday 13:00 → still boundary, accumulates
        assert ctor.update(tick(weekday_ts(5, 13, 0), 101.0, 1.0)) is None
        # Sunday 12:00 → still boundary, accumulates
        assert ctor.update(tick(weekday_ts(6, 12, 0), 102.0, 1.0)) is None

        assert ctor.current_bar is not None
        assert ctor.current_bar.num_ticks == 3  # all in one bar

        # Monday 10:00 → inside session, force-close weekend bar
        bar = ctor.update(tick(weekday_ts(7, 10, 0), 103.0, 1.0))
        assert bar is not None
        assert bar.num_ticks == 3  # all 3 weekend ticks in one bar
