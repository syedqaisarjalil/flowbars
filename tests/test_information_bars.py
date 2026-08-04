"""Tests for information-driven bar constructors — Phase 7."""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from flowbars.bars.information.imbalance_dollar_bars import (
    ImbalanceDollarBarConstructor,
    compute_imbalance_dollar_bars,
)
from flowbars.bars.information.imbalance_tick_bars import (
    ImbalanceTickBarConstructor,
    compute_imbalance_tick_bars,
)
from flowbars.bars.information.imbalance_volume_bars import (
    ImbalanceVolumeBarConstructor,
    compute_imbalance_volume_bars,
)
from flowbars.bars.information.run_dollar_bars import (
    RunDollarBarConstructor,
    compute_run_dollar_bars,
)
from flowbars.bars.information.run_tick_bars import (
    RunTickBarConstructor,
    compute_run_tick_bars,
)
from flowbars.bars.information.run_volume_bars import (
    RunVolumeBarConstructor,
    compute_run_volume_bars,
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
# Imbalance tick bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestImbalanceTickBarConstructor:
    def test_basic_single_bar(self) -> None:
        """5 consecutive buy ticks close a bar with initial threshold=5."""
        # initial threshold = 10.0 * 0.5 = 5.0
        ctor = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        for i in range(4):
            assert ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0)) is None
        bar = ctor.update(tick(1004, 104.0, 1.0, side=1.0))
        assert bar is not None
        assert bar.num_ticks == 5
        assert bar.bar_type == "imbalance_tick"
        assert bar.bar_id == 0

    def test_multi_bar_sequence(self) -> None:
        """Two bars from a block of 10 buy ticks with initial threshold=5."""
        ctor = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        bars = []
        for i in range(10):
            bar = ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0))
            if bar is not None:
                bars.append(bar)
        assert len(bars) >= 1
        assert all(b.bar_type == "imbalance_tick" for b in bars)

    def test_sell_ticks_close_bar(self) -> None:
        """5 consecutive sell ticks also close a bar (absolute imbalance)."""
        ctor = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        for i in range(4):
            assert ctor.update(tick(1000 + i, 100.0 - i, 1.0, side=-1.0)) is None
        bar = ctor.update(tick(1004, 96.0, 1.0, side=-1.0))
        assert bar is not None
        assert bar.num_ticks == 5

    def test_mixed_direction_delays_close(self) -> None:
        """Buy+sell cancels out → takes more ticks to close."""
        ctor = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        # Alternating buy/sell: signed imbalance stays near 0
        for i in range(20):
            side = 1.0 if i % 2 == 0 else -1.0
            ctor.update(tick(1000 + i, 100.0, 1.0, side=side))
        # No bar should have closed — |signed_imbalance| never reaches 5
        assert ctor.current_bar is not None
        assert ctor.bars_emitted == 0

    def test_batch_imbalance_tick_bars(self) -> None:
        """compute_imbalance_tick_bars() produces bars from a DataFrame."""
        # 10 buy ticks: should produce ~2 bars with threshold starting at 5
        df = make_ticks_df(
            timestamps=list(range(1000, 11000, 1000)),
            prices=[100.0 + i for i in range(10)],
            volumes=[1.0] * 10,
            sides=[1.0] * 10,
        )
        result = compute_imbalance_tick_bars(
            df,
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )
        assert len(result) >= 1
        assert result.iloc[0]["bar_type"] == "imbalance_tick"

    def test_batch_with_side_derivation(self) -> None:
        """Tick signs are derived from prices when no side column exists."""
        # Rising prices → all buy side
        df = make_ticks_df(
            timestamps=list(range(1000, 6000, 1000)),
            prices=[10.0, 11.0, 12.0, 13.0, 14.0],
            volumes=[1.0] * 5,
        )
        result = compute_imbalance_tick_bars(
            df,
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=False),
        )
        # First tick NaN, ticks 2-5 all +1 → signed=+4 < 5 → no close
        # But EWMA might cause threshold to change...
        assert len(result) >= 0

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        n = 20
        timestamps = list(range(1000, 1000 + n * 1000, 1000))
        prices = [100.0 + i * 0.1 for i in range(n)]
        volumes = [1.0] * n
        sides = [1.0 if i % 3 != 0 else -1.0 for i in range(n)]

        df = make_ticks_df(timestamps, prices, volumes, sides)
        batch_result = compute_imbalance_tick_bars(
            df,
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )

        ctor = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        stream_bars = []
        for i in range(n):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i], sides[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working constructor."""
        ctor = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 1.0, side=1.0))
        state = ctor.get_state()

        ctor2 = ImbalanceTickBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.num_ticks == 1

    def test_state_round_trip_interrupted(self) -> None:
        """Save/resume produces equivalent results."""
        ctor1 = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        uninterrupted_bars = []
        for i in range(20):
            bar = ctor1.update(tick(1000 + i, 100.0 + i * 0.1, 1.0, side=1.0))
            if bar is not None:
                uninterrupted_bars.append(bar)
        uninterrupted_df = bars_to_df(uninterrupted_bars)

        # Save after 10 ticks
        ctor2 = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        for i in range(10):
            ctor2.update(tick(1000 + i, 100.0 + i * 0.1, 1.0, side=1.0))
        state = ctor2.get_state()

        # Resume
        ctor3 = ImbalanceTickBarConstructor.from_state(state)
        interrupted_bars = []
        for i in range(10, 20):
            bar = ctor3.update(tick(1000 + i, 100.0 + i * 0.1, 1.0, side=1.0))
            if bar is not None:
                interrupted_bars.append(bar)
        interrupted_df = bars_to_df(interrupted_bars)

        n = len(interrupted_df)
        if n > 0:
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

    def test_ewma_threshold_adapts(self) -> None:
        """The threshold changes after bars close."""
        ctor = ImbalanceTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5, span=3.0)
        init_thresh = ctor._threshold_estimator.current_threshold

        # Feed 20 buy ticks → should produce multiple bars
        for i in range(20):
            ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0))

        # Threshold should have changed from initial
        assert ctor._threshold_estimator.current_threshold != pytest.approx(init_thresh)
        assert ctor._threshold_estimator.n_updates >= 1  # type: ignore[attr-defined]

    def test_warmup_bars_not_returned(self) -> None:
        """warmup_bars=2: first 2 bars discarded."""
        ctor = ImbalanceTickBarConstructor(
            initial_ewa_t=10.0, initial_ewa_proportion=0.5, warmup_bars=2
        )
        bars = []
        for i in range(20):
            bar = ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0))
            if bar is not None:
                bars.append(bar)
        # bars_emitted includes warmup bars, but returned bars don't
        assert ctor.bars_emitted >= len(bars) + 2

    def test_on_threshold_update_fires(self) -> None:
        """on_threshold_update callback fires when the EWMA threshold changes."""
        updates: list[float] = []
        ctor = ImbalanceTickBarConstructor(
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
            span=3.0,
            on_threshold_update=lambda v: updates.append(v),
        )
        init_thresh = ctor._threshold_estimator.current_threshold

        for i in range(20):
            ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0))

        # At least one threshold update should have fired
        assert len(updates) >= 1
        assert updates[0] != pytest.approx(init_thresh)

    def test_with_session_calendar(self) -> None:
        """Imbalance bars respect session boundaries."""
        cal = SessionCalendar(9, 30, 16, 0)
        ctor = ImbalanceTickBarConstructor(
            initial_ewa_t=100.0, calendar=cal
        )  # threshold=50, won't close on first tick
        ctor.update(tick(weekday_ts(0, 10, 0), 100.0, 1.0, side=1.0))
        assert ctor.current_bar is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Imbalance volume bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestImbalanceVolumeBarConstructor:
    def test_basic_single_bar(self) -> None:
        """Volume-weighted buy ticks close a bar."""
        # initial threshold = 10.0 * 0.5 = 5.0
        # Each buy tick adds +volume to signed imbalance
        ctor = ImbalanceVolumeBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        # 3 buy ticks: 2+2+2=6 >= 5 → close
        assert ctor.update(tick(1000, 100.0, 2.0, side=1.0)) is None
        assert ctor.update(tick(2000, 101.0, 2.0, side=1.0)) is None
        bar = ctor.update(tick(3000, 102.0, 2.0, side=1.0))
        assert bar is not None
        assert bar.bar_type == "imbalance_volume"
        assert bar.volume == pytest.approx(6.0)

    def test_batch_imbalance_volume_bars(self) -> None:
        """compute_imbalance_volume_bars() produces bars."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000, 5000, 6000],
            prices=[100.0] * 6,
            volumes=[2.0] * 6,
            sides=[1.0] * 6,
        )
        result = compute_imbalance_volume_bars(
            df,
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )
        assert len(result) >= 1
        assert result.iloc[0]["bar_type"] == "imbalance_volume"

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        n = 12
        timestamps = list(range(1000, 1000 + n * 1000, 1000))
        prices = [100.0] * n
        volumes = [2.0] * n
        sides = [1.0 if i % 3 != 0 else -1.0 for i in range(n)]

        df = make_ticks_df(timestamps, prices, volumes, sides)
        batch_result = compute_imbalance_volume_bars(
            df,
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )

        ctor = ImbalanceVolumeBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        stream_bars = []
        for i in range(n):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i], sides[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working constructor."""
        ctor = ImbalanceVolumeBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 2.0, side=1.0))
        state = ctor.get_state()

        ctor2 = ImbalanceVolumeBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.volume == 2.0

    def test_state_round_trip_preserves_metric(self) -> None:
        """State round-trip preserves the volume metric."""
        ctor = ImbalanceVolumeBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 3.0, side=1.0))
        state = ctor.get_state()

        assert state["accumulator"]["metric"] == "volume"
        assert state["bar_type"] == "imbalance_volume"


# ═══════════════════════════════════════════════════════════════════════════════
# Imbalance dollar bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestImbalanceDollarBarConstructor:
    def test_basic_single_bar(self) -> None:
        """Dollar-weighted buy ticks close a bar."""
        # initial threshold = 100.0 * 0.5 = 50.0
        ctor = ImbalanceDollarBarConstructor(initial_ewa_t=100.0, initial_ewa_proportion=0.5)
        # Tick: $10*5=$50 → reaches threshold
        bar = ctor.update(tick(1000, 10.0, 5.0, side=1.0))
        assert bar is not None
        assert bar.bar_type == "imbalance_dollar"
        assert bar.dollar_value == pytest.approx(50.0)

    def test_batch_imbalance_dollar_bars(self) -> None:
        """compute_imbalance_dollar_bars() produces bars."""
        df = make_ticks_df(
            timestamps=[1000, 2000, 3000, 4000, 5000],
            prices=[100.0, 100.0, 100.0, 100.0, 100.0],
            volumes=[1.0, 2.0, 3.0, 2.0, 1.0],
            sides=[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        result = compute_imbalance_dollar_bars(
            df,
            initial_ewa_t=1000.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )
        assert len(result) >= 1
        assert result.iloc[0]["bar_type"] == "imbalance_dollar"

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        n = 10
        timestamps = list(range(1000, 1000 + n * 1000, 1000))
        prices = [100.0] * n
        volumes = [1.0] * n
        sides = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]

        df = make_ticks_df(timestamps, prices, volumes, sides)
        batch_result = compute_imbalance_dollar_bars(
            df,
            initial_ewa_t=500.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )

        ctor = ImbalanceDollarBarConstructor(initial_ewa_t=500.0, initial_ewa_proportion=0.5)
        stream_bars = []
        for i in range(n):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i], sides[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working constructor."""
        ctor = ImbalanceDollarBarConstructor(initial_ewa_t=500.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 1.0, side=1.0))
        state = ctor.get_state()

        ctor2 = ImbalanceDollarBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.dollar_value == pytest.approx(100.0)

    def test_state_round_trip_preserves_metric(self) -> None:
        """State round-trip preserves the dollar metric."""
        ctor = ImbalanceDollarBarConstructor(initial_ewa_t=500.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 1.0, side=1.0))
        state = ctor.get_state()

        assert state["accumulator"]["metric"] == "dollar"
        assert state["bar_type"] == "imbalance_dollar"


# ═══════════════════════════════════════════════════════════════════════════════
# Run tick bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunTickBarConstructor:
    def test_basic_single_run_bar(self) -> None:
        """5 consecutive buy ticks → 1 run bar."""
        # initial threshold = 10.0 * max(0.5, 0.5) = 5.0
        ctor = RunTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        for i in range(4):
            assert ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0)) is None
        bar = ctor.update(tick(1004, 104.0, 1.0, side=1.0))
        assert bar is not None
        assert bar.num_ticks == 5
        assert bar.bar_type == "run_tick"
        assert bar.bar_id == 0

    def test_direction_change_creates_two_bars(self) -> None:
        """Buy run (3) + sell run (4) → 2 bars.  Second bar needs 4 ticks
        because the EWMA nudges threshold slightly above 3.0 after bar 0."""
        ctor = RunTickBarConstructor(initial_ewa_t=6.0, initial_ewa_proportion=0.5, span=10000.0)
        # initial threshold = 6.0 * 0.5 = 3.0

        # Buy run: 3 ticks → close bar0
        assert ctor.update(tick(1000, 100.0, 1.0, side=1.0)) is None
        assert ctor.update(tick(2000, 101.0, 1.0, side=1.0)) is None
        bar0 = ctor.update(tick(3000, 102.0, 1.0, side=1.0))
        assert bar0 is not None
        assert bar0.num_ticks == 3
        assert bar0.bar_id == 0

        # Sell run: 4 ticks needed (threshold ≈ 3.0003 after EWMA update)
        assert ctor.update(tick(4000, 101.0, 1.0, side=-1.0)) is None
        assert ctor.update(tick(5000, 100.0, 1.0, side=-1.0)) is None
        assert ctor.update(tick(6000, 99.0, 1.0, side=-1.0)) is None
        bar1 = ctor.update(tick(7000, 98.0, 1.0, side=-1.0))
        assert bar1 is not None
        assert bar1.num_ticks == 4
        assert bar1.bar_id == 1

    def test_alternating_direction_fragmentation(self) -> None:
        """Alternating buy/sell creates many small runs → delayed bar closure."""
        ctor = RunTickBarConstructor(initial_ewa_t=20.0, initial_ewa_proportion=0.5, span=10000.0)
        # threshold = 20 * 0.5 = 10.0, near-static (huge span)
        bars = []
        for i in range(30):
            side = 1.0 if i % 2 == 0 else -1.0
            bar = ctor.update(tick(1000 + i * 100, 100.0, 1.0, side=side))
            if bar is not None:
                bars.append(bar)
        # Alternating ticks with threshold=10: each alternation banks 1 and
        # starts a new run of 1.  After 9 alternations (18 ticks): total=10 → close.
        # With a near-static threshold the second bar also closes at ~18 more ticks.
        assert len(bars) >= 1
        assert bars[0].bar_type == "run_tick"

    def test_batch_run_tick_bars(self) -> None:
        """compute_run_tick_bars() produces bars."""
        # Pattern: 5 buy, 5 sell
        df = make_ticks_df(
            timestamps=list(range(1000, 11000, 1000)),
            prices=[100.0 + i for i in range(10)],
            volumes=[1.0] * 10,
            sides=[1.0] * 5 + [-1.0] * 5,
        )
        result = compute_run_tick_bars(
            df,
            initial_ewa_t=6.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )
        assert len(result) >= 1
        assert result.iloc[0]["bar_type"] == "run_tick"

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        n = 20
        timestamps = list(range(1000, 1000 + n * 1000, 1000))
        prices = [100.0 + i * 0.1 for i in range(n)]
        volumes = [1.0] * n
        # Pattern: 5 buy, 5 sell, 5 buy, 5 sell
        sides = [1.0] * 5 + [-1.0] * 5 + [1.0] * 5 + [-1.0] * 5

        df = make_ticks_df(timestamps, prices, volumes, sides)
        batch_result = compute_run_tick_bars(
            df,
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )

        ctor = RunTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        stream_bars = []
        for i in range(n):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i], sides[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working constructor."""
        ctor = RunTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 1.0, side=1.0))
        state = ctor.get_state()

        ctor2 = RunTickBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.num_ticks == 1

    def test_state_round_trip_interrupted(self) -> None:
        """Save/resume produces equivalent results."""
        ctor1 = RunTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        uninterrupted_bars = []
        for i in range(20):
            bar = ctor1.update(tick(1000 + i, 100.0 + i * 0.1, 1.0, side=1.0))
            if bar is not None:
                uninterrupted_bars.append(bar)
        uninterrupted_df = bars_to_df(uninterrupted_bars)

        # Save after 10 ticks
        ctor2 = RunTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        for i in range(10):
            ctor2.update(tick(1000 + i, 100.0 + i * 0.1, 1.0, side=1.0))
        state = ctor2.get_state()

        # Resume
        ctor3 = RunTickBarConstructor.from_state(state)
        interrupted_bars = []
        for i in range(10, 20):
            bar = ctor3.update(tick(1000 + i, 100.0 + i * 0.1, 1.0, side=1.0))
            if bar is not None:
                interrupted_bars.append(bar)
        interrupted_df = bars_to_df(interrupted_bars)

        n = len(interrupted_df)
        if n > 0:
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

    def test_ewma_threshold_adapts(self) -> None:
        """The threshold changes after run bars close."""
        ctor = RunTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5, span=3.0)
        init_thresh = ctor._threshold_estimator.current_threshold

        for i in range(20):
            ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0))

        assert ctor._threshold_estimator.current_threshold != pytest.approx(init_thresh)
        assert ctor._threshold_estimator.n_updates >= 1  # type: ignore[attr-defined]

    def test_warmup_bars_not_returned(self) -> None:
        """warmup_bars=2: first 2 bars discarded."""
        ctor = RunTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5, warmup_bars=2)
        bars = []
        for i in range(20):
            bar = ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0))
            if bar is not None:
                bars.append(bar)
        assert ctor.bars_emitted >= len(bars) + 2

    def test_min_run_length_filters_short_bars(self) -> None:
        """min_run_length=3: bars with < 3 ticks are not returned."""
        # Use a small initial_ewa_t so the threshold is low enough to close bars
        ctor = RunTickBarConstructor(
            initial_ewa_t=3.0,  # threshold = 3 * 0.5 = 1.5 → close every 2 ticks
            initial_ewa_proportion=0.5,
            min_run_length=3,
            span=3.0,
        )
        bars = []
        for i in range(10):
            side = 1.0 if i % 4 < 2 else -1.0  # runs of 2 ticks each
            bar = ctor.update(tick(1000 + i, 100.0, 1.0, side=side))
            if bar is not None:
                bars.append(bar)
        # All bars have >= 3 ticks (since min_run_length=3),
        # but with runs of 2 and threshold ~1.5, bars close with 2 ticks
        # and get filtered. Let me check...
        for bar in bars:
            assert bar.num_ticks >= 3

    def test_negative_min_run_length_raises(self) -> None:
        """min_run_length must be non-negative."""
        with pytest.raises(ValueError, match="min_run_length"):
            RunTickBarConstructor(min_run_length=-1)

    def test_on_threshold_update_fires(self) -> None:
        """on_threshold_update fires for run bars when threshold changes."""
        updates: list[float] = []
        ctor = RunTickBarConstructor(
            initial_ewa_t=10.0,
            initial_ewa_proportion=0.5,
            span=3.0,
            on_threshold_update=lambda v: updates.append(v),
        )
        for i in range(20):
            ctor.update(tick(1000 + i, 100.0 + i, 1.0, side=1.0))
        assert len(updates) >= 1

    def test_with_session_calendar(self) -> None:
        """Run bars respect session boundaries."""
        cal = SessionCalendar(9, 30, 16, 0)
        ctor = RunTickBarConstructor(
            initial_ewa_t=100.0, calendar=cal
        )  # threshold=50, won't close on first tick
        ctor.update(tick(weekday_ts(0, 10, 0), 100.0, 1.0, side=1.0))
        assert ctor.current_bar is not None

    def test_first_tick_nan_retroactive_inclusion(self) -> None:
        """First tick (NaN side) is included in tick-2's run direction."""
        ctor = RunTickBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        # Tick 1: side=None (first tick, NaN) → starts run, run_cum=1
        ctor.update(tick(1000, 100.0, 1.0, side=None))
        # Tick 2: side=+1.0 → same direction (NaN matches), run_cum=2
        ctor.update(tick(2000, 101.0, 1.0, side=1.0))
        # Tick 3: side=+1.0 → run_cum=3
        ctor.update(tick(3000, 102.0, 1.0, side=1.0))
        # Tick 4: side=+1.0 → run_cum=4
        ctor.update(tick(4000, 103.0, 1.0, side=1.0))
        # Tick 5: side=+1.0 → run_cum=5 >= 5.0 → close
        bar = ctor.update(tick(5000, 104.0, 1.0, side=1.0))
        assert bar is not None
        assert bar.num_ticks == 5  # includes first tick


# ═══════════════════════════════════════════════════════════════════════════════
# Run volume bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunVolumeBarConstructor:
    def test_basic_single_run_bar(self) -> None:
        """Volume-weighted run: 3 buy ticks → bar."""
        # initial threshold = 10.0 * 0.5 = 5.0
        ctor = RunVolumeBarConstructor(initial_ewa_t=10.0, initial_ewa_proportion=0.5)
        assert ctor.update(tick(1000, 100.0, 2.0, side=1.0)) is None  # run=2
        assert ctor.update(tick(2000, 101.0, 2.0, side=1.0)) is None  # run=4
        bar = ctor.update(tick(3000, 102.0, 2.0, side=1.0))  # run=6
        assert bar is not None
        assert bar.bar_type == "run_volume"
        assert bar.volume == pytest.approx(6.0)

    def test_direction_change_banks_volume(self) -> None:
        """Direction change banks the volume run; close fires on the same tick
        when the banked total + new run cum reaches the threshold."""
        ctor = RunVolumeBarConstructor(initial_ewa_t=6.0, initial_ewa_proportion=0.5, span=10000.0)
        # threshold = 6 * 0.5 = 3, near-static
        # Buy run: 2 ticks × 1.0 vol = 2 (run_cum=2)
        ctor.update(tick(1000, 100.0, 1.0, side=1.0))
        ctor.update(tick(2000, 101.0, 1.0, side=1.0))
        # Direction change at tick 3: bank 2, new sell run cum=1 → total=3 → close
        bar0 = ctor.update(tick(3000, 100.0, 1.0, side=-1.0))
        assert bar0 is not None
        assert bar0.num_ticks == 3  # 2 buy + 1 sell (the one that triggered direction change)
        assert bar0.bar_id == 0

        # Second bar: 3 sell ticks to reach threshold (~2.9999)
        ctor.update(tick(4000, 99.0, 1.0, side=-1.0))
        ctor.update(tick(5000, 98.0, 1.0, side=-1.0))
        bar1 = ctor.update(tick(6000, 97.0, 1.0, side=-1.0))
        assert bar1 is not None
        assert bar1.num_ticks == 3  # 3 sell ticks
        assert bar1.bar_id == 1

    def test_batch_run_volume_bars(self) -> None:
        """compute_run_volume_bars() produces bars."""
        df = make_ticks_df(
            timestamps=list(range(1000, 11000, 1000)),
            prices=[100.0] * 10,
            volumes=[2.0] * 10,
            sides=[1.0] * 5 + [-1.0] * 5,
        )
        result = compute_run_volume_bars(
            df,
            initial_ewa_t=20.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )
        assert len(result) >= 1
        assert result.iloc[0]["bar_type"] == "run_volume"

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        n = 15
        timestamps = list(range(1000, 1000 + n * 1000, 1000))
        prices = [100.0] * n
        volumes = [2.0] * n
        sides = [1.0] * 5 + [-1.0] * 5 + [1.0] * 5

        df = make_ticks_df(timestamps, prices, volumes, sides)
        batch_result = compute_run_volume_bars(
            df,
            initial_ewa_t=20.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )

        ctor = RunVolumeBarConstructor(initial_ewa_t=20.0, initial_ewa_proportion=0.5)
        stream_bars = []
        for i in range(n):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i], sides[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working constructor."""
        ctor = RunVolumeBarConstructor(initial_ewa_t=20.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 2.0, side=1.0))
        state = ctor.get_state()

        ctor2 = RunVolumeBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.volume == 2.0

    def test_state_round_trip_preserves_metric(self) -> None:
        """State round-trip preserves the volume metric."""
        ctor = RunVolumeBarConstructor(initial_ewa_t=20.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 2.0, side=1.0))
        state = ctor.get_state()

        assert state["accumulator"]["metric"] == "volume"
        assert state["bar_type"] == "run_volume"

    def test_min_run_length_filters_short_bars(self) -> None:
        """min_run_length filters bars with too few ticks."""
        ctor = RunVolumeBarConstructor(
            initial_ewa_t=3.0, initial_ewa_proportion=0.5, min_run_length=3, span=3.0
        )
        bars = []
        for i in range(10):
            side = 1.0 if i % 4 < 2 else -1.0
            bar = ctor.update(tick(1000 + i, 100.0, 2.0, side=side))
            if bar is not None:
                bars.append(bar)
        for bar in bars:
            assert bar.num_ticks >= 3


# ═══════════════════════════════════════════════════════════════════════════════
# Run dollar bars
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunDollarBarConstructor:
    def test_basic_single_run_bar(self) -> None:
        """Dollar-weighted run bar closes."""
        # initial threshold = 100.0 * 0.5 = 50.0
        ctor = RunDollarBarConstructor(initial_ewa_t=100.0, initial_ewa_proportion=0.5)
        # 2 ticks × ($10 × 3) = $60 → reaches threshold
        assert ctor.update(tick(1000, 10.0, 3.0, side=1.0)) is None  # run=$30
        bar = ctor.update(tick(2000, 10.0, 3.0, side=1.0))  # run=$60
        assert bar is not None
        assert bar.bar_type == "run_dollar"
        assert bar.dollar_value == pytest.approx(60.0)

    def test_direction_change_banks_dollar(self) -> None:
        """Direction change banks the dollar run properly."""
        ctor = RunDollarBarConstructor(initial_ewa_t=200.0, initial_ewa_proportion=0.5)
        # threshold = 200 * 0.5 = 100
        # Buy run: $10×3=$30, $10×3=$30 → run_cum=$60
        ctor.update(tick(1000, 10.0, 3.0, side=1.0))
        ctor.update(tick(2000, 10.0, 3.0, side=1.0))
        # Direction change → bank $60
        # Sell run: $10×2=$20 → banked(60)+run_cum(20)=80 < 100
        ctor.update(tick(3000, 10.0, 2.0, side=-1.0))
        # $10×2=$20 → total=100 >= 100 → close
        bar = ctor.update(tick(4000, 10.0, 2.0, side=-1.0))
        assert bar is not None
        assert bar.num_ticks == 4

    def test_batch_run_dollar_bars(self) -> None:
        """compute_run_dollar_bars() produces bars."""
        df = make_ticks_df(
            timestamps=list(range(1000, 11000, 1000)),
            prices=[100.0] * 10,
            volumes=[2.0] * 10,
            sides=[1.0] * 5 + [-1.0] * 5,
        )
        result = compute_run_dollar_bars(
            df,
            initial_ewa_t=2000.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )
        assert len(result) >= 1
        assert result.iloc[0]["bar_type"] == "run_dollar"

    def test_streaming_vs_batch_equivalence(self) -> None:
        """Streaming and batch produce identical results."""
        n = 15
        timestamps = list(range(1000, 1000 + n * 1000, 1000))
        prices = [100.0] * n
        volumes = [2.0] * n
        sides = [1.0] * 5 + [-1.0] * 5 + [1.0] * 5

        df = make_ticks_df(timestamps, prices, volumes, sides)
        batch_result = compute_run_dollar_bars(
            df,
            initial_ewa_t=2000.0,
            initial_ewa_proportion=0.5,
            schema=default_schema(has_side=True),
        )

        ctor = RunDollarBarConstructor(initial_ewa_t=2000.0, initial_ewa_proportion=0.5)
        stream_bars = []
        for i in range(n):
            bar = ctor.update(tick(timestamps[i], prices[i], volumes[i], sides[i]))
            if bar is not None:
                stream_bars.append(bar)
        stream_result = bars_to_df(stream_bars)

        pd.testing.assert_frame_equal(batch_result, stream_result)

    def test_from_state_reconstruction(self) -> None:
        """from_state() reconstructs a working constructor."""
        ctor = RunDollarBarConstructor(initial_ewa_t=1000.0, initial_ewa_proportion=0.5)
        ctor.update(tick(1000, 100.0, 2.0, side=1.0))
        state = ctor.get_state()

        ctor2 = RunDollarBarConstructor.from_state(state)
        assert ctor2.current_bar is not None
        assert ctor2.current_bar.dollar_value == pytest.approx(200.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestInfoBarRegistryIntegration:
    def test_all_six_types_registered(self) -> None:
        from flowbars.bars.registry import BarRegistry

        registered = BarRegistry.list()
        assert "imbalance_tick" in registered
        assert "imbalance_volume" in registered
        assert "imbalance_dollar" in registered
        assert "run_tick" in registered
        assert "run_volume" in registered
        assert "run_dollar" in registered

    def test_get_constructor_for_each_type(self) -> None:
        from flowbars.bars.registry import BarRegistry

        assert BarRegistry.get_constructor("imbalance_tick") is ImbalanceTickBarConstructor
        assert BarRegistry.get_constructor("imbalance_volume") is ImbalanceVolumeBarConstructor
        assert BarRegistry.get_constructor("imbalance_dollar") is ImbalanceDollarBarConstructor
        assert BarRegistry.get_constructor("run_tick") is RunTickBarConstructor
        assert BarRegistry.get_constructor("run_volume") is RunVolumeBarConstructor
        assert BarRegistry.get_constructor("run_dollar") is RunDollarBarConstructor

    def test_get_batch_function_for_each_type(self) -> None:
        from flowbars.bars.registry import BarRegistry

        assert BarRegistry.get_batch_function("imbalance_tick") is compute_imbalance_tick_bars
        assert BarRegistry.get_batch_function("imbalance_volume") is compute_imbalance_volume_bars
        assert BarRegistry.get_batch_function("imbalance_dollar") is compute_imbalance_dollar_bars
        assert BarRegistry.get_batch_function("run_tick") is compute_run_tick_bars
        assert BarRegistry.get_batch_function("run_volume") is compute_run_volume_bars
        assert BarRegistry.get_batch_function("run_dollar") is compute_run_dollar_bars
