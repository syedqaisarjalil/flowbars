"""Tests for bar accumulators — Phase 2."""

from __future__ import annotations

import numpy as np
import pytest

from flowbars.bars.accumulators import (
    DollarAccumulator,
    ImbalanceAccumulator,
    RunAccumulator,
    TickAccumulator,
    TimeAccumulator,
    VolumeAccumulator,
)
from flowbars.core import TickInfo

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def tick(ts: int, price: float, volume: float, side: float | None = None) -> TickInfo:
    """Shorthand to create a TickInfo with optional side."""
    return TickInfo(timestamp=ts, price=price, volume=volume, side=side)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared OHLCV behaviour (exercised via TickAccumulator, the simplest variant)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBaseAccumulator:
    """Shared OHLCV tracking, partial bar, and state persistence."""

    def test_empty_accumulator_has_no_current_bar(self) -> None:
        acc = TickAccumulator()
        assert acc.current_bar is None

    def test_single_tick(self) -> None:
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 50.0, 10.0))

        bar = acc.current_bar
        assert bar is not None
        assert bar.open == 50.0
        assert bar.high == 50.0
        assert bar.low == 50.0
        assert bar.close == 50.0
        assert bar.volume == 10.0
        assert bar.dollar_value == 500.0
        assert bar.num_ticks == 1
        assert bar.open_ts == 1000
        assert bar.close_ts == 1000

    def test_multiple_ticks_ohlcv(self) -> None:
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 100.0, 1.0))
        acc.add_tick(tick(2000, 102.0, 2.0))
        acc.add_tick(tick(3000, 99.0, 3.0))
        acc.add_tick(tick(4000, 101.0, 1.0))

        bar = acc.current_bar
        assert bar is not None
        assert bar.open == 100.0
        assert bar.high == 102.0
        assert bar.low == 99.0
        assert bar.close == 101.0
        assert bar.volume == 7.0
        assert bar.dollar_value == 100.0 * 1.0 + 102.0 * 2.0 + 99.0 * 3.0 + 101.0 * 1.0
        assert bar.num_ticks == 4
        assert bar.open_ts == 1000
        assert bar.close_ts == 4000

    def test_vwap_zero_volume(self) -> None:
        """VWAP with zero total volume should be 0.0."""
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 100.0, 0.0))
        acc.add_tick(tick(2000, 101.0, 0.0))
        bar = acc.current_bar
        assert bar is not None
        assert bar.vwap == 0.0

    def test_vwap_computation(self) -> None:
        """VWAP = dollar_value / volume."""
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 10.0, 2.0))  # $20
        acc.add_tick(tick(2000, 20.0, 3.0))  # $60
        bar = acc.current_bar
        assert bar is not None
        # vwap = 80/5 = 16.0
        assert bar.vwap == pytest.approx(16.0)

    def test_high_low_tracking(self) -> None:
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 100.0, 1.0))
        assert acc.current_bar.high == 100.0  # type: ignore[union-attr]
        assert acc.current_bar.low == 100.0  # type: ignore[union-attr]

        acc.add_tick(tick(2000, 110.0, 1.0))
        assert acc.current_bar.high == 110.0  # type: ignore[union-attr]
        assert acc.current_bar.low == 100.0  # type: ignore[union-attr]

        acc.add_tick(tick(3000, 95.0, 1.0))
        assert acc.current_bar.high == 110.0  # type: ignore[union-attr]
        assert acc.current_bar.low == 95.0  # type: ignore[union-attr]

    def test_bar_type_label(self) -> None:
        acc = TickAccumulator(bar_type="tick")
        acc.add_tick(tick(1000, 100.0, 1.0))
        assert acc.current_bar is not None
        assert acc.current_bar.bar_type == "tick"

    def test_current_bar_does_not_mutate_state(self) -> None:
        """Reading current_bar multiple times returns identical results."""
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 100.0, 5.0))
        acc.add_tick(tick(2000, 102.0, 3.0))

        bar1 = acc.current_bar
        bar2 = acc.current_bar
        assert bar1 == bar2  # dataclass equality
        assert bar1 is not bar2  # but they're different objects (safety)

    def test_bar_id_increments_on_close(self) -> None:
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 100.0, 1.0))
        assert acc.current_bar.bar_id == 0  # type: ignore[union-attr]

        # threshold=1 → close on first tick
        assert acc.should_close(1.0)
        bar = acc.close(1.0)
        assert bar.bar_id == 0

        acc.add_tick(tick(2000, 101.0, 1.0))
        assert acc.current_bar.bar_id == 1  # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════════
# TickAccumulator
# ═══════════════════════════════════════════════════════════════════════════════


class TestTickAccumulator:
    def test_should_close_at_exact_threshold(self) -> None:
        acc = TickAccumulator()
        for i in range(5):
            acc.add_tick(tick(1000 + i, 100.0, 1.0))
        assert acc.should_close(5.0)
        assert not acc.should_close(6.0)

    def test_should_close_above_threshold(self) -> None:
        acc = TickAccumulator()
        for i in range(5):
            acc.add_tick(tick(1000 + i, 100.0, 1.0))
        assert acc.should_close(4.0)
        assert acc.should_close(3.0)

    def test_single_tick_exceeds_threshold(self) -> None:
        """One tick, threshold=1 — closes immediately."""
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 100.0, 10.0))
        assert acc.should_close(1.0)
        bar = acc.close(1.0)
        assert bar.num_ticks == 1
        assert bar.open == 100.0

    def test_bar_close_then_next_tick(self) -> None:
        """Close a bar, then the next tick starts a new bar."""
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 10.0, 1.0))
        acc.add_tick(tick(2000, 12.0, 2.0))
        assert acc.should_close(2.0)

        bar0 = acc.close(2.0)
        assert bar0.num_ticks == 2
        assert bar0.bar_id == 0

        acc.add_tick(tick(3000, 14.0, 1.0))
        assert acc.current_bar is not None
        assert acc.current_bar.bar_id == 1
        assert acc.current_bar.open == 14.0

    def test_empty_should_not_close(self) -> None:
        acc = TickAccumulator()
        assert not acc.should_close(0.0)
        assert not acc.should_close(1.0)

    def test_close_raises_on_empty(self) -> None:
        """Closing an empty accumulator should assert-fail."""
        acc = TickAccumulator()
        with pytest.raises(AssertionError):
            acc.close(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# VolumeAccumulator
# ═══════════════════════════════════════════════════════════════════════════════


class TestVolumeAccumulator:
    def test_closes_when_volume_meets_threshold(self) -> None:
        acc = VolumeAccumulator()
        acc.add_tick(tick(1000, 100.0, 300.0))
        acc.add_tick(tick(2000, 101.0, 400.0))
        assert not acc.should_close(1000.0)  # 700 < 1000
        acc.add_tick(tick(3000, 102.0, 500.0))
        assert acc.should_close(1000.0)  # 1200 >= 1000

    def test_overflow_excess_volume_rolls_into_next_bar(self) -> None:
        """Hand-computed overflow: threshold=1000, ticks sum to 1200 → excess 200."""
        acc = VolumeAccumulator()
        acc.add_tick(tick(1000, 100.0, 300.0))
        acc.add_tick(tick(2000, 101.0, 400.0))
        acc.add_tick(tick(3000, 102.0, 500.0))  # total = 1200 >= 1000

        bar0 = acc.close(1000.0)
        assert bar0.volume == 1200.0
        assert bar0.num_ticks == 3

        # Next tick starts with 200 overflow
        acc.add_tick(tick(4000, 103.0, 100.0))  # cum = 200 + 100 = 300
        assert not acc.should_close(1000.0)
        acc.add_tick(tick(5000, 104.0, 600.0))  # cum = 300 + 600 = 900
        assert not acc.should_close(1000.0)
        acc.add_tick(tick(6000, 105.0, 200.0))  # cum = 900 + 200 = 1100
        assert acc.should_close(1000.0)

        bar1 = acc.close(1000.0)
        assert bar1.volume == pytest.approx(100.0 + 600.0 + 200.0)
        assert bar1.bar_id == 1

        # Excess from bar1: 1100 - 1000 = 100 rolls into bar2
        acc.add_tick(tick(7000, 106.0, 50.0))
        assert acc.current_bar is not None
        # cumulative volume should include overflow
        assert acc.should_close(155.0)  # 100 overflow + 50 = 150 >= 155? No
        assert not acc.should_close(151.0)  # 150 < 151

    def test_exact_threshold_no_overflow(self) -> None:
        """When cumulative volume equals threshold exactly, no overflow."""
        acc = VolumeAccumulator()
        acc.add_tick(tick(1000, 100.0, 500.0))
        acc.add_tick(tick(2000, 101.0, 500.0))  # exactly 1000

        bar = acc.close(1000.0)
        assert bar.volume == 1000.0

        # Next tick starts from 0
        acc.add_tick(tick(3000, 102.0, 100.0))
        assert acc.should_close(100.0)
        assert not acc.should_close(101.0)


# ═══════════════════════════════════════════════════════════════════════════════
# DollarAccumulator
# ═══════════════════════════════════════════════════════════════════════════════


class TestDollarAccumulator:
    def test_closes_when_dollar_value_meets_threshold(self) -> None:
        acc = DollarAccumulator()
        acc.add_tick(tick(1000, 100.0, 2.0))  # $200
        acc.add_tick(tick(2000, 100.0, 3.0))  # $300 → total $500
        assert not acc.should_close(600.0)
        acc.add_tick(tick(3000, 100.0, 2.0))  # $200 → total $700
        assert acc.should_close(600.0)

    def test_overflow_rolls_into_next_bar(self) -> None:
        """Spec example: threshold=$1,000,000, ticks: $300k, $400k, $500k."""
        acc = DollarAccumulator()
        acc.add_tick(tick(1000, 100.0, 3000.0))  # $300,000
        acc.add_tick(tick(2000, 100.0, 4000.0))  # $400,000 → $700,000
        acc.add_tick(tick(3000, 100.0, 5000.0))  # $500,000 → $1,200,000

        bar0 = acc.close(1_000_000.0)
        assert bar0.dollar_value == 1_200_000.0
        assert bar0.num_ticks == 3

        # Overflow: $200,000
        acc.add_tick(tick(4000, 100.0, 1000.0))  # $200,000 + $100,000 = $300,000
        assert not acc.should_close(1_000_000.0)
        bar_partial = acc.current_bar
        assert bar_partial is not None
        assert bar_partial.dollar_value == pytest.approx(100_000.0)

    def test_dollar_value_equals_price_times_volume(self) -> None:
        acc = DollarAccumulator()
        acc.add_tick(tick(1000, 50.0, 10.0))
        bar = acc.current_bar
        assert bar is not None
        assert bar.dollar_value == 500.0


# ═══════════════════════════════════════════════════════════════════════════════
# TimeAccumulator
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimeAccumulator:
    def test_clock_anchor_first_boundary(self) -> None:
        """A tick at 14:03:17 (ms past midnight = 50597000) with 5-min bars."""
        interval = 300_000  # 5 minutes
        # 50597000 ms = 14:03:17.000
        # Next 5-min boundary = 14:05:00.000 = 50700000
        acc = TimeAccumulator(interval_ms=interval, anchor="clock")
        acc.add_tick(tick(50_597_000, 100.0, 1.0))
        assert not acc.should_close(0.0)  # threshold ignored

    def test_clock_anchor_boundary_crossing(self) -> None:
        """Tick at 14:03, then tick at 14:06 crosses 14:05 boundary."""
        interval = 300_000  # 5 min
        acc = TimeAccumulator(interval_ms=interval, anchor="clock")
        acc.add_tick(tick(50_580_000, 100.0, 1.0))  # 14:03:00
        assert not acc.should_close(0.0)
        acc.add_tick(tick(50_700_000, 101.0, 1.0))  # 14:05:00 exactly
        assert acc.should_close(0.0)  # crossed boundary

    def test_clock_anchor_exact_boundary(self) -> None:
        """Tick exactly on a boundary closes the bar."""
        interval = 300_000
        acc = TimeAccumulator(interval_ms=interval, anchor="clock")
        acc.add_tick(tick(50_700_000, 100.0, 1.0))  # 14:05:00
        # First boundary = 14:05:00 + interval? No, next boundary after first tick.
        # first_ts = 50700000, remainder = 50700000 % 300000 = 0
        # So first_boundary = first_ts + interval = 51000000 (14:10:00)
        # So this tick doesn't cross yet
        assert not acc.should_close(0.0)
        acc.add_tick(tick(51_000_001, 101.0, 1.0))  # past 14:10:00
        assert acc.should_close(0.0)

    def test_clock_anchor_multi_bar_sequence(self) -> None:
        """Produce multiple time bars and verify timestamps."""
        interval = 60_000  # 1 minute
        acc = TimeAccumulator(interval_ms=interval, anchor="clock")

        # Tick at 10:00:30 (ts = 36030000 ms past midnight)
        acc.add_tick(tick(36_030_000, 10.0, 1.0))
        assert not acc.should_close(0.0)

        # Tick at 10:01:00 → crosses 10:01:00 boundary
        acc.add_tick(tick(36_060_000, 11.0, 1.0))
        assert acc.should_close(0.0)
        bar0 = acc.close(0.0)
        assert bar0.num_ticks == 2

        # Next bar: tick at 10:01:15
        acc.add_tick(tick(36_075_000, 12.0, 1.0))
        assert not acc.should_close(0.0)

        # Tick at 10:02:10 → crosses 10:02:00 boundary
        acc.add_tick(tick(36_130_000, 13.0, 1.0))
        assert acc.should_close(0.0)
        bar1 = acc.close(0.0)
        assert bar1.num_ticks == 2
        assert bar1.bar_id == 1

    def test_first_tick_anchor(self) -> None:
        """Anchor='first_tick' — bars measured from the first tick."""
        interval = 60_000  # 1 minute
        acc = TimeAccumulator(interval_ms=interval, anchor="first_tick")

        # First tick at arbitrary time
        first_ts = 36_030_000  # 10:00:30
        acc.add_tick(tick(first_ts, 10.0, 1.0))
        assert not acc.should_close(0.0)

        # Tick within the first bar
        acc.add_tick(tick(first_ts + 30_000, 11.0, 1.0))
        assert not acc.should_close(0.0)

        # Tick past the first bar boundary (60s after first tick)
        acc.add_tick(tick(first_ts + 60_001, 12.0, 1.0))
        assert acc.should_close(0.0)

        bar = acc.close(0.0)
        assert bar.num_ticks == 3
        assert bar.open_ts == first_ts

    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeAccumulator(interval_ms=-1000)

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeAccumulator(interval_ms=0)

    def test_invalid_anchor_raises(self) -> None:
        with pytest.raises(ValueError):
            TimeAccumulator(anchor="midnight")

    def test_empty_time_bar(self) -> None:
        """A time interval with no ticks — next tick after gap starts new bar."""
        interval = 60_000
        acc = TimeAccumulator(interval_ms=interval, anchor="clock")
        acc.add_tick(tick(36_000_000, 10.0, 1.0))  # 10:00:00
        # Next boundary = 10:01:00
        # Tick arrives at 10:05:00 (skips 4 boundaries)
        acc.add_tick(tick(36_300_000, 20.0, 1.0))  # 10:05:00
        assert acc.should_close(0.0)
        bar = acc.close(0.0)
        assert bar.open == 10.0
        assert bar.close == 20.0


# ═══════════════════════════════════════════════════════════════════════════════
# ImbalanceAccumulator
# ═══════════════════════════════════════════════════════════════════════════════


class TestImbalanceAccumulator:
    # ── tick metric ─────────────────────────────────────────────────

    def test_tick_imbalance_basic(self) -> None:
        """Pure buy stream: imbalance = +num_ticks."""
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))
        acc.add_tick(tick(3000, 102.0, 1.0, side=1.0))

        assert acc.should_close(3.0)
        assert not acc.should_close(4.0)

    def test_tick_imbalance_mixed(self) -> None:
        """Buy, buy, sell → imbalance = +1."""
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))
        acc.add_tick(tick(3000, 99.0, 1.0, side=-1.0))

        assert acc.should_close(1.0)
        assert not acc.should_close(2.0)

    def test_first_tick_nan_side_excluded(self) -> None:
        """First tick with NaN side: imbalance contribution = 0."""
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=np.nan))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))
        acc.add_tick(tick(3000, 102.0, 1.0, side=1.0))

        # tick 1 excluded, ticks 2-3 = +2 imbalance
        assert acc.should_close(2.0)
        assert not acc.should_close(3.0)

    def test_first_tick_none_side_excluded(self) -> None:
        """First tick with None side → same as NaN."""
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=None))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))
        acc.add_tick(tick(3000, 102.0, 1.0, side=1.0))

        assert acc.should_close(2.0)

    def test_first_tick_ohlcv_still_tracked(self) -> None:
        """First tick with NaN side is still in OHLCV, just not in imbalance."""
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 5.0, side=np.nan))

        bar = acc.current_bar
        assert bar is not None
        assert bar.open == 100.0
        assert bar.volume == 5.0
        assert bar.num_ticks == 1

    # ── volume metric ───────────────────────────────────────────────

    def test_volume_imbalance(self) -> None:
        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        acc.add_tick(tick(1000, 100.0, 10.0, side=1.0))  # +10
        acc.add_tick(tick(2000, 101.0, 30.0, side=-1.0))  # -30 → total -20
        acc.add_tick(tick(3000, 102.0, 5.0, side=-1.0))  # -5 → total -25

        assert acc.should_close(25.0)

    # ── dollar metric ───────────────────────────────────────────────

    def test_dollar_imbalance(self) -> None:
        acc = ImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
        # $100 * 10 = $1000 buy
        acc.add_tick(tick(1000, 100.0, 10.0, side=1.0))
        # $200 * 2 = $400 sell
        acc.add_tick(tick(2000, 200.0, 2.0, side=-1.0))
        # net: +1000 - 400 = +600
        assert acc.should_close(600.0)
        assert not acc.should_close(601.0)

    # ── overflow ────────────────────────────────────────────────────

    def test_overflow_positive_imbalance(self) -> None:
        """Imbalance crosses threshold, excess carries with sign."""
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        for _ in range(7):
            acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))  # +7

        bar0 = acc.close(5.0)
        assert bar0.num_ticks == 7
        # excess = 7 - 5 = +2

        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))  # +2 + 1 = +3
        assert acc.should_close(3.0)
        assert not acc.should_close(4.0)

    def test_overflow_negative_imbalance(self) -> None:
        """Negative imbalance crosses threshold, excess carries sign."""
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        for _ in range(7):
            acc.add_tick(tick(1000, 100.0, 1.0, side=-1.0))  # -7

        bar0 = acc.close(5.0)
        assert bar0.num_ticks == 7
        # excess = -7 - (-5)?  Let's compute: abs(-7) = 7, excess = 7 - 5 = 2
        # sign = -1, so overflow = -2

        acc.add_tick(tick(2000, 101.0, 1.0, side=-1.0))  # -2 + (-1) = -3
        assert acc.should_close(3.0)
        assert not acc.should_close(4.0)

    def test_overflow_exact_boundary(self) -> None:
        """Exactly at threshold: excess = 0."""
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        for _ in range(5):
            acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))

        bar = acc.close(5.0)
        assert bar.num_ticks == 5

        # Next tick starts with 0
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))
        assert acc.should_close(1.0)

    # ── error ───────────────────────────────────────────────────────

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError):
            ImbalanceAccumulator(metric="shares")


# ═══════════════════════════════════════════════════════════════════════════════
# RunAccumulator
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunAccumulator:
    # ── basic ───────────────────────────────────────────────────────

    def test_single_run_ticks(self) -> None:
        """All same direction — one continuous run."""
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))
        acc.add_tick(tick(3000, 102.0, 1.0, side=1.0))

        assert acc.should_close(3.0)
        assert not acc.should_close(4.0)

    def test_direction_change_banks_run(self) -> None:
        """Buy, buy, sell, sell — two runs, total = 4."""
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))  # run 1: 2 ticks
        acc.add_tick(tick(3000, 99.0, 1.0, side=-1.0))  # direction change
        acc.add_tick(tick(4000, 98.0, 1.0, side=-1.0))  # run 2: 2 ticks

        # total = 2 (banked) + 2 (current) = 4
        assert acc.should_close(4.0)
        assert not acc.should_close(5.0)

    def test_alternating_tiny_runs(self) -> None:
        """Every tick alternates direction — all runs of length 1."""
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=-1.0))
        acc.add_tick(tick(3000, 99.0, 1.0, side=1.0))
        acc.add_tick(tick(4000, 100.0, 1.0, side=-1.0))
        acc.add_tick(tick(5000, 101.0, 1.0, side=1.0))

        # Run 1: 1 tick (banked at tick 2)
        # Run 2: 1 tick (banked at tick 3)
        # Run 3: 1 tick (banked at tick 4)
        # Run 4: 1 tick (banked at tick 5)
        # Current run 5: 1 tick
        # Total = 4 + 1 = 5
        assert acc.should_close(5.0)
        assert not acc.should_close(6.0)

    # ── first-tick special case ─────────────────────────────────────

    def test_first_tick_nan_side_retroactively_included(self) -> None:
        """Tick 1 NaN + tick 2 +1: tick 1 is retroactively included in tick 2's run.

        Per spec: the first run's direction is determined by tick 2's sign,
        and tick 1 is included in that run (NaN matches anything via _same_direction).
        """
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=np.nan))  # run sign = NaN
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))  # same dir (NaN matches +1)

        # Both ticks in same run, dir retroactively set to +1
        assert acc.should_close(2.0)
        assert not acc.should_close(3.0)

    def test_first_tick_nan_then_sell_retroactively_assigned(self) -> None:
        """NaN first tick + sell tick 2: run dir becomes -1, tick 1 included."""
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=np.nan))
        acc.add_tick(tick(2000, 99.0, 1.0, side=-1.0))

        # Same run, dir = -1, total = 2
        assert acc.should_close(2.0)

    def test_first_tick_none_side(self) -> None:
        """First tick has None side — same as NaN."""
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=None))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))

        # side=None becomes NaN → same as above
        assert acc.should_close(2.0)

    # ── volume metric ───────────────────────────────────────────────

    def test_volume_run(self) -> None:
        acc = RunAccumulator(bar_type="run_volume", metric="volume")
        acc.add_tick(tick(1000, 100.0, 10.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 20.0, side=1.0))  # run 1: 30
        acc.add_tick(tick(3000, 102.0, 5.0, side=-1.0))  # bank 30, new run: 5

        # total = 30 + 5 = 35
        assert acc.should_close(35.0)

    # ── dollar metric ───────────────────────────────────────────────

    def test_dollar_run(self) -> None:
        acc = RunAccumulator(bar_type="run_dollar", metric="dollar")
        acc.add_tick(tick(1000, 10.0, 100.0, side=1.0))  # $1000
        acc.add_tick(tick(2000, 20.0, 50.0, side=-1.0))  # bank $1000, new run: $1000

        # total = 1000 + 1000 = 2000
        assert acc.should_close(2000.0)

    # ── overflow ────────────────────────────────────────────────────

    def test_overflow_single_run(self) -> None:
        """Continuous buy run crosses threshold, excess continues."""
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        for _ in range(7):
            acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))

        bar0 = acc.close(5.0)
        assert bar0.num_ticks == 7
        # excess = 7 - 5 = 2 carried in current run

        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))  # run continues: 2+1=3
        assert acc.should_close(3.0)

    def test_overflow_multi_run(self) -> None:
        """Overflow with direction changes."""
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))  # run 1: 2
        acc.add_tick(tick(3000, 99.0, 1.0, side=-1.0))  # bank 2, new run: 1
        acc.add_tick(tick(4000, 98.0, 1.0, side=-1.0))  # run 2: 2
        acc.add_tick(tick(5000, 100.0, 1.0, side=1.0))  # bank 2, new run: 1
        # total = 2 + 2 + 1 = 5

        bar0 = acc.close(4.0)
        assert bar0.num_ticks == 5
        # excess = 5 - 4 = 1

        acc.add_tick(tick(6000, 101.0, 1.0, side=1.0))  # run continues: 1+1=2
        assert acc.should_close(2.0)

    # ── directed run accumulation (run bars track runs, not signed imbalance) ──

    def test_run_accumulation_is_not_signed(self) -> None:
        """Run bars accumulate run sizes, not signed imbalance.

        Buy run of 3, sell run of 2 → total = 5 (not 1).
        """
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))
        acc.add_tick(tick(3000, 102.0, 1.0, side=1.0))  # run 1: 3
        acc.add_tick(tick(4000, 101.0, 1.0, side=-1.0))
        acc.add_tick(tick(5000, 100.0, 1.0, side=-1.0))  # run 2: 2

        # total = 3 + 2 = 5 (NOT |3-2| = 1)
        assert acc.should_close(5.0)
        assert not acc.should_close(1.0)  # if it were signed, this would close

    # ── error ───────────────────────────────────────────────────────

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError):
            RunAccumulator(metric="shares")


# ═══════════════════════════════════════════════════════════════════════════════
# State persistence (round-trip)
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    def test_tick_accumulator_state_round_trip(self) -> None:
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 10.0, 1.0))
        acc.add_tick(tick(2000, 12.0, 2.0))

        state = acc.get_state()
        assert state["bar_id"] == 0
        assert state["has_tick"] is True
        assert state["num_ticks"] == 2
        assert state["cum_ticks"] == 2.0

        # Reconstruct
        acc2 = TickAccumulator()
        acc2.load_state(state)

        bar = acc2.current_bar
        assert bar is not None
        assert bar.open == 10.0
        assert bar.high == 12.0
        assert bar.num_ticks == 2
        assert acc2.should_close(2.0)

    def test_volume_accumulator_state_round_trip(self) -> None:
        acc = VolumeAccumulator()
        acc.add_tick(tick(1000, 100.0, 300.0))
        acc.add_tick(tick(2000, 101.0, 200.0))

        state = acc.get_state()

        acc2 = VolumeAccumulator()
        acc2.load_state(state)

        assert acc2.should_close(500.0)
        bar = acc2.current_bar
        assert bar is not None
        assert bar.volume == 500.0

    def test_state_after_close(self) -> None:
        """State after a close should reflect the next bar correctly."""
        acc = VolumeAccumulator()
        acc.add_tick(tick(1000, 100.0, 600.0))
        acc.add_tick(tick(2000, 101.0, 500.0))  # total 1100 >= 1000

        bar0 = acc.close(1000.0)
        assert bar0.bar_id == 0

        state = acc.get_state()
        assert state["bar_id"] == 1  # next bar
        assert state["has_tick"] is False  # no ticks in current bar
        assert state["cum_volume"] == 100.0  # overflow 1100-1000

        acc2 = VolumeAccumulator()
        acc2.load_state(state)

        acc2.add_tick(tick(3000, 102.0, 50.0))  # overflow 100 + 50 = 150
        assert not acc2.should_close(1000.0)
        assert acc2.current_bar is not None
        assert acc2.current_bar.bar_id == 1

    def test_imbalance_accumulator_state_round_trip(self) -> None:
        acc = ImbalanceAccumulator(bar_type="imbalance_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=-1.0))

        state = acc.get_state()
        assert state["signed_imbalance"] == 0.0  # +1 -1 = 0

        acc2 = ImbalanceAccumulator()
        acc2.load_state(state)
        assert not acc2.should_close(1.0)

    def test_run_accumulator_state_round_trip(self) -> None:
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        acc.add_tick(tick(1000, 100.0, 1.0, side=1.0))
        acc.add_tick(tick(2000, 101.0, 1.0, side=1.0))
        acc.add_tick(tick(3000, 99.0, 1.0, side=-1.0))
        # run 1 banked (2 ticks), current run dir=-1, cum=1

        state = acc.get_state()
        acc2 = RunAccumulator()
        acc2.load_state(state)

        assert acc2.should_close(3.0)
        assert not acc2.should_close(4.0)

    def test_time_accumulator_state_round_trip(self) -> None:
        interval = 60_000
        acc = TimeAccumulator(interval_ms=interval, anchor="clock")
        acc.add_tick(tick(36_000_000, 10.0, 1.0))
        acc.add_tick(tick(36_030_000, 11.0, 1.0))

        state = acc.get_state()

        acc2 = TimeAccumulator()
        acc2.load_state(state)

        # Should behave identically after restore
        assert acc2._next_boundary_ms == acc._next_boundary_ms
        bar = acc2.current_bar
        assert bar is not None
        assert bar.num_ticks == 2

    def test_empty_state_round_trip(self) -> None:
        """Round-trip state before any ticks."""
        acc = TickAccumulator()
        state = acc.get_state()

        acc2 = TickAccumulator()
        acc2.load_state(state)

        assert acc2.current_bar is None
        assert not acc2.should_close(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_zero_volume_ticks_accepted(self) -> None:
        """Zero-volume ticks are accepted — they count for tick/run but not volume/dollar."""
        acc = VolumeAccumulator()
        acc.add_tick(tick(1000, 100.0, 0.0))
        acc.add_tick(tick(2000, 101.0, 0.0))
        assert not acc.should_close(1.0)
        assert acc.current_bar is not None
        assert acc.current_bar.num_ticks == 2
        assert acc.current_bar.volume == 0.0
        assert acc.current_bar.dollar_value == 0.0

    def test_high_precision_prices(self) -> None:
        """Float64 precision is maintained."""
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 12345.678901234567, 1.0))
        bar = acc.current_bar
        assert bar is not None
        assert bar.open == pytest.approx(12345.678901234567)

    def test_threshold_zero_closes_immediately(self) -> None:
        """Threshold 0: first tick closes the bar (for non-time accumulators)."""
        acc = TickAccumulator()
        acc.add_tick(tick(1000, 100.0, 1.0))
        assert acc.should_close(0.0)

    def test_dollar_accumulator_dtype(self) -> None:
        """Verify that dollar_value is float64 precision."""
        acc = DollarAccumulator()
        acc.add_tick(tick(1000, 100.0, 2.0))
        bar = acc.current_bar
        assert bar is not None
        assert isinstance(bar.dollar_value, float)

    def test_many_ticks_single_bar(self) -> None:
        """Large number of ticks in a single bar doesn't degrade."""
        acc = VolumeAccumulator()
        rng = np.random.default_rng(42)
        total_vol = 0.0
        for i in range(1000):
            vol = float(rng.uniform(1, 10))
            total_vol += vol
            acc.add_tick(tick(1000 + i, 100.0, vol))

        bar = acc.current_bar
        assert bar is not None
        assert bar.num_ticks == 1000
        assert bar.volume == pytest.approx(total_vol)
