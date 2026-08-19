"""Tests for the watermark / dedup-on-resume feature.

Covers idempotent resume: re-feeding already-processed ticks is silently
discarded, keyed by a configurable watermark (timestamp by default, any
column on request).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowbars import SchemaMapping, TickBarConstructor
from flowbars.bars.numba_backend import _NUMBA_AVAILABLE
from flowbars.core import StateValidationError, TickInfo

schema = SchemaMapping({"timestamp": "ts", "price": "px", "volume": "vol"})


def _feed(ctor: TickBarConstructor, tss: list[int]) -> list:
    """Feed ticks one at a time, return the list of completed bars."""
    out = []
    for ts in tss:
        bar = ctor.update(TickInfo(ts, float(ts), 1.0))
        if bar is not None:
            out.append(bar)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Resume + dedup
# ═══════════════════════════════════════════════════════════════════════════════


class TestResumeDedup:
    def test_resume_with_overlap_matches_uninterrupted(self) -> None:
        """Re-feeding an overlap after resume yields identical bars to an
        uninterrupted run (idempotent resume)."""
        tss = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10_000]

        uninterrupted = TickBarConstructor(threshold=3, schema=schema, stream_id="x")
        bars_a = _feed(uninterrupted, tss)

        interrupted = TickBarConstructor(threshold=3, schema=schema, stream_id="x")
        bars_before = _feed(interrupted, tss[:5])
        state = interrupted.get_state()

        resumed = TickBarConstructor.from_state(state, schema=schema)
        # overlap: re-feed the already-processed 5000 plus the remaining new ticks
        bars_after = _feed(resumed, tss[4:])

        bars_b = bars_before + bars_after
        assert [b.bar_id for b in bars_a] == [b.bar_id for b in bars_b]
        for a, b in zip(bars_a, bars_b):
            assert a.open_ts == b.open_ts
            assert a.close_ts == b.close_ts
            assert a.num_ticks == b.num_ticks
            assert a.close == b.close

    def test_duplicate_and_equal_watermark_dropped(self) -> None:
        """Ticks with watermark <= the saved watermark are dropped."""
        ctor = TickBarConstructor(threshold=3, schema=schema, stream_id="s")
        _feed(ctor, [1000, 2000, 3000])  # closes bar 0 exactly
        state = ctor.get_state()
        assert state["last_watermark"] == 3000

        ctor2 = TickBarConstructor.from_state(state, schema=schema)
        assert ctor2.update(TickInfo(1000, 1.0, 1.0)) is None  # stale
        assert ctor2.update(TickInfo(3000, 1.0, 1.0)) is None  # equal boundary
        assert ctor2.update(TickInfo(4000, 1.0, 1.0)) is None  # new (no close yet)
        assert ctor2.update(TickInfo(5000, 1.0, 1.0)) is None  # new (no close yet)
        bar = ctor2.update(TickInfo(6000, 1.0, 1.0))  # 3rd new tick
        assert bar is not None
        assert bar.num_ticks == 3
        assert bar.open_ts == 4000

    def test_fresh_constructor_does_not_dedup(self) -> None:
        """A fresh constructor has no saved watermark, so nothing is dropped."""
        ctor = TickBarConstructor(threshold=100, schema=schema, stream_id="f")
        assert ctor.get_state()["last_watermark"] is None
        bars = _feed(ctor, [1000, 2000, 3000])
        assert len(bars) == 0  # threshold not reached, but ticks were accepted
        assert ctor.get_state()["last_watermark"] == 3000

    def test_watermark_none_disables_dedup(self) -> None:
        """watermark=None restores legacy (no dedup) behavior."""
        ctor = TickBarConstructor(threshold=3, schema=schema, stream_id="n", watermark=None)
        _feed(ctor, [1000, 2000, 3000])
        state = ctor.get_state()
        assert state["watermark_key"] is None

        # Resume and re-feed the same ticks: without dedup they are counted again
        ctor2 = TickBarConstructor.from_state(state, schema=schema)
        # two of the three duplicate ticks now close a bar again (double-counted)
        assert ctor2.update(TickInfo(1000, 1.0, 1.0)) is None
        assert ctor2.update(TickInfo(2000, 1.0, 1.0)) is None
        bar = ctor2.update(TickInfo(3000, 1.0, 1.0))  # 3rd tick -> closes
        assert bar is not None
        assert bar.num_ticks == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Custom watermark column (batch)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCustomWatermark:
    def _seq_df(self, n: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts": [1000] * n,  # same timestamp — exercises the seq-key case
                "px": [100.0 + i for i in range(n)],
                "vol": [1.0] * n,
                "seq": list(range(1, n + 1)),
            }
        )

    def test_batch_custom_seq_watermark_dedup(self) -> None:
        """Dedup on a custom 'seq' column, even with identical timestamps."""
        df = self._seq_df(6)
        ctor = TickBarConstructor(threshold=2, schema=schema, watermark="seq")
        bars1 = ctor.batch(df.iloc[:4])  # seq 1-4 -> 2 bars
        assert len(bars1) == 2
        assert ctor.get_state()["last_watermark"] == 4

        ctor2 = TickBarConstructor.from_state(ctor.get_state(), schema=schema)
        bars2 = ctor2.batch(df.iloc[2:])  # seq 3,4 (dup) + 5,6 (new) -> 1 bar
        assert len(bars2) == 1
        assert bars2.iloc[0]["num_ticks"] == 2
        assert bars2.iloc[0]["open_ts"] == 1000  # ts unchanged
        assert bars2.iloc[0]["close_ts"] == 1000

    def test_missing_custom_column_raises(self) -> None:
        """A watermark column that does not exist raises SchemaError."""
        df = pd.DataFrame({"ts": [1000, 2000, 3000], "px": [100.0, 101.0, 102.0], "vol": [1.0] * 3})
        ctor = TickBarConstructor(threshold=2, schema=schema, watermark="seq")
        with pytest.raises(Exception, match="Watermark column"):
            ctor.batch(df)


# ═══════════════════════════════════════════════════════════════════════════════
# State persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    def test_state_round_trip_preserves_watermark(self) -> None:
        ctor = TickBarConstructor(threshold=3, schema=schema, stream_id="w")
        _feed(ctor, [1000, 2000, 3000, 4000])
        state = ctor.get_state()
        assert state["watermark_key"] == "timestamp"
        assert state["last_watermark"] == 4000

        ctor2 = TickBarConstructor(threshold=3, schema=schema, stream_id="w")
        ctor2.load_state(state)
        assert ctor2.get_state()["last_watermark"] == 4000

    def test_watermark_key_mismatch_raises(self) -> None:
        ctor = TickBarConstructor(threshold=3, schema=schema, stream_id="m")
        _feed(ctor, [1000, 2000, 3000])
        state = ctor.get_state()

        ctor2 = TickBarConstructor(threshold=3, schema=schema, stream_id="m", watermark="seq")
        with pytest.raises(StateValidationError, match="watermark_key mismatch"):
            ctor2.load_state(state)


# ═══════════════════════════════════════════════════════════════════════════════
# numba backend equivalence
# ═══════════════════════════════════════════════════════════════════════════════


class TestNumbaWatermark:
    def test_numba_batch_applies_watermark(self) -> None:
        """The numba path honors the saved watermark just like the Python path."""
        if not _NUMBA_AVAILABLE:
            pytest.skip("numba not installed")

        from flowbars import BaseBarConstructor
        from flowbars.bars.accumulators import TickAccumulator
        from flowbars.thresholds import StaticThresholdEstimator

        def make(backend: str) -> BaseBarConstructor:
            return BaseBarConstructor(
                TickAccumulator("tick"),
                StaticThresholdEstimator(5.0),
                schema=schema,
                stream_id="nb",
                backend=backend,
            )

        df = pd.DataFrame(
            {
                "ts": np.arange(0, 2000, 100, dtype=np.int64),
                "px": 100.0 + np.arange(0, 2000, 100) / 100.0,
                "vol": np.ones(20),
            }
        )
        ctor = make("numba")
        ctor.batch(df.iloc[:10])
        state = ctor.get_state()
        assert state["last_watermark"] == 900

        ctor2 = BaseBarConstructor.from_state(
            state, TickAccumulator("tick"), StaticThresholdEstimator(5.0), schema=schema
        )
        bars = ctor2.batch(df.iloc[5:])  # ts 500-900 dup + 1000-1900 new
        # only the new ticks (ts 1000..1900) should be processed
        assert len(bars) == 2  # 10 new ticks / 5 per bar
