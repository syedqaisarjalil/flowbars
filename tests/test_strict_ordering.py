"""Tests for the ``strict_ordering`` input-validation feature.

``strict_ordering=True`` raises :class:`TickDataError` when a tick arrives
with a timestamp earlier than the previous tick's.  Default ``False``
processes ticks in the order given (caller-beware).
"""

from __future__ import annotations

import pandas as pd
import pytest

from flowbars import SchemaMapping, TickBarConstructor
from flowbars.core import TickDataError, TickInfo

schema = SchemaMapping({"timestamp": "ts", "price": "px", "volume": "vol"})


def _feed(ctor: TickBarConstructor, tss: list[int]) -> None:
    for ts in tss:
        ctor.update(TickInfo(ts, float(ts), 1.0))


class TestStrictOrderingUpdate:
    def test_update_raises_on_out_of_order(self) -> None:
        ctor = TickBarConstructor(threshold=3, schema=schema, watermark=None, strict_ordering=True)
        _feed(ctor, [1000, 2000, 3000])
        with pytest.raises(TickDataError, match="Out-of-order"):
            ctor.update(TickInfo(2000, 2.0, 1.0))

    def test_update_default_does_not_raise(self) -> None:
        ctor = TickBarConstructor(
            threshold=100, schema=schema, watermark=None, strict_ordering=False
        )
        _feed(ctor, [1000, 2000, 3000])
        # out-of-order tick is silently accepted (no dedup, no ordering check)
        assert ctor.update(TickInfo(2000, 2.0, 1.0)) is None


class TestStrictOrderingBatch:
    def test_batch_raises_on_out_of_order(self) -> None:
        df = pd.DataFrame(
            {
                "ts": [3000, 2000, 4000],
                "px": [100.0, 101.0, 102.0],
                "vol": [1.0, 1.0, 1.0],
            }
        )
        ctor = TickBarConstructor(threshold=2, schema=schema, strict_ordering=True)
        with pytest.raises(TickDataError, match="Out-of-order"):
            ctor.batch(df)

    def test_batch_default_does_not_raise(self) -> None:
        df = pd.DataFrame(
            {
                "ts": [3000, 2000, 4000],
                "px": [100.0, 101.0, 102.0],
                "vol": [1.0, 1.0, 1.0],
            }
        )
        ctor = TickBarConstructor(threshold=2, schema=schema)
        bars = ctor.batch(df)
        assert len(bars) == 1

    def test_batch_watermark_seq_catches_out_of_order_timestamp(self) -> None:
        """strict_ordering catches a timestamp that is out of order even when
        the dedup watermark (a monotonic seq column) would not."""
        df = pd.DataFrame(
            {
                "ts": [3000, 2000, 4000],
                "px": [100.0, 101.0, 102.0],
                "vol": [1.0, 1.0, 1.0],
                "seq": [1, 2, 3],
            }
        )
        ctor = TickBarConstructor(threshold=2, schema=schema, watermark="seq", strict_ordering=True)
        with pytest.raises(TickDataError, match="Out-of-order"):
            ctor.batch(df)


class TestStrictOrderingState:
    def test_state_round_trip_preserves_last_timestamp(self) -> None:
        ctor = TickBarConstructor(threshold=3, schema=schema, stream_id="s")
        _feed(ctor, [1000, 2000, 3000])
        state = ctor.get_state()
        assert state["last_timestamp"] == 3000

        ctor2 = TickBarConstructor(threshold=3, schema=schema, stream_id="s")
        ctor2.load_state(state)
        assert ctor2.get_state()["last_timestamp"] == 3000

    def test_resume_continues_ordering_check(self) -> None:
        """After resume, the ordering check compares against the pre-resume
        last timestamp, not just the post-resume window."""
        ctor = TickBarConstructor(
            threshold=3, schema=schema, stream_id="r", watermark=None, strict_ordering=True
        )
        _feed(ctor, [1000, 2000, 3000])
        state = ctor.get_state()

        ctor2 = TickBarConstructor(
            threshold=3, schema=schema, stream_id="r", watermark=None, strict_ordering=True
        )
        ctor2.load_state(state)
        with pytest.raises(TickDataError, match="Out-of-order"):
            ctor2.update(TickInfo(2000, 2.0, 1.0))
