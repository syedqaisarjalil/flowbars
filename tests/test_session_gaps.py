"""Session-gap handling for feeds that never print during the gap.

A session-aware calendar must split a bar at the session close even when the
feed contains only in-session inputs.  The engine detects the jump on the
first input of the later session and force-closes the open bar at the prior
session's close, so no bar silently spans a weekend or holiday.

Also covered:

* identical behavior for tick and minute paths;
* ``session_policy`` close / discard / carry on a skipped gap;
* minute volatility/hybrid gap returns and tick-rule signs are excluded;
* chunked and resumed runs equal an uninterrupted run;
* ``flush()`` closes an open bar at end of stream.
"""

from __future__ import annotations

import datetime
import json
import math

import pandas as pd
import pytest

import flowbars as fb
from flowbars.calendars import ExchangeCalendar, SessionCalendar, WeekdayCalendar
from flowbars.core import MinuteInfo, TickInfo

TICK_SCHEMA = fb.SchemaMapping({"timestamp": "ts", "price": "price", "volume": "volume"})
MINUTE_SCHEMA = fb.MinuteSchemaMapping(
    {
        "timestamp": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
)

CAL = SessionCalendar(9, 30, 16, 0)


def _ts(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    dt = datetime.datetime(year, month, day, hour, minute, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


FRI_1558 = _ts(2024, 1, 5, 15, 58)  # Friday
FRI_1559 = _ts(2024, 1, 5, 15, 59)
FRI_1600 = _ts(2024, 1, 5, 16, 0)  # session close
MON_0930 = _ts(2024, 1, 8, 9, 30)  # Monday, next session open
MON_0931 = _ts(2024, 1, 8, 9, 31)


def _tick(timestamp: int, price: float, volume: float = 1.0) -> TickInfo:
    return TickInfo(timestamp=timestamp, price=price, volume=volume)


def _mi(timestamp: int, close: float, volume: float = 1.0) -> MinuteInfo:
    return MinuteInfo(
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def _tick_df(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "price": [r[1] for r in rows],
            "volume": [r[2] for r in rows],
        }
    )


def _minute_df(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[1] for r in rows],
            "close": [r[1] for r in rows],
            "volume": [r[2] for r in rows],
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tick path
# ═══════════════════════════════════════════════════════════════════════════════


class TestTickSessionGaps:
    def test_session_calendar_closes_at_prior_close(self) -> None:
        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA)
        assert ctor.update(_tick(FRI_1558, 100.0)) is None
        assert ctor.update(_tick(FRI_1559, 101.0)) is None

        bar = ctor.update(_tick(MON_0930, 110.0))
        assert bar is not None
        assert bar.partial is True
        assert bar.open_ts == FRI_1558
        assert bar.close_ts == FRI_1600  # prior session close, not Monday's tick
        assert bar.num_ticks == 2
        assert bar.volume == pytest.approx(2.0)
        assert bar.open == pytest.approx(100.0) and bar.close == pytest.approx(101.0)

        # Monday's tick starts a fresh bar.
        monday = ctor.flush()
        assert monday is not None and monday.partial is True
        assert monday.open_ts == MON_0930
        assert monday.num_ticks == 1

    def test_no_split_within_one_session(self) -> None:
        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA)
        assert ctor.update(_tick(_ts(2024, 1, 5, 10, 0), 100.0)) is None
        # A four-hour intraday halt is not a session boundary.
        assert ctor.update(_tick(_ts(2024, 1, 5, 14, 0), 101.0)) is None
        bar = ctor.flush()
        assert bar is not None and bar.num_ticks == 2
        assert bar.open_ts == _ts(2024, 1, 5, 10, 0)
        assert bar.close_ts == FRI_1600

    def test_overnight_session_is_not_split(self) -> None:
        cal = SessionCalendar(22, 0, 6, 0)
        ctor = fb.VolumeBarConstructor(1e12, calendar=cal, schema=TICK_SCHEMA)
        assert ctor.update(_tick(_ts(2024, 1, 8, 23, 0), 100.0)) is None
        # 02:00 is still inside the session opened Monday 22:00.
        assert ctor.update(_tick(_ts(2024, 1, 9, 2, 0), 101.0)) is None
        bar = ctor.flush()
        assert bar is not None and bar.num_ticks == 2

    def test_overnight_session_gap_closes_at_prior_close(self) -> None:
        cal = SessionCalendar(22, 0, 6, 0)
        ctor = fb.VolumeBarConstructor(1e12, calendar=cal, schema=TICK_SCHEMA)
        assert ctor.update(_tick(_ts(2024, 1, 8, 23, 0), 100.0)) is None
        bar = ctor.update(_tick(_ts(2024, 1, 9, 23, 0), 101.0))
        assert bar is not None and bar.partial is True
        assert bar.close_ts == _ts(2024, 1, 9, 6, 0)

    def test_weekday_calendar_splits_at_weekend(self) -> None:
        ctor = fb.VolumeBarConstructor(1e12, calendar=WeekdayCalendar(), schema=TICK_SCHEMA)
        assert ctor.update(_tick(_ts(2024, 1, 5, 23, 59), 100.0)) is None
        bar = ctor.update(_tick(_ts(2024, 1, 8, 0, 0), 101.0))
        assert bar is not None and bar.partial is True
        assert bar.close_ts == _ts(2024, 1, 6, 0, 0)  # Saturday 00:00 UTC

    def test_exchange_calendar_splits_at_holiday(self) -> None:
        pytest.importorskip("exchange_calendars")
        cal = ExchangeCalendar("XNYS")
        ctor = fb.VolumeBarConstructor(1e12, calendar=cal, schema=TICK_SCHEMA)
        # Friday 2024-01-12 15:00 UTC; next session is Tuesday (MLK Monday).
        assert ctor.update(_tick(_ts(2024, 1, 12, 15, 0), 100.0)) is None
        bar = ctor.update(_tick(_ts(2024, 1, 16, 15, 0), 101.0))
        assert bar is not None and bar.partial is True
        assert bar.close_ts == _ts(2024, 1, 12, 21, 0)  # 16:00 ET close

    def test_continuous_calendar_never_splits(self) -> None:
        ctor = fb.VolumeBarConstructor(1e12, schema=TICK_SCHEMA)
        assert ctor.update(_tick(FRI_1559, 100.0)) is None
        assert ctor.update(_tick(MON_0930, 110.0)) is None
        bar = ctor.flush()
        assert bar is not None and bar.num_ticks == 2
        assert bar.partial is True  # flush, not a session boundary

    def test_on_bar_callback_fires_for_gap_close(self) -> None:
        seen: list[object] = []
        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA, on_bar=seen.append)
        ctor.update(_tick(FRI_1558, 100.0))
        ctor.update(_tick(MON_0930, 110.0))
        assert len(seen) == 1
        assert seen[0].partial is True
        assert seen[0].close_ts == FRI_1600


class TestMinuteSessionPoliciesOnSkippedGap:
    def test_discard_drops_the_hanging_bar(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(
            1e12, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="discard"
        )
        assert ctor.update(_mi(FRI_1558, 100.0)) is None
        assert ctor.update(_mi(MON_0930, 110.0)) is None
        assert ctor.bars_emitted == 0
        assert ctor._accumulator._cum_volume == pytest.approx(1.0)  # Monday only

    def test_carry_keeps_the_bar_open(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(
            1e12, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="carry"
        )
        assert ctor.update(_mi(FRI_1558, 100.0)) is None
        assert ctor.update(_mi(MON_0930, 110.0)) is None
        assert ctor.bars_emitted == 0
        bar = ctor.flush()
        assert bar is not None and bar.num_ticks == 2


class TestTickGapContinuity:
    def test_chunked_update_many_matches_single_batch(self) -> None:
        friday = _tick_df([(FRI_1558, 100.0, 1.0), (FRI_1559, 101.0, 1.0)])
        monday = _tick_df([(MON_0930, 110.0, 1.0), (MON_0931, 111.0, 1.0)])

        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA)
        first = ctor.update_many(friday)
        second = ctor.update_many(monday)
        chunked = pd.concat([f for f in (first, second) if len(f)], ignore_index=True)

        full = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA).update_many(
            pd.concat([friday, monday], ignore_index=True)
        )
        pd.testing.assert_frame_equal(chunked, full)
        assert len(full) == 1  # Friday partial; Monday bar still open at EOF

    def test_resume_across_gap_matches_uninterrupted(self) -> None:
        friday = _tick_df([(FRI_1558, 100.0, 1.0), (FRI_1559, 101.0, 1.0)])
        monday = _tick_df([(MON_0930, 110.0, 1.0), (MON_0931, 111.0, 1.0)])

        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA)
        first = ctor.update_many(friday)
        state = json.loads(json.dumps(ctor.get_state()))
        resumed = fb.VolumeBarConstructor.from_state(state, calendar=CAL, schema=TICK_SCHEMA)
        second = resumed.update_many(monday)

        full = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA).update_many(
            pd.concat([friday, monday], ignore_index=True)
        )
        pd.testing.assert_frame_equal(
            pd.concat([f for f in (first, second) if len(f)], ignore_index=True), full
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Minute path
# ═══════════════════════════════════════════════════════════════════════════════


class TestMinuteSessionGaps:
    def test_volume_bar_closes_at_prior_close(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(1e12, calendar=CAL, schema=MINUTE_SCHEMA)
        assert ctor.update(_mi(FRI_1559, 100.0, volume=5.0)) is None
        bar = ctor.update(_mi(MON_0930, 110.0, volume=7.0))
        assert bar is not None and bar.partial is True
        assert bar.open_ts == FRI_1559 and bar.close_ts == FRI_1600
        assert bar.volume == pytest.approx(5.0)

    def test_volatility_gap_contributes_no_return(self) -> None:
        ctor = fb.MinuteVolatilityBarConstructor(1e9, calendar=CAL, schema=MINUTE_SCHEMA)
        ctor.update(_mi(FRI_1558, 100.0))
        ctor.update(_mi(FRI_1559, 100.0))
        ctor.update(_mi(MON_0930, 130.0))
        ctor.update(_mi(MON_0931, 143.0))

        # Only the in-session 130 -> 143 move counts; the 30% overnight gap
        # and the first minute of each session contribute nothing.
        assert ctor._accumulator._cum_metric == pytest.approx(13.0 / 130.0)
        bar = ctor.flush()
        assert bar is not None and bar.num_ticks == 2

    def test_first_minute_after_gap_has_no_derived_sign(self) -> None:
        ctor = fb.MinuteImbalanceVolumeBarConstructor(span=3.0, calendar=CAL, schema=MINUTE_SCHEMA)
        ctor.update(_mi(FRI_1558, 100.0))
        ctor.update(_mi(FRI_1559, 99.0))
        assert ctor._prev_sign == -1.0  # derived within the session

        ctor.update(_mi(MON_0930, 130.0))  # +31% but not a tick-rule move
        assert math.isnan(ctor._prev_sign)
        assert ctor._prev_close == pytest.approx(130.0)

    def test_supplied_side_survives_gap(self) -> None:
        ctor = fb.MinuteImbalanceVolumeBarConstructor(span=3.0, calendar=CAL, schema=MINUTE_SCHEMA)
        ctor.update(_mi(FRI_1559, 100.0))
        ctor.update(
            MinuteInfo(
                timestamp=MON_0930,
                open=130.0,
                high=130.0,
                low=130.0,
                close=130.0,
                volume=1.0,
                side=1.0,
            )
        )
        assert ctor._prev_sign == 1.0

    def test_chunked_update_many_matches_single_batch(self) -> None:
        friday = _minute_df([(FRI_1558, 100.0, 1.0), (FRI_1559, 101.0, 1.0)])
        monday = _minute_df([(MON_0930, 110.0, 1.0), (MON_0931, 111.0, 1.0)])

        ctor = fb.MinuteVolumeBarConstructor(1e12, calendar=CAL, schema=MINUTE_SCHEMA)
        first = ctor.update_many(friday)
        second = ctor.update_many(monday)

        full = fb.MinuteVolumeBarConstructor(1e12, calendar=CAL, schema=MINUTE_SCHEMA).update_many(
            pd.concat([friday, monday], ignore_index=True)
        )
        pd.testing.assert_frame_equal(
            pd.concat([f for f in (first, second) if len(f)], ignore_index=True), full
        )
        assert len(full) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# flush()
# ═══════════════════════════════════════════════════════════════════════════════


class TestFlush:
    def test_flush_on_empty_returns_none(self) -> None:
        assert fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA).flush() is None

    def test_flush_defaults_to_session_end(self) -> None:
        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA)
        ctor.update(_tick(_ts(2024, 1, 5, 14, 0), 100.0))
        bar = ctor.flush()
        assert bar is not None and bar.partial is True
        assert bar.close_ts == FRI_1600

    def test_flush_with_explicit_timestamp(self) -> None:
        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA)
        ctor.update(_tick(_ts(2024, 1, 5, 14, 0), 100.0))
        close = _ts(2024, 1, 5, 14, 30)
        bar = ctor.flush(close)
        assert bar is not None and bar.close_ts == close

    def test_flush_then_continue_starts_a_new_bar(self) -> None:
        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA)
        ctor.update(_tick(FRI_1558, 100.0))
        first = ctor.flush()
        assert first is not None and first.volume == pytest.approx(1.0)

        ctor.update(_tick(FRI_1559, 101.0))
        second = ctor.flush()
        assert second is not None
        assert second.open_ts == FRI_1559
        assert second.num_ticks == 1
        assert second.bar_id == first.bar_id + 1

    def test_flush_respects_warmup(self) -> None:
        ctor = fb.VolumeBarConstructor(1e12, calendar=CAL, schema=TICK_SCHEMA, warmup_bars=1)
        ctor.update(_tick(FRI_1558, 100.0))
        assert ctor.flush() is None  # first bar is a warmup bar
        ctor.update(_tick(FRI_1559, 101.0))
        bar = ctor.flush()
        assert bar is not None and bar.open_ts == FRI_1559
