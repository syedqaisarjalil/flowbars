"""Tests for trading calendars — Phase 4."""

from __future__ import annotations

import datetime

import pytest

from flowbars.calendars import (
    ContinuousCalendar,
    SessionCalendar,
    TradingCalendar,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def utc_ms(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> int:
    """Convert a naive UTC datetime to Unix milliseconds."""
    dt = datetime.datetime(year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


def weekday_ts(weekday: int, hour: int, minute: int, second: int = 0) -> int:
    """Return a Unix-ms timestamp for the given UTC weekday and time.

    Uses 2024-01-15 (Monday) as the reference point.  *weekday* 0=Mon … 6=Sun.
    """
    base = datetime.datetime(2024, 1, 15, hour, minute, second, tzinfo=datetime.timezone.utc)
    target = base + datetime.timedelta(days=weekday)
    return int(target.timestamp() * 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# TradingCalendar ABC
# ═══════════════════════════════════════════════════════════════════════════════


class TestTradingCalendarABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            TradingCalendar()  # type: ignore[abstract]

    def test_must_implement_both_abstracts(self) -> None:
        class Incomplete(TradingCalendar):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_both(self) -> None:
        class OnlyOne(TradingCalendar):
            def is_session_boundary(self, timestamp: int) -> bool:
                return False

        with pytest.raises(TypeError):
            OnlyOne()  # type: ignore[abstract]


# ═══════════════════════════════════════════════════════════════════════════════
# ContinuousCalendar
# ═══════════════════════════════════════════════════════════════════════════════


class TestContinuousCalendar:
    def test_never_a_boundary(self) -> None:
        cal = ContinuousCalendar()
        assert not cal.is_session_boundary(0)
        assert not cal.is_session_boundary(1705311000000)

    def test_next_session_open_is_identity(self) -> None:
        cal = ContinuousCalendar()
        assert cal.next_session_open(42) == 42
        assert cal.next_session_open(0) == 0
        assert cal.next_session_open(1705311000000) == 1705311000000


# ═══════════════════════════════════════════════════════════════════════════════
# SessionCalendar — regular session (intraday)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionCalendarIntraday:
    """Regular session: 09:30–16:00 UTC (all within one UTC day)."""

    @pytest.fixture
    def cal(self) -> SessionCalendar:
        return SessionCalendar(9, 30, 16, 0)

    # ── is_session_boundary ────────────────────────────────────────────

    def test_during_session_not_boundary(self, cal: SessionCalendar) -> None:
        # Monday 10:00 UTC — inside session
        ts = weekday_ts(0, 10, 0)
        assert not cal.is_session_boundary(ts)

    def test_before_session_is_boundary(self, cal: SessionCalendar) -> None:
        # Monday 08:00 UTC — before open
        ts = weekday_ts(0, 8, 0)
        assert cal.is_session_boundary(ts)

    def test_after_session_is_boundary(self, cal: SessionCalendar) -> None:
        # Monday 17:00 UTC — after close
        ts = weekday_ts(0, 17, 0)
        assert cal.is_session_boundary(ts)

    def test_exactly_at_open_not_boundary(self, cal: SessionCalendar) -> None:
        ts = weekday_ts(0, 9, 30)
        assert not cal.is_session_boundary(ts)

    def test_exactly_at_close_is_boundary(self, cal: SessionCalendar) -> None:
        # At 16:00:00 the session is closed — tick triggers boundary
        ts = weekday_ts(0, 16, 0)
        assert cal.is_session_boundary(ts)

    def test_weekend_is_boundary(self, cal: SessionCalendar) -> None:
        # Saturday 12:00 UTC
        ts = weekday_ts(5, 12, 0)
        assert cal.is_session_boundary(ts)

    def test_sunday_is_boundary(self, cal: SessionCalendar) -> None:
        ts = weekday_ts(6, 12, 0)
        assert cal.is_session_boundary(ts)

    def test_friday_during_session_not_boundary(self, cal: SessionCalendar) -> None:
        # Friday 10:00 UTC
        ts = weekday_ts(4, 10, 0)
        assert not cal.is_session_boundary(ts)

    # ── next_session_open ──────────────────────────────────────────────

    def test_next_open_within_session_returns_next_day(self, cal: SessionCalendar) -> None:
        # Monday 10:00 → next open is Tuesday 09:30
        ts = weekday_ts(0, 10, 0)
        expected = weekday_ts(1, 9, 30)
        assert cal.next_session_open(ts) == expected

    def test_next_open_after_close_returns_next_day(self, cal: SessionCalendar) -> None:
        # Monday 17:00 → Tuesday 09:30
        ts = weekday_ts(0, 17, 0)
        expected = weekday_ts(1, 9, 30)
        assert cal.next_session_open(ts) == expected

    def test_next_open_before_open_returns_today(self, cal: SessionCalendar) -> None:
        # Monday 08:00 → Monday 09:30
        ts = weekday_ts(0, 8, 0)
        expected = weekday_ts(0, 9, 30)
        assert cal.next_session_open(ts) == expected

    def test_next_open_from_friday_after_close_returns_monday(self, cal: SessionCalendar) -> None:
        # Friday 17:00 → Monday 09:30
        ts = weekday_ts(4, 17, 0)
        expected = weekday_ts(7, 9, 30)  # +3 days to Monday
        assert cal.next_session_open(ts) == expected

    def test_next_open_from_saturday_returns_monday(self, cal: SessionCalendar) -> None:
        ts = weekday_ts(5, 12, 0)
        expected = weekday_ts(7, 9, 30)
        assert cal.next_session_open(ts) == expected

    def test_next_open_from_sunday_returns_monday(self, cal: SessionCalendar) -> None:
        ts = weekday_ts(6, 12, 0)
        expected = weekday_ts(7, 9, 30)
        assert cal.next_session_open(ts) == expected

    def test_next_open_at_exactly_open_returns_next_day(self, cal: SessionCalendar) -> None:
        # At exactly 09:30:00, the session is already open → next open is tomorrow
        ts = weekday_ts(0, 9, 30)
        expected = weekday_ts(1, 9, 30)
        assert cal.next_session_open(ts) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# SessionCalendar — overnight session
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionCalendarOvernight:
    """Overnight session: 22:00–06:00 UTC (spans midnight)."""

    @pytest.fixture
    def cal(self) -> SessionCalendar:
        return SessionCalendar(22, 0, 6, 0)

    def test_during_session_before_midnight(self, cal: SessionCalendar) -> None:
        # Monday 23:00 UTC — session started Monday 22:00
        ts = weekday_ts(0, 23, 0)
        assert not cal.is_session_boundary(ts)

    def test_during_session_after_midnight(self, cal: SessionCalendar) -> None:
        # Tuesday 02:00 UTC — still in Monday's overnight session
        ts = weekday_ts(1, 2, 0)
        assert not cal.is_session_boundary(ts)

    def test_exactly_at_close_is_boundary(self, cal: SessionCalendar) -> None:
        # Tuesday 06:00 UTC
        ts = weekday_ts(1, 6, 0)
        assert cal.is_session_boundary(ts)

    def test_after_close_before_next_open(self, cal: SessionCalendar) -> None:
        # Tuesday 12:00 UTC — after Monday-night session closed, before Tuesday-night
        ts = weekday_ts(1, 12, 0)
        assert cal.is_session_boundary(ts)

    def test_next_open_after_close_same_day(self, cal: SessionCalendar) -> None:
        # Tuesday 07:00 → Tuesday 22:00
        ts = weekday_ts(1, 7, 0)
        expected = weekday_ts(1, 22, 0)
        assert cal.next_session_open(ts) == expected

    def test_next_open_during_session_returns_next_day(self, cal: SessionCalendar) -> None:
        # Tuesday 02:00 → Tuesday 22:00 (next session, not today's already-open one)
        ts = weekday_ts(1, 2, 0)
        expected = weekday_ts(1, 22, 0)
        assert cal.next_session_open(ts) == expected

    def test_friday_overnight_not_boundary(self, cal: SessionCalendar) -> None:
        # Saturday 03:00 UTC — inside Friday-night session
        ts = weekday_ts(5, 3, 0)
        assert not cal.is_session_boundary(ts)

    def test_saturday_after_close_is_boundary(self, cal: SessionCalendar) -> None:
        # Saturday 07:00 — after Friday-night close
        ts = weekday_ts(5, 7, 0)
        assert cal.is_session_boundary(ts)

    def test_next_open_from_saturday_returns_monday(self, cal: SessionCalendar) -> None:
        # Saturday 07:00 → Monday 22:00
        ts = weekday_ts(5, 7, 0)
        expected = weekday_ts(7, 22, 0)  # Monday
        assert cal.next_session_open(ts) == expected

    def test_sunday_evening_during_session(self, cal: SessionCalendar) -> None:
        """Sunday 22:00 UTC: session start is Sunday, but Sunday is a weekend.
        Therefore this is NOT a valid session — it's a boundary."""
        ts = weekday_ts(6, 22, 0)
        # Session would start Sunday 22:00, but Sunday is weekend → invalid
        assert cal.is_session_boundary(ts)

    def test_next_open_sunday_night_returns_monday(self, cal: SessionCalendar) -> None:
        """Sunday 23:00 → Monday 22:00 (Sunday session skipped)."""
        ts = weekday_ts(6, 23, 0)
        expected = weekday_ts(7, 22, 0)
        assert cal.next_session_open(ts) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# SessionCalendar — minute precision
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionCalendarMinutePrecision:
    """Session with non-zero minutes: 08:45–15:15 UTC."""

    @pytest.fixture
    def cal(self) -> SessionCalendar:
        return SessionCalendar(8, 45, 15, 15)

    def test_opens_at_correct_minute(self, cal: SessionCalendar) -> None:
        ts = weekday_ts(0, 8, 45)
        assert not cal.is_session_boundary(ts)

    def test_before_open_by_one_minute(self, cal: SessionCalendar) -> None:
        ts = weekday_ts(0, 8, 44)
        assert cal.is_session_boundary(ts)

    def test_closes_at_correct_minute(self, cal: SessionCalendar) -> None:
        ts = weekday_ts(0, 15, 15)
        assert cal.is_session_boundary(ts)

    def test_one_minute_before_close_not_boundary(self, cal: SessionCalendar) -> None:
        ts = weekday_ts(0, 15, 14)
        assert not cal.is_session_boundary(ts)

    def test_next_open_preserves_minutes(self, cal: SessionCalendar) -> None:
        # Monday 16:00 → Tuesday 08:45
        ts = weekday_ts(0, 16, 0)
        expected = weekday_ts(1, 8, 45)
        assert cal.next_session_open(ts) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# SessionCalendar — constructor validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionCalendarValidation:
    def test_invalid_start_hour_too_high(self) -> None:
        with pytest.raises(ValueError):
            SessionCalendar(24, 0, 16, 0)

    def test_invalid_start_hour_negative(self) -> None:
        with pytest.raises(ValueError):
            SessionCalendar(-1, 0, 16, 0)

    def test_invalid_end_hour_too_high(self) -> None:
        with pytest.raises(ValueError):
            SessionCalendar(9, 30, 24, 0)

    def test_invalid_start_minute(self) -> None:
        with pytest.raises(ValueError):
            SessionCalendar(9, 60, 16, 0)

    def test_invalid_end_minute(self) -> None:
        with pytest.raises(ValueError):
            SessionCalendar(9, 30, 16, 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_midnight_to_midnight_session(self) -> None:
        """00:00–00:00 is a full 24h session (duration=0 → bumped to 24h)."""
        cal = SessionCalendar(0, 0, 0, 0)
        # Monday 12:00 — inside session
        ts = weekday_ts(0, 12, 0)
        assert not cal.is_session_boundary(ts)
        # But weekends are still closed
        assert cal.is_session_boundary(weekday_ts(5, 12, 0))

    def test_one_second_session(self) -> None:
        """09:30:00–09:30:00 is a zero-duration session → 24h.
        Use 09:30–09:31 for a 1-minute session."""
        cal = SessionCalendar(9, 30, 9, 31)
        # At 09:30:30 — inside
        ts = weekday_ts(0, 9, 30, 30)
        assert not cal.is_session_boundary(ts)
        # At 09:31:00 — boundary
        assert cal.is_session_boundary(weekday_ts(0, 9, 31))

    def test_midnight_crossing_with_minutes(self) -> None:
        """Overnight: 23:45–00:15 (30-minute session)."""
        cal = SessionCalendar(23, 45, 0, 15)
        # Monday 23:50 — inside
        assert not cal.is_session_boundary(weekday_ts(0, 23, 50))
        # Tuesday 00:10 — inside
        assert not cal.is_session_boundary(weekday_ts(1, 0, 10))
        # Tuesday 00:15 — boundary
        assert cal.is_session_boundary(weekday_ts(1, 0, 15))
        # Tuesday 00:20 — boundary
        assert cal.is_session_boundary(weekday_ts(1, 0, 20))

    def test_continuous_calendar_large_timestamps(self) -> None:
        cal = ContinuousCalendar()
        huge = 2**62
        assert not cal.is_session_boundary(huge)
        assert cal.next_session_open(huge) == huge

    def test_session_calendar_repr(self) -> None:
        cal = SessionCalendar(9, 30, 16, 0)
        # Just ensure it doesn't crash
        assert isinstance(repr(cal), str)
