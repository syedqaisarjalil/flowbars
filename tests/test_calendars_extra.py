"""Tests for WeekdayCalendar and ExchangeCalendar (Phase: calendars)."""

from __future__ import annotations

import datetime
import importlib.util

import pytest

from flowbars.calendars import ExchangeCalendar, WeekdayCalendar

_HAS_XCALS = importlib.util.find_spec("exchange_calendars") is not None


def _ms(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    """Unix-ms for a UTC datetime."""
    return int(datetime.datetime(y, mo, d, h, mi, tzinfo=datetime.timezone.utc).timestamp() * 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# WeekdayCalendar (no optional deps — always runs)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeekdayCalendar:
    def test_weekday_is_not_boundary(self) -> None:
        cal = WeekdayCalendar()
        # 2024-01-01 is a Monday; 2024-01-05 is a Friday.
        assert cal.is_session_boundary(_ms(2024, 1, 1)) is False
        assert cal.is_session_boundary(_ms(2024, 1, 5, 23, 59)) is False

    def test_weekend_is_boundary(self) -> None:
        cal = WeekdayCalendar()
        assert cal.is_session_boundary(_ms(2024, 1, 6)) is True  # Saturday
        assert cal.is_session_boundary(_ms(2024, 1, 7)) is True  # Sunday

    def test_next_session_open_skips_weekend(self) -> None:
        cal = WeekdayCalendar()
        monday = _ms(2024, 1, 8)
        assert cal.next_session_open(_ms(2024, 1, 5)) == monday  # Friday
        assert cal.next_session_open(_ms(2024, 1, 6)) == monday  # Saturday
        assert cal.next_session_open(_ms(2024, 1, 7)) == monday  # Sunday

    def test_next_session_open_at_monday_advances(self) -> None:
        cal = WeekdayCalendar()
        # At exactly Monday 00:00, the next open is the following Monday.
        assert cal.next_session_open(_ms(2024, 1, 1)) == _ms(2024, 1, 8)

    def test_has_sessions_true(self) -> None:
        assert WeekdayCalendar().has_sessions is True


# ═══════════════════════════════════════════════════════════════════════════════
# ExchangeCalendar
# ═══════════════════════════════════════════════════════════════════════════════


class TestExchangeCalendarMissingDep:
    def test_raises_helpful_error(self, monkeypatch) -> None:
        """A clear error is raised when exchange_calendars is not importable."""
        import builtins
        import sys

        real_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):
            if name == "exchange_calendars":
                raise ImportError("simulated missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(sys.modules, "exchange_calendars", raising=False)
        with pytest.raises(ImportError, match="calendars"):
            ExchangeCalendar("XNYS")


@pytest.mark.skipif(not _HAS_XCALS, reason="exchange_calendars not installed")
class TestExchangeCalendar:
    def test_us_holiday_is_boundary(self) -> None:
        cal = ExchangeCalendar("XNYS")
        # July 4 2023 was a Tuesday (US Independence Day) — market closed.
        assert cal.is_session_boundary(_ms(2023, 7, 4, 15, 0)) is True

    def test_weekday_during_hours_is_open(self) -> None:
        cal = ExchangeCalendar("XNYS")
        # July 11 2023 (Tuesday) 14:00 UTC = 10:00 ET — market open.
        assert cal.is_session_boundary(_ms(2023, 7, 11, 14, 0)) is False

    def test_weekend_is_boundary(self) -> None:
        cal = ExchangeCalendar("XNYS")
        assert cal.is_session_boundary(_ms(2023, 7, 8)) is True  # Saturday

    def test_next_session_open_is_in_future(self) -> None:
        cal = ExchangeCalendar("XNYS")
        friday_close = _ms(2023, 7, 7, 21, 0)  # Friday 17:00 ET, after close
        assert cal.next_session_open(friday_close) > friday_close

    def test_has_sessions_true(self) -> None:
        assert ExchangeCalendar("XNYS").has_sessions is True
