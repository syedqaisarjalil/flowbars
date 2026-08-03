"""Trading calendars for session-boundary detection.

Calendars tell the bar constructor when to force-close a bar at session
boundaries (e.g., NYSE close at 21:00 UTC) and when the next session opens.
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod


class TradingCalendar(ABC):
    """Abstract base class for trading calendars.

    All timestamps are Unix milliseconds (UTC).  Calendars are stateless —
    they hold configuration only and have no mutable state to persist.
    """

    @abstractmethod
    def is_session_boundary(self, timestamp: int) -> bool:
        """Return ``True`` if *timestamp* falls outside a trading session.

        The bar constructor uses this to decide whether to force-close
        the in-progress bar before the next session begins.
        """
        ...

    @abstractmethod
    def next_session_open(self, timestamp: int) -> int:
        """Return the Unix-ms timestamp of the next session open ≥ *timestamp*.

        When a tick lands outside a session, the bar constructor closes the
        current bar and uses this to know when to resume.
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Always-open calendar (crypto / forex)
# ═══════════════════════════════════════════════════════════════════════════════


class ContinuousCalendar(TradingCalendar):
    """A calendar with no session boundaries — always open.

    This is the default for crypto and forex markets that trade 24/7.
    """

    def is_session_boundary(self, timestamp: int) -> bool:
        return False

    def next_session_open(self, timestamp: int) -> int:
        return timestamp


# ═══════════════════════════════════════════════════════════════════════════════
# Configurable-session calendar (equities, futures, …)
# ═══════════════════════════════════════════════════════════════════════════════


class SessionCalendar(TradingCalendar):
    """A calendar with fixed daily session hours, Monday–Friday.

    Sessions are defined in UTC.  Users must convert their local market
    hours to UTC themselves (e.g. NYSE 09:30–16:00 EST → 14:30–21:00 UTC).

    Overnight sessions are supported — when *end_hour* < *start_hour* the
    session spans midnight UTC.

    Weekends (Saturday and Sunday) are always non-trading.  No holiday
    calendar is included in this version; an adapter for
    ``exchange_calendars`` can provide full holiday support.

    Parameters
    ----------
    start_hour : int
        Session open hour in UTC (0–23).
    start_minute : int
        Session open minute in UTC (0–59).
    end_hour : int
        Session close hour in UTC (0–23).
    end_minute : int
        Session close minute in UTC (0–59).
    """

    def __init__(
        self,
        start_hour: int,
        start_minute: int,
        end_hour: int,
        end_minute: int,
    ) -> None:
        if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
            raise ValueError(
                f"Hours must be 0–23, got start={start_hour}, end={end_hour}"
            )
        if not (0 <= start_minute <= 59 and 0 <= end_minute <= 59):
            raise ValueError(
                f"Minutes must be 0–59, got start={start_minute}, end={end_minute}"
            )

        self._start_hour = start_hour
        self._start_minute = start_minute
        self._end_hour = end_hour
        self._end_minute = end_minute

        # Session duration in seconds (handles overnight)
        duration_seconds = (end_hour - start_hour) * 3600 + (end_minute - start_minute) * 60
        if duration_seconds <= 0:
            duration_seconds += 24 * 3600
        self._session_duration = datetime.timedelta(seconds=duration_seconds)

    # ── TradingCalendar interface ──────────────────────────────────────────

    def is_session_boundary(self, timestamp: int) -> bool:
        """Return ``True`` if *timestamp* is not during a trading session.

        Checks both today's and yesterday's session (for overnight sessions
        that started on the prior UTC day).  Only sessions whose *open* falls
        on a weekday (Mon–Fri) are considered valid.
        """
        dt = _utcfromtimestamp(timestamp)

        # Check yesterday's session (overnight sessions can start yesterday)
        yesterday = dt - datetime.timedelta(days=1)
        y_open = _session_open(yesterday, self._start_hour, self._start_minute)
        y_close = y_open + self._session_duration
        if y_open.weekday() < 5 and y_open <= dt < y_close:
            return False

        # Check today's session
        t_open = _session_open(dt, self._start_hour, self._start_minute)
        t_close = t_open + self._session_duration
        if t_open.weekday() < 5 and t_open <= dt < t_close:
            return False

        return True

    def next_session_open(self, timestamp: int) -> int:
        """Return the next session-open timestamp ≥ *timestamp*.

        Skips Saturday and Sunday — a session whose open falls on a weekend
        is never returned.
        """
        dt = _utcfromtimestamp(timestamp)

        # Start with today's session open
        candidate = _session_open(dt, self._start_hour, self._start_minute)

        # If already past today's open, advance at least one day
        if candidate <= dt:
            candidate += datetime.timedelta(days=1)

        # Skip weekends
        while candidate.weekday() >= 5:
            candidate += datetime.timedelta(days=1)

        return _to_utc_ms(candidate)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _utcfromtimestamp(ts_ms: int) -> datetime.datetime:
    """Convert a Unix-ms timestamp to a naive UTC datetime."""
    return datetime.datetime.utcfromtimestamp(ts_ms / 1000.0)


def _session_open(
    dt: datetime.datetime, start_hour: int, start_minute: int
) -> datetime.datetime:
    """Return the session-open datetime for the UTC day of *dt*."""
    return dt.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)


def _to_utc_ms(dt: datetime.datetime) -> int:
    """Convert a naive UTC datetime to Unix milliseconds."""
    # Attach UTC tzinfo so .timestamp() computes the correct Unix time
    # regardless of the local timezone.
    return int(dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
