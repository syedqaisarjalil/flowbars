"""State and identity correctness (calendar/schema/registry/session identity).

Covers the P0 checklist:

* calendars expose a stable ``to_dict``/``fingerprint`` identity;
* state records the calendar, schema mapping, registry key, and session;
* ``load_state`` rejects a mismatched calendar/schema/registry with
  ``StateValidationError`` unless ``verify_identity=False``;
* session identity survives a save/resume across a weekend;
* every registry key, row ``bar_type``, frame attr, and state value agree;
* both registries offer a uniform ``constructor_from_state`` entry point.
"""

from __future__ import annotations

import datetime
import json

import pandas as pd
import pytest

import flowbars as fb
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.bars.registry import BarRegistry
from flowbars.core import MinuteInfo, StateValidationError

BASE_TS = 1_704_067_200_000  # 2024-01-01T00:00:00Z (Monday)
MINUTE_MS = 60_000

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

OTHER_MINUTE_SCHEMA = fb.MinuteSchemaMapping(
    {
        "timestamp": "t",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    }
)


def _ms(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(datetime.datetime(y, mo, d, h, mi, tzinfo=datetime.timezone.utc).timestamp() * 1000)


def _ticks_df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [BASE_TS + i * 1000 for i in range(n)],
            "price": [100.0 + (i % 5) for i in range(n)],
            "volume": [1.0] * n,
        }
    )


def _minutes_df(n: int, start: int = BASE_TS, closes: list[float] | None = None) -> pd.DataFrame:
    if closes is None:
        closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "ts": [start + i * MINUTE_MS for i in range(n)],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * n,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Calendar / schema identity (P0-1)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCalendarIdentity:
    def test_fingerprints_are_stable_and_distinct(self) -> None:
        calendars = [
            fb.ContinuousCalendar(),
            fb.SessionCalendar(9, 30, 16, 0),
            fb.SessionCalendar(9, 30, 16, 30),
            fb.WeekdayCalendar(),
        ]
        fingerprints = [cal.fingerprint() for cal in calendars]
        assert all(len(fp) == 16 for fp in fingerprints)
        assert fingerprints == [cal.fingerprint() for cal in calendars]
        assert len(set(fingerprints)) == len(fingerprints)
        for cal in calendars:
            json.dumps(cal.to_dict())  # JSON-serializable

    def test_exchange_calendar_identity(self) -> None:
        pytest.importorskip("exchange_calendars")
        cal = fb.ExchangeCalendar("XNYS")
        assert cal.to_dict() == {"kind": "exchange", "exchange": "XNYS"}
        assert cal.fingerprint() == fb.ExchangeCalendar("XNYS").fingerprint()
        assert cal.fingerprint() != fb.ExchangeCalendar("CME").fingerprint()

    def test_state_records_calendar_and_schema(self) -> None:
        calendar = fb.WeekdayCalendar()
        ctor = fb.MinuteVolumeBarConstructor(1e9, schema=MINUTE_SCHEMA, calendar=calendar)
        ctor.update_many(_minutes_df(2))
        state = ctor.get_state()

        assert state["calendar"] == calendar.to_dict()
        assert state["calendar_fingerprint"] == calendar.fingerprint()
        assert state["schema_mapping"] == MINUTE_SCHEMA.to_dict()
        assert state["schema_fingerprint"] == MINUTE_SCHEMA.fingerprint()

    def test_resume_with_different_calendar_raises(self) -> None:
        calendar = fb.WeekdayCalendar()
        ctor = fb.MinuteVolumeBarConstructor(1e9, schema=MINUTE_SCHEMA, calendar=calendar)
        ctor.update_many(_minutes_df(2))
        state = json.loads(json.dumps(ctor.get_state()))

        with pytest.raises(StateValidationError, match="calendar mismatch"):
            fb.MinuteVolumeBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
        with pytest.raises(StateValidationError, match="calendar mismatch"):
            fb.MinuteVolumeBarConstructor.from_state(
                state, calendar=fb.SessionCalendar(9, 30, 16, 0), schema=MINUTE_SCHEMA
            )

    def test_resume_with_different_schema_raises(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(1e9, schema=MINUTE_SCHEMA)
        ctor.update_many(_minutes_df(2))
        state = json.loads(json.dumps(ctor.get_state()))

        with pytest.raises(StateValidationError, match="schema mapping mismatch"):
            fb.MinuteVolumeBarConstructor.from_state(state, schema=OTHER_MINUTE_SCHEMA)

    def test_registry_name_mismatch_raises(self) -> None:
        ctor = fb.VolumeBarConstructor(1e9, schema=TICK_SCHEMA)
        ctor.update_many(_ticks_df(2))
        state = json.loads(json.dumps(ctor.get_state()))
        state["registry_name"] = "minute_volume"

        with pytest.raises(StateValidationError, match="registry_name mismatch"):
            fb.VolumeBarConstructor.from_state(state, schema=TICK_SCHEMA)

    def test_verify_identity_false_allows_calendar_change(self) -> None:
        calendar = fb.WeekdayCalendar()
        df = _minutes_df(8)
        ctor = fb.MinuteVolumeBarConstructor(4.0, schema=MINUTE_SCHEMA, calendar=calendar)
        first = ctor.update_many(df.iloc[:3])
        state = json.loads(json.dumps(ctor.get_state()))
        assert len(first) == 0

        resumed = fb.MinuteVolumeBarConstructor.from_state(
            state,
            schema=MINUTE_SCHEMA,
            verify_identity=False,  # continuous default
        )
        second = resumed.update_many(df.iloc[3:])

        full = fb.MinuteVolumeBarConstructor(
            4.0, schema=MINUTE_SCHEMA, calendar=calendar
        ).update_many(df)
        pd.testing.assert_frame_equal(second, full)

    def test_legacy_state_without_identity_keys_still_loads(self) -> None:
        ctor = fb.VolumeBarConstructor(1e9, schema=TICK_SCHEMA)
        ctor.update_many(_ticks_df(2))
        state = json.loads(json.dumps(ctor.get_state()))
        for key in (
            "calendar",
            "calendar_fingerprint",
            "schema_mapping",
            "schema_fingerprint",
            "registry_name",
            "session_id",
            "last_session_end",
        ):
            state.pop(key)

        resumed = fb.VolumeBarConstructor.from_state(state, schema=TICK_SCHEMA)
        assert resumed.get_state()["last_watermark"] == ctor.get_state()["last_watermark"]


# ═══════════════════════════════════════════════════════════════════════════════
# Session state persistence (P0-2)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionState:
    def test_state_records_session_identity(self) -> None:
        cal = fb.SessionCalendar(9, 30, 16, 0)
        ctor = fb.MinuteVolumeBarConstructor(1e9, schema=MINUTE_SCHEMA, calendar=cal)
        ctor.update(
            MinuteInfo(
                timestamp=_ms(2024, 1, 8, 10, 0),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1.0,
            )
        )
        state = ctor.get_state()
        assert state["session_id"] == "2024-01-08T09:30:00Z"
        assert state["last_session_end"] == _ms(2024, 1, 8, 16, 0)
        assert state["in_session"] is True

    def test_resume_across_weekend_matches_uninterrupted(self) -> None:
        friday = _minutes_df(2, start=_ms(2024, 1, 5, 14, 30))
        monday = _minutes_df(4, start=_ms(2024, 1, 8, 14, 30))

        cal = fb.WeekdayCalendar()
        ctor = fb.MinuteVolumeBarConstructor(4.0, schema=MINUTE_SCHEMA, calendar=cal)
        first = ctor.update_many(friday)
        state = json.loads(json.dumps(ctor.get_state()))
        assert state["session_id"] is not None
        assert len(first) == 0

        resumed = fb.MinuteVolumeBarConstructor.from_state(
            state, calendar=cal, schema=MINUTE_SCHEMA
        )
        after_weekend = resumed.update_many(monday)

        full_df = pd.concat([friday, monday], ignore_index=True)
        full = fb.MinuteVolumeBarConstructor(4.0, schema=MINUTE_SCHEMA, calendar=cal).update_many(
            full_df
        )
        pd.testing.assert_frame_equal(after_weekend, full)


# ═══════════════════════════════════════════════════════════════════════════════
# Canonical registry identity (P0-3)
# ═══════════════════════════════════════════════════════════════════════════════

# Expected emitted ``bar_type`` label for each registry key.
_TICK_TYPES = {
    "time": "time",
    "tick": "tick",
    "volume": "volume",
    "dollar": "dollar",
    "imbalance_tick": "imbalance_tick",
    "imbalance_volume": "imbalance_volume",
    "imbalance_dollar": "imbalance_dollar",
    "run_tick": "run_tick",
    "run_volume": "run_volume",
    "run_dollar": "run_dollar",
}

_MINUTE_TYPES = {
    "minute_volume": "volume",
    "minute_dollar": "dollar",
    "minute_imbalance_volume": "imbalance_volume",
    "minute_imbalance_dollar": "imbalance_dollar",
    "minute_run_volume": "run_volume",
    "minute_run_dollar": "run_dollar",
    "minute_volatility": "volatility",
    "minute_range": "range",
    "minute_renko": "renko",
    "minute_hybrid": "hybrid",
}


def _tick_ctor(name: str) -> fb.BaseBarConstructor:
    if name == "time":
        return fb.TimeBarConstructor(interval_ms=2000, schema=TICK_SCHEMA)
    if name == "tick":
        return fb.TickBarConstructor(2, schema=TICK_SCHEMA)
    if name == "volume":
        return fb.VolumeBarConstructor(1.0, schema=TICK_SCHEMA)
    if name == "dollar":
        return fb.DollarBarConstructor(100.0, schema=TICK_SCHEMA)
    if name == "imbalance_tick":
        return fb.ImbalanceTickBarConstructor(schema=TICK_SCHEMA)
    if name == "imbalance_volume":
        return fb.ImbalanceVolumeBarConstructor(schema=TICK_SCHEMA)
    if name == "imbalance_dollar":
        return fb.ImbalanceDollarBarConstructor(schema=TICK_SCHEMA)
    if name == "run_tick":
        return fb.RunTickBarConstructor(schema=TICK_SCHEMA)
    if name == "run_volume":
        return fb.RunVolumeBarConstructor(schema=TICK_SCHEMA)
    return fb.RunDollarBarConstructor(schema=TICK_SCHEMA)


def _minute_ctor(name: str) -> fb.BaseMinuteBarConstructor:
    tiny = 1e-9
    if name == "minute_volume":
        return fb.MinuteVolumeBarConstructor(1.0, schema=MINUTE_SCHEMA)
    if name == "minute_dollar":
        return fb.MinuteDollarBarConstructor(100.0, schema=MINUTE_SCHEMA)
    if name == "minute_imbalance_volume":
        return fb.MinuteImbalanceVolumeBarConstructor(schema=MINUTE_SCHEMA)
    if name == "minute_imbalance_dollar":
        return fb.MinuteImbalanceDollarBarConstructor(schema=MINUTE_SCHEMA)
    if name == "minute_run_volume":
        return fb.MinuteRunVolumeBarConstructor(schema=MINUTE_SCHEMA)
    if name == "minute_run_dollar":
        return fb.MinuteRunDollarBarConstructor(schema=MINUTE_SCHEMA)
    if name == "minute_volatility":
        return fb.MinuteVolatilityBarConstructor(tiny, schema=MINUTE_SCHEMA)
    if name == "minute_range":
        return fb.MinuteRangeBarConstructor(tiny, schema=MINUTE_SCHEMA)
    if name == "minute_renko":
        return fb.MinuteRenkoBarConstructor(tiny, schema=MINUTE_SCHEMA)
    return fb.MinuteHybridBarConstructor(tiny, tiny, schema=MINUTE_SCHEMA)


class TestRegistryIdentity:
    def test_tick_registry_identity_agrees_everywhere(self) -> None:
        assert set(_TICK_TYPES) == set(BarRegistry.list())
        for name, label in _TICK_TYPES.items():
            ctor = _tick_ctor(name)
            assert type(ctor).registry_name == name
            frame = ctor.update_many(_ticks_df(8))
            state = ctor.get_state()

            assert state["registry_name"] == name
            assert state["bar_type"] == label
            assert frame.attrs["registry_name"] == name
            assert frame.attrs["bar_type"] == label
            if len(frame):
                assert set(frame["bar_type"]) == {label}

    def test_minute_registry_identity_agrees_everywhere(self) -> None:
        assert set(_MINUTE_TYPES) == set(MinuteBarRegistry.list())
        closes = [100.0, 101.5, 99.0, 102.5, 98.0, 103.5, 97.0, 104.5]
        df = _minutes_df(8, closes=closes)
        for name, label in _MINUTE_TYPES.items():
            ctor = _minute_ctor(name)
            assert type(ctor).registry_name == name
            frame = ctor.update_many(df)
            state = ctor.get_state()

            assert state["registry_name"] == name
            assert state["bar_type"] == label
            assert frame.attrs["registry_name"] == name
            assert frame.attrs["bar_type"] == label
            if len(frame):
                assert set(frame["bar_type"]) == {label}

    def test_spec_key_matches_registry_name(self) -> None:
        for registry in (BarRegistry, MinuteBarRegistry):
            for name in registry.list():
                assert registry.get_spec(name).bar_type == name
                assert registry.get_constructor(name).registry_name == name


# ═══════════════════════════════════════════════════════════════════════════════
# Uniform constructor-from-state entry point (P1-7)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryConstructorFromState:
    def test_tick_round_trip(self) -> None:
        df = _ticks_df(8)
        ctor = fb.VolumeBarConstructor(4.0, schema=TICK_SCHEMA)
        first = ctor.update_many(df.iloc[:3])
        state = json.loads(json.dumps(ctor.get_state()))
        assert len(first) == 0

        resumed = BarRegistry.constructor_from_state("volume", state, schema=TICK_SCHEMA)
        second = resumed.update_many(df.iloc[3:])

        full = fb.VolumeBarConstructor(4.0, schema=TICK_SCHEMA).update_many(df)
        pd.testing.assert_frame_equal(second, full)

    def test_minute_round_trip(self) -> None:
        closes = [100.0, 100.5, 102.5, 101.0, 99.0, 103.0]
        df = _minutes_df(6, closes=closes)
        ctor = fb.MinuteRenkoBarConstructor(0.01, schema=MINUTE_SCHEMA)
        first = ctor.update_many(df.iloc[:2])
        state = json.loads(json.dumps(ctor.get_state()))
        assert len(first) == 0

        resumed = MinuteBarRegistry.constructor_from_state(
            "minute_renko", state, schema=MINUTE_SCHEMA
        )
        second = resumed.update_many(df.iloc[2:])

        full = fb.MinuteRenkoBarConstructor(0.01, schema=MINUTE_SCHEMA).update_many(df)
        pd.testing.assert_frame_equal(second, full)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown"):
            BarRegistry.constructor_from_state("nope", {})
        with pytest.raises(KeyError, match="Unknown"):
            MinuteBarRegistry.constructor_from_state("nope", {})
