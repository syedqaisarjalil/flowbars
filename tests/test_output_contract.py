"""Tests for the public bar output contract (columns, metadata, validation)."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

import flowbars as fb
from flowbars import __version__
from flowbars.bars.constructor import _OUTPUT_COLUMNS, _STATE_SCHEMA_VERSION, _params_hash
from flowbars.calendars import SessionCalendar
from flowbars.core import BarContractError, TickInfo
from flowbars.schema import MinuteSchemaMapping, SchemaMapping

TICK_SCHEMA = SchemaMapping({"timestamp": "ts", "price": "px", "volume": "vol"})
MINUTE_SCHEMA = MinuteSchemaMapping(
    {
        "timestamp": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
)

UTC = datetime.timezone.utc


def _ticks(n: int = 40, start_ts: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "ts": start_ts + np.arange(n, dtype=np.int64) * 1_000,
            "px": 100.0 + np.cumsum(rng.normal(0.0, 0.05, n)),
            "vol": np.full(n, 10.0),
        }
    )


def _minutes(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype=np.int64) * 60_000,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.002,
            "low": np.minimum(opens, closes) * 0.998,
            "close": closes,
            "volume": np.full(n, 10.0),
        }
    )


def _ms(dt: datetime.datetime) -> int:
    return int(dt.timestamp() * 1_000)


class TestFrameContract:
    def test_tick_frame_columns_and_partial(self) -> None:
        bars = fb.VolumeBarConstructor(50.0, schema=TICK_SCHEMA).batch(_ticks())
        assert list(bars.columns) == _OUTPUT_COLUMNS
        assert bars["partial"].dtype == bool
        assert not bars["partial"].any()

    def test_minute_frame_columns_and_partial(self) -> None:
        bars = fb.MinuteRenkoBarConstructor(0.01, schema=MINUTE_SCHEMA).batch(_minutes())
        assert list(bars.columns) == _OUTPUT_COLUMNS
        assert not bars["partial"].any()

    def test_empty_frame_keeps_contract_columns(self) -> None:
        bars = fb.VolumeBarConstructor(1e12, schema=TICK_SCHEMA).batch(_ticks())
        assert bars.empty
        assert list(bars.columns) == _OUTPUT_COLUMNS
        assert bars.attrs["bar_type"] == "volume"

    def test_frame_metadata(self) -> None:
        ctor = fb.VolumeBarConstructor(50.0, schema=TICK_SCHEMA, stream_id="abc")
        bars = ctor.batch(_ticks())
        assert bars.attrs["bar_type"] == "volume"
        assert bars.attrs["input_kind"] == "tick"
        assert bars.attrs["engine_version"] == __version__
        assert bars.attrs["state_schema_version"] == _STATE_SCHEMA_VERSION
        assert bars.attrs["params_hash"] == _params_hash(ctor.get_params())

    def test_minute_frame_metadata(self) -> None:
        ctor = fb.MinuteVolatilityBarConstructor(0.01, schema=MINUTE_SCHEMA)
        bars = ctor.update_many(_minutes())
        assert bars.attrs["bar_type"] == "volatility"
        assert bars.attrs["input_kind"] == "minute"
        assert bars.attrs["params_hash"] == _params_hash(ctor.get_params())


class TestSessionPartial:
    def test_session_force_close_sets_partial_and_close_ts(self) -> None:
        cal = SessionCalendar(9, 30, 16, 0)
        schema = SchemaMapping({"timestamp": "ts", "price": "px", "volume": "vol"})
        ctor = fb.TickBarConstructor(threshold=100, schema=schema, calendar=cal)

        day = datetime.datetime(2024, 1, 8, tzinfo=UTC)  # Monday
        ts1 = _ms(day.replace(hour=9, minute=31))
        ts2 = _ms(day.replace(hour=9, minute=32))
        ts3 = _ms(day.replace(hour=16, minute=1))

        assert ctor.update(TickInfo(timestamp=ts1, price=10.0, volume=1.0)) is None
        assert ctor.update(TickInfo(timestamp=ts2, price=11.0, volume=1.0)) is None
        bar = ctor.update(TickInfo(timestamp=ts3, price=12.0, volume=1.0))
        assert bar is not None
        assert bar.partial is True
        assert bar.open_ts == ts1
        assert bar.close_ts == ts2  # last in-session tick, not the boundary tick
        assert bar.num_ticks == 2

    def test_threshold_bar_not_partial(self) -> None:
        ctor = fb.TickBarConstructor(threshold=2, schema=TICK_SCHEMA)
        assert ctor.update(TickInfo(timestamp=1_000, price=10.0, volume=1.0)) is None
        bar = ctor.update(TickInfo(timestamp=2_000, price=11.0, volume=1.0))
        assert bar is not None
        assert bar.partial is False


class TestValidateBarFrame:
    def _frame(self) -> pd.DataFrame:
        return fb.VolumeBarConstructor(50.0, schema=TICK_SCHEMA).batch(_ticks())

    def test_accepts_produced_frames(self) -> None:
        fb.validate_bar_frame(self._frame())
        fb.validate_bar_frame(
            fb.MinuteRenkoBarConstructor(0.01, schema=MINUTE_SCHEMA).batch(_minutes())
        )

    def test_accepts_numeric_extras(self) -> None:
        frame = self._frame()
        frame["spread"] = np.linspace(0.0, 1.0, len(frame))
        fb.validate_bar_frame(frame)

    def test_accepts_declared_spec(self) -> None:
        fb.validate_bar_frame(self._frame(), spec=fb.BarRegistry.get_spec("volume"))

    def test_missing_column_rejected(self) -> None:
        frame = self._frame().drop(columns=["close_ts"])
        with pytest.raises(BarContractError, match="missing required columns"):
            fb.validate_bar_frame(frame)

    def test_non_boolean_partial_rejected(self) -> None:
        frame = self._frame()
        frame["partial"] = frame["partial"].astype(int)
        with pytest.raises(BarContractError, match="partial"):
            fb.validate_bar_frame(frame)

    def test_non_numeric_extra_rejected(self) -> None:
        frame = self._frame()
        frame["comment"] = "x"
        with pytest.raises(BarContractError, match="must be numeric"):
            fb.validate_bar_frame(frame)

    def test_duplicate_open_ts_rejected(self) -> None:
        frame = self._frame()
        frame.loc[1, "open_ts"] = frame.loc[0, "open_ts"]
        with pytest.raises(BarContractError, match="open_ts must be unique"):
            fb.validate_bar_frame(frame)
        fb.validate_bar_frame(frame, require_unique_open_ts=False)

    def test_backwards_close_ts_rejected(self) -> None:
        frame = self._frame()
        frame.loc[0, "close_ts"] = frame.loc[0, "open_ts"] - 1
        with pytest.raises(BarContractError, match="close_ts must be"):
            fb.validate_bar_frame(frame)
