"""Output-contract metadata and ergonomics (P0-4, P1-5/6/8, P2-9/10/11).

Covers:

* session force-closed (``partial``) bars end at the session boundary;
* ``core_columns`` / ``extras_columns`` partition the output contract;
* ``spec_version`` is advertised in the spec and frame attrs;
* the recommended-warmup metadata is explicit (``0`` = no warmup required);
* the public ``params_hash`` helper is canonical and exported;
* non-integer timestamps are rejected with an explicit ``SchemaError``;
* ``batch`` / ``update_many`` accept an opt-in ``validate=True`` flag.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

import flowbars as fb
from flowbars.bars.constructor import _OUTPUT_COLUMNS
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.bars.registry import BarRegistry
from flowbars.bars.spec import BarTypeSpec
from flowbars.core import BarContractError, SchemaError, TickInfo

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


def _ms(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> int:
    return int(datetime.datetime(y, mo, d, h, mi, tzinfo=datetime.timezone.utc).timestamp() * 1000)


def _ticks_df(n: int, start: int = BASE_TS, step: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [start + i * step for i in range(n)],
            "price": [100.0 + (i % 3) for i in range(n)],
            "volume": [1.0] * n,
        }
    )


def _minutes_df(n: int) -> pd.DataFrame:
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "ts": [BASE_TS + i * MINUTE_MS for i in range(n)],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * n,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Partial-bar close_ts = session boundary (P0-4)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPartialBarCloseTs:
    def test_session_calendar_leaving_session_ends_at_close(self) -> None:
        cal = fb.SessionCalendar(9, 30, 16, 0)
        ctor = fb.TickBarConstructor(threshold=100, schema=TICK_SCHEMA, calendar=cal)

        assert (
            ctor.update(TickInfo(timestamp=_ms(2024, 1, 8, 15, 59), price=10.0, volume=1.0)) is None
        )
        bar = ctor.update(TickInfo(timestamp=_ms(2024, 1, 8, 16, 1), price=11.0, volume=1.0))

        assert bar is not None and bar.partial is True
        assert bar.close_ts == _ms(2024, 1, 8, 16, 0)
        assert bar.close_ts > bar.open_ts

    def test_session_calendar_entering_session_ends_at_open(self) -> None:
        cal = fb.SessionCalendar(9, 30, 16, 0)
        ctor = fb.TickBarConstructor(threshold=100, schema=TICK_SCHEMA, calendar=cal)

        # An after-hours bar accumulates until the next session opens.
        assert (
            ctor.update(TickInfo(timestamp=_ms(2024, 1, 8, 16, 1), price=10.0, volume=1.0)) is None
        )
        bar = ctor.update(TickInfo(timestamp=_ms(2024, 1, 9, 9, 30), price=11.0, volume=1.0))

        assert bar is not None and bar.partial is True
        assert bar.num_ticks == 1
        assert bar.close_ts == _ms(2024, 1, 9, 9, 30)

    def test_weekday_calendar_partial_ends_at_midnight(self) -> None:
        cal = fb.WeekdayCalendar()
        ctor = fb.TickBarConstructor(threshold=100, schema=TICK_SCHEMA, calendar=cal)

        assert (
            ctor.update(TickInfo(timestamp=_ms(2024, 1, 5, 23, 59), price=10.0, volume=1.0)) is None
        )
        bar = ctor.update(TickInfo(timestamp=_ms(2024, 1, 6, 0, 0), price=11.0, volume=1.0))

        assert bar is not None and bar.partial is True
        assert bar.close_ts == _ms(2024, 1, 6, 0, 0)

    def test_threshold_bars_keep_last_input_ts(self) -> None:
        ctor = fb.TickBarConstructor(threshold=2, schema=TICK_SCHEMA)
        assert ctor.update(TickInfo(timestamp=1000, price=10.0, volume=1.0)) is None
        bar = ctor.update(TickInfo(timestamp=2000, price=11.0, volume=1.0))
        assert bar is not None and bar.partial is False
        assert bar.close_ts == 2000


# ═══════════════════════════════════════════════════════════════════════════════
# core/extras partition + spec version (P1-5, P1-6)
# ═══════════════════════════════════════════════════════════════════════════════


def _all_specs() -> list[fb.BarTypeSpec]:
    return [BarRegistry.get_spec(n) for n in BarRegistry.list()] + [
        MinuteBarRegistry.get_spec(n) for n in MinuteBarRegistry.list()
    ]


class TestSpecColumns:
    def test_every_type_partitions_core_and_extras(self) -> None:
        for spec in _all_specs():
            assert tuple(spec.output_columns) == tuple(spec.core_columns) + tuple(
                spec.extras_columns
            )
            assert spec.extras_columns == {}
            assert set(spec.core_columns) == set(_OUTPUT_COLUMNS)

    def test_dollar_value_vwap_num_ticks_are_core(self) -> None:
        for spec in _all_specs():
            for col in ("dollar_value", "vwap", "num_ticks"):
                assert col in spec.core_columns
                assert col not in spec.extras_columns

    def test_inconsistent_partition_rejected(self) -> None:
        with pytest.raises(ValueError, match="core_columns"):
            BarTypeSpec(
                bar_type="probe",
                input_kinds=("tick",),
                params_schema={},
                output_columns=("open",),
                core_columns=("bar_id",),
            )

    def test_spec_version_in_spec_and_frame_attrs(self) -> None:
        spec = BarRegistry.get_spec("volume")
        assert spec.spec_version == 1
        payload = spec.to_dict()
        assert payload["spec_version"] == 1
        assert "core_columns" in payload

        ctor = fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA)
        frame = ctor.update_many(_ticks_df(4))
        assert frame.attrs["spec_version"] == spec.spec_version
        assert frame.attrs["params_hash"] == fb.params_hash(ctor.get_params())
        assert frame.attrs["state_schema_version"] == spec.state_schema_version


# ═══════════════════════════════════════════════════════════════════════════════
# Warmup metadata (P1-8)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWarmupMetadata:
    def test_builtin_types_require_no_warmup(self) -> None:
        for registry in (BarRegistry, MinuteBarRegistry):
            for name in registry.list():
                spec = registry.get_spec(name)
                ctor_cls = registry.get_constructor(name)
                # 0 is an intentional "no warmup required" advertisement.
                assert spec.warmup_minutes == 0
                assert ctor_cls.recommended_warmup_minutes == 0
                assert isinstance(spec.to_dict()["warmup_minutes"], int)

    def test_runtime_param_is_distinct_from_metadata(self) -> None:
        for registry in (BarRegistry, MinuteBarRegistry):
            for name in registry.list():
                props = registry.get_spec(name).params_schema["properties"]
                # Runtime knob (N minutes) defaults to disabled …
                assert props["warmup_minutes"] == {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "default": None,
                }
                # … while bar-count warmup defaults to 0 (disabled).
                assert props["warmup_bars"]["default"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Public params_hash (P2-9)
# ═══════════════════════════════════════════════════════════════════════════════


class TestParamsHash:
    def test_exported_and_canonical(self) -> None:
        assert "params_hash" in fb.__all__
        digest = fb.params_hash({"b": 2, "a": 1})
        assert len(digest) == 64
        assert digest == fb.params_hash({"a": 1, "b": 2})

    def test_distinguishes_values(self) -> None:
        assert fb.params_hash({"threshold": 1.0}) != fb.params_hash({"threshold": 2.0})

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError):
            fb.params_hash({"threshold": float("nan")})


# ═══════════════════════════════════════════════════════════════════════════════
# Timestamp dtype validation (P2-10)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTimestampValidation:
    def test_tz_aware_datetime64_rejected(self) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
                "price": [100.0, 101.0, 102.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        with pytest.raises(SchemaError, match="integer Unix epoch milliseconds"):
            fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA).batch(df)

    def test_tz_naive_datetime64_rejected(self) -> None:
        df = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=3, freq="1min"),
                "price": [100.0, 101.0, 102.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        with pytest.raises(SchemaError, match="integer Unix epoch milliseconds"):
            fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA).batch(df)

    def test_minute_path_rejects_datetime64(self) -> None:
        df = _minutes_df(3).assign(ts=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"))
        with pytest.raises(SchemaError, match="integer Unix epoch milliseconds"):
            fb.MinuteVolumeBarConstructor(1.0, schema=MINUTE_SCHEMA).batch(df)

    def test_integral_float_timestamps_accepted(self) -> None:
        int_df = _ticks_df(6)
        float_df = int_df.assign(ts=int_df["ts"].astype(float))
        int_bars = fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA).update_many(int_df)
        float_bars = fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA).update_many(float_df)
        pd.testing.assert_frame_equal(int_bars, float_bars)

    def test_non_integral_float_rejected(self) -> None:
        df = _ticks_df(3).assign(ts=[1000.5, 2000.5, 3000.5])
        with pytest.raises(SchemaError, match="integer Unix epoch milliseconds"):
            fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA).batch(df)


# ═══════════════════════════════════════════════════════════════════════════════
# Opt-in output validation (P2-11)
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidateFlag:
    def test_validate_flag_passes_on_good_frame(self) -> None:
        df = _ticks_df(6)
        ctor = fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA)
        assert len(ctor.batch(df, validate=True)) == 3
        assert (
            len(fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA).update_many(df, validate=True))
            == 3
        )

    def test_validate_flag_invokes_contract_check(self, monkeypatch) -> None:
        from flowbars.bars import output

        calls: list[int] = []

        def boom(frame: pd.DataFrame) -> None:
            calls.append(len(frame))
            raise BarContractError("injected")

        monkeypatch.setattr(output, "validate_bar_frame", boom)

        df = _ticks_df(4)
        ctor = fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA)
        with pytest.raises(BarContractError, match="injected"):
            ctor.batch(df, validate=True)
        assert calls

        ctor2 = fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA)
        with pytest.raises(BarContractError, match="injected"):
            ctor2.update_many(df, validate=True)

    def test_validation_is_off_by_default(self, monkeypatch) -> None:
        from flowbars.bars import output

        def boom(frame: pd.DataFrame) -> None:
            raise BarContractError("should not run")

        monkeypatch.setattr(output, "validate_bar_frame", boom)
        df = _ticks_df(4)
        assert len(fb.VolumeBarConstructor(2.0, schema=TICK_SCHEMA).batch(df)) == 2
