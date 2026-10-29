"""Tests for the bar-type registry spec API (``list_specs`` / ``describe``)."""

from __future__ import annotations

import json

import pytest

import flowbars as fb
from flowbars.bars.constructor import _OUTPUT_COLUMNS, _STATE_SCHEMA_VERSION
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.bars.registry import BarRegistry
from flowbars.bars.spec import BarTypeSpec, BarTypeSpecRegistry

TICK_TYPES = [
    "dollar",
    "imbalance_dollar",
    "imbalance_tick",
    "imbalance_volume",
    "run_dollar",
    "run_tick",
    "run_volume",
    "tick",
    "time",
    "volume",
]
MINUTE_TYPES = [
    "minute_dollar",
    "minute_hybrid",
    "minute_imbalance_dollar",
    "minute_imbalance_volume",
    "minute_range",
    "minute_renko",
    "minute_run_dollar",
    "minute_run_volume",
    "minute_volatility",
    "minute_volume",
]


class TestSpecInventory:
    def test_list_specs_matches_registry(self) -> None:
        assert [s.bar_type for s in BarRegistry.list_specs()] == TICK_TYPES
        assert [s.bar_type for s in MinuteBarRegistry.list_specs()] == MINUTE_TYPES
        assert BarTypeSpec is not None

    def test_every_spec_is_complete_and_json_safe(self) -> None:
        for registry, names, kind in (
            (BarRegistry, TICK_TYPES, "tick"),
            (MinuteBarRegistry, MINUTE_TYPES, "minute"),
        ):
            for name in names:
                spec = registry.get_spec(name)
                assert isinstance(spec, BarTypeSpec)
                assert spec.bar_type == name
                assert spec.input_kinds == (kind,)
                assert spec.output_columns == tuple(_OUTPUT_COLUMNS)
                assert spec.streaming is True
                assert spec.state_schema_version == _STATE_SCHEMA_VERSION
                assert spec.session_policies
                assert spec.volume_types
                assert spec.warmup_minutes >= 0
                assert isinstance(spec.extras_columns, dict)
                assert spec.description
                assert spec.batch_function == registry.get_batch_function(name).__name__
                assert json.loads(json.dumps(spec.to_dict())) == spec.to_dict()

    def test_describe_round_trips(self) -> None:
        described = BarRegistry.describe("volume")
        assert described["bar_type"] == "volume"
        assert described["input_kinds"] == ["tick"]
        assert described["output_columns"] == list(_OUTPUT_COLUMNS)
        assert described["params_schema"]["required"] == ["threshold"]

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown bar type"):
            BarRegistry.get_spec("not_a_type")
        with pytest.raises(KeyError, match="Unknown minute bar type"):
            MinuteBarRegistry.describe("minute_not_a_type")


class TestParamsSchema:
    def test_required_params_match_signatures(self) -> None:
        required = {
            name: BarRegistry.get_spec(name).params_schema["required"] for name in TICK_TYPES
        }
        assert required["tick"] == ["threshold"]
        assert required["volume"] == ["threshold"]
        assert required["dollar"] == ["threshold"]
        assert required["time"] == ["interval_ms"]
        assert required["imbalance_tick"] == []
        assert required["run_tick"] == []

        assert MinuteBarRegistry.get_spec("minute_hybrid").params_schema["required"] == [
            "dollar_threshold",
            "volatility_threshold",
        ]
        assert MinuteBarRegistry.get_spec("minute_renko").params_schema["required"] == ["threshold"]

    def test_scalar_and_optional_types(self) -> None:
        schema = BarRegistry.get_spec("imbalance_volume").params_schema
        props = schema["properties"]
        assert props["span"]["type"] == "number"
        assert props["span"]["default"] == 20.0
        assert props["halflife"]["type"] == ["number", "null"]
        assert props["initial_ewa_t"]["type"] == "number"
        assert props["backend"]["enum"] == ["python", "numba"]
        assert props["warmup_bars"]["type"] == "integer"
        assert props["watermark"]["type"] == ["string", "null"]
        assert schema["additionalProperties"] is False

    def test_run_specs_expose_min_run_length(self) -> None:
        for name in ("run_tick", "run_volume", "run_dollar"):
            props = BarRegistry.get_spec(name).params_schema["properties"]
            assert props["min_run_length"]["type"] == "integer"

    def test_minute_dollar_family_marked_approximate(self) -> None:
        approximate = {s.bar_type for s in MinuteBarRegistry.list_specs() if s.approximate}
        assert approximate == {
            "minute_dollar",
            "minute_hybrid",
            "minute_imbalance_dollar",
            "minute_run_dollar",
        }
        assert not any(s.approximate for s in BarRegistry.list_specs())


class TestAddingBarTypeNeedsNoDownstreamChanges:
    def test_new_registration_gets_spec_automatically(self) -> None:
        class _DummyConstructor:
            """Dummy bar type for registry discovery tests."""

            input_kind = "tick"

            def __init__(
                self,
                threshold: float,
                warmup_bars: int = 0,
                calendar: object | None = None,
            ) -> None:
                self.threshold = threshold

        class _IsolatedRegistry(BarTypeSpecRegistry):
            _constructors: dict = {}
            _batch_functions: dict = {}
            _specs: dict = {}

        def _batch(df: object) -> object:
            return df

        _IsolatedRegistry.register("dummy", _DummyConstructor, batch_fn=_batch)

        spec = _IsolatedRegistry.get_spec("dummy")
        assert spec.bar_type == "dummy"
        assert spec.input_kinds == ("tick",)
        assert spec.params_schema["required"] == ["threshold"]
        assert spec.params_schema["properties"]["threshold"]["type"] == "number"
        assert "calendar" not in spec.params_schema["properties"]
        assert spec.batch_function == "_batch"
        assert spec.output_columns == tuple(_OUTPUT_COLUMNS)
        assert _IsolatedRegistry.describe("dummy")["bar_type"] == "dummy"

    def test_top_level_exports(self) -> None:
        assert fb.BarTypeSpec is BarTypeSpec
        assert BarRegistry.describe("tick")["input_kinds"] == ["tick"]
