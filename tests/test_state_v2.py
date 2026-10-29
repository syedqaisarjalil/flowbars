"""Tests for the versioned (v2) JSON state contract.

v2 adds the bar-defining ``params`` and their SHA-256 hash, the engine
version, and the persisted ``warmup_bars`` / ``strict_ordering`` settings to
the serialized constructor state.  v1 states remain loadable.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import flowbars as fb
from flowbars import __version__
from flowbars.bars.accumulators import ImbalanceAccumulator, RunAccumulator
from flowbars.bars.constructor import BaseBarConstructor, _params_hash
from flowbars.core import MinuteInfo, StateValidationError, TickDataError, TickInfo
from flowbars.schema import MinuteSchemaMapping, SchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator

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


def _ticks(n: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype=np.int64) * 1_000,
            "px": 100.0 + np.cumsum(rng.normal(0.0, 0.05, n)),
            "vol": rng.integers(1, 10, n).astype(float),
        }
    )


def _minutes(n: int = 60, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, n)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype=np.int64) * 60_000,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.001,
            "low": np.minimum(opens, closes) * 0.999,
            "close": closes,
            "volume": rng.integers(1, 30, n).astype(float),
        }
    )


class TestStateContents:
    def test_v2_keys_and_metadata(self) -> None:
        ctor = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA, warmup_bars=2)
        for i in range(20):
            ctor.update(TickInfo(timestamp=i * 1_000, price=10.0 + i, volume=50.0))
        state = ctor.get_state()

        assert state["schema_version"] == 2
        assert state["engine_version"] == __version__
        assert state["input_kind"] == "tick"
        assert state["bar_type"] == "volume"
        assert state["warmup_bars"] == 2
        assert state["strict_ordering"] is False
        assert state["params"]["bar_type"] == "volume"
        assert state["params"]["accumulator"]["bar_type"] == "volume"
        assert state["params_hash"] == _params_hash(state["params"])
        assert len(state["params_hash"]) == 64
        assert "last_timestamp" in state
        assert "last_watermark" in state
        assert "in_session" in state
        assert "bars_emitted" in state
        assert "accumulator" in state and "threshold_estimator" in state

    def test_state_is_json_safe_without_nan(self) -> None:
        ctor = fb.MinuteHybridBarConstructor(12_000.0, 0.05, schema=MINUTE_SCHEMA, warmup_bars=1)
        for rec in _minutes().to_dict("records"):
            ctor.update(
                MinuteInfo(
                    timestamp=int(rec["ts"]),
                    open=float(rec["open"]),
                    high=float(rec["high"]),
                    low=float(rec["low"]),
                    close=float(rec["close"]),
                    volume=float(rec["volume"]),
                )
            )
        state = ctor.get_state()
        text = json.dumps(state, allow_nan=False)
        assert json.loads(text) == state

    def test_params_hash_is_stable_and_order_independent(self) -> None:
        a = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA)
        b = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA)
        assert a.get_params() == b.get_params()

        acc = ImbalanceAccumulator(bar_type="imbalance_volume", metric="volume")
        est = EWMAThresholdEstimator("imbalance", span=10.0, halflife=None)
        ctor = BaseBarConstructor(acc, est)
        params = ctor.get_params()
        reordered = dict(reversed(list(params.items())))
        assert _params_hash(params) == _params_hash(reordered)


class TestResume:
    def test_tick_resume_round_trip(self) -> None:
        df = _ticks()
        full = fb.VolumeBarConstructor(60.0, schema=TICK_SCHEMA).batch(df)

        ctor = fb.VolumeBarConstructor(60.0, schema=TICK_SCHEMA)
        first = ctor.batch(df.iloc[:20])
        state = json.loads(json.dumps(ctor.get_state(), allow_nan=False))
        resumed = fb.VolumeBarConstructor.from_state(state, schema=TICK_SCHEMA)
        second = resumed.batch(df.iloc[20:])

        combined = pd.concat([first, second], ignore_index=True)
        pd.testing.assert_frame_equal(full.reset_index(drop=True), combined)

    def test_minute_hybrid_resume_round_trip(self) -> None:
        df = _minutes()
        kwargs = {"schema": MINUTE_SCHEMA}
        full = fb.MinuteHybridBarConstructor(12_000.0, 0.04, **kwargs).batch(df)

        ctor = fb.MinuteHybridBarConstructor(12_000.0, 0.04, **kwargs)
        first = ctor.batch(df.iloc[:25])
        state = json.loads(json.dumps(ctor.get_state(), allow_nan=False))
        resumed = fb.MinuteHybridBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
        second = resumed.batch(df.iloc[25:])

        combined = pd.concat([first, second], ignore_index=True)
        pd.testing.assert_frame_equal(full.reset_index(drop=True), combined)
        assert resumed.get_state()["params"]["accumulator"]["dollar_threshold"] == 12_000.0
        assert resumed.get_state()["params"]["accumulator"]["volatility_threshold"] == 0.04

    def test_run_min_run_length_survives_round_trip(self) -> None:
        df = _ticks(n=80)
        ctor = fb.RunVolumeBarConstructor(
            initial_ewa_t=100.0,
            initial_ewa_proportion=0.5,
            min_run_length=2,
            schema=TICK_SCHEMA,
        )
        ctor.batch(df.iloc[:30])
        state = json.loads(json.dumps(ctor.get_state(), allow_nan=False))
        assert state["params"]["min_run_length"] == 2

        resumed = fb.RunVolumeBarConstructor.from_state(state, schema=TICK_SCHEMA)
        assert resumed._min_run_length == 2
        assert resumed.get_params() == ctor.get_params()

    def test_warmup_and_ordering_persisted(self) -> None:
        ctor = fb.VolumeBarConstructor(
            500.0, schema=TICK_SCHEMA, warmup_bars=2, strict_ordering=True, watermark=None
        )
        for i in range(4):
            ctor.update(TickInfo(timestamp=i * 1_000, price=10.0, volume=200.0))
        state = ctor.get_state()

        resumed = fb.VolumeBarConstructor.from_state(state, schema=TICK_SCHEMA)
        assert resumed._warmup_bars == 2
        assert resumed._strict_ordering is True
        with pytest.raises(TickDataError, match="Out-of-order"):
            resumed.update(TickInfo(timestamp=2_000, price=10.0, volume=200.0))


class TestValidation:
    def _state(self, **kwargs: object) -> dict:
        ctor = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA, **kwargs)
        ctor.update(TickInfo(timestamp=1_000, price=10.0, volume=500.0))
        return ctor.get_state()

    def test_threshold_mismatch_requires_rebuild(self) -> None:
        state = self._state()
        ctor = fb.VolumeBarConstructor(900.0, schema=TICK_SCHEMA)
        with pytest.raises(StateValidationError, match="params mismatch"):
            ctor.load_state(state)

    def test_bar_type_param_mismatch(self) -> None:
        state = self._state()
        ctor = fb.DollarBarConstructor(500.0, schema=TICK_SCHEMA)
        with pytest.raises(StateValidationError, match="bar_type mismatch"):
            ctor.load_state(state)

    def test_min_run_length_mismatch(self) -> None:
        ctor = fb.RunTickBarConstructor(
            initial_ewa_t=20.0, initial_ewa_proportion=0.5, min_run_length=2, schema=TICK_SCHEMA
        )
        ctor.update(TickInfo(timestamp=1_000, price=10.0, volume=1.0, side=1.0))
        state = ctor.get_state()

        other = fb.RunTickBarConstructor(
            initial_ewa_t=20.0, initial_ewa_proportion=0.5, min_run_length=5, schema=TICK_SCHEMA
        )
        with pytest.raises(StateValidationError, match="params mismatch"):
            other.load_state(state)

    def test_estimator_family_mismatch(self) -> None:
        acc = RunAccumulator(bar_type="run_tick", metric="tick")
        est = EWMAThresholdEstimator("run", span=10.0)
        ctor = BaseBarConstructor(acc, est)
        state = ctor.get_state()

        acc2 = RunAccumulator(bar_type="run_tick", metric="tick")
        est2 = EWMAThresholdEstimator("imbalance", span=10.0)
        ctor2 = BaseBarConstructor(acc2, est2)
        with pytest.raises(StateValidationError, match="params mismatch"):
            ctor2.load_state(state)

    def test_tampered_params_hash_rejected(self) -> None:
        state = self._state()
        state["params"]["accumulator"]["bar_type"] = "dollar"
        ctor = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA)
        with pytest.raises(StateValidationError, match="corrupt"):
            ctor.load_state(state)

    def test_missing_params_v2_rejected(self) -> None:
        state = self._state()
        del state["params"]
        del state["params_hash"]
        ctor = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA)
        with pytest.raises(StateValidationError, match="missing 'params'"):
            ctor.load_state(state)

    def test_engine_version_incompatibility_rejected(self) -> None:
        state = self._state()
        state["engine_version"] = "9.0.0"
        ctor = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA)
        with pytest.raises(StateValidationError, match="engine version"):
            ctor.load_state(state)

    def test_future_schema_version_requires_rebuild(self) -> None:
        state = self._state()
        state["schema_version"] = 999
        ctor = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA)
        with pytest.raises(StateValidationError, match="newer version"):
            ctor.load_state(state)


class TestV1Compatibility:
    def _strip_v2(self, state: dict) -> dict:
        v1 = {
            k: v
            for k, v in state.items()
            if k
            not in {"params", "params_hash", "engine_version", "warmup_bars", "strict_ordering"}
        }
        v1["schema_version"] = 1
        return v1

    def test_v1_state_resumes_identically(self) -> None:
        df = _ticks(n=60)
        full = fb.VolumeBarConstructor(60.0, schema=TICK_SCHEMA).batch(df)

        ctor = fb.VolumeBarConstructor(60.0, schema=TICK_SCHEMA)
        first = ctor.batch(df.iloc[:25])
        v1 = json.loads(json.dumps(self._strip_v2(ctor.get_state()), allow_nan=False))

        resumed = fb.VolumeBarConstructor.from_state(v1, schema=TICK_SCHEMA)
        second = resumed.batch(df.iloc[25:])
        combined = pd.concat([first, second], ignore_index=True)
        pd.testing.assert_frame_equal(full.reset_index(drop=True), combined)
        # A resumed constructor writes the current schema version again.
        assert resumed.get_state()["schema_version"] == 2

    def test_v1_state_skips_params_validation(self) -> None:
        ctor = fb.VolumeBarConstructor(500.0, schema=TICK_SCHEMA)
        ctor.update(TickInfo(timestamp=1_000, price=10.0, volume=100.0))
        v1 = self._strip_v2(ctor.get_state())

        other = fb.VolumeBarConstructor(900.0, schema=TICK_SCHEMA)
        other.load_state(v1)  # v1 has no params — accepted for backward compat
