"""Tests for volume/notional semantics: multiplier, volume_type, approximate.

The contract under test:

* ``multiplier`` scales every notional quantity (``dollar_value``, dollar
  thresholds, dollar-metric imbalance/run statistics) but never the VWAP,
  which stays a price.
* ``volume_type`` is metadata only — it is validated and carried through
  params/state/frame attrs without changing any arithmetic.
* Dollar-family minute bar types are flagged ``approximate``; their tick
  counterparts and volume-based types are not.
* Defaults (``multiplier=1.0``, ``volume_type="base"``) reproduce the
  pre-existing output exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import flowbars as fb
from flowbars.bars.accumulators import ImbalanceAccumulator
from flowbars.bars.minute.accumulators import MinuteHybridAccumulator, MinuteImbalanceAccumulator
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.bars.registry import BarRegistry
from flowbars.core import (
    VOLUME_TYPES,
    MinuteInfo,
    StateValidationError,
    TickInfo,
    validate_volume_semantics,
)
from flowbars.schema import MinuteSchemaMapping, SchemaMapping
from flowbars.thresholds import MinuteCalibrationHelper, StaticCalibrationHelper

TICK_SCHEMA = SchemaMapping({"timestamp": "ts", "price": "px", "volume": "vol"})
TICK_SCHEMA_SIDED = SchemaMapping(
    {"timestamp": "ts", "price": "px", "volume": "vol", "side": "side"}
)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _ticks(n: int = 24, seed: int = 3, with_side: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "ts": np.arange(n, dtype=np.int64) * 1_000,
        "px": 100.0 + np.cumsum(rng.normal(0.0, 0.05, n)),
        "vol": rng.integers(1, 10, n).astype(float),
    }
    if with_side:
        data["side"] = rng.choice([-1.0, 1.0], n)
    return pd.DataFrame(data)


def _minutes(n: int = 30, seed: int = 4) -> pd.DataFrame:
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


def _assert_frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for col in a.columns:
        if col == "bar_type":
            assert (a[col] == b[col]).all()
        elif col in ("bar_id", "num_ticks", "open_ts", "close_ts"):
            assert (a[col].astype(int) == b[col].astype(int)).all()
        else:
            assert np.allclose(
                np.asarray(a[col], dtype=np.float64),
                np.asarray(b[col], dtype=np.float64),
                rtol=1e-12,
                atol=1e-14,
            ), col


def _prefix(df: pd.DataFrame) -> list[str]:
    return [
        "bar_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "num_ticks",
        "open_ts",
        "close_ts",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_accepts_valid_semantics(self) -> None:
        assert validate_volume_semantics(2, "quote") == (2.0, "quote")
        assert validate_volume_semantics(1.0, "base") == (1.0, "base")
        for volume_type in VOLUME_TYPES:
            assert validate_volume_semantics(1.0, volume_type) == (1.0, volume_type)

    @pytest.mark.parametrize(
        "multiplier",
        [0, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "2"],
    )
    def test_rejects_invalid_multiplier(self, multiplier: object) -> None:
        with pytest.raises(ValueError):
            validate_volume_semantics(multiplier, "base")  # type: ignore[arg-type]

    @pytest.mark.parametrize("volume_type", ["lots", "BASE", "", "usd"])
    def test_rejects_unknown_volume_type(self, volume_type: str) -> None:
        with pytest.raises(ValueError):
            validate_volume_semantics(1.0, volume_type)

    def test_constructor_rejects_invalid_semantics(self) -> None:
        with pytest.raises(ValueError):
            fb.VolumeBarConstructor(threshold=10.0, multiplier=0.0)
        with pytest.raises(ValueError):
            fb.VolumeBarConstructor(threshold=10.0, volume_type="lots")
        with pytest.raises(ValueError):
            fb.MinuteVolumeBarConstructor(threshold=10.0, multiplier=-2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Tick path
# ═══════════════════════════════════════════════════════════════════════════════


class TestTickNotional:
    def test_volume_bars_scale_notional_but_not_vwap(self) -> None:
        df = _ticks()
        base = fb.VolumeBarConstructor(threshold=15.0, schema=TICK_SCHEMA).batch(df)
        scaled = fb.VolumeBarConstructor(threshold=15.0, multiplier=2.0, schema=TICK_SCHEMA).batch(
            df
        )

        # Volume is unscaled, so the bar partition is identical.
        _assert_frames_equal(base[_prefix(base)], scaled[_prefix(scaled)])
        assert np.allclose(scaled["dollar_value"], 2.0 * base["dollar_value"])
        assert np.allclose(scaled["vwap"], base["vwap"])

    def test_dollar_bars_threshold_is_scaled_notional(self) -> None:
        df = _ticks()
        base = fb.DollarBarConstructor(threshold=300.0, schema=TICK_SCHEMA).batch(df)
        scaled = fb.DollarBarConstructor(threshold=600.0, multiplier=2.0, schema=TICK_SCHEMA).batch(
            df
        )

        _assert_frames_equal(base[_prefix(base)], scaled[_prefix(scaled)])
        assert np.allclose(scaled["dollar_value"], 2.0 * base["dollar_value"])
        assert np.allclose(scaled["vwap"], base["vwap"])

    def test_dollar_imbalance_metric_scales(self) -> None:
        acc = ImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
        acc.configure_volume(2.5, "contracts")
        acc.add_tick(TickInfo(timestamp=1, price=10.0, volume=4.0, side=1.0))
        assert acc.get_state()["signed_imbalance"] == pytest.approx(10.0 * 4.0 * 2.5)
        assert acc.get_params()["multiplier"] == 2.5
        assert acc.get_params()["volume_type"] == "contracts"

    def test_dollar_bars_numba_matches_python_with_multiplier(self) -> None:
        pytest.importorskip("numba")
        from flowbars.bars.numba_backend import _NUMBA_AVAILABLE

        if not _NUMBA_AVAILABLE:
            pytest.skip("numba is not available")

        df = _ticks()
        py = fb.DollarBarConstructor(
            threshold=500.0, schema=TICK_SCHEMA, multiplier=3.5, volume_type="quote"
        ).batch(df)
        nb = fb.DollarBarConstructor(
            threshold=500.0,
            schema=TICK_SCHEMA,
            backend="numba",
            multiplier=3.5,
            volume_type="quote",
        ).batch(df)
        _assert_frames_equal(py, nb)


# ═══════════════════════════════════════════════════════════════════════════════
# Minute path
# ═══════════════════════════════════════════════════════════════════════════════


class TestMinuteNotional:
    def test_minute_volume_bars_scale_notional_but_not_vwap(self) -> None:
        df = _minutes()
        base = fb.MinuteVolumeBarConstructor(threshold=200.0, schema=MINUTE_SCHEMA).batch(df)
        scaled = fb.MinuteVolumeBarConstructor(
            threshold=200.0, schema=MINUTE_SCHEMA, multiplier=2.0
        ).batch(df)

        _assert_frames_equal(base[_prefix(base)], scaled[_prefix(scaled)])
        assert np.allclose(scaled["dollar_value"], 2.0 * base["dollar_value"])
        assert np.allclose(scaled["vwap"], base["vwap"])

    def test_minute_dollar_bars_threshold_is_scaled_notional(self) -> None:
        df = _minutes()
        base = fb.MinuteDollarBarConstructor(threshold=20_000.0, schema=MINUTE_SCHEMA).batch(df)
        scaled = fb.MinuteDollarBarConstructor(
            threshold=60_000.0, schema=MINUTE_SCHEMA, multiplier=3.0
        ).batch(df)

        _assert_frames_equal(base[_prefix(base)], scaled[_prefix(scaled)])
        assert np.allclose(scaled["dollar_value"], 3.0 * base["dollar_value"])
        assert np.allclose(scaled["vwap"], base["vwap"])

    def test_minute_dollar_imbalance_metric_scales(self) -> None:
        acc = MinuteImbalanceAccumulator(bar_type="imbalance_dollar", metric="dollar")
        acc.configure_volume(4.0, "shares")
        acc.add_minute(
            MinuteInfo(
                timestamp=0,
                open=99.0,
                high=101.0,
                low=98.0,
                close=100.0,
                volume=5.0,
                side=1.0,
            )
        )
        assert acc.get_state()["signed_imbalance"] == pytest.approx(100.0 * 5.0 * 4.0)

    def test_hybrid_accumulator_scales_dollar_leg_only(self) -> None:
        acc = MinuteHybridAccumulator(
            bar_type="hybrid", dollar_threshold=1e9, volatility_threshold=1e9
        )
        acc.configure_volume(2.0, "quote")
        acc.add_minute(
            MinuteInfo(
                timestamp=0,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10.0,
            )
        )
        state = acc.get_state()
        assert state["cum_dollar"] == pytest.approx(2.0 * 100.0 * 10.0)
        assert state["cum_vol"] == 0.0  # first minute has no return

    def test_minute_dollar_bars_numba_matches_python_with_multiplier(self) -> None:
        pytest.importorskip("numba")
        from flowbars.bars.numba_backend import _NUMBA_AVAILABLE

        if not _NUMBA_AVAILABLE:
            pytest.skip("numba is not available")

        df = _minutes()
        py = fb.MinuteDollarBarConstructor(
            threshold=25_000.0, schema=MINUTE_SCHEMA, multiplier=2.5
        ).batch(df)
        nb = fb.MinuteDollarBarConstructor(
            threshold=25_000.0, schema=MINUTE_SCHEMA, backend="numba", multiplier=2.5
        ).batch(df)
        _assert_frames_equal(py, nb)

    def test_minute_hybrid_numba_matches_python_with_multiplier(self) -> None:
        pytest.importorskip("numba")
        from flowbars.bars.numba_backend import _NUMBA_AVAILABLE

        if not _NUMBA_AVAILABLE:
            pytest.skip("numba is not available")

        df = _minutes()
        kwargs = {"dollar_threshold": 30_000.0, "volatility_threshold": 0.01, "multiplier": 2.0}
        py = fb.MinuteHybridBarConstructor(schema=MINUTE_SCHEMA, **kwargs).batch(df)
        nb = fb.MinuteHybridBarConstructor(schema=MINUTE_SCHEMA, backend="numba", **kwargs).batch(
            df
        )
        _assert_frames_equal(py, nb)


# ═══════════════════════════════════════════════════════════════════════════════
# Metadata: params, state, frame attrs, specs
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_frame_attrs_carry_semantics(self) -> None:
        df = _ticks()
        frame = fb.DollarBarConstructor(
            threshold=300.0, schema=TICK_SCHEMA, multiplier=2.0, volume_type="contracts"
        ).batch(df)
        assert frame.attrs["multiplier"] == 2.0
        assert frame.attrs["volume_type"] == "contracts"
        assert frame.attrs["approximate"] is False

    def test_minute_dollar_frame_flagged_approximate(self) -> None:
        frame = fb.MinuteDollarBarConstructor(threshold=20_000.0, schema=MINUTE_SCHEMA).batch(
            _minutes()
        )
        assert frame.attrs["approximate"] is True

    def test_defaults_appear_in_params(self) -> None:
        ctor = fb.VolumeBarConstructor(threshold=10.0)
        acc_params = ctor.get_params()["accumulator"]
        assert acc_params["multiplier"] == 1.0
        assert acc_params["volume_type"] == "base"
        assert ctor.multiplier == 1.0
        assert ctor.volume_type == "base"

    def test_state_roundtrip_preserves_semantics(self) -> None:
        df = _ticks()
        ctor = fb.DollarBarConstructor(
            threshold=300.0, schema=TICK_SCHEMA, multiplier=2.0, volume_type="quote"
        )
        first = ctor.batch(df.iloc[:12])
        state = json.loads(json.dumps(ctor.get_state()))

        assert state["params"]["accumulator"]["multiplier"] == 2.0
        assert state["params"]["accumulator"]["volume_type"] == "quote"
        assert state["accumulator"]["multiplier"] == 2.0
        assert state["accumulator"]["volume_type"] == "quote"

        resumed = fb.DollarBarConstructor.from_state(state, schema=TICK_SCHEMA)
        assert resumed.multiplier == 2.0
        assert resumed.volume_type == "quote"
        second = resumed.batch(df.iloc[12:])

        full = fb.DollarBarConstructor(
            threshold=300.0, schema=TICK_SCHEMA, multiplier=2.0, volume_type="quote"
        ).batch(df)
        _assert_frames_equal(pd.concat([first, second], ignore_index=True), full)

    def test_minute_state_roundtrip_preserves_semantics(self) -> None:
        df = _minutes()
        ctor = fb.MinuteDollarBarConstructor(
            threshold=20_000.0, schema=MINUTE_SCHEMA, multiplier=3.0, volume_type="tick"
        )
        first = ctor.batch(df.iloc[:14])
        state = json.loads(json.dumps(ctor.get_state()))
        resumed = fb.MinuteDollarBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
        assert resumed.multiplier == 3.0
        assert resumed.volume_type == "tick"
        second = resumed.batch(df.iloc[14:])

        full = fb.MinuteDollarBarConstructor(
            threshold=20_000.0, schema=MINUTE_SCHEMA, multiplier=3.0, volume_type="tick"
        ).batch(df)
        _assert_frames_equal(pd.concat([first, second], ignore_index=True), full)

    def test_resume_with_different_multiplier_is_rejected(self) -> None:
        ctor = fb.DollarBarConstructor(
            threshold=300.0, schema=TICK_SCHEMA, multiplier=2.0, volume_type="quote"
        )
        ctor.batch(_ticks().iloc[:8])
        state = ctor.get_state()

        with pytest.raises(StateValidationError):
            fb.DollarBarConstructor(
                threshold=300.0, schema=TICK_SCHEMA, multiplier=3.0, volume_type="quote"
            ).load_state(state)

        with pytest.raises(StateValidationError):
            fb.DollarBarConstructor(
                threshold=300.0, schema=TICK_SCHEMA, multiplier=2.0, volume_type="shares"
            ).load_state(state)

    def test_specs_expose_semantics(self) -> None:
        props = BarRegistry.get_spec("volume").params_schema["properties"]
        assert props["multiplier"]["default"] == 1.0
        assert props["volume_type"]["default"] == "base"
        assert BarRegistry.get_spec("volume").approximate is False
        assert BarRegistry.get_spec("dollar").approximate is False

    @pytest.mark.parametrize(
        ("name", "approximate"),
        [
            ("minute_volume", False),
            ("minute_dollar", True),
            ("minute_imbalance_volume", False),
            ("minute_imbalance_dollar", True),
            ("minute_run_volume", False),
            ("minute_run_dollar", True),
            ("minute_volatility", False),
            ("minute_range", False),
            ("minute_renko", False),
            ("minute_hybrid", True),
        ],
    )
    def test_minute_specs_flag_approximate(self, name: str, approximate: bool) -> None:
        assert MinuteBarRegistry.get_spec(name).approximate is approximate


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration honors the multiplier
# ═══════════════════════════════════════════════════════════════════════════════


def _tick_infos(df: pd.DataFrame, sided: bool = False) -> list[TickInfo]:
    return [
        TickInfo(
            timestamp=int(row["ts"]),
            price=float(row["px"]),
            volume=float(row["vol"]),
            side=float(row["side"]) if sided else None,
        )
        for row in df.to_dict("records")
    ]


def _minute_infos(df: pd.DataFrame) -> list[MinuteInfo]:
    return [
        MinuteInfo(
            timestamp=int(row["ts"]),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        for row in df.to_dict("records")
    ]


class TestCalibrationMultiplier:
    def test_tick_dollar_threshold_scales(self) -> None:
        ticks = _tick_infos(_ticks())
        base = StaticCalibrationHelper.estimate_fixed_threshold(ticks, "dollar", 5.0)
        scaled = StaticCalibrationHelper.estimate_fixed_threshold(
            ticks, "dollar", 5.0, multiplier=3.0
        )
        assert scaled == pytest.approx(3.0 * base)

    def test_tick_volume_threshold_ignores_multiplier(self) -> None:
        ticks = _tick_infos(_ticks())
        base = StaticCalibrationHelper.estimate_fixed_threshold(ticks, "volume", 5.0)
        scaled = StaticCalibrationHelper.estimate_fixed_threshold(
            ticks, "volume", 5.0, multiplier=3.0
        )
        assert scaled == base

    def test_tick_dollar_imbalance_threshold_scales(self) -> None:
        ticks = _tick_infos(_ticks(with_side=True), sided=True)
        base = StaticCalibrationHelper.estimate_fixed_threshold(ticks, "imbalance_dollar", 4.0)
        scaled = StaticCalibrationHelper.estimate_fixed_threshold(
            ticks, "imbalance_dollar", 4.0, multiplier=2.0
        )
        assert scaled == pytest.approx(2.0 * base)

    def test_tick_dollar_ewma_seeds_scale(self) -> None:
        ticks = _tick_infos(_ticks(with_side=True), sided=True)
        base = StaticCalibrationHelper.estimate_ewma_seeds(ticks, "imbalance", metric="dollar")
        scaled = StaticCalibrationHelper.estimate_ewma_seeds(
            ticks, "imbalance", metric="dollar", multiplier=2.0
        )
        assert scaled["initial_ewa_t"] == pytest.approx(2.0 * base["initial_ewa_t"])
        assert scaled["initial_ewa_proportion"] == pytest.approx(base["initial_ewa_proportion"])

    def test_minute_dollar_threshold_scales(self) -> None:
        minutes = _minute_infos(_minutes())
        base = MinuteCalibrationHelper.estimate_dollar_threshold(minutes, 6.0)
        scaled = MinuteCalibrationHelper.estimate_dollar_threshold(minutes, 6.0, multiplier=2.0)
        assert scaled == pytest.approx(2.0 * base)

    def test_minute_hybrid_fixed_dollar_scales_with_multiplier(self) -> None:
        minutes = _minute_infos(_minutes())
        _, vol_base = MinuteCalibrationHelper.estimate_hybrid_thresholds(
            minutes, 4.0, dollar_threshold=20_000.0
        )
        _, vol_scaled = MinuteCalibrationHelper.estimate_hybrid_thresholds(
            minutes, 4.0, dollar_threshold=60_000.0, multiplier=3.0
        )
        # Both sides of the dollar comparison scale, so the volatility leg is
        # unchanged and the ray's dollar target scales exactly.
        assert vol_scaled == vol_base

        dollar_base, vol_ray_base = MinuteCalibrationHelper.estimate_hybrid_thresholds(minutes, 4.0)
        dollar_scaled, vol_ray_scaled = MinuteCalibrationHelper.estimate_hybrid_thresholds(
            minutes, 4.0, multiplier=3.0
        )
        assert vol_ray_scaled == pytest.approx(vol_ray_base)
        assert dollar_scaled == pytest.approx(3.0 * dollar_base, rel=0.05)

    def test_minute_dollar_ewma_seeds_scale(self) -> None:
        minutes = _minute_infos(_minutes())
        base = MinuteCalibrationHelper.estimate_ewma_seeds(minutes, "run", metric="dollar")
        scaled = MinuteCalibrationHelper.estimate_ewma_seeds(
            minutes, "run", metric="dollar", multiplier=2.0
        )
        assert scaled["initial_ewa_t"] == pytest.approx(2.0 * base["initial_ewa_t"])
        assert scaled["initial_ewa_proportion"] == pytest.approx(base["initial_ewa_proportion"])


# ═══════════════════════════════════════════════════════════════════════════════
# Polars mirrors
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolarsMirror:
    def test_minute_dollar_polars_scales_like_pandas(self) -> None:
        pl = pytest.importorskip("polars")
        from flowbars.adapters import polars as fb_pl

        pdf = _minutes()
        pl_df = pl.from_pandas(pdf)
        pandas_bars = fb.MinuteDollarBarConstructor(
            threshold=60_000.0, schema=MINUTE_SCHEMA, multiplier=3.0
        ).batch(pdf)
        polars_bars = fb_pl.compute_dollar_bars_from_minutes(
            pl_df, 60_000.0, schema=MINUTE_SCHEMA, multiplier=3.0
        ).to_pandas()
        _assert_frames_equal(pandas_bars, polars_bars)
