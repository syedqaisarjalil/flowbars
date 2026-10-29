"""Tests for the ``update_many`` streaming API (tick and minute paths)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from flowbars.bars.accumulators import (
    DollarAccumulator,
    ImbalanceAccumulator,
    RunAccumulator,
    TickAccumulator,
    TimeAccumulator,
    VolumeAccumulator,
)
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.core import MinuteInfo, TickInfo
from flowbars.schema import MinuteSchemaMapping, SchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator, StaticThresholdEstimator

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


def _ticks(n: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype=np.int64) * 1_000,
            "px": prices,
            "vol": rng.integers(1, 20, n).astype(float),
            "side": np.where(np.arange(n) % 3, 1.0, -1.0),
        }
    )


def _minutes(n: int = 240, seed: int = 5) -> pd.DataFrame:
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


def _tick_ctor(kind: str) -> BaseBarConstructor:
    if kind == "tick":
        acc, est = TickAccumulator(), StaticThresholdEstimator(10.0)
    elif kind == "time":
        acc, est = TimeAccumulator(interval_ms=60_000), StaticThresholdEstimator(0.0)
    elif kind == "volume":
        acc, est = VolumeAccumulator(), StaticThresholdEstimator(100.0)
    elif kind == "dollar":
        acc, est = DollarAccumulator(), StaticThresholdEstimator(10_000.0)
    elif kind == "imbalance_tick":
        acc = ImbalanceAccumulator(bar_type=kind, metric="tick")
        est = EWMAThresholdEstimator("imbalance", span=10.0, initial_ewa_t=15.0)
    elif kind == "imbalance_volume":
        acc = ImbalanceAccumulator(bar_type=kind, metric="volume")
        est = EWMAThresholdEstimator("imbalance", span=10.0, initial_ewa_t=100.0)
    elif kind == "imbalance_dollar":
        acc = ImbalanceAccumulator(bar_type=kind, metric="dollar")
        est = EWMAThresholdEstimator("imbalance", span=10.0, initial_ewa_t=10_000.0)
    elif kind == "run_tick":
        acc = RunAccumulator(bar_type=kind, metric="tick")
        est = EWMAThresholdEstimator("run", span=10.0, initial_ewa_t=15.0)
    elif kind == "run_volume":
        acc = RunAccumulator(bar_type=kind, metric="volume")
        est = EWMAThresholdEstimator("run", span=10.0, initial_ewa_t=100.0)
    elif kind == "run_dollar":
        acc = RunAccumulator(bar_type=kind, metric="dollar")
        est = EWMAThresholdEstimator("run", span=10.0, initial_ewa_t=10_000.0)
    else:  # pragma: no cover
        raise ValueError(kind)
    return BaseBarConstructor(acc, est, schema=TICK_SCHEMA)


TICK_KINDS = [
    "tick",
    "time",
    "volume",
    "dollar",
    "imbalance_tick",
    "imbalance_volume",
    "imbalance_dollar",
    "run_tick",
    "run_volume",
    "run_dollar",
]


def _minute_ctors() -> dict[str, tuple[object, dict]]:
    import flowbars as fb

    return {
        "volume": (fb.MinuteVolumeBarConstructor, {"threshold": 120.0}),
        "dollar": (fb.MinuteDollarBarConstructor, {"threshold": 12_000.0}),
        "imbalance_volume": (
            fb.MinuteImbalanceVolumeBarConstructor,
            {"initial_ewa_t": 150.0, "initial_ewa_proportion": 0.5},
        ),
        "imbalance_dollar": (
            fb.MinuteImbalanceDollarBarConstructor,
            {"initial_ewa_t": 15_000.0, "initial_ewa_proportion": 0.5},
        ),
        "run_volume": (
            fb.MinuteRunVolumeBarConstructor,
            {"initial_ewa_t": 300.0, "initial_ewa_proportion": 0.5},
        ),
        "run_dollar": (
            fb.MinuteRunDollarBarConstructor,
            {"initial_ewa_t": 30_000.0, "initial_ewa_proportion": 0.5},
        ),
        "volatility": (fb.MinuteVolatilityBarConstructor, {"threshold": 0.03}),
        "range": (fb.MinuteRangeBarConstructor, {"threshold": 0.03}),
        "renko": (fb.MinuteRenkoBarConstructor, {"threshold": 0.02}),
        "hybrid": (
            fb.MinuteHybridBarConstructor,
            {"dollar_threshold": 12_000.0, "volatility_threshold": 0.03},
        ),
    }


@pytest.mark.parametrize("kind", TICK_KINDS)
def test_tick_update_many_matches_batch(kind: str) -> None:
    df = _ticks()
    batched = _tick_ctor(kind).batch(df)
    streamed = _tick_ctor(kind).update_many(df)
    assert len(batched) > 0
    pd.testing.assert_frame_equal(batched, streamed)


@pytest.mark.parametrize("kind", list(_minute_ctors()))
def test_minute_update_many_matches_batch(kind: str) -> None:
    cls, kwargs = _minute_ctors()[kind]
    df = _minutes()
    batched = cls(**kwargs, schema=MINUTE_SCHEMA).batch(df)
    streamed = cls(**kwargs, schema=MINUTE_SCHEMA).update_many(df)
    assert len(batched) > 0
    pd.testing.assert_frame_equal(batched, streamed)


def test_update_many_matches_explicit_update_loop_tick() -> None:
    from flowbars.tick_rule import derive_tick_sign

    df = _ticks()
    signs = derive_tick_sign(df["px"].to_numpy())
    items = [
        TickInfo(
            timestamp=int(rec["ts"]),
            price=float(rec["px"]),
            volume=float(rec["vol"]),
            side=None if np.isnan(signs[i]) else float(signs[i]),
        )
        for i, rec in enumerate(df.to_dict("records"))
    ]
    loop_ctor = _tick_ctor("imbalance_volume")
    loop_bars = [bar for bar in (loop_ctor.update(item) for item in items) if bar is not None]
    loop = pd.DataFrame([vars(bar) for bar in loop_bars])
    streamed = _tick_ctor("imbalance_volume").update_many(df)
    pd.testing.assert_frame_equal(loop, streamed)


def test_update_many_matches_explicit_update_loop_minute() -> None:
    df = _minutes()
    items = [
        MinuteInfo(
            timestamp=int(rec["ts"]),
            open=float(rec["open"]),
            high=float(rec["high"]),
            low=float(rec["low"]),
            close=float(rec["close"]),
            volume=float(rec["volume"]),
        )
        for rec in df.to_dict("records")
    ]
    import flowbars as fb

    loop_ctor = fb.MinuteVolatilityBarConstructor(0.03, schema=MINUTE_SCHEMA)
    loop_bars = [bar for bar in (loop_ctor.update(item) for item in items) if bar is not None]
    loop = pd.DataFrame([vars(bar) for bar in loop_bars])
    streamed = fb.MinuteVolatilityBarConstructor(0.03, schema=MINUTE_SCHEMA).update_many(df)
    pd.testing.assert_frame_equal(loop, streamed)


def test_update_many_split_resume_matches_one_pass() -> None:
    import flowbars as fb

    df = _minutes()
    full = fb.MinuteVolumeBarConstructor(120.0, schema=MINUTE_SCHEMA).update_many(df)

    ctor = fb.MinuteVolumeBarConstructor(120.0, schema=MINUTE_SCHEMA)
    first = ctor.update_many(df.iloc[:100])
    state = json.loads(json.dumps(ctor.get_state(), allow_nan=False))
    resumed = fb.MinuteVolumeBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
    second = resumed.update_many(df.iloc[100:])
    combined = pd.concat([first, second], ignore_index=True)
    pd.testing.assert_frame_equal(full.reset_index(drop=True), combined)


def test_update_many_replay_is_noop() -> None:
    import flowbars as fb

    df = _minutes()
    ctor = fb.MinuteVolumeBarConstructor(120.0, schema=MINUTE_SCHEMA)
    first = ctor.update_many(df)
    second = ctor.update_many(df)
    assert len(first) > 0
    assert second.empty


def test_update_many_empty_frame() -> None:
    import flowbars as fb

    out = fb.MinuteVolumeBarConstructor(120.0, schema=MINUTE_SCHEMA).update_many(
        _minutes().iloc[:0]
    )
    assert out.empty
    assert list(out.columns) == [
        "bar_id",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_value",
        "vwap",
        "num_ticks",
        "open_ts",
        "close_ts",
        "bar_type",
        "partial",
    ]
