"""State save/resume tests for the minute path.

Every minute type must survive ``get_state`` → ``json.dumps``/``loads`` →
``from_state`` and continue streaming with output identical to an
uninterrupted stream.  Type-specific state (renko anchor, hybrid thresholds,
minute tick-rule carry, ``min_run_length``) is checked explicitly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from flowbars.bars.minute import (
    MinuteDollarBarConstructor,
    MinuteHybridBarConstructor,
    MinuteImbalanceDollarBarConstructor,
    MinuteImbalanceVolumeBarConstructor,
    MinuteRangeBarConstructor,
    MinuteRenkoBarConstructor,
    MinuteRunDollarBarConstructor,
    MinuteRunVolumeBarConstructor,
    MinuteVolatilityBarConstructor,
    MinuteVolumeBarConstructor,
)
from flowbars.core import MinuteInfo, StateValidationError
from flowbars.schema import MinuteSchemaMapping

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

ALL_TYPES = [
    "volume",
    "dollar",
    "imbalance_volume",
    "imbalance_dollar",
    "run_volume",
    "run_dollar",
    "volatility",
    "range",
    "renko",
    "hybrid",
]

BAR_COLUMNS = [
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


def _minutes(n: int = 240, seed: int = 11, start_ts: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.003, n)))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    volumes = rng.integers(1, 31, n).astype(np.float64)
    return pd.DataFrame(
        {
            "ts": start_ts + np.arange(n, dtype=np.int64) * 60_000,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


def _constructor(kind: str, **kwargs: object) -> object:
    common = {"schema": MINUTE_SCHEMA, **kwargs}
    if kind == "volume":
        return MinuteVolumeBarConstructor(threshold=60.0, **common)  # type: ignore[arg-type]
    if kind == "dollar":
        return MinuteDollarBarConstructor(threshold=6_000.0, **common)  # type: ignore[arg-type]
    if kind == "imbalance_volume":
        return MinuteImbalanceVolumeBarConstructor(
            span=10.0,
            initial_ewa_t=100.0,
            initial_ewa_proportion=0.5,
            **common,  # type: ignore[arg-type]
        )
    if kind == "imbalance_dollar":
        return MinuteImbalanceDollarBarConstructor(
            span=10.0,
            initial_ewa_t=10_000.0,
            initial_ewa_proportion=0.5,
            **common,  # type: ignore[arg-type]
        )
    if kind == "run_volume":
        return MinuteRunVolumeBarConstructor(
            span=10.0,
            initial_ewa_t=200.0,
            initial_ewa_proportion=0.5,
            **common,  # type: ignore[arg-type]
        )
    if kind == "run_dollar":
        return MinuteRunDollarBarConstructor(
            span=10.0,
            initial_ewa_t=20_000.0,
            initial_ewa_proportion=0.5,
            **common,  # type: ignore[arg-type]
        )
    if kind == "volatility":
        return MinuteVolatilityBarConstructor(threshold=0.02, **common)  # type: ignore[arg-type]
    if kind == "range":
        return MinuteRangeBarConstructor(threshold=0.02, **common)  # type: ignore[arg-type]
    if kind == "renko":
        return MinuteRenkoBarConstructor(threshold=0.01, **common)  # type: ignore[arg-type]
    if kind == "hybrid":
        return MinuteHybridBarConstructor(
            dollar_threshold=6_000.0,
            volatility_threshold=0.02,
            **common,  # type: ignore[arg-type]
        )
    raise AssertionError(f"unknown kind {kind!r}")


def _minute_info(rec: dict[str, object]) -> MinuteInfo:
    return MinuteInfo(
        timestamp=int(rec["ts"]),  # type: ignore[arg-type]
        open=float(rec["open"]),  # type: ignore[arg-type]
        high=float(rec["high"]),  # type: ignore[arg-type]
        low=float(rec["low"]),  # type: ignore[arg-type]
        close=float(rec["close"]),  # type: ignore[arg-type]
        volume=float(rec["volume"]),  # type: ignore[arg-type]
    )


def _stream(ctor: object, df: pd.DataFrame) -> pd.DataFrame:
    bars = []
    for rec in df.to_dict("records"):
        bar = ctor.update(_minute_info(rec))  # type: ignore[attr-defined]
        if bar is not None:
            bars.append(bar)
    if not bars:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return pd.DataFrame([vars(b) for b in bars])[BAR_COLUMNS]


@pytest.mark.parametrize("kind", ALL_TYPES)
def test_resume_matches_uninterrupted(kind: str) -> None:
    df = _minutes()
    uninterrupted = _stream(_constructor(kind), df)

    ctor = _constructor(kind)
    first = _stream(ctor, df.iloc[:101])
    state = json.loads(json.dumps(ctor.get_state()))  # type: ignore[attr-defined]

    resumed = type(ctor).from_state(state, schema=MINUTE_SCHEMA)
    second = _stream(resumed, df.iloc[101:])

    combined = pd.concat([first, second], ignore_index=True)
    pd.testing.assert_frame_equal(
        combined.reset_index(drop=True), uninterrupted.reset_index(drop=True)
    )


@pytest.mark.parametrize("kind", ALL_TYPES)
def test_state_is_json_safe(kind: str) -> None:
    ctor = _constructor(kind)
    _stream(ctor, _minutes(17))
    state = ctor.get_state()  # type: ignore[attr-defined]
    text = json.dumps(state, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == state


def test_input_kind_mismatch_raises() -> None:
    ctor = _constructor("volume")
    _stream(ctor, _minutes(5))
    state = ctor.get_state()  # type: ignore[attr-defined]
    state["input_kind"] = "tick"
    with pytest.raises(StateValidationError, match="input_kind"):
        MinuteVolumeBarConstructor.from_state(state, schema=MINUTE_SCHEMA)


def test_renko_anchor_and_prev_close_restored() -> None:
    df = _minutes(80, seed=3)
    uninterrupted = _stream(MinuteRenkoBarConstructor(threshold=0.01, schema=MINUTE_SCHEMA), df)

    ctor = MinuteRenkoBarConstructor(threshold=0.01, schema=MINUTE_SCHEMA)
    first = _stream(ctor, df.iloc[:13])
    state = json.loads(json.dumps(ctor.get_state()))
    anchor = state["accumulator"]["anchor"]

    resumed = MinuteRenkoBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
    assert resumed._accumulator.get_state()["anchor"] == anchor
    assert resumed._prev_close == ctor._prev_close
    second = _stream(resumed, df.iloc[13:])

    combined = pd.concat([first, second], ignore_index=True)
    pd.testing.assert_frame_equal(
        combined.reset_index(drop=True), uninterrupted.reset_index(drop=True)
    )


def test_hybrid_thresholds_restored() -> None:
    df = _minutes(60, seed=5)
    ctor = MinuteHybridBarConstructor(
        dollar_threshold=1_000.0, volatility_threshold=0.05, schema=MINUTE_SCHEMA
    )
    _stream(ctor, df.iloc[:23])
    state = json.loads(json.dumps(ctor.get_state()))

    resumed = MinuteHybridBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
    acc = resumed._accumulator
    assert acc._dollar_threshold == 1_000.0
    assert acc._volatility_threshold == 0.05
    assert resumed._prev_close == ctor._prev_close


def test_min_run_length_restored() -> None:
    df = _minutes(30, seed=7)
    ctor = MinuteRunVolumeBarConstructor(
        span=10.0,
        initial_ewa_t=200.0,
        initial_ewa_proportion=0.5,
        min_run_length=3,
        schema=MINUTE_SCHEMA,
    )
    _stream(ctor, df.iloc[:11])
    state = json.loads(json.dumps(ctor.get_state()))

    resumed = MinuteRunVolumeBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
    assert resumed._min_run_length == 3
    bars = _stream(resumed, df.iloc[11:])
    assert (bars["num_ticks"].astype(int) >= 3).all()


def test_minute_sign_carry_survives_resume() -> None:
    """A flat close across the resume boundary keeps the carried sign."""
    df = pd.DataFrame(
        {
            "ts": [0, 60_000, 120_000, 180_000],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 101.0, 101.0, 101.0],
            "volume": [1.0, 1.0, 1.0, 1.0],
        }
    )
    ctor = MinuteImbalanceVolumeBarConstructor(
        span=10.0,
        initial_ewa_t=100.0,
        initial_ewa_proportion=0.5,
        schema=MINUTE_SCHEMA,
    )
    _stream(ctor, df.iloc[:2])  # last derived sign is +1
    state = json.loads(json.dumps(ctor.get_state()))
    resumed = MinuteImbalanceVolumeBarConstructor.from_state(state, schema=MINUTE_SCHEMA)

    item = _minute_info(df.iloc[2].to_dict())
    resumed.update(item)
    # Flat close carries +1; the accumulator saw a +1 side for that minute.
    assert resumed._prev_sign == 1.0
