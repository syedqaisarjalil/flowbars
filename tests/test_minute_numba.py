"""Numba backend tests for the minute path.

Python-vs-numba equivalence for all ten minute bar types, warmup ``bar_id``
and callback parity, ``min_run_length`` filtering, and the pristine/session
fallback warnings.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba", reason="numba not installed")

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
from flowbars.calendars import SessionCalendar
from flowbars.core import MinuteInfo
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


def _bars_equal(df1: pd.DataFrame, df2: pd.DataFrame, float_tol: float = 1e-12) -> bool:
    if list(df1.columns) != list(df2.columns) or len(df1) != len(df2):
        return False
    for col in df1.columns:
        if col == "bar_type":
            if not (df1[col] == df2[col]).all():
                return False
        elif col in ("bar_id", "num_ticks", "open_ts", "close_ts"):
            if not (df1[col].astype(np.int64) == df2[col].astype(np.int64)).all():
                return False
        else:
            a = np.asarray(df1[col], dtype=np.float64)
            b = np.asarray(df2[col], dtype=np.float64)
            if not np.allclose(a, b, rtol=float_tol, atol=1e-14):
                return False
    return True


def _constructor(kind: str, backend: str, **kwargs: object) -> object:
    common = {"schema": MINUTE_SCHEMA, "backend": backend, **kwargs}
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


@pytest.mark.parametrize("kind", ALL_TYPES)
def test_python_numba_equivalence(kind: str) -> None:
    df = _minutes()
    py = _constructor(kind, "python").batch(df)  # type: ignore[attr-defined]
    nb = _constructor(kind, "numba").batch(df)  # type: ignore[attr-defined]

    assert len(py) > 0, f"{kind} produced no bars"
    assert _bars_equal(py, nb), f"{kind}: numba output differs from Python"


def test_supplied_sides_equivalence() -> None:
    df = _minutes()
    prices = df["close"].to_numpy(dtype=np.float64)
    sides = np.full(len(df), np.nan, dtype=np.float64)
    last = np.nan
    for i in range(1, len(df)):
        if prices[i] > prices[i - 1]:
            last = 1.0
        elif prices[i] < prices[i - 1]:
            last = -1.0
        sides[i] = last
    df = df.assign(side=sides)
    schema = MinuteSchemaMapping(
        {
            "timestamp": "ts",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "side": "side",
        }
    )
    py = MinuteImbalanceVolumeBarConstructor(
        span=10.0, initial_ewa_t=100.0, initial_ewa_proportion=0.5, schema=schema
    ).batch(df)
    nb = MinuteImbalanceVolumeBarConstructor(
        span=10.0,
        initial_ewa_t=100.0,
        initial_ewa_proportion=0.5,
        schema=schema,
        backend="numba",
    ).batch(df)
    assert len(py) > 0
    assert _bars_equal(py, nb)


def test_warmup_ids_and_callbacks_match() -> None:
    df = _minutes()
    py_callbacks: list[object] = []
    nb_callbacks: list[object] = []

    py_ctor = MinuteRunVolumeBarConstructor(
        span=10.0,
        initial_ewa_t=200.0,
        initial_ewa_proportion=0.5,
        warmup_bars=2,
        schema=MINUTE_SCHEMA,
        on_bar=py_callbacks.append,
    )
    nb_ctor = MinuteRunVolumeBarConstructor(
        span=10.0,
        initial_ewa_t=200.0,
        initial_ewa_proportion=0.5,
        warmup_bars=2,
        schema=MINUTE_SCHEMA,
        backend="numba",
        on_bar=nb_callbacks.append,
    )

    py = py_ctor.batch(df)
    nb = nb_ctor.batch(df)

    assert _bars_equal(py, nb)
    assert [int(x) for x in py["bar_id"]] == list(range(2, 2 + len(py)))
    assert len(nb_callbacks) == nb_ctor.bars_emitted
    assert len(py_callbacks) == py_ctor.bars_emitted
    assert [b.bar_id for b in py_callbacks] == [b.bar_id for b in nb_callbacks]


def test_min_run_length_parity() -> None:
    df = _minutes()
    py = MinuteRunVolumeBarConstructor(
        span=10.0,
        initial_ewa_t=200.0,
        initial_ewa_proportion=0.5,
        min_run_length=3,
        schema=MINUTE_SCHEMA,
    ).batch(df)
    nb = MinuteRunVolumeBarConstructor(
        span=10.0,
        initial_ewa_t=200.0,
        initial_ewa_proportion=0.5,
        min_run_length=3,
        schema=MINUTE_SCHEMA,
        backend="numba",
    ).batch(df)
    assert _bars_equal(py, nb)
    assert len(nb) > 0
    assert (nb["num_ticks"].astype(int) >= 3).all()


def test_non_pristine_falls_back_to_python() -> None:
    """A constructor already driven by ``update()`` continues on the Python path."""
    df = _minutes()

    def feed(ctor: object) -> None:
        for rec in df.head(97).to_dict("records"):
            ctor.update(  # type: ignore[attr-defined]
                MinuteInfo(
                    timestamp=int(rec["ts"]),  # type: ignore[arg-type]
                    open=float(rec["open"]),  # type: ignore[arg-type]
                    high=float(rec["high"]),  # type: ignore[arg-type]
                    low=float(rec["low"]),  # type: ignore[arg-type]
                    close=float(rec["close"]),  # type: ignore[arg-type]
                    volume=float(rec["volume"]),  # type: ignore[arg-type]
                )
            )

    nb_ctor = MinuteVolumeBarConstructor(threshold=60.0, schema=MINUTE_SCHEMA, backend="numba")
    py_ctor = MinuteVolumeBarConstructor(threshold=60.0, schema=MINUTE_SCHEMA)
    feed(nb_ctor)
    feed(py_ctor)

    with pytest.warns(UserWarning, match="pristine"):
        nb = nb_ctor.batch(df.iloc[97:])
    py = py_ctor.batch(df.iloc[97:])

    assert len(nb) > 0
    assert _bars_equal(nb, py)


def test_second_batch_after_numba_warns() -> None:
    df = _minutes()
    nb_ctor = MinuteVolumeBarConstructor(threshold=60.0, schema=MINUTE_SCHEMA, backend="numba")
    first = nb_ctor.batch(df.iloc[:120])
    assert len(first) > 0

    with pytest.warns(UserWarning, match="pristine"):
        nb_ctor.batch(df.iloc[120:])


def _weekday_ts(weekday: int, hour: int, minute: int) -> int:
    base = datetime.datetime(2024, 1, 15, hour, minute, tzinfo=datetime.timezone.utc)
    return int((base + datetime.timedelta(days=weekday)).timestamp() * 1000)


def test_session_calendar_falls_back_to_python() -> None:
    cal = SessionCalendar(9, 30, 16, 0)
    df = _minutes(120, start_ts=_weekday_ts(0, 10, 0))

    py = MinuteVolatilityBarConstructor(threshold=0.02, calendar=cal, schema=MINUTE_SCHEMA).batch(
        df
    )
    with pytest.warns(UserWarning, match="session boundaries"):
        nb = MinuteVolatilityBarConstructor(
            threshold=0.02, calendar=cal, schema=MINUTE_SCHEMA, backend="numba"
        ).batch(df)
    assert _bars_equal(py, nb)
