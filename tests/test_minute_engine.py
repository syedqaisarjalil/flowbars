"""Tests for the shared minute bar engine — full mode parity.

Covers batch vs streaming ``update()`` equivalence for all ten minute bar
types, callbacks (including warmup bars), watermark dedup, strict ordering,
empty/single-row input, and determinism.
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
    compute_volume_bars_from_minutes,
)
from flowbars.core import MinuteInfo, TickDataError
from flowbars.schema import MinuteSchemaMapping

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


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _minutes(n: int = 240, seed: int = 11, start_ts: int = 0) -> pd.DataFrame:
    """Deterministic synthetic minute OHLCV data."""
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
    """Build one of the ten minute constructors with test-friendly params."""
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
        side=None,
    )


def _stream(ctor: object, df: pd.DataFrame) -> pd.DataFrame:
    bars = []
    for rec in df.to_dict("records"):
        bar = ctor.update(_minute_info(rec))  # type: ignore[attr-defined]
        if bar is not None:
            bars.append(bar)
    return _bars_frame(bars)


def _bars_frame(bars: list[object]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(columns=BAR_COLUMNS)
    return pd.DataFrame([vars(b) for b in bars])[BAR_COLUMNS]


# ═══════════════════════════════════════════════════════════════════════════════
# Batch vs streaming
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("kind", ALL_TYPES)
def test_batch_matches_streaming(kind: str) -> None:
    """``batch(df)`` and per-minute ``update()`` produce identical frames."""
    df = _minutes()
    batch = _constructor(kind).batch(df)  # type: ignore[attr-defined]
    stream = _stream(_constructor(kind), df)

    assert len(batch) > 0, f"{kind} produced no bars"
    pd.testing.assert_frame_equal(batch.reset_index(drop=True), stream.reset_index(drop=True))


@pytest.mark.parametrize(
    "kind", ["imbalance_volume", "imbalance_dollar", "run_volume", "run_dollar"]
)
def test_supplied_sides_match_streaming(kind: str) -> None:
    """Supplied side columns flow through identically on both paths."""
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
    batch = _constructor(kind, schema=schema).batch(df)  # type: ignore[attr-defined]

    ctor = _constructor(kind, schema=schema)
    bars = []
    for rec in df.to_dict("records"):
        item = _minute_info(rec)
        item.side = None if np.isnan(rec["side"]) else float(rec["side"])  # type: ignore[arg-type]
        bar = ctor.update(item)  # type: ignore[attr-defined]
        if bar is not None:
            bars.append(bar)
    stream = _bars_frame(bars)

    assert len(batch) > 0
    pd.testing.assert_frame_equal(batch.reset_index(drop=True), stream.reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks and warmup
# ═══════════════════════════════════════════════════════════════════════════════


class TestCallbacksAndWarmup:
    def test_on_bar_fires_including_warmup(self) -> None:
        captured: list[object] = []
        ctor = MinuteVolumeBarConstructor(
            threshold=60.0,
            schema=MINUTE_SCHEMA,
            warmup_bars=2,
            on_bar=captured.append,
        )
        bars = ctor.batch(_minutes())

        assert len(bars) > 0
        assert len(captured) == ctor.bars_emitted
        assert ctor.bars_emitted == len(bars) + 2
        # Returned bars keep their original ids (no renumbering).
        assert [int(b) for b in bars["bar_id"]] == list(range(2, 2 + len(bars)))

    def test_on_bar_fires_for_warmup_in_streaming(self) -> None:
        captured: list[object] = []
        ctor = MinuteVolumeBarConstructor(
            threshold=60.0,
            schema=MINUTE_SCHEMA,
            warmup_bars=1,
            on_bar=captured.append,
        )
        returned = _stream(ctor, _minutes(60))
        assert ctor.bars_emitted == len(captured)
        assert len(returned) == len(captured) - 1

    def test_on_threshold_update_fires_for_minute_ewma(self) -> None:
        updates: list[float] = []
        ctor = MinuteImbalanceVolumeBarConstructor(
            span=10.0,
            initial_ewa_t=100.0,
            initial_ewa_proportion=0.5,
            schema=MINUTE_SCHEMA,
            on_threshold_update=updates.append,
        )
        ctor.batch(_minutes())
        assert len(updates) > 0
        assert all(value > 0.0 for value in updates)

    def test_on_threshold_update_silent_for_static(self) -> None:
        updates: list[float] = []
        ctor = MinuteVolumeBarConstructor(
            threshold=60.0,
            schema=MINUTE_SCHEMA,
            on_threshold_update=updates.append,
        )
        ctor.batch(_minutes())
        assert updates == []


# ═══════════════════════════════════════════════════════════════════════════════
# Watermark dedup
# ═══════════════════════════════════════════════════════════════════════════════


class TestWatermark:
    def test_streaming_dedup_is_idempotent(self) -> None:
        df = _minutes(12)
        ctor = MinuteVolumeBarConstructor(threshold=60.0, schema=MINUTE_SCHEMA)
        for rec in df.head(6).to_dict("records"):
            ctor.update(_minute_info(rec))

        before = json.dumps(ctor.get_state(), sort_keys=True)
        # Re-feeding the last processed minute is a no-op.
        assert ctor.update(_minute_info(df.iloc[5].to_dict())) is None
        assert json.dumps(ctor.get_state(), sort_keys=True) == before

    def test_streaming_resume_matches_uninterrupted(self) -> None:
        df = _minutes(40)
        uninterrupted = _stream(
            MinuteVolumeBarConstructor(threshold=60.0, schema=MINUTE_SCHEMA), df
        )

        ctor = MinuteVolumeBarConstructor(threshold=60.0, schema=MINUTE_SCHEMA)
        first = []
        for rec in df.head(12).to_dict("records"):
            bar = ctor.update(_minute_info(rec))
            if bar is not None:
                first.append(bar)
        state = json.loads(json.dumps(ctor.get_state()))

        resumed = MinuteVolumeBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
        second = _stream(resumed, df.iloc[12:])

        combined = pd.concat([_bars_frame(first), second], ignore_index=True)
        pd.testing.assert_frame_equal(
            combined.reset_index(drop=True), uninterrupted.reset_index(drop=True)
        )

    def test_batch_watermark_custom_column(self) -> None:
        df = _minutes(30).assign(seq=np.arange(30, dtype=np.int64))
        kwargs = {"threshold": 60.0, "schema": MINUTE_SCHEMA, "watermark": "seq"}

        ctor = MinuteVolumeBarConstructor(**kwargs)  # type: ignore[arg-type]
        first = ctor.batch(df.iloc[:15])
        second = ctor.batch(df.iloc[15:])
        partial = pd.concat([first, second], ignore_index=True)

        full = MinuteVolumeBarConstructor(
            threshold=60.0, schema=MINUTE_SCHEMA, watermark="seq"
        ).batch(df)
        pd.testing.assert_frame_equal(partial, full)

    def test_batch_overlap_ignored_after_resume(self) -> None:
        df = _minutes(30).assign(seq=np.arange(30, dtype=np.int64))
        ctor = MinuteVolumeBarConstructor(threshold=60.0, schema=MINUTE_SCHEMA, watermark="seq")
        first = ctor.batch(df.iloc[:15])
        # Re-sending rows 10..14 (already seen) must not duplicate them.
        second = ctor.batch(df.iloc[10:])
        full = MinuteVolumeBarConstructor(
            threshold=60.0, schema=MINUTE_SCHEMA, watermark="seq"
        ).batch(df)
        pd.testing.assert_frame_equal(pd.concat([first, second], ignore_index=True), full)


# ═══════════════════════════════════════════════════════════════════════════════
# Strict ordering
# ═══════════════════════════════════════════════════════════════════════════════


class TestStrictOrdering:
    def test_streaming_out_of_order_raises(self) -> None:
        ctor = MinuteVolumeBarConstructor(
            threshold=1_000.0,
            schema=MINUTE_SCHEMA,
            strict_ordering=True,
            watermark=None,  # disable dedup so the ordering check is reached
        )
        df = _minutes(5)
        for rec in df.head(3).to_dict("records"):
            ctor.update(_minute_info(rec))
        with pytest.raises(TickDataError, match="Out-of-order"):
            ctor.update(_minute_info(df.iloc[1].to_dict()))

    def test_batch_out_of_order_raises(self) -> None:
        df = _minutes(6).iloc[[0, 2, 1, 3, 4, 5]]
        ctor = MinuteVolumeBarConstructor(
            threshold=1_000.0, schema=MINUTE_SCHEMA, strict_ordering=True
        )
        with pytest.raises(TickDataError, match="Out-of-order"):
            ctor.batch(df)

    def test_non_monotonic_allowed_by_default(self) -> None:
        df = _minutes(6).iloc[[0, 2, 1, 3, 4, 5]]
        bars = MinuteVolumeBarConstructor(threshold=1_000.0, schema=MINUTE_SCHEMA).batch(df)
        assert len(bars) == 0  # threshold never reached, but no raise


# ═══════════════════════════════════════════════════════════════════════════════
# Empty / single row / determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_empty_input(self) -> None:
        empty = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        bars = compute_volume_bars_from_minutes(empty, threshold=5.0, schema=MINUTE_SCHEMA)
        assert len(bars) == 0
        assert list(bars.columns) == BAR_COLUMNS

    def test_single_row(self) -> None:
        df = pd.DataFrame(
            {
                "ts": [0],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1.0],
            }
        )
        bars = compute_volume_bars_from_minutes(df, threshold=1.0, schema=MINUTE_SCHEMA)
        assert len(bars) == 1
        assert bars.iloc[0]["num_ticks"] == 1

    @pytest.mark.parametrize(
        "kind,threshold",
        [
            ("volume", 60.0),
            ("volatility", 0.02),
            ("renko", 0.01),
        ],
    )
    def test_repeated_runs_are_identical(self, kind: str, threshold: float) -> None:
        df = _minutes()
        cls = {
            "volume": MinuteVolumeBarConstructor,
            "volatility": MinuteVolatilityBarConstructor,
            "renko": MinuteRenkoBarConstructor,
        }[kind]

        def run() -> pd.DataFrame:
            return cls(threshold, schema=MINUTE_SCHEMA).batch(df)

        pd.testing.assert_frame_equal(run(), run())
