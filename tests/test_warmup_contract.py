"""Warmup contract: ``warmup_bars`` vs ``warmup_minutes``, ``warmup_done``.

The contract under test:

* ``warmup_bars`` and ``warmup_minutes`` are mutually exclusive; negative
  values are rejected.
* ``warmup_minutes=N`` suppresses every bar whose closing input arrives
  before N minutes have elapsed since the first accepted input.  The bar
  that closes at or after the boundary is returned.
* Warmup bars keep their original ``bar_id`` (no renumbering), still fire
  ``on_bar``, and still feed the threshold estimator.
* ``warmup_done`` reports when the configured warmup window has elapsed.
* The setting and its anchor timestamp survive a JSON state round-trip.
* The numba batch path applies the same duration-warmup mask as streaming.
* The registry advertises the runtime parameter and the recommended duration.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import flowbars as fb
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.bars.registry import BarRegistry
from flowbars.core import MinuteInfo, TickInfo
from flowbars.schema import MinuteSchemaMapping, SchemaMapping

BASE_TS = 1_704_067_200_000  # 2024-01-01T00:00:00Z
MINUTE_MS = 60_000

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

TICK_SCHEMA = SchemaMapping({"timestamp": "ts", "price": "price", "volume": "volume"})


def _ts(minute: int) -> int:
    return BASE_TS + minute * MINUTE_MS


def _tick(minute: float, price: float = 100.0, volume: float = 1.0) -> TickInfo:
    return TickInfo(timestamp=int(_ts(minute)), price=price, volume=volume)


def _minute(minute: int, close: float = 100.0, volume: float = 1.0) -> MinuteInfo:
    return MinuteInfo(
        timestamp=_ts(minute),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def _ticks_df(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "price": [r[1] for r in rows],
            "volume": [r[2] for r in rows],
        }
    )


def _minutes_df(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[1] for r in rows],
            "close": [r[1] for r in rows],
            "volume": [r[2] for r in rows],
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Parameter validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            fb.VolumeBarConstructor(1.0, warmup_bars=2, warmup_minutes=5)
        with pytest.raises(ValueError, match="mutually exclusive"):
            fb.MinuteVolumeBarConstructor(1.0, warmup_bars=2, warmup_minutes=5)

    def test_negative_minutes_rejected(self) -> None:
        with pytest.raises(ValueError, match="warmup_minutes"):
            fb.VolumeBarConstructor(1.0, warmup_minutes=-1)

    def test_zero_minutes_with_bars_allowed(self) -> None:
        ctor = fb.VolumeBarConstructor(1.0, warmup_bars=2, warmup_minutes=0)
        assert ctor.warmup_done is False
        assert ctor.get_state()["warmup_minutes"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# warmup_done
# ═══════════════════════════════════════════════════════════════════════════════


class TestWarmupDone:
    def test_no_warmup_is_done_immediately(self) -> None:
        ctor = fb.VolumeBarConstructor(1.0)
        assert ctor.warmup_done is True

    def test_warmup_bars_progress(self) -> None:
        ctor = fb.VolumeBarConstructor(1.0, warmup_bars=2)
        assert ctor.warmup_done is False
        ctor.update(_tick(0))
        assert ctor.warmup_done is False  # 1 bar, still inside warmup
        ctor.update(_tick(1))
        assert ctor.warmup_done is False  # 2 bars, first real bar still to come
        ctor.update(_tick(2))
        assert ctor.warmup_done is True

    def test_warmup_minutes_progress(self) -> None:
        ctor = fb.VolumeBarConstructor(1.0, warmup_minutes=2)
        assert ctor.warmup_done is False
        ctor.update(_tick(0))
        assert ctor.warmup_done is False
        ctor.update(_tick(1.5))
        assert ctor.warmup_done is False
        ctor.update(_tick(2.0))
        assert ctor.warmup_done is True


# ═══════════════════════════════════════════════════════════════════════════════
# Duration warmup behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestWarmupMinutesStreaming:
    def test_bars_closing_before_window_are_suppressed(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(2.0, schema=MINUTE_SCHEMA, warmup_minutes=2)
        # Each minute carries volume 1 -> a bar closes every two minutes.
        assert ctor.update(_minute(0)) is None
        assert ctor.update(_minute(1)) is None  # closes at ts(1), inside warmup
        assert ctor.update(_minute(2)) is None
        bar = ctor.update(_minute(3))  # closes at ts(3) >= ts(0) + 2 min
        assert bar is not None
        assert bar.bar_id == 1  # not renumbered
        assert bar.num_ticks == 2
        assert bar.close_ts == _ts(3)

    def test_bar_closing_exactly_at_boundary_is_returned(self) -> None:
        ctor = fb.VolumeBarConstructor(1.0, warmup_minutes=1)
        assert ctor.update(_tick(0)) is None  # bar 0 closes at ts(0)
        bar = ctor.update(_tick(1))  # bar 1 closes exactly at ts(0) + 1 min
        assert bar is not None
        assert bar.bar_id == 1

    def test_callbacks_and_estimator_see_warmup_bars(self) -> None:
        seen: list[int] = []
        ctor = fb.MinuteImbalanceVolumeBarConstructor(
            span=2.0,
            schema=MINUTE_SCHEMA,
            warmup_minutes=2,
            on_bar=lambda b: seen.append(b.bar_id),
        )
        for i, close in enumerate([100.0, 101.0, 102.0, 103.0, 104.0]):
            ctor.update(_minute(i, close))
        # Bars closing inside the warmup window still fire the callback and
        # feed the estimator; only their return is suppressed.
        assert seen[0] == 0
        assert len(seen) >= 2
        assert ctor.warmup_done is True
        assert ctor._threshold_estimator.get_state()["n_updates"] == len(seen)

    def test_equivalence_with_unrestricted_stream(self) -> None:
        rows = [(_ts(i), 100.0 + i, 1.0) for i in range(8)]
        df = _minutes_df(rows)
        warmed = fb.compute_volume_bars_from_minutes(
            df, threshold=2.0, schema=MINUTE_SCHEMA, warmup_minutes=2
        )
        full = fb.compute_volume_bars_from_minutes(df, threshold=2.0, schema=MINUTE_SCHEMA)
        expected = full[full["close_ts"] >= _ts(2)].reset_index(drop=True)
        pd.testing.assert_frame_equal(warmed, expected)

    def test_compute_function_matches_streaming(self) -> None:
        rows = [(i, 100.0 + i, 1.0) for i in range(7)]
        df = _minutes_df(rows)
        batch = fb.compute_volume_bars_from_minutes(
            df, threshold=1.0, schema=MINUTE_SCHEMA, warmup_minutes=3
        )
        ctor = fb.MinuteVolumeBarConstructor(1.0, schema=MINUTE_SCHEMA, warmup_minutes=3)
        streamed = ctor.update_many(df)
        pd.testing.assert_frame_equal(batch, streamed)


# ═══════════════════════════════════════════════════════════════════════════════
# State round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestWarmupState:
    def test_state_carries_anchor_and_resumes(self) -> None:
        df = _minutes_df([(_ts(i), 100.0 + i, 1.0) for i in range(8)])
        ctor = fb.MinuteVolumeBarConstructor(1.0, schema=MINUTE_SCHEMA, warmup_minutes=2)
        first = ctor.update_many(df.iloc[:2])
        state = json.loads(json.dumps(ctor.get_state()))
        assert state["warmup_minutes"] == 2
        assert state["warmup_start_ts"] == _ts(0)

        resumed = fb.MinuteVolumeBarConstructor.from_state(state, schema=MINUTE_SCHEMA)
        assert resumed.warmup_done is False  # ts(1) is still inside the window
        second = resumed.update_many(df.iloc[2:])

        full = fb.MinuteVolumeBarConstructor(
            1.0, schema=MINUTE_SCHEMA, warmup_minutes=2
        ).update_many(df)
        assert len(first) == 0  # all first-half bars were warmup bars
        # Every bar the full stream returns was produced in the second half,
        # so the resumed stream must reproduce it exactly (ids included).
        pd.testing.assert_frame_equal(second, full)


# ═══════════════════════════════════════════════════════════════════════════════
# Numba parity
# ═══════════════════════════════════════════════════════════════════════════════


class TestWarmupNumba:
    @pytest.fixture(autouse=True)
    def _require_numba(self) -> None:
        backends = pytest.importorskip("flowbars.bars.numba_backend")
        if not backends._NUMBA_AVAILABLE:
            pytest.skip("numba is not installed")

    def test_minute_warmup_matches_python(self) -> None:
        df = _minutes_df([(i, 100.0 + i, 1.0) for i in range(10)])
        python = fb.compute_volume_bars_from_minutes(
            df, threshold=1.0, schema=MINUTE_SCHEMA, warmup_minutes=3
        )
        numba = fb.compute_volume_bars_from_minutes(
            df, threshold=1.0, schema=MINUTE_SCHEMA, warmup_minutes=3, backend="numba"
        )
        pd.testing.assert_frame_equal(python, numba)

    def test_tick_warmup_matches_python(self) -> None:
        df = _ticks_df([(int(_ts(i)), 100.0 + i, 1.0) for i in range(10)])
        python = fb.compute_volume_bars(df, threshold=1.0, schema=TICK_SCHEMA, warmup_minutes=3)
        numba = fb.compute_volume_bars(
            df, threshold=1.0, schema=TICK_SCHEMA, warmup_minutes=3, backend="numba"
        )
        pd.testing.assert_frame_equal(python, numba)


# ═══════════════════════════════════════════════════════════════════════════════
# Registry metadata
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistryWarmupMetadata:
    def test_runtime_parameter_in_params_schema(self) -> None:
        for spec in (
            BarRegistry.get_spec("volume"),
            MinuteBarRegistry.get_spec("minute_volume"),
        ):
            props = spec.params_schema["properties"]
            assert props["warmup_minutes"] == {
                "type": ["integer", "null"],
                "minimum": 0,
                "default": None,
            }
            assert props["warmup_bars"]["default"] == 0

    def test_recommended_warmup_minutes_advertised(self) -> None:
        for spec in (BarRegistry.get_spec("volume"), MinuteBarRegistry.get_spec("minute_volume")):
            assert spec.warmup_minutes == 0
            assert isinstance(spec.to_dict()["warmup_minutes"], int)

    def test_class_attribute_feeds_spec(self) -> None:
        from flowbars.bars.constructor import BaseBarConstructor
        from flowbars.bars.spec import build_bar_type_spec

        class _Recommended(BaseBarConstructor):  # pragma: no cover - metadata only
            recommended_warmup_minutes = 30
            input_kind = "minute"

        spec = build_bar_type_spec("recommended_probe", _Recommended)
        assert spec.warmup_minutes == 30
