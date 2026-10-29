"""Minute session policies: close / discard / carry, estimator reset, DST.

The contract under test:

* ``"close"`` (default) force-closes the in-progress bar at a session
  boundary as a ``partial`` bar, so no bar spans two sessions.
* ``"discard"`` drops the in-progress bar silently (no callback, no
  ``bar_id``/``bars_emitted`` advance, no metric or overflow carried).
* ``"carry"`` keeps the bar open across the boundary (explicit opt-in to
  session-spanning bars); the gap still contributes no price return.
* ``session_estimator="reset"`` restores the adaptive threshold seeds at each
  boundary; ``"persist"`` (default) keeps learning.
* Exchange-calendar DST and early-close boundaries are honoured because the
  engine asks the calendar on every input.
"""

from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
import pytest

import flowbars as fb
from flowbars.bars.minute.registry import MinuteBarRegistry
from flowbars.bars.registry import BarRegistry
from flowbars.calendars import ExchangeCalendar, SessionCalendar
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

CAL = SessionCalendar(9, 30, 16, 0)


def _ts(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    dt = datetime.datetime(year, month, day, hour, minute, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


MON_IN_SESSION = _ts(2024, 1, 15, 10, 0)  # Monday
MON_BOUNDARY = _ts(2024, 1, 15, 16, 30)
TUE_IN_SESSION = _ts(2024, 1, 16, 10, 0)


def _mi(timestamp: int, close: float, volume: float = 1.0) -> MinuteInfo:
    return MinuteInfo(
        timestamp=timestamp,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
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


# ═══════════════════════════════════════════════════════════════════════════════
# Policy validation and metadata
# ═══════════════════════════════════════════════════════════════════════════════


class TestPolicyMetadata:
    def test_invalid_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_policy"):
            fb.MinuteVolumeBarConstructor(1.0, session_policy="merge")
        with pytest.raises(ValueError, match="session_estimator"):
            fb.MinuteVolumeBarConstructor(1.0, session_estimator="keep")

    def test_defaults_are_close_and_persist(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(1.0)
        params = ctor.get_params()
        assert params["session_policy"] == "close"
        assert params["session_estimator"] == "persist"

    def test_specs_advertise_policies(self) -> None:
        minute_spec = MinuteBarRegistry.get_spec("minute_volume")
        assert minute_spec.session_policies == ("close", "discard", "carry")
        assert "session_policy" in minute_spec.params_schema["properties"]
        assert minute_spec.params_schema["properties"]["session_policy"]["default"] == "close"
        assert BarRegistry.get_spec("volume").session_policies == ("close",)


# ═══════════════════════════════════════════════════════════════════════════════
# close / discard / carry
# ═══════════════════════════════════════════════════════════════════════════════


class TestClosePolicy:
    def test_boundary_emits_partial_bar(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(
            threshold=1e12, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="close"
        )
        assert ctor.update(_mi(MON_IN_SESSION, 100.0, volume=5.0)) is None

        bar = ctor.update(_mi(MON_BOUNDARY, 101.0, volume=7.0))
        assert bar is not None
        assert bar.partial is True
        assert bar.num_ticks == 1
        assert bar.volume == pytest.approx(5.0)
        assert bar.open_ts == MON_IN_SESSION
        assert ctor.bars_emitted == 1

        # The boundary minute opened an off-hours bar; the Tuesday open
        # force-closes it (still never spanning a session).
        bar2 = ctor.update(_mi(TUE_IN_SESSION, 102.0, volume=1.0))
        assert bar2 is not None
        assert bar2.partial is True
        assert bar2.volume == pytest.approx(7.0)

    def test_no_bar_spans_two_sessions(self) -> None:
        rows = [
            (MON_IN_SESSION, 100.0, 5.0),
            (_ts(2024, 1, 15, 10, 1), 100.5, 4.0),
            (MON_BOUNDARY, 101.0, 7.0),
            (TUE_IN_SESSION, 102.0, 1.0),
        ]
        bars = fb.compute_volume_bars_from_minutes(
            _minutes_df(rows),
            threshold=1e12,
            calendar=CAL,
            schema=MINUTE_SCHEMA,
            session_policy="close",
        )
        assert len(bars) == 2
        assert list(bars["num_ticks"]) == [2, 1]
        assert bool(bars["partial"].all())

    def test_policy_is_per_series(self) -> None:
        df = _minutes_df(
            [
                (MON_IN_SESSION, 100.0, 5.0),
                (MON_BOUNDARY, 101.0, 7.0),
                (TUE_IN_SESSION, 102.0, 1.0),
            ]
        )
        close_bars = fb.compute_volume_bars_from_minutes(
            df, threshold=1e12, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="close"
        )
        discard_bars = fb.compute_volume_bars_from_minutes(
            df, threshold=1e12, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="discard"
        )
        # Same calendar and input, different per-series policy.
        assert len(close_bars) == 2
        assert len(discard_bars) == 0


class TestDiscardPolicy:
    def test_boundary_drops_in_progress_bar(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(
            threshold=1e12, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="discard"
        )
        assert ctor.update(_mi(MON_IN_SESSION, 100.0, volume=5.0)) is None

        # The boundary minute discards the 5 and starts an off-hours bar.
        assert ctor.update(_mi(MON_BOUNDARY, 101.0, volume=7.0)) is None
        assert ctor._accumulator._cum_volume == pytest.approx(7.0)
        assert ctor.bars_emitted == 0
        assert ctor._accumulator._bar_id == 0

        # The Tuesday open discards the off-hours bar and starts fresh.
        assert ctor.update(_mi(TUE_IN_SESSION, 102.0, volume=3.0)) is None
        assert ctor._accumulator._cum_volume == pytest.approx(3.0)
        assert ctor.bars_emitted == 0

    def test_overflow_does_not_carry(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(
            threshold=4.0, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="discard"
        )
        assert ctor.update(_mi(MON_IN_SESSION, 100.0, volume=3.0)) is None
        # No threshold close yet; the boundary discards the 3 without an
        # overflow roll of 3 - 4.
        assert ctor.update(_mi(MON_BOUNDARY, 100.0, volume=1.0)) is None
        assert ctor._accumulator._cum_volume == pytest.approx(1.0)

    def test_callbacks_do_not_fire(self) -> None:
        seen: list[object] = []
        ctor = fb.MinuteVolumeBarConstructor(
            threshold=1e12,
            calendar=CAL,
            schema=MINUTE_SCHEMA,
            session_policy="discard",
            on_bar=seen.append,
        )
        ctor.update(_mi(MON_IN_SESSION, 100.0, volume=5.0))
        ctor.update(_mi(MON_BOUNDARY, 100.0, volume=5.0))
        ctor.update(_mi(TUE_IN_SESSION, 100.0, volume=5.0))
        assert seen == []


class TestCarryPolicy:
    def test_bar_spans_sessions(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(
            threshold=1e12, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="carry"
        )
        assert ctor.update(_mi(MON_IN_SESSION, 100.0, volume=5.0)) is None
        assert ctor.update(_mi(MON_BOUNDARY, 101.0, volume=7.0)) is None
        assert ctor.update(_mi(TUE_IN_SESSION, 102.0, volume=3.0)) is None

        assert ctor.bars_emitted == 0
        assert ctor._accumulator._cum_volume == pytest.approx(15.0)
        assert ctor._accumulator._num_ticks == 3

    def test_gap_contributes_no_volatility_return(self) -> None:
        ctor = fb.MinuteVolatilityBarConstructor(
            threshold=1e9, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="carry"
        )
        assert ctor.update(_mi(MON_IN_SESSION, 100.0)) is None
        assert ctor.update(_mi(MON_BOUNDARY, 200.0)) is None  # gap: no return
        assert ctor.update(_mi(TUE_IN_SESSION, 210.0)) is None  # session open: no return
        assert ctor.update(_mi(_ts(2024, 1, 16, 10, 1), 220.0)) is None

        # Only the in-session 210 -> 220 return counts (10 / 210); both
        # session-gap transitions contribute nothing, though the bar stays open.
        assert ctor._accumulator._cum_metric == pytest.approx(10.0 / 210.0)
        assert ctor._accumulator._num_ticks == 4

    def test_threshold_close_still_fires(self) -> None:
        ctor = fb.MinuteVolumeBarConstructor(
            threshold=10.0, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="carry"
        )
        assert ctor.update(_mi(MON_IN_SESSION, 100.0, volume=6.0)) is None
        bar = ctor.update(_mi(MON_BOUNDARY, 100.0, volume=6.0))
        assert bar is not None
        assert bar.partial is False
        assert bar.volume == pytest.approx(12.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Estimator persist / reset
# ═══════════════════════════════════════════════════════════════════════════════


def _feed_imbalance_minutes(ctor: fb.MinuteImbalanceVolumeBarConstructor) -> None:
    for i, close in enumerate([100.0, 101.0, 102.0, 103.0, 104.0, 105.0]):
        ctor.update(_mi(_ts(2024, 1, 15, 10, i), close, volume=1.0))


class TestEstimatorPolicy:
    def test_reset_restores_seeds_at_boundary(self) -> None:
        ctor = fb.MinuteImbalanceVolumeBarConstructor(
            span=3.0,
            calendar=CAL,
            schema=MINUTE_SCHEMA,
            session_estimator="reset",
        )
        _feed_imbalance_minutes(ctor)
        before = ctor._threshold_estimator.get_state()
        assert before["n_updates"] >= 2

        ctor.update(_mi(MON_BOUNDARY, 106.0, volume=1.0))
        state = ctor._threshold_estimator.get_state()
        # Reset happens before the boundary minute is ingested, so at most
        # one post-reset bar close may have updated the estimator again.
        assert state["n_updates"] <= 1
        assert state["ewa_t"] == pytest.approx(before["initial_ewa_t"])
        assert state["ewa_proportion"] != pytest.approx(before["ewa_proportion"])

    def test_persist_keeps_learning(self) -> None:
        ctor = fb.MinuteImbalanceVolumeBarConstructor(
            span=3.0,
            calendar=CAL,
            schema=MINUTE_SCHEMA,
            session_estimator="persist",
        )
        _feed_imbalance_minutes(ctor)
        before = ctor._threshold_estimator.get_state()["n_updates"]
        assert before > 0

        ctor.update(_mi(MON_BOUNDARY, 106.0, volume=1.0))
        assert ctor._threshold_estimator.get_state()["n_updates"] == before


# ═══════════════════════════════════════════════════════════════════════════════
# State persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionState:
    def test_policy_survives_state_roundtrip(self) -> None:
        df = _minutes_df(
            [
                (MON_IN_SESSION, 100.0, 1.0),
                (_ts(2024, 1, 15, 10, 1), 100.5, 1.0),
                (MON_BOUNDARY, 101.0, 2.0),
                (TUE_IN_SESSION, 102.0, 1.0),
                (_ts(2024, 1, 16, 10, 1), 102.5, 1.0),
            ]
        )
        ctor = fb.MinuteVolumeBarConstructor(
            threshold=2.5,
            calendar=CAL,
            schema=MINUTE_SCHEMA,
            session_policy="discard",
            session_estimator="persist",
        )
        first = ctor.update_many(df.iloc[:3])
        state = json.loads(json.dumps(ctor.get_state()))
        assert state["session_policy"] == "discard"
        assert state["params"]["session_policy"] == "discard"

        resumed = fb.MinuteVolumeBarConstructor.from_state(
            state, calendar=CAL, schema=MINUTE_SCHEMA
        )
        assert resumed.get_params()["session_policy"] == "discard"
        assert resumed.get_params()["session_estimator"] == "persist"
        second = resumed.update_many(df.iloc[3:])

        full = fb.MinuteVolumeBarConstructor(
            threshold=2.5,
            calendar=CAL,
            schema=MINUTE_SCHEMA,
            session_policy="discard",
            session_estimator="persist",
        ).update_many(df)
        _assert_frames_equal(pd.concat([first, second], ignore_index=True), full)

    def test_resume_with_mismatched_policy_rejected(self) -> None:
        from flowbars.core import StateValidationError

        ctor = fb.MinuteVolumeBarConstructor(
            threshold=2.5, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="discard"
        )
        ctor.update(_mi(MON_IN_SESSION, 100.0))
        state = ctor.get_state()

        with pytest.raises(StateValidationError):
            fb.MinuteVolumeBarConstructor(
                threshold=2.5, calendar=CAL, schema=MINUTE_SCHEMA, session_policy="close"
            ).load_state(state)


# ═══════════════════════════════════════════════════════════════════════════════
# Exchange calendars: DST and early closes
# ═══════════════════════════════════════════════════════════════════════════════


class TestExchangeCalendarBoundaries:
    @pytest.fixture(autouse=True)
    def _require_exchange_calendars(self) -> None:
        pytest.importorskip("exchange_calendars")

    def test_early_close_is_a_boundary(self) -> None:
        cal = ExchangeCalendar("XNYS")
        # 2024-11-29 (day after Thanksgiving) closes 13:00 ET = 18:00 UTC.
        assert cal.is_session_boundary(_ts(2024, 11, 29, 18, 30)) is True
        # A regular Monday is still open at 18:30 UTC (13:30 ET).
        assert cal.is_session_boundary(_ts(2024, 12, 2, 18, 30)) is False

    def test_dst_shift_moves_the_utc_close(self) -> None:
        cal = ExchangeCalendar("XNYS")
        # 2024-03-08 (EST): 20:30 UTC = 15:30 ET -> open.
        assert cal.is_session_boundary(_ts(2024, 3, 8, 20, 30)) is False
        # 2024-03-11 (EDT): 20:30 UTC = 16:30 ET -> closed.
        assert cal.is_session_boundary(_ts(2024, 3, 11, 20, 30)) is True

    def test_constructor_force_closes_at_early_close(self) -> None:
        cal = ExchangeCalendar("XNYS")
        ctor = fb.MinuteVolumeBarConstructor(
            threshold=1e12, calendar=cal, schema=MINUTE_SCHEMA, session_policy="close"
        )
        assert ctor.update(_mi(_ts(2024, 11, 29, 17, 0), 100.0, volume=4.0)) is None
        bar = ctor.update(_mi(_ts(2024, 11, 29, 18, 30), 101.0, volume=2.0))
        assert bar is not None and bar.partial is True
        assert bar.volume == pytest.approx(4.0)
