"""Shared engine for the minute-OHLCV → bars path.

The minute path reuses :class:`~flowbars.bars.constructor.BaseBarConstructor`
for all orchestration (streaming ``update``, batch, warmup, watermark dedup,
strict ordering, callbacks, session calendars, and state save/resume).  This
module supplies the minute-specific input hooks:

* :meth:`BaseMinuteBarConstructor._extract_batch_arrays` — pulls
  ``(timestamps, opens, highs, lows, closes, volumes, signs)`` from a
  :class:`~flowbars.schema.MinuteSchemaMapping` DataFrame and derives minute
  signs with the tick rule applied to closes.
* :meth:`BaseMinuteBarConstructor._prepare_input` — derives the minute tick
  rule for streaming inputs whose ``side`` is ``None``.
* :meth:`BaseMinuteBarConstructor._ingest_input` — calls
  ``accumulator.add_minute``.

Design decisions are specified in the repo's ``SPEC.md`` under
"Minute-OHLCV → bars path — design". Key points: direct OHLCV accumulation,
whole-minute attribution (only the scalar metric excess rolls), the minute
tick rule ``sign(close_i − close_{i−1})`` with first-minute ``NaN``,
``num_ticks`` counts **minutes**, and ``dollar_value``/``vwap`` use
``close × volume`` as the notional proxy.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd

from flowbars.bars.constructor import BaseBarConstructor
from flowbars.core import MinuteInfo, TickInfo
from flowbars.schema import MinuteSchemaMapping
from flowbars.thresholds import EWMAThresholdEstimator, ThresholdEstimator
from flowbars.tick_rule import resolve_tick_signs

_DEFAULT_MINUTE_SCHEMA = MinuteSchemaMapping(
    {
        "timestamp": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
)


class BaseMinuteBarConstructor(BaseBarConstructor):
    """Shared engine for bar types built from minute OHLCV candles."""

    input_kind: ClassVar[str] = "minute"

    # Session-end behavior for the in-progress bar.  Minute bars opt in to
    # the full policy set; tick bars support force-close only.
    session_policies: ClassVar[tuple[str, ...]] = ("close", "discard", "carry")

    def __init__(
        self,
        accumulator: Any,
        threshold_estimator: ThresholdEstimator,
        calendar: Any = None,
        schema: MinuteSchemaMapping | None = None,
        watermark: str | None = "timestamp",
        stream_id: str = "default",
        strict_ordering: bool = False,
        warmup_bars: int = 0,
        warmup_minutes: int | None = None,
        backend: Any = "python",
        on_bar: Any = None,
        on_threshold_update: Any = None,
        multiplier: float = 1.0,
        volume_type: str = "base",
        session_policy: Literal["close", "carry", "discard"] = "close",
        session_estimator: Literal["persist", "reset"] = "persist",
    ) -> None:
        super().__init__(
            accumulator=accumulator,
            threshold_estimator=threshold_estimator,
            calendar=calendar,
            schema=schema,
            watermark=watermark,
            stream_id=stream_id,
            strict_ordering=strict_ordering,
            warmup_bars=warmup_bars,
            warmup_minutes=warmup_minutes,
            backend=backend,
            on_bar=on_bar,
            on_threshold_update=on_threshold_update,
            multiplier=multiplier,
            volume_type=volume_type,
            session_policy=session_policy,
            session_estimator=session_estimator,
        )
        # Previous minute close/derived sign for the streaming tick rule.
        self._prev_close: float | None = None
        self._prev_sign: float = np.nan

    # ── input hooks ─────────────────────────────────────────────────────

    def _default_schema(self) -> MinuteSchemaMapping:
        return _DEFAULT_MINUTE_SCHEMA

    def _extract_batch_arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, ...]:
        assert isinstance(self._schema, MinuteSchemaMapping)
        timestamps, opens, highs, lows, closes, volumes, sides = self._schema.extract_arrays(df)
        signs = resolve_tick_signs(closes, sides)
        return (timestamps, opens, highs, lows, closes, volumes, signs)

    def _make_input(
        self, arrays: tuple[np.ndarray, ...], i: int, watermark: np.ndarray | None
    ) -> MinuteInfo:
        timestamps, opens, highs, lows, closes, volumes, signs = arrays
        return MinuteInfo(
            timestamp=int(timestamps[i]),
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
            volume=float(volumes[i]),
            side=float(signs[i]),
            watermark=int(watermark[i]) if watermark is not None else None,
        )

    def _prepare_input(self, item: TickInfo | MinuteInfo) -> TickInfo | MinuteInfo:
        """Derive the minute tick rule when no side was supplied."""
        if not isinstance(item, MinuteInfo):
            raise TypeError(f"Minute constructors require MinuteInfo, got {type(item).__name__}")

        close = item.close
        if item.side is None:
            if self._prev_close is None:
                sign = np.nan
            else:
                diff = close - self._prev_close
                if diff > 0.0:
                    sign = 1.0
                elif diff < 0.0:
                    sign = -1.0
                else:
                    sign = self._prev_sign
            if not math.isnan(sign):
                item = replace(item, side=sign)
            self._prev_sign = sign
        else:
            self._prev_sign = item.side

        self._prev_close = close
        return item

    def _ingest_input(self, item: TickInfo | MinuteInfo) -> None:
        assert isinstance(item, MinuteInfo)
        self._accumulator.add_minute(item)

    # ── state ───────────────────────────────────────────────────────────

    def _extra_state(self) -> dict[str, Any]:
        prev_sign = self._prev_sign
        return {
            "prev_close": self._prev_close,
            "prev_sign": None if math.isnan(prev_sign) else prev_sign,
        }

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        self._prev_close = state.get("prev_close")
        prev_sign = state.get("prev_sign")
        self._prev_sign = np.nan if prev_sign is None else float(prev_sign)

    # ── numba dispatch ──────────────────────────────────────────────────

    def _numba_config(self) -> dict[str, float]:
        estimator = self._threshold_estimator
        config: dict[str, float] = {"multiplier": self.multiplier}
        if isinstance(estimator, EWMAThresholdEstimator):
            config["alpha"] = estimator.alpha
            config["initial_ewa_t"] = estimator.ewa_t
            config["initial_ewa_proportion"] = estimator.ewa_proportion
        else:
            config["threshold"] = estimator.current_threshold
        volatility_threshold = getattr(self._accumulator, "_volatility_threshold", None)
        if volatility_threshold is not None:
            config["volatility_threshold"] = float(volatility_threshold)
        return config

    def _batch_numba(
        self,
        arrays: tuple[np.ndarray, ...],
        watermark: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Run the batch loop through the minute numba kernels.

        Falls back to the Python engine when numba is unavailable, the
        calendar produces session boundaries, or the constructor is not
        pristine.  The kernels are stateless — they do not advance the Python
        accumulator/estimator — so incremental batches must use the Python
        backend, which is what the non-pristine fallback does.
        """
        import warnings

        from flowbars.bars.numba_backend import _NUMBA_AVAILABLE, _get_compilation_warning

        if not _NUMBA_AVAILABLE:
            _get_compilation_warning()
            return self._batch_python(arrays, watermark)

        if self._calendar.has_sessions:
            warnings.warn(
                "This calendar produces session boundaries, which the numba "
                "backend does not support. Falling back to the Python path.",
                stacklevel=2,
            )
            return self._batch_python(arrays, watermark)

        if not self._is_pristine():
            warnings.warn(
                "The numba backend requires a pristine constructor (no prior "
                "update()/batch() calls and no restored state). Falling back "
                "to the Python path.",
                stacklevel=2,
            )
            return self._batch_python(arrays, watermark)

        from flowbars.bars.minute.numba import numba_batch_minute

        bar_type = self._accumulator._bar_type
        try:
            bar_data, bt = numba_batch_minute(bar_type, arrays, self._numba_config())
        except Exception:
            warnings.warn(
                f"numba compilation/execution failed for {bar_type}. "
                "Falling back to the Python path.",
                stacklevel=2,
            )
            return self._batch_python(arrays, watermark)

        return self._finalize_numba_result(bar_data, bt)
