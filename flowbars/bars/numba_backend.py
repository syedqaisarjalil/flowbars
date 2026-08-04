"""Numba-accelerated batch bar construction.

Provides JIT-compiled versions of the hot tick-ingestion loop for each
bar type.  Falls back gracefully to the Python backend when numba is
not installed or compilation fails.

Architecture
------------
Each bar type has a dedicated ``@numba.njit`` function that processes raw
numpy arrays directly — no Python objects in the inner loop.  All helper
logic is inlined to satisfy numba's nopython-mode restrictions.

The Python wrapper functions (:func:`numba_batch_static`,
:func:`numba_batch_ewma`) handle DataFrame conversion, tick-sign
derivation, and callback invocation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# ── numba availability ──────────────────────────────────────────────────

_NUMBA_AVAILABLE = False
_numba: Any = None

try:
    import numba

    _numba = numba
    _NUMBA_AVAILABLE = True
except ImportError:
    pass


def is_numba_available() -> bool:
    """Return ``True`` if numba is installed and importable."""
    return _NUMBA_AVAILABLE


# ── Output column indices ───────────────────────────────────────────────

_COL_BAR_ID = 0
_COL_OPEN = 1
_COL_HIGH = 2
_COL_LOW = 3
_COL_CLOSE = 4
_COL_VOLUME = 5
_COL_DOLLAR_VALUE = 6
_COL_VWAP = 7
_COL_NUM_TICKS = 8
_COL_OPEN_TS = 9
_COL_CLOSE_TS = 10
_NUM_COLS = 11

# ── Bar-type dispatch registry ──────────────────────────────────────────

_NUMBA_DISPATCH: dict[str, Any] = {}


def _register(bar_type: str, func: Any) -> None:
    """Register a numba batch function for *bar_type*."""
    _NUMBA_DISPATCH[bar_type] = func


# ═══════════════════════════════════════════════════════════════════════════════
# JIT compilation helper
# ═══════════════════════════════════════════════════════════════════════════════


def _njit(func):  # type: ignore[no-untyped-def]
    """JIT-compile *func* with numba, or return it as-is if numba is absent."""
    if _NUMBA_AVAILABLE and _numba is not None:
        return _numba.njit(cache=True)(func)
    return func


# ═══════════════════════════════════════════════════════════════════════════════
# Standard-bar JIT functions
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each function takes raw float64 arrays and a threshold, runs a tight
# tick-by-tick loop, and returns a (N, 11) float64 array of bar rows.
# The emit logic is inlined because numba nopython mode can only call
# other @njit functions — the _njit decorator returns a plain Python
# function when numba is absent, which would break the nopython chain.


@_njit  # type: ignore[untyped-decorator]
def _tick_bars_numba(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Numba kernel for tick bars (static threshold)."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    cum_ticks = 0.0
    has_tick = False
    bar_id = 0
    bar_count = 0
    open_val = 0.0
    high_val = 0.0
    low_val = 0.0
    close_val = 0.0
    vol = 0.0
    dollar_val = 0.0
    n_ticks = 0
    open_ts_val = 0
    close_ts_val = 0

    for i in range(n):
        price = prices[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])

        if not has_tick:
            open_val = price
            high_val = price
            low_val = price
            open_ts_val = ts
            has_tick = True
        else:
            if price > high_val:
                high_val = price
            if price < low_val:
                low_val = price

        close_val = price
        vol += vol_i
        dollar_val += price * vol_i
        n_ticks += 1
        close_ts_val = ts
        cum_ticks += 1.0

        if cum_ticks >= threshold:
            vwap = dollar_val / vol if vol > 0.0 else 0.0
            # Inline emit
            out[bar_count, _COL_BAR_ID] = bar_id
            out[bar_count, _COL_OPEN] = open_val
            out[bar_count, _COL_HIGH] = high_val
            out[bar_count, _COL_LOW] = low_val
            out[bar_count, _COL_CLOSE] = close_val
            out[bar_count, _COL_VOLUME] = vol
            out[bar_count, _COL_DOLLAR_VALUE] = dollar_val
            out[bar_count, _COL_VWAP] = vwap
            out[bar_count, _COL_NUM_TICKS] = n_ticks
            out[bar_count, _COL_OPEN_TS] = open_ts_val
            out[bar_count, _COL_CLOSE_TS] = close_ts_val
            bar_count += 1
            bar_id += 1
            # Reset with overflow
            excess = cum_ticks - threshold
            cum_ticks = excess if excess > 0.0 else 0.0
            has_tick = False
            open_val = 0.0
            high_val = 0.0
            low_val = 0.0
            close_val = 0.0
            vol = 0.0
            dollar_val = 0.0
            n_ticks = 0
            open_ts_val = 0
            close_ts_val = 0

    if bar_count == 0:
        return out[:0]
    return out[:bar_count]


@_njit  # type: ignore[untyped-decorator]
def _volume_bars_numba(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Numba kernel for volume bars (static threshold)."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    cum_volume = 0.0
    has_tick = False
    bar_id = 0
    bar_count = 0
    open_val = 0.0
    high_val = 0.0
    low_val = 0.0
    close_val = 0.0
    vol = 0.0
    dollar_val = 0.0
    n_ticks = 0
    open_ts_val = 0
    close_ts_val = 0

    for i in range(n):
        price = prices[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])

        if not has_tick:
            open_val = price
            high_val = price
            low_val = price
            open_ts_val = ts
            has_tick = True
        else:
            if price > high_val:
                high_val = price
            if price < low_val:
                low_val = price

        close_val = price
        vol += vol_i
        dollar_val += price * vol_i
        n_ticks += 1
        close_ts_val = ts
        cum_volume += vol_i

        if cum_volume >= threshold:
            vwap = dollar_val / vol if vol > 0.0 else 0.0
            out[bar_count, _COL_BAR_ID] = bar_id
            out[bar_count, _COL_OPEN] = open_val
            out[bar_count, _COL_HIGH] = high_val
            out[bar_count, _COL_LOW] = low_val
            out[bar_count, _COL_CLOSE] = close_val
            out[bar_count, _COL_VOLUME] = vol
            out[bar_count, _COL_DOLLAR_VALUE] = dollar_val
            out[bar_count, _COL_VWAP] = vwap
            out[bar_count, _COL_NUM_TICKS] = n_ticks
            out[bar_count, _COL_OPEN_TS] = open_ts_val
            out[bar_count, _COL_CLOSE_TS] = close_ts_val
            bar_count += 1
            bar_id += 1
            # Reset with overflow
            excess = cum_volume - threshold
            cum_volume = excess
            has_tick = False
            open_val = 0.0
            high_val = 0.0
            low_val = 0.0
            close_val = 0.0
            vol = 0.0
            dollar_val = 0.0
            n_ticks = 0
            open_ts_val = 0
            close_ts_val = 0

    if bar_count == 0:
        return out[:0]
    return out[:bar_count]


@_njit  # type: ignore[untyped-decorator]
def _dollar_bars_numba(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Numba kernel for dollar bars (static threshold)."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    cum_dollar = 0.0
    has_tick = False
    bar_id = 0
    bar_count = 0
    open_val = 0.0
    high_val = 0.0
    low_val = 0.0
    close_val = 0.0
    vol = 0.0
    dollar_val = 0.0
    n_ticks = 0
    open_ts_val = 0
    close_ts_val = 0

    for i in range(n):
        price = prices[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])

        if not has_tick:
            open_val = price
            high_val = price
            low_val = price
            open_ts_val = ts
            has_tick = True
        else:
            if price > high_val:
                high_val = price
            if price < low_val:
                low_val = price

        close_val = price
        vol += vol_i
        dollar_val += price * vol_i
        n_ticks += 1
        close_ts_val = ts
        cum_dollar += price * vol_i

        if cum_dollar >= threshold:
            vwap = dollar_val / vol if vol > 0.0 else 0.0
            out[bar_count, _COL_BAR_ID] = bar_id
            out[bar_count, _COL_OPEN] = open_val
            out[bar_count, _COL_HIGH] = high_val
            out[bar_count, _COL_LOW] = low_val
            out[bar_count, _COL_CLOSE] = close_val
            out[bar_count, _COL_VOLUME] = vol
            out[bar_count, _COL_DOLLAR_VALUE] = dollar_val
            out[bar_count, _COL_VWAP] = vwap
            out[bar_count, _COL_NUM_TICKS] = n_ticks
            out[bar_count, _COL_OPEN_TS] = open_ts_val
            out[bar_count, _COL_CLOSE_TS] = close_ts_val
            bar_count += 1
            bar_id += 1
            # Reset with overflow
            excess = cum_dollar - threshold
            cum_dollar = excess
            has_tick = False
            open_val = 0.0
            high_val = 0.0
            low_val = 0.0
            close_val = 0.0
            vol = 0.0
            dollar_val = 0.0
            n_ticks = 0
            open_ts_val = 0
            close_ts_val = 0

    if bar_count == 0:
        return out[:0]
    return out[:bar_count]


@_njit  # type: ignore[untyped-decorator]
def _time_bars_numba(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    interval_ms: int,
    anchor_code: int,
) -> np.ndarray:
    """Numba kernel for time bars.

    Parameters
    ----------
    anchor_code : int
        0 = clock, 1 = first_tick.
    """
    n = len(timestamps)
    if n == 0:
        return np.zeros((0, _NUM_COLS), dtype=np.float64)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    has_tick = False
    bar_id = 0
    bar_count = 0
    open_val = 0.0
    high_val = 0.0
    low_val = 0.0
    close_val = 0.0
    vol = 0.0
    dollar_val = 0.0
    n_ticks = 0
    open_ts_val = 0
    close_ts_val = 0

    # Compute first boundary
    first_ts = int(timestamps[0])
    if anchor_code == 1:  # first_tick
        next_boundary = first_ts + interval_ms
    else:  # clock
        remainder = first_ts % interval_ms
        if remainder == 0:
            next_boundary = first_ts + interval_ms
        else:
            next_boundary = first_ts - remainder + interval_ms

    for i in range(n):
        price = prices[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])

        if not has_tick:
            open_val = price
            high_val = price
            low_val = price
            open_ts_val = ts
            has_tick = True
        else:
            if price > high_val:
                high_val = price
            if price < low_val:
                low_val = price

        close_val = price
        vol += vol_i
        dollar_val += price * vol_i
        n_ticks += 1
        close_ts_val = ts

        if ts >= next_boundary:
            vwap = dollar_val / vol if vol > 0.0 else 0.0
            out[bar_count, _COL_BAR_ID] = bar_id
            out[bar_count, _COL_OPEN] = open_val
            out[bar_count, _COL_HIGH] = high_val
            out[bar_count, _COL_LOW] = low_val
            out[bar_count, _COL_CLOSE] = close_val
            out[bar_count, _COL_VOLUME] = vol
            out[bar_count, _COL_DOLLAR_VALUE] = dollar_val
            out[bar_count, _COL_VWAP] = vwap
            out[bar_count, _COL_NUM_TICKS] = n_ticks
            out[bar_count, _COL_OPEN_TS] = open_ts_val
            out[bar_count, _COL_CLOSE_TS] = close_ts_val
            bar_count += 1
            bar_id += 1
            # Reset
            has_tick = False
            open_val = 0.0
            high_val = 0.0
            low_val = 0.0
            close_val = 0.0
            vol = 0.0
            dollar_val = 0.0
            n_ticks = 0
            open_ts_val = 0
            close_ts_val = 0
            # Advance boundary
            next_boundary += interval_ms

    if bar_count == 0:
        return out[:0]
    return out[:bar_count]


# Register standard bars
_register("tick", _tick_bars_numba)
_register("volume", _volume_bars_numba)
_register("dollar", _dollar_bars_numba)
# time bars use a different signature — handled specially in numba_batch_static


# ═══════════════════════════════════════════════════════════════════════════════
# Information-driven-bar JIT functions (static threshold)
# ═══════════════════════════════════════════════════════════════════════════════


@_njit  # type: ignore[untyped-decorator]
def _imbalance_bars_numba(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    threshold: float,
    metric_code: int,
) -> np.ndarray:
    """Numba kernel for imbalance bars (static threshold).

    Parameters
    ----------
    metric_code : int
        0 = tick, 1 = volume, 2 = dollar.
    """
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    signed_imbalance = 0.0
    has_tick = False
    bar_id = 0
    bar_count = 0
    open_val = 0.0
    high_val = 0.0
    low_val = 0.0
    close_val = 0.0
    vol = 0.0
    dollar_val = 0.0
    n_ticks = 0
    open_ts_val = 0
    close_ts_val = 0

    for i in range(n):
        price = prices[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])
        side = sides[i]

        if not has_tick:
            open_val = price
            high_val = price
            low_val = price
            open_ts_val = ts
            has_tick = True
        else:
            if price > high_val:
                high_val = price
            if price < low_val:
                low_val = price

        close_val = price
        vol += vol_i
        dollar_val += price * vol_i
        n_ticks += 1
        close_ts_val = ts

        # First-tick exclusion: if side is NaN, imbalance contribution is 0
        if not np.isnan(side):
            # Inline metric computation
            if metric_code == 0:
                metric_val = 1.0
            elif metric_code == 1:
                metric_val = vol_i
            else:  # dollar
                metric_val = price * vol_i
            signed_imbalance += side * metric_val

        if abs(signed_imbalance) >= threshold:
            vwap = dollar_val / vol if vol > 0.0 else 0.0
            out[bar_count, _COL_BAR_ID] = bar_id
            out[bar_count, _COL_OPEN] = open_val
            out[bar_count, _COL_HIGH] = high_val
            out[bar_count, _COL_LOW] = low_val
            out[bar_count, _COL_CLOSE] = close_val
            out[bar_count, _COL_VOLUME] = vol
            out[bar_count, _COL_DOLLAR_VALUE] = dollar_val
            out[bar_count, _COL_VWAP] = vwap
            out[bar_count, _COL_NUM_TICKS] = n_ticks
            out[bar_count, _COL_OPEN_TS] = open_ts_val
            out[bar_count, _COL_CLOSE_TS] = close_ts_val
            bar_count += 1
            bar_id += 1
            # Overflow: signed excess carries into next bar
            if signed_imbalance >= 0.0:
                sign = 1.0
            else:
                sign = -1.0
            excess = abs(signed_imbalance) - threshold
            signed_imbalance = sign * excess if excess > 0.0 else 0.0
            has_tick = False
            open_val = 0.0
            high_val = 0.0
            low_val = 0.0
            close_val = 0.0
            vol = 0.0
            dollar_val = 0.0
            n_ticks = 0
            open_ts_val = 0
            close_ts_val = 0

    if bar_count == 0:
        return out[:0]
    return out[:bar_count]


@_njit  # type: ignore[untyped-decorator]
def _run_bars_numba(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    threshold: float,
    metric_code: int,
) -> np.ndarray:
    """Numba kernel for run bars (static threshold).

    Parameters
    ----------
    metric_code : int
        0 = tick, 1 = volume, 2 = dollar.
    """
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    has_tick = False
    bar_id = 0
    bar_count = 0
    open_val = 0.0
    high_val = 0.0
    low_val = 0.0
    close_val = 0.0
    vol = 0.0
    dollar_val = 0.0
    n_ticks = 0
    open_ts_val = 0
    close_ts_val = 0

    first_tick = True
    banked = 0.0
    run_sign = np.nan
    run_cum = 0.0

    for i in range(n):
        price = prices[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])
        if np.isnan(sides[i]):
            side = np.nan
        else:
            side = sides[i]

        if not has_tick:
            open_val = price
            high_val = price
            low_val = price
            open_ts_val = ts
            has_tick = True
        else:
            if price > high_val:
                high_val = price
            if price < low_val:
                low_val = price

        close_val = price
        vol += vol_i
        dollar_val += price * vol_i
        n_ticks += 1
        close_ts_val = ts

        # Inline metric computation
        if metric_code == 0:
            metric_val = 1.0
        elif metric_code == 1:
            metric_val = vol_i
        else:  # dollar
            metric_val = price * vol_i

        if first_tick:
            first_tick = False
            run_sign = side
            run_cum = metric_val
        elif np.isnan(run_sign) or np.isnan(side) or run_sign == side:
            # Same direction — continue the run
            run_cum += metric_val
            if np.isnan(run_sign) and not np.isnan(side):
                run_sign = side
        else:
            # Direction change: bank current run, start new one
            banked += run_cum
            run_sign = side
            run_cum = metric_val

        total = banked + run_cum
        if total >= threshold:
            vwap = dollar_val / vol if vol > 0.0 else 0.0
            out[bar_count, _COL_BAR_ID] = bar_id
            out[bar_count, _COL_OPEN] = open_val
            out[bar_count, _COL_HIGH] = high_val
            out[bar_count, _COL_LOW] = low_val
            out[bar_count, _COL_CLOSE] = close_val
            out[bar_count, _COL_VOLUME] = vol
            out[bar_count, _COL_DOLLAR_VALUE] = dollar_val
            out[bar_count, _COL_VWAP] = vwap
            out[bar_count, _COL_NUM_TICKS] = n_ticks
            out[bar_count, _COL_OPEN_TS] = open_ts_val
            out[bar_count, _COL_CLOSE_TS] = close_ts_val
            bar_count += 1
            bar_id += 1
            # Overflow: excess from total carries into next bar
            excess = total - threshold
            banked = 0.0
            if excess > 0.0:
                run_cum = excess
            else:
                run_cum = 0.0
                run_sign = np.nan
            has_tick = False
            open_val = 0.0
            high_val = 0.0
            low_val = 0.0
            close_val = 0.0
            vol = 0.0
            dollar_val = 0.0
            n_ticks = 0
            open_ts_val = 0
            close_ts_val = 0

    if bar_count == 0:
        return out[:0]
    return out[:bar_count]


# Register information-driven bars (static threshold)
_register("imbalance_tick", _imbalance_bars_numba)
_register("imbalance_volume", _imbalance_bars_numba)
_register("imbalance_dollar", _imbalance_bars_numba)
_register("run_tick", _run_bars_numba)
_register("run_volume", _run_bars_numba)
_register("run_dollar", _run_bars_numba)


# ═══════════════════════════════════════════════════════════════════════════════
# EWMA-enabled JIT functions for information-driven bars
# ═══════════════════════════════════════════════════════════════════════════════


@_njit  # type: ignore[untyped-decorator]
def _imbalance_bars_ewma_numba(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    metric_code: int,
    alpha: float,
    initial_ewa_t: float,
    initial_ewa_proportion: float,
) -> np.ndarray:
    """Numba kernel for imbalance bars with in-loop EWMA threshold updates."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    signed_imbalance = 0.0
    has_tick = False
    bar_id = 0
    bar_count = 0
    open_val = 0.0
    high_val = 0.0
    low_val = 0.0
    close_val = 0.0
    vol = 0.0
    dollar_val = 0.0
    n_ticks = 0
    open_ts_val = 0
    close_ts_val = 0

    # EWMA state — proportion tracked signed, abs() only for threshold
    ewa_t = initial_ewa_t
    ewa_proportion = initial_ewa_proportion  # signed
    threshold_val = ewa_t * abs(ewa_proportion)
    if threshold_val <= 0.0:
        threshold_val = 1.0

    for i in range(n):
        price = prices[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])
        side = sides[i]

        if not has_tick:
            open_val = price
            high_val = price
            low_val = price
            open_ts_val = ts
            has_tick = True
        else:
            if price > high_val:
                high_val = price
            if price < low_val:
                low_val = price

        close_val = price
        vol += vol_i
        dollar_val += price * vol_i
        n_ticks += 1
        close_ts_val = ts

        if not np.isnan(side):
            if metric_code == 0:
                metric_val = 1.0
            elif metric_code == 1:
                metric_val = vol_i
            else:
                metric_val = price * vol_i
            signed_imbalance += side * metric_val

        if abs(signed_imbalance) >= threshold_val:
            vwap = dollar_val / vol if vol > 0.0 else 0.0
            out[bar_count, _COL_BAR_ID] = bar_id
            out[bar_count, _COL_OPEN] = open_val
            out[bar_count, _COL_HIGH] = high_val
            out[bar_count, _COL_LOW] = low_val
            out[bar_count, _COL_CLOSE] = close_val
            out[bar_count, _COL_VOLUME] = vol
            out[bar_count, _COL_DOLLAR_VALUE] = dollar_val
            out[bar_count, _COL_VWAP] = vwap
            out[bar_count, _COL_NUM_TICKS] = n_ticks
            out[bar_count, _COL_OPEN_TS] = open_ts_val
            out[bar_count, _COL_CLOSE_TS] = close_ts_val
            bar_count += 1
            bar_id += 1

            # Save old threshold for overflow computation
            old_threshold = threshold_val

            # Compute bar statistics for EWMA update
            if metric_code == 0:
                t_stat = float(n_ticks)
            elif metric_code == 1:
                t_stat = vol
            else:
                t_stat = dollar_val

            proportion_stat = signed_imbalance / t_stat if t_stat > 0.0 else 0.0

            # EWMA update — produces the threshold for the NEXT bar
            ewa_t = alpha * t_stat + (1.0 - alpha) * ewa_t
            ewa_proportion = alpha * proportion_stat + (1.0 - alpha) * ewa_proportion
            threshold_val = ewa_t * abs(ewa_proportion)
            if threshold_val <= 0.0:
                threshold_val = 1.0

            # Overflow — uses the OLD threshold (the one that triggered the close)
            if signed_imbalance >= 0.0:
                sign = 1.0
            else:
                sign = -1.0
            excess = abs(signed_imbalance) - old_threshold
            signed_imbalance = sign * excess if excess > 0.0 else 0.0
            has_tick = False
            open_val = 0.0
            high_val = 0.0
            low_val = 0.0
            close_val = 0.0
            vol = 0.0
            dollar_val = 0.0
            n_ticks = 0
            open_ts_val = 0
            close_ts_val = 0

    if bar_count == 0:
        return out[:0]
    return out[:bar_count]


@_njit  # type: ignore[untyped-decorator]
def _run_bars_ewma_numba(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    metric_code: int,
    alpha: float,
    initial_ewa_t: float,
    initial_ewa_proportion: float,
) -> np.ndarray:
    """Numba kernel for run bars with in-loop EWMA threshold updates."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    has_tick = False
    bar_id = 0
    bar_count = 0
    open_val = 0.0
    high_val = 0.0
    low_val = 0.0
    close_val = 0.0
    vol = 0.0
    dollar_val = 0.0
    n_ticks = 0
    open_ts_val = 0
    close_ts_val = 0

    first_tick = True
    banked = 0.0
    run_sign = np.nan
    run_cum = 0.0
    buy_cum = 0.0
    sell_cum = 0.0

    # EWMA state
    ewa_t = initial_ewa_t
    ewa_proportion = initial_ewa_proportion
    if ewa_proportion >= 1.0 - ewa_proportion:
        threshold_val = ewa_t * ewa_proportion
    else:
        threshold_val = ewa_t * (1.0 - ewa_proportion)
    if threshold_val <= 0.0:
        threshold_val = 1.0

    for i in range(n):
        price = prices[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])
        if np.isnan(sides[i]):
            side = np.nan
        else:
            side = sides[i]

        if not has_tick:
            open_val = price
            high_val = price
            low_val = price
            open_ts_val = ts
            has_tick = True
        else:
            if price > high_val:
                high_val = price
            if price < low_val:
                low_val = price

        close_val = price
        vol += vol_i
        dollar_val += price * vol_i
        n_ticks += 1
        close_ts_val = ts

        if metric_code == 0:
            metric_val = 1.0
        elif metric_code == 1:
            metric_val = vol_i
        else:
            metric_val = price * vol_i

        if first_tick:
            first_tick = False
            run_sign = side
            run_cum = metric_val
        elif np.isnan(run_sign) or np.isnan(side) or run_sign == side:
            run_cum += metric_val
            if np.isnan(run_sign) and not np.isnan(side):
                run_sign = side
        else:
            # Direction change: bank current run
            banked += run_cum
            if run_sign > 0.0:
                buy_cum += run_cum
            elif run_sign < 0.0:
                sell_cum += run_cum
            run_sign = side
            run_cum = metric_val

        total = banked + run_cum
        if total >= threshold_val:
            vwap = dollar_val / vol if vol > 0.0 else 0.0
            out[bar_count, _COL_BAR_ID] = bar_id
            out[bar_count, _COL_OPEN] = open_val
            out[bar_count, _COL_HIGH] = high_val
            out[bar_count, _COL_LOW] = low_val
            out[bar_count, _COL_CLOSE] = close_val
            out[bar_count, _COL_VOLUME] = vol
            out[bar_count, _COL_DOLLAR_VALUE] = dollar_val
            out[bar_count, _COL_VWAP] = vwap
            out[bar_count, _COL_NUM_TICKS] = n_ticks
            out[bar_count, _COL_OPEN_TS] = open_ts_val
            out[bar_count, _COL_CLOSE_TS] = close_ts_val
            bar_count += 1
            bar_id += 1

            # Save old threshold for overflow computation
            old_threshold = threshold_val

            # Compute bar statistics for EWMA update
            t_stat = total

            # P⁺ computation
            cur_buy = buy_cum
            cur_sell = sell_cum
            if run_sign > 0.0:
                cur_buy += run_cum
            elif run_sign < 0.0:
                cur_sell += run_cum

            total_buysell = cur_buy + cur_sell
            if total_buysell > 0.0:
                p_plus = cur_buy / total_buysell
            else:
                p_plus = 0.5

            # EWMA update
            ewa_t = alpha * t_stat + (1.0 - alpha) * ewa_t
            ewa_proportion = alpha * p_plus + (1.0 - alpha) * ewa_proportion
            if ewa_proportion >= 1.0 - ewa_proportion:
                threshold_val = ewa_t * ewa_proportion
            else:
                threshold_val = ewa_t * (1.0 - ewa_proportion)
            if threshold_val <= 0.0:
                threshold_val = 1.0

            # Overflow — uses the OLD threshold (the one that triggered the close)
            excess = total - old_threshold
            banked = 0.0
            buy_cum = 0.0
            sell_cum = 0.0
            if excess > 0.0:
                run_cum = excess
            else:
                run_cum = 0.0
                run_sign = np.nan
            has_tick = False
            open_val = 0.0
            high_val = 0.0
            low_val = 0.0
            close_val = 0.0
            vol = 0.0
            dollar_val = 0.0
            n_ticks = 0
            open_ts_val = 0
            close_ts_val = 0

    if bar_count == 0:
        return out[:0]
    return out[:bar_count]


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — data conversion
# ═══════════════════════════════════════════════════════════════════════════════


def _bar_data_to_columns(bar_data: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a (N, 11) numba output array to a dict of named columns."""
    return {
        "bar_id": bar_data[:, _COL_BAR_ID].astype(np.int64),
        "open": bar_data[:, _COL_OPEN],
        "high": bar_data[:, _COL_HIGH],
        "low": bar_data[:, _COL_LOW],
        "close": bar_data[:, _COL_CLOSE],
        "volume": bar_data[:, _COL_VOLUME],
        "dollar_value": bar_data[:, _COL_DOLLAR_VALUE],
        "vwap": bar_data[:, _COL_VWAP],
        "num_ticks": bar_data[:, _COL_NUM_TICKS].astype(np.int64),
        "open_ts": bar_data[:, _COL_OPEN_TS].astype(np.int64),
        "close_ts": bar_data[:, _COL_CLOSE_TS].astype(np.int64),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — batch dispatch
# ═══════════════════════════════════════════════════════════════════════════════


def numba_batch_static(
    bar_type: str,
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray | None,
    threshold: float,
    interval_ms: int = 60000,
    anchor: str = "clock",
) -> tuple[np.ndarray, str]:
    """Run the numba batch path for a static (fixed) threshold.

    Parameters
    ----------
    bar_type : str
        One of ``"tick"``, ``"volume"``, ``"dollar"``, ``"time"``,
        ``"imbalance_tick"``, ``"imbalance_volume"``, ``"imbalance_dollar"``,
        ``"run_tick"``, ``"run_volume"``, ``"run_dollar"``.
    timestamps, prices, volumes : ndarray of float64
        Tick data arrays (1-D).
    sides : ndarray of float64 or None
        Derived tick signs.  Required for information-driven bars.
    threshold : float
        The fixed threshold.
    interval_ms : int
        For time bars only.
    anchor : str
        For time bars only: ``"clock"`` or ``"first_tick"``.

    Returns
    -------
    bar_data : ndarray
        (N, 11) float64 array of bar data.
    bar_type_str : str
    """
    if bar_type == "time":
        anchor_code = 1 if anchor == "first_tick" else 0
        bar_data = _time_bars_numba(
            timestamps.astype(np.float64),
            prices.astype(np.float64),
            volumes.astype(np.float64),
            interval_ms,
            anchor_code,
        )
        return bar_data, "time"

    if bar_type in (
        "imbalance_tick",
        "imbalance_volume",
        "imbalance_dollar",
        "run_tick",
        "run_volume",
        "run_dollar",
    ):
        if sides is None:
            raise ValueError(f"Bar type {bar_type!r} requires side/direction information.")
        # Map bar_type to metric_code
        if "tick" in bar_type:
            metric_code = 0
        elif "volume" in bar_type:
            metric_code = 1
        else:  # dollar
            metric_code = 2

        if bar_type.startswith("imbalance_"):
            bar_data = _imbalance_bars_numba(
                timestamps.astype(np.float64),
                prices.astype(np.float64),
                volumes.astype(np.float64),
                sides.astype(np.float64),
                threshold,
                metric_code,
            )
        else:  # run
            bar_data = _run_bars_numba(
                timestamps.astype(np.float64),
                prices.astype(np.float64),
                volumes.astype(np.float64),
                sides.astype(np.float64),
                threshold,
                metric_code,
            )
        return bar_data, bar_type

    # Standard bars: tick, volume, dollar
    dispatch_fn = _NUMBA_DISPATCH.get(bar_type)
    if dispatch_fn is None:
        raise KeyError(f"No numba backend registered for bar type {bar_type!r}")
    bar_data = dispatch_fn(
        timestamps.astype(np.float64),
        prices.astype(np.float64),
        volumes.astype(np.float64),
        threshold,
    )
    return bar_data, bar_type


def numba_batch_ewma(
    bar_type: str,
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray | None,
    alpha: float,
    initial_ewa_t: float,
    initial_ewa_proportion: float,
) -> tuple[np.ndarray, str]:
    """Run the numba batch path with in-loop EWMA threshold updates.

    Only for information-driven bars (``imbalance_*``, ``run_*``).

    Parameters
    ----------
    bar_type : str
        One of ``"imbalance_tick"``, ``"imbalance_volume"``,
        ``"imbalance_dollar"``, ``"run_tick"``, ``"run_volume"``,
        ``"run_dollar"``.
    timestamps, prices, volumes : ndarray of float64
        Tick data arrays (1-D).
    sides : ndarray of float64
        Derived tick signs  required for information-driven bars.
    alpha : float
        EWMA smoothing factor (0 < alpha <= 1).
    initial_ewa_t : float
        Initial seed for :math:`E[T]` (expected tick count / volume per bar).
    initial_ewa_proportion : float
        Initial seed for the proportion component
        (:math:`|E[\\theta]|` for imbalance, :math:`E[P^+]` for run).

    Returns
    -------
    bar_data : ndarray
        (N, 11) float64 array of bar data.
    bar_type_str : str
    """
    if sides is None:
        raise ValueError(f"Bar type {bar_type!r} requires side/direction information.")

    if "tick" in bar_type:
        metric_code = 0
    elif "volume" in bar_type:
        metric_code = 1
    else:  # dollar
        metric_code = 2

    if bar_type.startswith("imbalance_"):
        bar_data = _imbalance_bars_ewma_numba(
            timestamps.astype(np.float64),
            prices.astype(np.float64),
            volumes.astype(np.float64),
            sides.astype(np.float64),
            metric_code,
            alpha,
            initial_ewa_t,
            initial_ewa_proportion,
        )
    elif bar_type.startswith("run_"):
        bar_data = _run_bars_ewma_numba(
            timestamps.astype(np.float64),
            prices.astype(np.float64),
            volumes.astype(np.float64),
            sides.astype(np.float64),
            metric_code,
            alpha,
            initial_ewa_t,
            initial_ewa_proportion,
        )
    else:
        raise KeyError(f"No EWMA numba backend for bar type {bar_type!r}")

    return bar_data, bar_type


def _get_compilation_warning() -> None:
    """Issue a one-shot warning when numba compilation is attempted."""
    if not _NUMBA_AVAILABLE:
        warnings.warn(
            "numba is not installed. The numba backend will fall back to the "
            "Python path. Install with: pip install flowbars[numba]",
            stacklevel=3,
        )
