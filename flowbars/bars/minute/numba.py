"""Numba-accelerated batch construction for the minute-OHLCV → bars path.

Mirrors :mod:`flowbars.bars.numba_backend` for minute candles.  Kernels take
whole-minute OHLCV arrays and emit the same 11-column internal output.  Signs
are always precomputed by the caller — kernels never derive them.

The Python engine falls back to the pure-Python path when numba is absent,
a session-producing calendar is in use, or the constructor is not pristine.
"""

from __future__ import annotations

import numpy as np

from flowbars.bars.numba_backend import (
    _COL_BAR_ID,
    _COL_CLOSE,
    _COL_CLOSE_TS,
    _COL_DOLLAR_VALUE,
    _COL_HIGH,
    _COL_LOW,
    _COL_NUM_TICKS,
    _COL_OPEN,
    _COL_OPEN_TS,
    _COL_VOLUME,
    _COL_VWAP,
    _NUM_COLS,
    _njit,
)

# Metric codes for the shared scalar-rollover kernel.
_METRIC_VOLUME = 0
_METRIC_DOLLAR = 1
_METRIC_VOLATILITY = 2
_METRIC_RANGE = 3

_METRIC_CODES = {
    "volume": _METRIC_VOLUME,
    "dollar": _METRIC_DOLLAR,
    "volatility": _METRIC_VOLATILITY,
    "range": _METRIC_RANGE,
    "imbalance_volume": _METRIC_VOLUME,
    "imbalance_dollar": _METRIC_DOLLAR,
    "run_volume": _METRIC_VOLUME,
    "run_dollar": _METRIC_DOLLAR,
}


@_njit  # type: ignore[untyped-decorator]
def _write_bar(
    out: np.ndarray,
    idx: int,
    bar_id: int,
    open_val: float,
    high_val: float,
    low_val: float,
    close_val: float,
    vol: float,
    dollar_val: float,
    n_ticks: int,
    open_ts_val: int,
    close_ts_val: int,
) -> None:
    """Write one completed bar into *out* at row *idx* (inline emit)."""
    out[idx, _COL_BAR_ID] = bar_id
    out[idx, _COL_OPEN] = open_val
    out[idx, _COL_HIGH] = high_val
    out[idx, _COL_LOW] = low_val
    out[idx, _COL_CLOSE] = close_val
    out[idx, _COL_VOLUME] = vol
    out[idx, _COL_DOLLAR_VALUE] = dollar_val
    out[idx, _COL_VWAP] = dollar_val / vol if vol > 0.0 else 0.0
    out[idx, _COL_NUM_TICKS] = n_ticks
    out[idx, _COL_OPEN_TS] = open_ts_val
    out[idx, _COL_CLOSE_TS] = close_ts_val


@_njit  # type: ignore[untyped-decorator]
def _minute_metric_bars_numba(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    threshold: float,
    metric_code: int,
) -> np.ndarray:
    """Scalar-rollover kernel: volume, dollar, volatility, relative range."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    has_minute = False
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

    cum = 0.0
    prev_close = np.nan  # volatility: None sentinel

    for i in range(n):
        o_i = opens[i]
        h_i = highs[i]
        l_i = lows[i]
        c_i = closes[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])

        if not has_minute:
            open_val = o_i
            high_val = h_i
            low_val = l_i
            open_ts_val = ts
            has_minute = True
        else:
            if h_i > high_val:
                high_val = h_i
            if l_i < low_val:
                low_val = l_i

        close_val = c_i
        vol += vol_i
        dollar_val += c_i * vol_i
        n_ticks += 1
        close_ts_val = ts

        if metric_code == _METRIC_VOLUME:
            metric = vol_i
        elif metric_code == _METRIC_DOLLAR:
            metric = c_i * vol_i
        elif metric_code == _METRIC_VOLATILITY:
            if np.isnan(prev_close) or prev_close == 0.0:
                metric = 0.0
            else:
                metric = abs(c_i - prev_close) / prev_close
            prev_close = c_i
        else:  # range
            if c_i == 0.0:
                metric = 0.0
            else:
                metric = (h_i - l_i) / c_i
        cum += metric

        if cum >= threshold:
            _write_bar(
                out,
                bar_count,
                bar_id,
                open_val,
                high_val,
                low_val,
                close_val,
                vol,
                dollar_val,
                n_ticks,
                open_ts_val,
                close_ts_val,
            )
            bar_count += 1
            bar_id += 1
            excess = cum - threshold
            cum = excess if excess > 0.0 else 0.0
            has_minute = False
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
def _minute_renko_bars_numba(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Renko-displacement kernel.  Displacement is not additive: no overflow."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    has_minute = False
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

    anchor = np.nan
    displacement = 0.0

    for i in range(n):
        o_i = opens[i]
        h_i = highs[i]
        l_i = lows[i]
        c_i = closes[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])

        if not has_minute:
            open_val = o_i
            high_val = h_i
            low_val = l_i
            open_ts_val = ts
            has_minute = True
        else:
            if h_i > high_val:
                high_val = h_i
            if l_i < low_val:
                low_val = l_i

        close_val = c_i
        vol += vol_i
        dollar_val += c_i * vol_i
        n_ticks += 1
        close_ts_val = ts

        if np.isnan(anchor):
            anchor = o_i
        if anchor <= 0.0:
            displacement = 0.0
        else:
            displacement = abs(c_i - anchor) / anchor

        if displacement >= threshold:
            _write_bar(
                out,
                bar_count,
                bar_id,
                open_val,
                high_val,
                low_val,
                close_val,
                vol,
                dollar_val,
                n_ticks,
                open_ts_val,
                close_ts_val,
            )
            bar_count += 1
            bar_id += 1
            anchor = np.nan
            displacement = 0.0
            has_minute = False
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
def _minute_hybrid_bars_numba(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    dollar_threshold: float,
    volatility_threshold: float,
) -> np.ndarray:
    """Hybrid kernel: close when both dollar and volatility targets are met."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    has_minute = False
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

    cum_dollar = 0.0
    cum_vol = 0.0
    prev_close = np.nan

    for i in range(n):
        o_i = opens[i]
        h_i = highs[i]
        l_i = lows[i]
        c_i = closes[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])

        if not has_minute:
            open_val = o_i
            high_val = h_i
            low_val = l_i
            open_ts_val = ts
            has_minute = True
        else:
            if h_i > high_val:
                high_val = h_i
            if l_i < low_val:
                low_val = l_i

        close_val = c_i
        vol += vol_i
        dollar_val += c_i * vol_i
        n_ticks += 1
        close_ts_val = ts

        cum_dollar += c_i * vol_i
        if np.isnan(prev_close) or prev_close == 0.0:
            metric = 0.0
        else:
            metric = abs(c_i - prev_close) / prev_close
        cum_vol += metric
        prev_close = c_i

        if cum_dollar >= dollar_threshold and cum_vol >= volatility_threshold:
            _write_bar(
                out,
                bar_count,
                bar_id,
                open_val,
                high_val,
                low_val,
                close_val,
                vol,
                dollar_val,
                n_ticks,
                open_ts_val,
                close_ts_val,
            )
            bar_count += 1
            bar_id += 1
            exc_dollar = cum_dollar - dollar_threshold
            exc_vol = cum_vol - volatility_threshold
            cum_dollar = exc_dollar if exc_dollar > 0.0 else 0.0
            cum_vol = exc_vol if exc_vol > 0.0 else 0.0
            has_minute = False
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
def _minute_imbalance_bars_numba(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    threshold: float,
    metric_code: int,
) -> np.ndarray:
    """Minute imbalance kernel (static threshold)."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    signed_imbalance = 0.0
    has_minute = False
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
        o_i = opens[i]
        h_i = highs[i]
        l_i = lows[i]
        c_i = closes[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])
        side = sides[i]

        if not has_minute:
            open_val = o_i
            high_val = h_i
            low_val = l_i
            open_ts_val = ts
            has_minute = True
        else:
            if h_i > high_val:
                high_val = h_i
            if l_i < low_val:
                low_val = l_i

        close_val = c_i
        vol += vol_i
        dollar_val += c_i * vol_i
        n_ticks += 1
        close_ts_val = ts

        if not np.isnan(side):
            if metric_code == _METRIC_VOLUME:
                metric = vol_i
            else:  # dollar
                metric = c_i * vol_i
            signed_imbalance += side * metric

        if abs(signed_imbalance) >= threshold:
            _write_bar(
                out,
                bar_count,
                bar_id,
                open_val,
                high_val,
                low_val,
                close_val,
                vol,
                dollar_val,
                n_ticks,
                open_ts_val,
                close_ts_val,
            )
            bar_count += 1
            bar_id += 1
            if signed_imbalance >= 0.0:
                sign = 1.0
            else:
                sign = -1.0
            excess = abs(signed_imbalance) - threshold
            signed_imbalance = sign * excess if excess > 0.0 else 0.0
            has_minute = False
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
def _minute_run_bars_numba(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    threshold: float,
    metric_code: int,
) -> np.ndarray:
    """Minute run kernel (static threshold)."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    has_minute = False
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

    first_minute = True
    banked = 0.0
    run_sign = np.nan
    run_cum = 0.0

    for i in range(n):
        o_i = opens[i]
        h_i = highs[i]
        l_i = lows[i]
        c_i = closes[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])
        side = sides[i]

        if not has_minute:
            open_val = o_i
            high_val = h_i
            low_val = l_i
            open_ts_val = ts
            has_minute = True
        else:
            if h_i > high_val:
                high_val = h_i
            if l_i < low_val:
                low_val = l_i

        close_val = c_i
        vol += vol_i
        dollar_val += c_i * vol_i
        n_ticks += 1
        close_ts_val = ts

        if metric_code == _METRIC_VOLUME:
            metric = vol_i
        else:  # dollar
            metric = c_i * vol_i

        if first_minute:
            first_minute = False
            run_sign = side
            run_cum = metric
        elif np.isnan(run_sign) or np.isnan(side) or run_sign == side:
            run_cum += metric
            if np.isnan(run_sign) and not np.isnan(side):
                run_sign = side
        else:
            banked += run_cum
            run_sign = side
            run_cum = metric

        total = banked + run_cum
        if total >= threshold:
            _write_bar(
                out,
                bar_count,
                bar_id,
                open_val,
                high_val,
                low_val,
                close_val,
                vol,
                dollar_val,
                n_ticks,
                open_ts_val,
                close_ts_val,
            )
            bar_count += 1
            bar_id += 1
            excess = total - threshold
            banked = 0.0
            if excess > 0.0:
                run_cum = excess
            else:
                run_cum = 0.0
                run_sign = np.nan
            has_minute = False
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
def _minute_imbalance_bars_ewma_numba(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    metric_code: int,
    alpha: float,
    initial_ewa_t: float,
    initial_ewa_proportion: float,
) -> np.ndarray:
    """Minute imbalance kernel with in-loop two-component EWMA updates."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    signed_imbalance = 0.0
    has_minute = False
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

    ewa_t = initial_ewa_t
    ewa_proportion = initial_ewa_proportion
    threshold_val = ewa_t * abs(ewa_proportion)
    if threshold_val <= 0.0:
        threshold_val = 1.0

    for i in range(n):
        o_i = opens[i]
        h_i = highs[i]
        l_i = lows[i]
        c_i = closes[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])
        side = sides[i]

        if not has_minute:
            open_val = o_i
            high_val = h_i
            low_val = l_i
            open_ts_val = ts
            has_minute = True
        else:
            if h_i > high_val:
                high_val = h_i
            if l_i < low_val:
                low_val = l_i

        close_val = c_i
        vol += vol_i
        dollar_val += c_i * vol_i
        n_ticks += 1
        close_ts_val = ts

        if not np.isnan(side):
            if metric_code == _METRIC_VOLUME:
                metric = vol_i
            else:
                metric = c_i * vol_i
            signed_imbalance += side * metric

        if abs(signed_imbalance) >= threshold_val:
            _write_bar(
                out,
                bar_count,
                bar_id,
                open_val,
                high_val,
                low_val,
                close_val,
                vol,
                dollar_val,
                n_ticks,
                open_ts_val,
                close_ts_val,
            )
            bar_count += 1
            bar_id += 1

            old_threshold = threshold_val

            if metric_code == _METRIC_VOLUME:
                t_stat = vol
            else:
                t_stat = dollar_val
            proportion_stat = signed_imbalance / t_stat if t_stat > 0.0 else 0.0

            ewa_t = alpha * t_stat + (1.0 - alpha) * ewa_t
            ewa_proportion = alpha * proportion_stat + (1.0 - alpha) * ewa_proportion
            threshold_val = ewa_t * abs(ewa_proportion)
            if threshold_val <= 0.0:
                threshold_val = 1.0

            if signed_imbalance >= 0.0:
                sign = 1.0
            else:
                sign = -1.0
            excess = abs(signed_imbalance) - old_threshold
            signed_imbalance = sign * excess if excess > 0.0 else 0.0
            has_minute = False
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
def _minute_run_bars_ewma_numba(
    timestamps: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    metric_code: int,
    alpha: float,
    initial_ewa_t: float,
    initial_ewa_proportion: float,
) -> np.ndarray:
    """Minute run kernel with in-loop two-component EWMA updates."""
    n = len(timestamps)
    out = np.zeros((n, _NUM_COLS), dtype=np.float64)

    has_minute = False
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

    first_minute = True
    banked = 0.0
    run_sign = np.nan
    run_cum = 0.0
    buy_cum = 0.0
    sell_cum = 0.0

    ewa_t = initial_ewa_t
    ewa_proportion = initial_ewa_proportion
    if ewa_proportion >= 1.0 - ewa_proportion:
        threshold_val = ewa_t * ewa_proportion
    else:
        threshold_val = ewa_t * (1.0 - ewa_proportion)
    if threshold_val <= 0.0:
        threshold_val = 1.0

    for i in range(n):
        o_i = opens[i]
        h_i = highs[i]
        l_i = lows[i]
        c_i = closes[i]
        vol_i = volumes[i]
        ts = int(timestamps[i])
        side = sides[i]

        if not has_minute:
            open_val = o_i
            high_val = h_i
            low_val = l_i
            open_ts_val = ts
            has_minute = True
        else:
            if h_i > high_val:
                high_val = h_i
            if l_i < low_val:
                low_val = l_i

        close_val = c_i
        vol += vol_i
        dollar_val += c_i * vol_i
        n_ticks += 1
        close_ts_val = ts

        if metric_code == _METRIC_VOLUME:
            metric = vol_i
        else:
            metric = c_i * vol_i

        if first_minute:
            first_minute = False
            run_sign = side
            run_cum = metric
        elif np.isnan(run_sign) or np.isnan(side) or run_sign == side:
            run_cum += metric
            if np.isnan(run_sign) and not np.isnan(side):
                run_sign = side
        else:
            banked += run_cum
            if run_sign > 0.0:
                buy_cum += run_cum
            elif run_sign < 0.0:
                sell_cum += run_cum
            run_sign = side
            run_cum = metric

        total = banked + run_cum
        if total >= threshold_val:
            _write_bar(
                out,
                bar_count,
                bar_id,
                open_val,
                high_val,
                low_val,
                close_val,
                vol,
                dollar_val,
                n_ticks,
                open_ts_val,
                close_ts_val,
            )
            bar_count += 1
            bar_id += 1

            old_threshold = threshold_val

            t_stat = total
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

            ewa_t = alpha * t_stat + (1.0 - alpha) * ewa_t
            ewa_proportion = alpha * p_plus + (1.0 - alpha) * ewa_proportion
            if ewa_proportion >= 1.0 - ewa_proportion:
                threshold_val = ewa_t * ewa_proportion
            else:
                threshold_val = ewa_t * (1.0 - ewa_proportion)
            if threshold_val <= 0.0:
                threshold_val = 1.0

            excess = total - old_threshold
            banked = 0.0
            buy_cum = 0.0
            sell_cum = 0.0
            if excess > 0.0:
                run_cum = excess
            else:
                run_cum = 0.0
                run_sign = np.nan
            has_minute = False
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


def numba_batch_minute(
    bar_type: str,
    arrays: tuple[np.ndarray, ...],
    config: dict[str, float],
) -> tuple[np.ndarray, str]:
    """Dispatch a minute batch to the matching JIT kernel.

    Parameters
    ----------
    bar_type : str
        Minute accumulator label (``"volume"``, ``"imbalance_volume"``, …).
    arrays : tuple
        ``(timestamps, opens, highs, lows, closes, volumes, sides)``.
    config : dict
        ``threshold`` for static bars, or ``alpha`` / ``initial_ewa_t`` /
        ``initial_ewa_proportion`` for EWMA bars.  Hybrid also reads
        ``volatility_threshold``.

    Returns
    -------
    (bar_data, bar_type) : (ndarray, str)
        ``(N, 11)`` float64 bar rows.
    """
    timestamps, opens, highs, lows, closes, volumes, sides = arrays
    ts = timestamps.astype(np.float64)
    o = opens.astype(np.float64)
    h = highs.astype(np.float64)
    low = lows.astype(np.float64)
    c = closes.astype(np.float64)
    v = volumes.astype(np.float64)

    if bar_type in ("volume", "dollar", "volatility", "range"):
        bar_data = _minute_metric_bars_numba(
            ts, o, h, low, c, v, float(config["threshold"]), _METRIC_CODES[bar_type]
        )
    elif bar_type == "renko":
        bar_data = _minute_renko_bars_numba(ts, o, h, low, c, v, float(config["threshold"]))
    elif bar_type == "hybrid":
        bar_data = _minute_hybrid_bars_numba(
            ts,
            o,
            h,
            low,
            c,
            v,
            float(config["threshold"]),
            float(config["volatility_threshold"]),
        )
    elif bar_type in ("imbalance_volume", "imbalance_dollar"):
        code = _METRIC_CODES[bar_type]
        if "alpha" in config:
            bar_data = _minute_imbalance_bars_ewma_numba(
                ts,
                o,
                h,
                low,
                c,
                v,
                sides.astype(np.float64),
                code,
                float(config["alpha"]),
                float(config["initial_ewa_t"]),
                float(config["initial_ewa_proportion"]),
            )
        else:
            bar_data = _minute_imbalance_bars_numba(
                ts, o, h, low, c, v, sides.astype(np.float64), float(config["threshold"]), code
            )
    elif bar_type in ("run_volume", "run_dollar"):
        code = _METRIC_CODES[bar_type]
        if "alpha" in config:
            bar_data = _minute_run_bars_ewma_numba(
                ts,
                o,
                h,
                low,
                c,
                v,
                sides.astype(np.float64),
                code,
                float(config["alpha"]),
                float(config["initial_ewa_t"]),
                float(config["initial_ewa_proportion"]),
            )
        else:
            bar_data = _minute_run_bars_numba(
                ts, o, h, low, c, v, sides.astype(np.float64), float(config["threshold"]), code
            )
    else:
        raise KeyError(f"No minute numba backend for bar type {bar_type!r}")

    return bar_data, bar_type


__all__ = ["numba_batch_minute"]
