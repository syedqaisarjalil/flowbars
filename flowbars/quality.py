"""Bar quality report — statistical diagnostics for constructed bar DataFrames.

Every metric is deterministic (fixed lag set, fixed histogram rule, no RNG).
``scipy`` is optional: without it the Ljung-Box / Jarque-Bera statistics are
still computed in numpy, but their p-values (and reject flags) are ``None``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def bar_quality_report(
    bars_df: Any,
    ljung_box_lags: int | None = None,
    day_length_ms: int = 86_400_000,
    *,
    variance_ratio_lags: tuple[int, ...] = (2, 5),
    fragmentation_threshold: int = 2,
    num_ticks_unit: str = "ticks",
) -> dict[str, Any]:
    """Run statistical diagnostics on a bar DataFrame.

    Parameters
    ----------
    bars_df : pd.DataFrame or pl.DataFrame
        Completed bars.  Must contain at least the columns ``close`` and
        ``bar_type``.
    ljung_box_lags : int or None, default None
        Number of lags for the Ljung-Box test.  When *None*, defaults to
        ``min(10, n_returns // 5)`` — capped at 10 for small samples.
    day_length_ms : int, default 86_400_000 (24 hours)
        Length of a "day" for bar-count stability, in milliseconds.
    variance_ratio_lags : tuple of int, default ``(2, 5)``
        Lags reported for the variance-ratio diagnostic.
    fragmentation_threshold : int, default 2
        A run bar is "fragmented" when its ``num_ticks`` is at or below this.
    num_ticks_unit : str, default ``"ticks"``
        Unit label for ``num_ticks`` (``"minutes"`` for minute bars).

    Returns
    -------
    dict
        Dictionary with these keys:

        * ``n_bars`` — total number of bars
        * ``scipy_available`` — whether scipy-backed p-values were computed
        * ``ljung_box`` — ``statistic``, ``p_value``, ``lags``,
          ``reject_autocorrelation``
        * ``jarque_bera`` — ``statistic``, ``p_value``, ``skewness``,
          ``kurtosis``, ``reject_normality``
        * ``bar_count_stability`` — ``per_day``, ``mean``, ``std``, ``cv``
        * ``run_fragmentation`` — fraction of run bars with
          ``num_ticks <= fragmentation_threshold`` (plus ``threshold`` and
          ``unit``), or ``None`` for non-run bar types
        * ``return_moments`` — ``mean``, ``std``, ``skewness``, ``kurtosis``
          of log returns (Pearson kurtosis; normal = 3)
        * ``return_autocorrelation`` — ``lag1``, ``lag5``
        * ``abs_return_autocorrelation`` — ``lag1``, ``lag5`` on ``|r|``
        * ``variance_ratio`` — ``lag2``, ``lag5``
        * ``entropy`` — ``shannon``, ``normalized``
        * ``bar_size_cv`` — ``cv`` and the ``metric`` used
        * ``duration_cv`` — CV of ``close_ts − open_ts``
        * ``effective_sample_size`` — ``n · (1−ρ₁)/(1+ρ₁)``
        * ``ohlc_integrity`` — ``n_violations``, ``fraction``
        * ``composite_score`` — ``score`` (mean of available sub-scores) and
          ``components``
    """
    # -- optional scipy ------------------------------------------------------
    try:
        from scipy import stats as scipy_stats  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        scipy_stats = None
    scipy_available = scipy_stats is not None

    # Accept both pandas and polars DataFrames.  We use pandas internally
    # because the bar pipeline already returns pandas.
    try:
        import polars as pl  # type: ignore  # noqa: PLC0415
    except ImportError:
        pl = None  # type: ignore[assignment]

    if pl is not None and isinstance(bars_df, pl.DataFrame):
        bars = bars_df.to_pandas()
    else:
        bars = bars_df

    n_bars = len(bars)
    result: dict[str, Any] = {
        "n_bars": n_bars,
        "scipy_available": scipy_available,
    }

    # -- returns ------------------------------------------------------------
    close = np.asarray(bars["close"], dtype=np.float64)
    log_returns = np.array([], dtype=np.float64)
    if n_bars >= 2:
        log_returns = np.diff(np.log(close))
        log_returns = log_returns[np.isfinite(log_returns)]
    n_returns = len(log_returns)
    rho1 = _acf_at(log_returns, 1)

    if n_returns < 2:
        result["ljung_box"] = None
        result["jarque_bera"] = None
    else:
        # ---- Ljung-Box ----------------------------------------------------
        lags = ljung_box_lags
        if lags is None:
            lags = max(1, min(10, n_returns // 5))
        lags = min(lags, n_returns - 1)

        acf = _autocorrelation(log_returns, lags)
        n = n_returns
        q_stat = n * (n + 2) * float(np.sum(acf**2 / (n - np.arange(1, lags + 1))))
        p_value = float(scipy_stats.chi2.sf(q_stat, lags)) if scipy_available else None

        result["ljung_box"] = {
            "statistic": q_stat,
            "p_value": p_value,
            "lags": lags,
            "reject_autocorrelation": (p_value < 0.05) if p_value is not None else None,
        }

        # ---- Jarque-Bera --------------------------------------------------
        if n_returns < 4:
            result["jarque_bera"] = None
        else:
            skew, kurt = _skew_kurtosis(log_returns)
            jb = (n / 6.0) * (skew**2 + ((kurt - 3.0) ** 2) / 4.0)
            p_value = float(scipy_stats.chi2.sf(jb, 2)) if scipy_available else None

            result["jarque_bera"] = {
                "statistic": jb,
                "p_value": p_value,
                "skewness": skew,
                "kurtosis": kurt,
                "reject_normality": (p_value < 0.05) if p_value is not None else None,
            }

    # -- richer return statistics --------------------------------------------
    result["return_moments"] = _return_moments(log_returns)
    result["return_autocorrelation"] = _autocorrelation_pairs(log_returns, (1, 5))
    result["abs_return_autocorrelation"] = _autocorrelation_pairs(np.abs(log_returns), (1, 5))
    result["variance_ratio"] = {
        f"lag{lag}": _variance_ratio(log_returns, lag) for lag in variance_ratio_lags
    }
    result["entropy"] = _entropy(log_returns)
    result["effective_sample_size"] = _effective_sample_size(n_returns, rho1)

    # -- bar size / duration dispersion -------------------------------------
    result["bar_size_cv"] = _bar_size_cv(bars)
    result["duration_cv"] = _duration_cv(bars)

    # -- OHLC integrity ------------------------------------------------------
    result["ohlc_integrity"] = _ohlc_integrity(bars)

    # -- composite score ------------------------------------------------------
    result["composite_score"] = _composite_score(result)

    # -- bar-count stability -------------------------------------------------
    if "open_ts" in bars.columns:
        open_ts = np.asarray(bars["open_ts"], dtype=np.int64)
    else:
        open_ts = np.array([], dtype=np.int64)
    if len(open_ts) > 0 and day_length_ms > 0:
        day_bucket = open_ts // day_length_ms
        _, counts = np.unique(day_bucket, return_counts=True)
        per_day = counts.tolist()
        mean_val = float(np.mean(counts))
        std_val = float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0
        cv = std_val / mean_val if mean_val > 0 else float("inf")

        result["bar_count_stability"] = {
            "per_day": per_day,
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
        }
    else:
        result["bar_count_stability"] = None

    # -- run-bar fragmentation -----------------------------------------------
    bar_type_col = bars["bar_type"]
    if hasattr(bar_type_col, "astype"):
        is_run = bar_type_col.astype(str).str.startswith("run_")
        run_mask = is_run.to_numpy()
    else:
        run_mask = np.zeros(n_bars, dtype=bool)

    if run_mask.any():
        num_ticks = np.asarray(bars.loc[run_mask, "num_ticks"], dtype=np.int64)
        n_run_bars = len(num_ticks)
        n_fragmented = int(np.sum(num_ticks <= fragmentation_threshold))
        fraction = n_fragmented / n_run_bars if n_run_bars > 0 else 0.0
        result["run_fragmentation"] = {
            "n_run_bars": n_run_bars,
            "n_fragmented": n_fragmented,
            "fraction_fragmented": fraction,
            "threshold": fragmentation_threshold,
            "unit": num_ticks_unit,
        }
    else:
        result["run_fragmentation"] = None

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _autocorrelation(x: np.ndarray, nlags: int) -> np.ndarray:
    r"""Compute sample autocorrelation at lags 1..*nlags*.

    Uses the standard estimator:

    .. math::

        \rho_k = \frac{\sum_{t=k+1}^n (x_t - \bar{x})(x_{t-k} - \bar{x})}
                       {\sum_{t=1}^n (x_t - \bar{x})^2}
    """
    n = len(x)
    x_demean = x - np.mean(x)
    denom = float(np.sum(x_demean**2))
    if denom == 0.0:
        return np.zeros(nlags, dtype=np.float64)

    acf = np.empty(nlags, dtype=np.float64)
    for k in range(1, nlags + 1):
        acf[k - 1] = float(np.sum(x_demean[k:] * x_demean[: n - k])) / denom
    return acf


def _acf_at(x: np.ndarray, lag: int) -> float | None:
    """Autocorrelation at a single lag, or ``None`` when undefined."""
    if len(x) <= lag:
        return None
    return float(_autocorrelation(x, lag)[lag - 1])


def _autocorrelation_pairs(x: np.ndarray, lags: tuple[int, ...]) -> dict[str, float | None]:
    """Autocorrelation at the requested lags (``None`` when insufficient)."""
    return {f"lag{lag}": _acf_at(x, lag) for lag in lags}


def _skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    """Biased (MLE) skewness and Pearson kurtosis (normal = 3)."""
    m2 = float(np.mean(x**2))
    if m2 == 0.0:
        return 0.0, 0.0
    m3 = float(np.mean(x**3))
    m4 = float(np.mean(x**4))
    return m3 / (m2**1.5), m4 / (m2**2)


def _return_moments(x: np.ndarray) -> dict[str, float] | None:
    """Mean, std, skewness, and kurtosis of log returns."""
    if len(x) == 0:
        return None
    skew, kurt = _skew_kurtosis(x) if len(x) >= 2 else (0.0, 0.0)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "skewness": skew,
        "kurtosis": kurt,
    }


def _variance_ratio(x: np.ndarray, k: int) -> float | None:
    """Overlapping-window variance ratio ``Var(sum of k) / (k · Var(r))``."""
    n = len(x)
    if k < 1 or n <= k:
        return None
    var1 = float(np.var(x, ddof=1))
    if var1 == 0.0:
        return None
    sums = np.convolve(x, np.ones(k), mode="valid")
    if len(sums) < 2:
        return None
    var_k = float(np.var(sums, ddof=1))
    return var_k / (k * var1)


def _entropy(x: np.ndarray) -> dict[str, float] | None:
    """Histogram entropy of *x* (``max(2, floor(sqrt(n)))`` bins)."""
    n = len(x)
    if n < 2:
        return None
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo:
        return None
    bins = max(2, int(math.floor(math.sqrt(n))))
    counts, _ = np.histogram(x, bins=bins, range=(lo, hi))
    p = counts[counts > 0] / counts.sum()
    shannon = float(-np.sum(p * np.log(p)))
    return {"shannon": shannon, "normalized": shannon / math.log(bins)}


def _effective_sample_size(n: int, rho1: float | None) -> float | None:
    """``n · (1 − ρ₁) / (1 + ρ₁)`` with ``ρ₁`` clipped to ±0.99."""
    if n < 2 or rho1 is None:
        return None
    rho = float(np.clip(rho1, -0.99, 0.99))
    return float(n * (1.0 - rho) / (1.0 + rho))


def _cv(values: np.ndarray) -> float | None:
    """Coefficient of variation ``std(ddof=1) / mean``; ``None`` if undefined."""
    if len(values) < 2:
        return None
    mean_val = float(np.mean(values))
    if mean_val == 0.0:
        return None
    return float(np.std(values, ddof=1) / mean_val)


def _bar_size_cv(bars: Any) -> dict[str, Any] | None:
    """CV of bar volume (fallback: ``num_ticks``), with the metric used."""
    if "volume" in bars.columns:
        metric = "volume"
    elif "num_ticks" in bars.columns:
        metric = "num_ticks"
    else:
        return None
    values = np.asarray(bars[metric], dtype=np.float64)
    cv = _cv(values)
    if cv is None:
        return None
    return {"cv": cv, "metric": metric}


def _duration_cv(bars: Any) -> float | None:
    """CV of ``close_ts − open_ts``; ``None`` if columns are absent."""
    if "close_ts" not in bars.columns or "open_ts" not in bars.columns:
        return None
    durations = np.asarray(bars["close_ts"], dtype=np.float64) - np.asarray(
        bars["open_ts"], dtype=np.float64
    )
    return _cv(durations)


def _ohlc_integrity(bars: Any) -> dict[str, Any] | None:
    """Count bars violating ``low <= open, close <= high``."""
    required = ("open", "high", "low", "close")
    if any(col not in bars.columns for col in required):
        return None
    o = np.asarray(bars["open"], dtype=np.float64)
    h = np.asarray(bars["high"], dtype=np.float64)
    low = np.asarray(bars["low"], dtype=np.float64)
    c = np.asarray(bars["close"], dtype=np.float64)
    if len(o) == 0:
        return {"n_violations": 0, "fraction": 0.0}
    violations = (o < low) | (o > h) | (c < low) | (c > h)
    n_violations = int(np.sum(violations))
    return {"n_violations": n_violations, "fraction": n_violations / len(o)}


def _composite_score(result: dict[str, Any]) -> dict[str, Any] | None:
    """Mean of available sub-scores in ``[0, 1]``, plus the components."""
    components: dict[str, float] = {}

    rho1 = result["return_autocorrelation"].get("lag1")
    if rho1 is not None:
        components["autocorrelation"] = 1.0 - min(1.0, abs(rho1))

    vr5 = result["variance_ratio"].get("lag5")
    if vr5 is not None:
        components["variance_ratio"] = max(0.0, 1.0 - abs(vr5 - 1.0))

    entropy = result["entropy"]
    if entropy is not None and entropy.get("normalized") is not None:
        components["entropy"] = max(0.0, min(1.0, entropy["normalized"]))

    bar_size_cv = result["bar_size_cv"]
    if bar_size_cv is not None:
        components["bar_size_cv"] = max(0.0, 1.0 - bar_size_cv["cv"])

    duration_cv = result["duration_cv"]
    if duration_cv is not None:
        components["duration_cv"] = max(0.0, 1.0 - duration_cv)

    if not components:
        return None
    return {
        "score": float(np.mean(list(components.values()))),
        "components": components,
    }
