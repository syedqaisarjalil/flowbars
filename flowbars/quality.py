"""Bar quality report — statistical diagnostics for constructed bar DataFrames.

This is what makes the comparison notebooks a real argument, not just a demonstration.

Requires ``scipy`` for hypothesis-test p-values.  Install with
``pip install flowbars[all]`` or ``pip install scipy``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_QUALITY_IMPORT_ERROR = (
    "bar_quality_report requires scipy.  Install it with:\n"
    "    pip install scipy\n"
    "or:\n"
    "    pip install flowbars[all]"
)


def bar_quality_report(
    bars_df: Any,
    ljung_box_lags: int | None = None,
    day_length_ms: int = 86_400_000,
) -> dict[str, Any]:
    """Run statistical diagnostics on a bar DataFrame.

    Parameters
    ----------
    bars_df : pd.DataFrame or pl.DataFrame
        Completed bars.  Must contain at least the columns ``close``,
        ``bar_type``, ``num_ticks``, and ``open_ts``.
    ljung_box_lags : int or None, default None
        Number of lags for the Ljung-Box test.  When *None*, defaults to
        ``min(10, n_returns // 5)`` — capped at 10 for small samples.
    day_length_ms : int, default 86_400_000 (24 hours)
        Length of a "day" for bar-count stability, in milliseconds.

    Returns
    -------
    dict
        Dictionary with these keys:

        * ``n_bars`` — total number of bars
        * ``ljung_box`` — dict with ``statistic``, ``p_value``, ``lags``,
          ``reject_autocorrelation`` (bool, True = returns ARE autocorrelated)
        * ``jarque_bera`` — dict with ``statistic``, ``p_value``,
          ``reject_normality`` (bool, True = returns are NOT normal)
        * ``bar_count_stability`` — dict with ``per_day`` (list of counts),
          ``mean``, ``std``, ``cv`` (coefficient of variation)
        * ``run_fragmentation`` — dict with ``fraction_fragmented`` (fraction
          of run bars with ≤ 2 ticks), or ``None`` for non-run bar types
    """
    # -- lazy scipy import so the rest of flowbars doesn't need it ----------
    try:
        from scipy import stats as scipy_stats  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        raise ImportError(_QUALITY_IMPORT_ERROR) from None

    # Accept both pandas and polars DataFrames.  We use pandas internally
    # because the bar pipeline already returns pandas.
    try:
        import polars as pl  # noqa: PLC0415
    except ImportError:
        pl = None  # type: ignore[assignment]

    if pl is not None and isinstance(bars_df, pl.DataFrame):
        bars = bars_df.to_pandas()
    else:
        bars = bars_df

    n_bars = len(bars)
    result: dict[str, Any] = {"n_bars": n_bars}

    # -- returns ------------------------------------------------------------
    close = np.asarray(bars["close"], dtype=np.float64)
    if n_bars < 2:
        # Not enough bars for any return-based diagnostic.
        result["ljung_box"] = None
        result["jarque_bera"] = None
    else:
        log_returns = np.diff(np.log(close))
        log_returns = log_returns[np.isfinite(log_returns)]
        n_returns = len(log_returns)

        # ---- Ljung-Box ----------------------------------------------------
        if n_returns < 2:
            result["ljung_box"] = None
        else:
            lags = ljung_box_lags
            if lags is None:
                lags = max(1, min(10, n_returns // 5))
            lags = min(lags, n_returns - 1)

            acf = _autocorrelation(log_returns, lags)
            n = n_returns
            q_stat = n * (n + 2) * float(np.sum(acf**2 / (n - np.arange(1, lags + 1))))
            p_value = float(scipy_stats.chi2.sf(q_stat, lags))

            result["ljung_box"] = {
                "statistic": q_stat,
                "p_value": p_value,
                "lags": lags,
                "reject_autocorrelation": p_value < 0.05,
            }

        # ---- Jarque-Bera --------------------------------------------------
        if n_returns < 4:
            result["jarque_bera"] = None
        else:
            n = n_returns
            m2 = float(np.mean(log_returns**2))
            m3 = float(np.mean(log_returns**3))
            m4 = float(np.mean(log_returns**4))

            # Use biased (MLE) estimators for consistency with scipy
            skew = m3 / (m2**1.5) if m2 > 0 else 0.0
            kurt = m4 / (m2**2) if m2 > 0 else 0.0

            jb = (n / 6.0) * (skew**2 + ((kurt - 3.0) ** 2) / 4.0)
            p_value = float(scipy_stats.chi2.sf(jb, 2))

            result["jarque_bera"] = {
                "statistic": jb,
                "p_value": p_value,
                "skewness": skew,
                "kurtosis": kurt,
                "reject_normality": p_value < 0.05,
            }

    # -- bar-count stability -------------------------------------------------
    open_ts = np.asarray(bars["open_ts"], dtype=np.int64)
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
    is_run = bar_type_col.str.startswith("run_").any() if hasattr(bar_type_col, "str") else False

    if is_run:
        num_ticks = np.asarray(bars["num_ticks"], dtype=np.int64)
        n_run_bars = len(num_ticks)
        n_fragmented = int(np.sum(num_ticks <= 2))
        fraction = n_fragmented / n_run_bars if n_run_bars > 0 else 0.0
        result["run_fragmentation"] = {
            "n_run_bars": n_run_bars,
            "n_fragmented": n_fragmented,
            "fraction_fragmented": fraction,
        }
    else:
        result["run_fragmentation"] = None

    return result


def _autocorrelation(x: np.ndarray, nlags: int) -> np.ndarray:
    """Compute sample autocorrelation at lags 1..*nlags*.

    Uses the standard estimator:

    .. math::

        \\rho_k = \\frac{\\sum_{t=k+1}^n (x_t - \\bar{x})(x_{t-k} - \\bar{x})}
                        {\\sum_{t=1}^n (x_t - \\bar{x})^2}

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Mean-zero or nearly-so series (e.g. log-returns).
    nlags : int
        Number of lags.

    Returns
    -------
    np.ndarray, shape (nlags,)
        Autocorrelation at lags 1 through *nlags*.
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
