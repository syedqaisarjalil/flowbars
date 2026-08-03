# flowbars

Production-grade bar construction for financial ML.

Standard bars (time, tick, volume, dollar) and information-driven bars
(imbalance, run) per Advances in Financial Machine Learning (López de Prado),
implemented correctly — not transcribed.

## What makes this different

- **Correct AFML-spec dynamic thresholds** — two-component EWMA (E₀[T] × |E₀[θ]|
  for imbalance, E₀[T] × max(E₀[P⁺], 1−E₀[P⁺]) for run), not a single-EWMA
  approximation
- **Streaming + resumable** — `BarConstructor.update()` with restart-safe
  `get_state()`/`load_state()`/`from_state()`, capturing both accumulator and
  threshold estimator state
- **Dual backend** — pure-Python (zero extra deps) + numba (optional, real speed)
- **Pluggable calendars** — `ContinuousCalendar` (crypto/forex) and
  `SessionCalendar` (equities/futures), session-boundary-aware
- **Honest benchmarks** — measured, not assumed; compilation cost reported
  separately; benchmark script in the repo
- **No silent guesses** — explicit schema mapping, no auto column-name detection

## Install

```bash
pip install flowbars                  # pure-Python backend
pip install flowbars[numba]           # + numba backend
pip install flowbars[polars]          # + polars adapter
pip install flowbars[all]             # everything
```

## Quickstart

```python
from flowbars import compute_dollar_bars, bar_quality_report

bars = compute_dollar_bars(
    tick_df,
    schema={"timestamp": "datetime", "price": "px", "volume": "qty"},
    threshold="ewma",
)

report = bar_quality_report(bars)
```

See the [comparison notebooks](notebooks/) for detailed benchmarks vs.
mlfinlab/mlfinpy and tick-built vs. minute-approximated bar divergence.

## Assumptions

- **UTC timestamps only** — no timezone conversion or detection
- **Ticks ordered** — non-decreasing timestamps (configurable strictness)
- **Explicit schema** — you tell the library your column names; it never guesses
- **Float64 internally** — float32 input is upcast

## License

MIT — see [LICENSE](LICENSE).
