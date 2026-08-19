# flowbars

Production-grade bar construction for financial ML — standard and
information-driven bars, correct per *Advances in Financial Machine Learning*
(López de Prado, 2018).

```python
from flowbars import load_sample_data, compute_tick_bars, bar_quality_report
from flowbars.schema import SchemaMapping

df = load_sample_data()
schema = SchemaMapping({"timestamp": "timestamp", "price": "price", "volume": "volume"})
bars = compute_tick_bars(df, threshold=100, schema=schema)
report = bar_quality_report(bars)
print(f"Ljung-Box p-value: {report['ljung_box']['p_value']:.4f}")
```

## What makes this different

### Correct AFML-spec thresholds

The two-component EWMA formula from AFML §2.4 is implemented exactly:

| Bar family | Threshold formula |
|---|---|
| Imbalance | *Tₙ = E₀[T]ₙ × \|E₀[θ]ₙ\|* |
| Run | *Tₙ = E₀[T]ₙ × max(E₀[P⁺]ₙ, 1 − E₀[P⁺]ₙ)* |

Both terms are updated **at bar-close time** using the bar-level statistics
(tick count or volume, signed-proportion or buy-fraction). This is not a
single-EWMA approximation — it is the specification.

Other libraries (mlfinlab, mlfinpy) use a single EWMA on the raw cumulative
metric, which produces different bar counts and different information content.
If you are reproducing research, this matters.

### Streaming + resumable

Every bar constructor supports `update(tick)` for real-time streaming and
`get_state()` / `load_state()` / `from_state()` for crash-safe resume:

```python
# Save mid-stream state
state = ctor.get_state()  # plain dict, JSON-serializable

# Resume later — identical output as uninterrupted
ctor2 = TickBarConstructor.from_state(state)
```

Resume is **idempotent** via a configurable watermark: re-feeding ticks that
were already processed is silently discarded. Defaults to the tick timestamp;
pass `watermark="seq"` (or any column name) for feeds that emit multiple ticks
per millisecond, and `watermark=None` to disable.

```python
ctor = TickBarConstructor(threshold=100, schema=schema, watermark="timestamp")
state = ctor.get_state()

# Re-feeding an already-processed tick is a no-op
ctor2 = TickBarConstructor.from_state(state, schema=schema)
ctor2.update(tick_already_processed)  # -> None (deduplicated)
```

### Dual backend

| Backend | Dependency | Speed vs. pure-Python |
|---|---|---|
| `"python"` | numpy, pandas | 1× (baseline) |
| `"numba"` | + numba | 85× – 237× |

The numba backend JIT-compiles the tick-ingestion loop. Compilation happens
on first call; subsequent calls run at native speed. If numba is not installed
or compilation fails, the Python path is used automatically with a warning.

### Honest benchmarks

Measured, not assumed. Compilation cost reported separately. Benchmark script
in the repo — run it yourself:

```bash
python -m flowbars.benchmarks
```

Results on 250,000 synthetic ticks (Intel/AMD x86-64, numba 0.57+):

| Bar type | Python (s) | numba (ms) | Speedup |
|---|---|---|---|
| tick | 0.56 | 6.6 | **86×** |
| volume | 0.56 | 5.1 | **109×** |
| dollar | 0.68 | 4.7 | **145×** |
| time (5-min) | 0.57 | 3.2 | **181×** |
| imbalance_tick | 0.61 | 3.5 | **177×** |
| imbalance_volume | 0.64 | 3.3 | **194×** |
| imbalance_dollar | 0.75 | 6.1 | **123×** |
| run_tick | 1.04 | 5.0 | **210×** |
| run_volume | 1.00 | 4.2 | **237×** |
| run_dollar | 1.24 | 6.7 | **185×** |
| imbalance_tick (EWMA) | 0.59 | 5.3 | **111×** |
| run_tick (EWMA) | 2.40 | 18.8 | **127×** |

### Pluggable calendars

Four calendars cover every asset class, with full holiday/DST support where it
matters:

| Asset class | Calendar |
|---|---|
| Crypto (spot/perp, 24/7) | `ContinuousCalendar` |
| FX majors/minors (24/5) | `WeekdayCalendar` |
| US equities | `ExchangeCalendar("XNYS")` |
| US futures / commodities | `ExchangeCalendar("CME")` |
| LSE / EU equities | `ExchangeCalendar("LSE")` / `("XETR")` |
| Custom fixed UTC session | `SessionCalendar(...)` |

- `ContinuousCalendar` — always open; never a boundary.
- `WeekdayCalendar` — open Mon–Fri; boundaries only at weekends.
- `SessionCalendar` — fixed daily UTC hours (no holidays/DST — use only for a
  constant UTC window).
- `ExchangeCalendar` — wraps `exchange_calendars` for per-exchange holidays,
  half-days, and DST-aware sessions (`pip install flowbars[calendars]`).

Session boundaries trigger automatic bar close and accumulator reset.

### No silent guesses

Explicit schema mapping — you tell the library your column names. No
auto-detection, no magic. If a required column is missing, you get a clear
`SchemaError`.

### Bar quality diagnostics

`bar_quality_report()` provides honest, standard statistical tests:

| Diagnostic | Test | What it flags |
|---|---|---|
| Autocorrelation | Ljung-Box | Serially-correlated returns |
| Normality | Jarque-Bera | Non-normal return distribution |
| Stability | CV of per-day bar counts | Uneven sampling rate |
| Fragmentation | Fraction of run-bars ≤ 2 ticks | Over-fragmented run bars |

## Bar types

### Standard (activity-based)

| Bar type | Closes when… | Constructor |
|---|---|---|
| tick | `num_ticks ≥ threshold` | `TickBarConstructor` |
| volume | `cumulative_volume ≥ threshold` | `VolumeBarConstructor` |
| dollar | `cumulative_dollar ≥ threshold` | `DollarBarConstructor` |
| time | tick crosses interval boundary | `TimeBarConstructor` |

### Information-driven (adaptive threshold)

| Bar type | Closes when… | Constructor |
|---|---|---|
| imbalance_tick | \|Σ signed ticks\| ≥ adaptive T | `ImbalanceTickBarConstructor` |
| imbalance_volume | \|Σ signed volume\| ≥ adaptive T | `ImbalanceVolumeBarConstructor` |
| imbalance_dollar | \|Σ signed dollar\| ≥ adaptive T | `ImbalanceDollarBarConstructor` |
| run_tick | same-sign tick run ≥ adaptive T | `RunTickBarConstructor` |
| run_volume | same-sign volume run ≥ adaptive T | `RunVolumeBarConstructor` |
| run_dollar | same-sign dollar run ≥ adaptive T | `RunDollarBarConstructor` |

All ten are available through convenience functions:

```python
from flowbars import (
    compute_tick_bars,
    compute_volume_bars,
    compute_dollar_bars,
    compute_time_bars,
    compute_imbalance_tick_bars,
    compute_imbalance_volume_bars,
    compute_imbalance_dollar_bars,
    compute_run_tick_bars,
    compute_run_volume_bars,
    compute_run_dollar_bars,
)
```

## Install

```bash
pip install flowbars                  # pure-Python backend
pip install flowbars[numba]           # + numba JIT backend
pip install flowbars[polars]          # + polars DataFrame adapter
pip install flowbars[calendars]       # + exchange_calendars (holidays/DST)
pip install flowbars[all]             # everything
```

## Quickstart

### Build tick bars from the bundled sample data

```python
from flowbars import load_sample_data, compute_tick_bars
from flowbars.schema import SchemaMapping

df = load_sample_data()  # 500 ticks with timestamp, price, volume
schema = SchemaMapping({"timestamp": "timestamp", "price": "price", "volume": "volume"})

bars = compute_tick_bars(df, threshold=50, schema=schema)
print(bars.head())
```

### Build dollar bars from your own data

```python
import pandas as pd
from flowbars import compute_dollar_bars
from flowbars.schema import SchemaMapping

ticks = pd.read_csv("my_ticks.csv")  # must have ts, px, qty columns
schema = SchemaMapping({"timestamp": "ts", "price": "px", "volume": "qty"})

bars = compute_dollar_bars(ticks, threshold=1_000_000, schema=schema)
```

### Build imbalance bars with adaptive threshold

```python
from flowbars import compute_imbalance_tick_bars
from flowbars.tick_rule import resolve_tick_signs

df["side"] = resolve_tick_signs(df["price"].values, None)
schema = SchemaMapping(
    {
        "timestamp": "ts",
        "price": "px",
        "volume": "qty",
        "side": "side",
    }
)

bars = compute_imbalance_tick_bars(
    df,
    span=20.0,  # EWMA span
    warmup_bars=5,  # discard first 5 bars while EWMA converges
    schema=schema,
)
```

### Use the numba backend

```python
from flowbars.bars.constructor import BaseBarConstructor
from flowbars.bars.accumulators import TickAccumulator
from flowbars.thresholds import StaticThresholdEstimator

ctor = BaseBarConstructor(
    accumulator=TickAccumulator(bar_type="tick"),
    threshold_estimator=StaticThresholdEstimator(threshold=100),
    schema=schema,
    backend="numba",  # flip this one switch
)
bars = ctor.batch(ticks_df)
```

### Polars adapter

```python
import polars as pl
from flowbars.adapters.polars import compute_volume_bars

pl_ticks = pl.read_csv("ticks.csv")
pl_bars = compute_volume_bars(pl_ticks, threshold=5000)
```

### Save and resume mid-stream

```python
# Process first batch
ctor = TickBarConstructor(threshold=100, schema=schema)
batch1 = ctor.batch(ticks_df.iloc[:1000])

# Save state
state = ctor.get_state()

# ... days later, resume with new data ...
ctor2 = TickBarConstructor.from_state(state, schema=schema)
batch2 = ctor2.batch(ticks_df.iloc[1000:])
```

### Check bar quality

```python
from flowbars import bar_quality_report

report = bar_quality_report(bars, ljung_box_lags=10)
print(f"Autocorrelated returns: {report['ljung_box']['reject_autocorrelation']}")
print(f"Non-normal returns:     {report['jarque_bera']['reject_normality']}")
print(f"Bar-count CV:           {report['bar_count_stability']['cv']:.3f}")
```

## Reproducing benchmarks

### Data sourcing

The benchmarks use synthetic data generated in-process (seeded random walk
prices + exponential volumes). No external data files are needed.

To benchmark against the same data used in the README table:

```bash
python -m flowbars.benchmarks
```

### Comparing with mlfinlab / mlfinpy

The comparison notebooks in `notebooks/` use this procedure:

1. Download Binance BTC/USDT tick data from
   [Binance Public Data](https://data.binance.vision/) (monthly, csv.zip).
2. Extract to `data/btcusdt/`.
3. Run `notebooks/01_compare_mlfinlab.ipynb`.

Date range used: 2024-01-01 through 2024-01-31 (January 2024).

## Assumptions

| Assumption | Detail |
|---|---|
| **UTC only** | No timezone conversion or detection. All timestamps are treated as Unix milliseconds in UTC. |
| **Ticks ordered** | Ticks must be non-decreasing in timestamp. The library does not sort for you — it processes in the order given. Set `strict_ordering=True` to raise on out-of-order input. |
| **Explicit schema** | You tell the library your column names via `SchemaMapping`. There is no auto-detection. |
| **Float64 internally** | All prices, volumes, and computed values use `float64`. Float32 input is upcast. |
| **NaN side = unknown** | A tick with `side=NaN` is treated as undetermined (first tick of a stream). It is included in OHLCV but excluded from signed-imbalance sums. Run-bar logic treats NaN as a wildcard that matches any direction. |
| **Thread safety** | Bar constructors are **not** thread-safe. Feed ticks from a single thread. |
| **Batch = streaming** | `ctor.batch(df)` and `ctor.update(tick)` in a loop produce identical output for the same data. |

## API reference

### Top-level functions

| Function | Returns | Description |
|---|---|---|
| `compute_tick_bars(df, threshold, ...)` | DataFrame | Tick bars |
| `compute_volume_bars(df, threshold, ...)` | DataFrame | Volume bars |
| `compute_dollar_bars(df, threshold, ...)` | DataFrame | Dollar bars |
| `compute_time_bars(df, interval_ms, ...)` | DataFrame | Time bars |
| `compute_imbalance_tick_bars(df, ...)` | DataFrame | Tick-imbalance bars |
| `compute_imbalance_volume_bars(df, ...)` | DataFrame | Volume-imbalance bars |
| `compute_imbalance_dollar_bars(df, ...)` | DataFrame | Dollar-imbalance bars |
| `compute_run_tick_bars(df, ...)` | DataFrame | Tick-run bars |
| `compute_run_volume_bars(df, ...)` | DataFrame | Volume-run bars |
| `compute_run_dollar_bars(df, ...)` | DataFrame | Dollar-run bars |
| `load_sample_data()` | DataFrame | Bundled 500-tick sample |
| `bar_quality_report(bars)` | dict | Statistical diagnostics |

### Core classes

| Class | Purpose |
|---|---|
| `BaseBarConstructor` | Configurable bar construction engine |
| `SchemaMapping` | Explicit column-name mapping |
| `BarRegistry` | Discover registered bar types |
| `EWMAThresholdEstimator` | Two-component adaptive threshold |
| `StaticThresholdEstimator` | Fixed threshold |
| `StaticCalibrationHelper` | Estimate good seeds from data |
| `ContinuousCalendar` | No session boundaries (24/7) |
| `WeekdayCalendar` | Mon–Fri, 24/5 |
| `SessionCalendar` | Fixed daily UTC session hours |
| `ExchangeCalendar` | Holiday/DST-aware (via `exchange_calendars`) |

### Tick rule

| Function | Description |
|---|---|
| `derive_tick_sign(prices)` | Tick rule on raw prices → side array |
| `resolve_tick_signs(prices, sides)` | Use supplied sides or derive |

## Comparison to other libraries

| | flowbars | mlfinlab | mlfinpy |
|---|---|---|---|
| **EWMA formula** | Two-component (AFML spec) | Single EWMA | Single EWMA |
| **Streaming** | `update(tick)` + state save/resume | Batch only | Batch only |
| **numba backend** | Yes (85×–237×) | Yes | No |
| **Polars** | First-class adapter | No | No |
| **Calendars** | Pluggable (continuous, weekday, sessions, exchange-backed) | Hardcoded NYSE | Hardcoded |
| **Bar quality** | Built-in diagnostics | None | None |
| **Schema** | Explicit mapping | Auto-inference | Auto-inference |
| **Benchmarks** | Reproducible, in-repo | None | None |

## License

MIT — see [LICENSE](LICENSE).
