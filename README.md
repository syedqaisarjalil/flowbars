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

State is versioned (`schema_version`/`engine_version`) and carries the full
params plus a SHA-256 params hash, so resuming with a mismatched threshold,
multiplier, volume type, or session policy raises `StateValidationError`
instead of silently changing bar semantics.  The state also records the
canonical registry key (`registry_name`), the calendar identity
(`calendar` + `calendar_fingerprint`), the input column mapping
(`schema_mapping` + `schema_fingerprint`), and the active session
(`session_id`, `last_session_end`, `in_session`).  Resuming with a different
calendar or schema mapping therefore fails loudly — pass
`verify_identity=False` to `load_state()` / `from_state()` to opt out.
Tick and minute state can never be cross-loaded (`input_kind` is validated).
`update_many(df)` streams a whole DataFrame through the same `update()` path.

### Warm-up contract

```python
ctor = TickBarConstructor(
    threshold=100,
    schema=schema,
    warmup_minutes=30,  # discard bars closing in the first 30 minutes
)
# or warmup_bars=N — the two are mutually exclusive
...
assert ctor.warmup_done  # True once the warm-up window has elapsed
```

Warm-up bars keep their original `bar_id`, still fire `on_bar`, and still feed
the adaptive estimator; only their return is suppressed.

### Session policies

On each session-boundary transition (weekends, holidays, early closes, DST —
whatever the calendar reports) the in-progress bar is handled per
`session_policy`:

| Policy | Behavior |
|---|---|
| `"close"` (default) | Force-close as a `partial` bar; no bar spans two sessions |
| `"discard"` | Drop the in-progress bar silently (no callback, no overflow carried) |
| `"carry"` | Keep the bar open across the boundary (opt-in); the gap adds no return |

Session-only feeds (equity/FX data that never prints outside the trading day)
are handled too: the engine compares each in-session input against the
calendar's next session open, so the first input of a later session
force-closes the hanging bar at the **prior session's close** (`partial=True`,
`close_ts` = last session end) instead of letting one bar span the weekend or a
holiday.  Minute volatility/hybrid bars exclude the gap return, and the minute
tick rule derives no sign across the gap.

When a stream simply ends mid-session (no later input to reveal the gap), call
`flush()` to emit the open bar as a `partial` bar:

```python
bar = ctor.flush()          # close_ts = calendar session end of the last input
bar = ctor.flush(now_ms)    # or an explicit close instant
```

`session_estimator="persist"` (default) keeps EWMA learning across sessions;
`"reset"` restores the seeds at each boundary.  Policies are per constructor,
so different series can choose independently.

### Discover bar types

```python
from flowbars.bars.registry import BarRegistry

BarRegistry.list()  # ['dollar', 'imbalance_tick', ...]
spec = BarRegistry.get_spec("volume")
spec.params_schema  # JSON Schema for constructor params
spec.output_columns  # core_columns + extras_columns
spec.core_columns  # the 13 contract columns
spec.extras_columns  # typed bar-specific extras ({} for built-ins)
spec.session_policies  # ('close',)
spec.state_schema_version  # 2
spec.spec_version  # output-contract version (pin with params_hash)
spec.warmup_minutes  # recommended warmup; 0 = none required
```

`MinuteBarRegistry` mirrors this for the `minute_*` types.  The registry key
is the canonical identity (`minute_volume`), while each row's `bar_type`
label is the shorter family name (`volume`); `frame.attrs["registry_name"]`
and `state["registry_name"]` carry the unambiguous key.

Rebuild any registered type from saved state without knowing its concrete
constructor:

```python
resumed = BarRegistry.constructor_from_state("volume", state, schema=schema)
resumed_minute = MinuteBarRegistry.constructor_from_state(
    "minute_renko", state, schema=minute_schema
)
```

Pin a downstream contract with `params_hash(params)` (canonical JSON: sorted
keys, no whitespace, NaN/Infinity rejected) — frame attrs expose
`(registry_name, spec_version, params_hash)` for exactly this.

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

`bar_quality_report()` provides honest, standard statistical tests.  `scipy`
is optional — without it the Ljung-Box / Jarque-Bera statistics are still
computed (in numpy) but their p-values and reject flags are `None`.

| Diagnostic | Test | What it flags |
|---|---|---|
| Autocorrelation | Ljung-Box | Serially-correlated returns |
| Normality | Jarque-Bera | Non-normal return distribution |
| Stability | CV of per-day bar counts | Uneven sampling rate |
| Fragmentation | Fraction of run-bars ≤ 2 ticks/minutes | Over-fragmented run bars |
| Return moments | mean, std, skewness, kurtosis | Distribution shape |
| Return ACF | lag-1, lag-5 (raw and `\|r\|`) | Serial dependence, volatility clustering |
| Variance ratio | lag-2, lag-5 | Random-walk departure |
| Entropy | Histogram Shannon entropy | Information content (normalized) |
| Bar-size CV | CV of volume (or `num_ticks`) | Uneven bar sizes |
| Duration CV | CV of `close_ts − open_ts` | Uneven bar durations |
| Effective sample size | `n·(1−ρ₁)/(1+ρ₁)` | Overstated sample size from autocorrelation |
| OHLC integrity | Violations of `low ≤ open, close ≤ high` | Malformed bars |
| Composite score | Mean of available sub-scores | One-number quality summary |

All metrics are deterministic (fixed lag set, fixed histogram rule, no RNG).
For minute bars pass `num_ticks_unit="minutes"`; the run-fragmentation
threshold is configurable with `fragmentation_threshold=`.  Run fragmentation
is masked to `run_*` bar types, so mixed frames report correctly.

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

Every tick constructor and `compute_*_bars` function accepts `warmup_bars` /
`warmup_minutes`, `backend`, `stream_id`, `multiplier`, `volume_type`,
`session_policy`, `session_estimator`, and the `on_bar` /
`on_threshold_update` callbacks, matching the minute path's signature surface.

## Minute-OHLCV path

Non-time bars can also be built from **minute OHLCV candles** instead of raw
ticks.  The minute path has full mode parity with the tick path — streaming
`update()`/`update_many()`, versioned state save/resume, callbacks,
`warmup_bars`/`warmup_minutes`, watermark dedup, `strict_ordering`, calendars
with the same session policies, the numba batch backend, and the polars
adapter — and produces the identical 13-column output schema.  `num_ticks`
counts **minutes**, and `dollar_value` / `vwap` use `close × volume` as the
notional proxy (so dollar-family minute bars are flagged
`approximate=True` in the registry spec and frame metadata).

| Minute type | Closes when… | Function |
|---|---|---|
| volume | `Σ volume ≥ threshold` | `compute_volume_bars_from_minutes` |
| dollar | `Σ close×volume ≥ threshold` | `compute_dollar_bars_from_minutes` |
| imbalance_volume | \|Σ sign×volume\| ≥ EWMA T | `compute_imbalance_volume_bars_from_minutes` |
| imbalance_dollar | \|Σ sign×close×volume\| ≥ EWMA T | `compute_imbalance_dollar_bars_from_minutes` |
| run_volume | same-sign volume run ≥ EWMA T | `compute_run_volume_bars_from_minutes` |
| run_dollar | same-sign dollar run ≥ EWMA T | `compute_run_dollar_bars_from_minutes` |
| volatility | `Σ \|closeₜ−closeₜ₋₁\|/closeₜ₋₁ ≥ threshold` | `compute_volatility_bars_from_minutes` |
| range | `Σ (high−low)/close ≥ threshold` | `compute_range_bars_from_minutes` |
| renko | `\|close−anchor\|/anchor ≥ threshold` (anchor = first open) | `compute_renko_bars_from_minutes` |
| hybrid | both `Σ close×volume` and volatility targets met | `compute_hybrid_bars_from_minutes` |

The four structural types (volatility, range, renko, hybrid) are minute-only
and use **static** thresholds.  Renko displacement is not additive, so its
anchor resets on every close with no overflow; the other three roll their
scalar excess into the next bar.  Volatility and hybrid reset `prev_close` at
session boundaries so an overnight gap contributes no fake return.

### Calibrate thresholds from minute data

`MinuteCalibrationHelper` is deterministic (no RNG, fixed bisection
iterations): repeated calls on the same input return identical values.  Targets
are bars **per UTC day**, scaled by the number of observed UTC days.

```python
from flowbars import MinuteCalibrationHelper

vol_threshold = MinuteCalibrationHelper.estimate_volume_threshold(minutes, 50.0)
renko_threshold = MinuteCalibrationHelper.estimate_renko_threshold(minutes, 50.0)
dollar_t, variance_t = MinuteCalibrationHelper.estimate_hybrid_thresholds(
    minutes, 50.0, volatility_threshold=0.01
)
seeds = MinuteCalibrationHelper.estimate_ewma_seeds(
    minutes, "run", metric="dollar", span=20.0, target_bars=50
)
```

Minute bars work with the polars adapter too:
`from flowbars.adapters.polars import compute_volume_bars_from_minutes`.

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
from flowbars import VolumeBarConstructor

ctor = VolumeBarConstructor(
    5_000.0,
    schema=schema,
    backend="numba",  # flip this one switch
)
bars = ctor.batch(ticks_df)
```

All ten tick constructors and all ten `compute_*_bars` functions accept
`backend="numba"`; minute constructors take the same argument.  The Python
backend remains the default, and `update()` always runs in Python.

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

The comparison notebook `notebooks/01_compare_mlfinpy.ipynb` compares flowbars
against [mlfinpy](https://pypi.org/project/mlfinpy/) — the free, open-source
reimplementation of the paid MlFinLab — on synthetic tick data (no external
data files needed). It requires Python ≥ 3.11:

```bash
conda create -n flowbars311 python=3.11 -y
conda activate flowbars311
pip install mlfinpy scikit-learn
pip install .                 # from the flowbars repo root
jupyter nbconvert --to notebook --execute notebooks/01_compare_mlfinpy.ipynb
```

A direct comparison against the paid MlFinLab is deferred; mlfinpy serves as
the free, drop-in proxy.

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
| `params_hash(params)` | str | Canonical SHA-256 of a params dict |

All `compute_*_bars` functions also accept keyword-only `warmup_bars` /
`warmup_minutes`, `backend`, `stream_id`, `multiplier`, `volume_type`,
`session_policy`, `session_estimator`, `on_bar`, and `on_threshold_update`.
The minute equivalents are `compute_*_bars_from_minutes`
(`compute_volume_bars_from_minutes`, …, `compute_hybrid_bars_from_minutes`)
with the same keyword surface.

### Output contract

Every bar — tick or minute — has the same 13 columns: `bar_id`, `open`,
`high`, `low`, `close`, `volume`, `dollar_value`, `vwap`, `num_ticks`,
`open_ts`, `close_ts`, `bar_type`, `partial`.  All are **core** columns
(`spec.core_columns`); registered extras, if any, are appended and typed
(`spec.extras_columns`).  `partial` is `True` only for session-boundary
force-closes.  Returned DataFrames carry series metadata in `DataFrame.attrs`
(`bar_type`, `registry_name`, `params_hash`, `spec_version`, input kind,
engine/state version, multiplier, volume type, `approximate`).  `multiplier`
scales `dollar_value` (and the dollar/imbalance/run metrics) while `vwap`
stays a price; `volume_type` (`base`/`quote`/`shares`/`contracts`/`tick`) is
metadata.

Timestamp conventions:

* `open_ts` is the timestamp of the bar's first input — the **stable bar
  identity**.  It is unique within a series (`validate_bar_frame` enforces
  this).  `bar_id` is run-local and resets when you rebuild from source, so
  key downstream rows on `open_ts`, not `bar_id`.
* `close_ts` is the last input that completed a threshold/time bar, or the
  crossing minute's open timestamp on the minute path.
* For `partial=True` bars, `close_ts` is the **session boundary** (session
  close or next open), i.e. the instant the bar's information became
  available.

Input timestamps must be integer Unix epoch **milliseconds** (integral
floats are accepted and cast; `datetime64`/tz-aware columns raise
`SchemaError` with a conversion hint).  Pass `validate=True` to
`batch()` / `update_many()` to run `validate_bar_frame` on the returned frame.

### Core classes

| Class | Purpose |
|---|---|
| `BaseBarConstructor` | Configurable bar construction engine |
| `SchemaMapping` / `MinuteSchemaMapping` | Explicit column-name mapping |
| `BarRegistry` / `MinuteBarRegistry` | Discover registered bar types |
| `BarTypeSpec` | Machine-readable spec (params schema, output columns, capabilities) |
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

MIT — see [LICENSE](https://github.com/syedqaisarjalil/flowbars/blob/main/LICENSE).
