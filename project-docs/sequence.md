# flowbars — Implementation Sequence

Ordered by dependency. Each phase must be complete and tested before the next begins.
Tests are written alongside implementation, not as a separate phase.

---

## Phase 0: Foundation (no dependencies) ✅ DONE

### 0.1 — Schema mapping ✅
- `SchemaMapping` class: validates user-supplied column-name dict, extracts required
  columns (timestamp, price, volume, optional side) from pandas DataFrames.
- Normalizes input to internal `TickInfo` representation.
- Raises `SchemaError` on missing required keys.
- Value validation: `validate_price/volume/side` (scalar) + `_array` variants (batch).
- Rejects NaN/Inf/negative in price and volume, malformed side values.
- Side NaN accepted (undetermined sign — first tick of a stream).
- **Files:** `flowbars/schema.py`, `tests/test_schema.py`
- **Tests:** 55 (construction 7, column validation 3, extract_arrays 15, normalize_tick 10,
  value validation 14, smoke 5, + schema smoke)

### 0.2 — Core types ✅
- `TickInfo`, `Bar` dataclasses finalized.
- `FlowbarsError` hierarchy: `SchemaError`, `ThresholdError`, `StateValidationError`,
  `TickDataError`.
- `SchemaMapping` re-exported from `flowbars/__init__.py`.
- **Files:** `flowbars/core.py`, `tests/test_smoke.py`

---

## Phase 1: Tick rule (depends on 0.2) ✅ DONE

### 1.1 — Tick-rule sign derivation ✅
- `derive_tick_sign(prices) -> ndarray[float64]`
- First tick → `NaN`, equal price → carry forward previous sign, otherwise
  sign(price_t − price_{t-1}).
- Pure numpy loop (no numba yet). O(n), correct, tested.
- Handles edge cases: empty, single tick, all equal, all rising, all falling,
  equal-at-start, equal-mid-stream, alternating, large gaps, float precision.
- **Files:** `flowbars/tick_rule.py`

### 1.2 — Side column passthrough ✅
- `resolve_tick_signs(prices, supplied_sides)` — uses supplied sides if provided
  (returns copy), otherwise derives via tick rule.
- Length validation (mismatch raises ValueError).
- Side validation already handled by SchemaMapping (Phase 0.1).
- **Files:** `flowbars/tick_rule.py`, `tests/test_tick_rule.py`
- **Tests:** 19 (derive_tick_sign 13, resolve_tick_signs 6)

---

## Phase 2: Bar accumulator (depends on 0.2, indirectly on 1.1) ✅ DONE

### 2.1 — Base accumulator class ✅
- Tracks: running OHLCV, volume, dollar_value, tick count, timestamps.
- `add_tick(tick) -> None`: update running statistics.
- `should_close(threshold) -> bool`: does the running total meet/exceed threshold?
- `close(threshold) -> Bar`: emit completed bar, reset with overflow rollover.
- `current_bar -> Bar | None`: read-only partial bar access.
- Overflow semantics: excess rolls into next bar. Shared base logic.
- `get_state()` / `load_state()`: JSON-serializable state persistence.
- **Tests:** 62 (base OHLCV 8, tick 6, volume 3, dollar 3, time 8, imbalance 10,
  run 11, state persistence 7, edge cases 6)

### 2.2 — Tick accumulator (for tick bars) ✅
- Closes when `num_ticks >= threshold`.
- TickAccumulator with `_cum_ticks` (discrete — overflow always 0 for integer thresholds).
- **Tests:** exact threshold match, overflow, single tick, close-then-next-tick.

### 2.3 — Volume accumulator (for volume bars) ✅
- Closes when `cumulative_volume >= threshold`.
- VolumeAccumulator with `_cum_volume`. Overflow: excess volume rolls into next bar.
- **Tests:** closure, overflow with hand-computed values (300+400+500 vs 1000 threshold),
  exact threshold (no overflow).

### 2.4 — Dollar accumulator (for dollar bars) ✅
- Closes when `cumulative_dollar_value >= threshold`.
- DollarAccumulator with `_cum_dollar`. Matches spec example ($300k+$400k+$500k).
- **Tests:** closure, overflow rollover per spec example, price×volume computation.

### 2.5 — Time accumulator (for time bars) ✅
- Closes when tick timestamp crosses a round-clock boundary.
- `anchor="clock"` (default) vs `anchor="first_tick"`.
- `_next_boundary_ms` computed on first tick; advances by `interval_ms` on each close.
- **Tests:** boundary crossing, exact boundary, multi-bar sequence, first-tick anchor,
  empty interval (gap), negative/zero interval raises, invalid anchor raises.

### 2.6 — Imbalance accumulator (for imbalance bars) ✅
- Tracks cumulative signed imbalance: Σ(b_t × metric) where metric ∈ {tick, volume, dollar}.
- First-tick sign is NaN → excluded from imbalance sum, included in OHLCV.
- Closes when |signed_imbalance| >= threshold.
- Overflow: signed excess carries into next bar (sign preserved).
- **Tests:** tick/volume/dollar metrics, first-tick exclusion (None + NaN), pure-buy,
  pure-sell, mixed, positive/negative overflow, exact boundary.

### 2.7 — Run accumulator (for run bars) ✅
- Tracks runs: consecutive same-sign ticks. `_same_direction()` helper: NaN matches
  anything (first-tick retroactively included in tick-2's run direction).
- Closes when `_banked + _run_cum >= threshold`.
- Direction change: banks current run, starts new run. Overflow: excess carried in
  current run's direction.
- **Tests:** single run, multi-run (direction changes), alternating tiny runs,
  first-tick NaN retroactive inclusion (buy + sell cases), volume/dollar metrics,
  overflow single/multi-run, run accumulation is NOT signed (buy 3 + sell 2 = 5, not 1).
- **Note:** `min_run_length` filter deferred to Phase 7 (bar constructor level).

---

## Phase 3: Threshold estimation (depends on 0.2) ✅ DONE

### 3.1 — `ThresholdEstimator` ABC ✅
- Interface: `update(tick_info)`, `current_threshold` (property), `get_state()`,
  `load_state()`, `from_state()` (classmethod), `reset()`, `on_bar_close()` (hook).
- **Files:** `flowbars/thresholds.py`, `tests/test_thresholds.py`

### 3.2 — `StaticThresholdEstimator` ✅
- Fixed threshold, never changes. `update()` is a no-op.
- Raises `ThresholdError` on negative threshold.
- Tests: returns same value always, state round-trip, from_state, reset.

### 3.3 — `EWMAThresholdEstimator` ✅
- Implements the two-component AFML formula.
- **Imbalance bars:** `T_n = E₀[T]_n × |E₀[θ]_n|` — two EWMA terms updated at
  bar-close time.
- **Run bars:** `T_n = E₀[T]_n × max(E₀[P⁺]_n, 1 − E₀[P⁺]_n)` — same E₀[T],
  different multiplier.
- Configurable span/halflife (halflife takes precedence), initial values.
- Read-only properties: `alpha`, `span`, `halflife`, `n_updates`, `ewa_t`,
  `ewa_proportion`.
- Tests: hand-computed EWMA over small series (3 bars each for imbalance and run),
  independently derived expected values. Span/halflife equivalence. Convergence.
  Negative proportion for imbalance, asymmetric P⁺ for run.

### 3.4 — `StaticCalibrationHelper` ✅
- `estimate_fixed_threshold(ticks, bar_type, target_bars_per_day)` — exact for
  standard bars, simulation-based for info-driven bars.
- `estimate_ewma_seeds(ticks, bar_family, metric, span, target_bars)` — simulates
  bars, collects per-bar stats, warms EWMA forward, returns seed dict.
- Returns a dict consumable by `EWMAThresholdEstimator` or `StaticThresholdEstimator`.
- Tests: known tick streams → expected bar count ≈ target (within tolerance).
  EWMA seeds for imbalance/run tick/volume. Edge: empty, single tick, invalid params.
- **Tests:** 57 (ABC 4, static 10, EWMA 22, state persistence 7, calibration 14)

---

## Phase 4: Calendars (depends on 0.2) ✅ DONE

### 4.1 — `TradingCalendar` ABC ✅
- `is_session_boundary(timestamp: int) -> bool`
- `next_session_open(timestamp: int) -> int`
- Stateless configuration — no `get_state()`/`load_state()` needed.

### 4.2 — `ContinuousCalendar` ✅
- Always returns False (no boundaries). Crypto/forex default.
- `next_session_open` returns the input timestamp (identity).
- Tests: never triggers boundary, identity holds for any timestamp.

### 4.3 — `SessionCalendar` ✅
- Configurable session hours in UTC (e.g., 09:30–16:00).
- Overnight sessions supported (e.g., 22:00–06:00).
- Weekends (Sat/Sun) hardcoded as non-trading. No holiday calendar.
- Session open must fall on a weekday for the session to be valid.
- **Tests:** 46 (ABC 3, continuous 2, intraday 16, overnight 11, minute precision 5,
  validation 5, edge cases 5)
- **Files:** `flowbars/calendars.py`, `tests/test_calendars.py`

---

## Phase 5: Base bar constructor (depends on 2.1, 3.1, 4.1) ✅ DONE

### 5.1 — `BaseBarConstructor` ✅
- Shared infrastructure for all bar types.
- Holds: accumulator, threshold estimator, calendar, backend, schema mapping.
- `update(tick) -> Bar | None`: feed one tick.
- `batch(ticks_df) -> DataFrame`: feed all ticks, return all bars.
- `get_state()` / `load_state()` / `from_state()`: state persistence.
- `current_bar` property.
- Callbacks: `on_bar`, `on_threshold_update`.
- `warmup_bars` parameter.
- Thread safety: not thread-safe (documented, not enforced).
- Tests: batch vs streaming equivalence (small dataset, all bar types).
- **Files:** `flowbars/bars/constructor.py`, `tests/test_constructor.py`
- **Tests:** 42 (basic 11, callbacks 5, batch 10, state persistence 9, edge cases 7)

### 5.2 — State persistence implementation ✅
- State dict: schema version, `stream_id`, bar_type, threshold config, backend,
  accumulator state, threshold estimator state, `next_bar_id`.
- Identity validation: mismatched stream_id/bar_type/threshold → `StateValidationError`.
- Version check: old version → clear error message.
- Tests: round-trip (interrupted stream = uninterrupted), from_state reconstruction,
  identity mismatch raises, version mismatch raises, bar_id continuity.

### Accumulator enhancements ✅
- `BaseAccumulator.get_close_stats()`: returns `(t_stat, proportion_stat)` for bar-close EWMA update.
- `ImbalanceAccumulator.get_close_stats()`: signed imbalance proportion θ.
- `RunAccumulator`: buy/sell tracking via `_buy_cum` and `_sell_cum` for P⁺ computation.
- RunAccumulator state persistence updated for buy/sell fields.

---

## Phase 6: Standard bars (depends on 5.1) ✅ DONE

### 6.1 — Time bars ✅
- `TimeBarConstructor`, `compute_time_bars()`.
- Register via `BarRegistry.register("time", ...)`.
- Tests: batch, streaming, equivalence, edge cases.
- **Files:** `flowbars/bars/standard/time_bars.py`

### 6.2 — Tick bars ✅
- `TickBarConstructor`, `compute_tick_bars()`.
- Register via `BarRegistry.register("tick", ...)`.
- Tests: same pattern.
- **Files:** `flowbars/bars/standard/tick_bars.py`

### 6.3 — Volume bars ✅
- `VolumeBarConstructor`, `compute_volume_bars()`.
- Register via `BarRegistry.register("volume", ...)`.
- Tests: same pattern.
- **Files:** `flowbars/bars/standard/volume_bars.py`

### 6.4 — Dollar bars ✅
- `DollarBarConstructor`, `compute_dollar_bars()`.
- Register via `BarRegistry.register("dollar", ...)`.
- Tests: same pattern.
- **Files:** `flowbars/bars/standard/dollar_bars.py`

- **Tests:** 39 (time 11, tick 10, volume 8, dollar 7, registry 3)
- **Files:** `flowbars/bars/standard/__init__.py` (re-exports), `tests/test_standard_bars.py`

---

## Phase 7: Information-driven bars (depends on 5.1, 1.1, 3.3) ✅ DONE

### 7.1 — Imbalance bars (tick) ✅
- `ImbalanceTickBarConstructor`, `compute_imbalance_tick_bars()`.
- Uses tick-rule + EWMA imbalance threshold.
- Tests: batch, streaming, equivalence, first-tick exclusion, pure-direction stream.

### 7.2 — Imbalance bars (volume) ✅
- Same, with volume-weighted imbalance.

### 7.3 — Imbalance bars (dollar) ✅
- Same, with dollar-value-weighted imbalance.

### 7.4 — Run bars (tick) ✅
- `RunTickBarConstructor`, `compute_run_tick_bars()`.
- Uses tick-rule + EWMA run threshold.
- Tests: batch, streaming, equivalence, alternating direction (fragmentation),
  min_run_length.

### 7.5 — Run bars (volume) ✅
- Same, with volume-weighted runs.

### 7.6 — Run bars (dollar) ✅
- Same, with dollar-value-weighted runs.

---

## Phase 8: Public API + registry wiring (depends on 6.x, 7.x) ✅ DONE

### 8.1 — Bar registry finalization ✅
- All bar types registered. `BarRegistry.list()` returns all 10.
- Top-level imports in `flowbars/__init__.py` wired up.
- Tests: registry listing, duplicate registration raises, unknown type lookup raises.

### 8.2 — Convenience re-exports ✅
- `from flowbars import compute_dollar_bars` etc. — one import per bar type.
- Tests: imports work, functions produce bars.

---

## Phase 9: numba backend (depends on 8.x) ✅ DONE

### 9.1 — numba-accelerated accumulator loop ✅
- JIT-compiled version of the hot loop (tick ingestion + accumulator update).
- Same accumulator logic, different backend.
- Tests: equivalence with Python backend (identical output, all bar types, ≥100k
  tick dataset).

### 9.2 — Benchmark script ✅
- Realistic data size (≥100k ticks), warm-up handled (dry run before timing),
  compilation time reported separately.
- Results go in README.
- Tests: `python -m flowbars.benchmarks` runs without error and produces numbers.

### 9.3 — numba graceful fallback ✅
- If numba not installed or compilation fails → Python backend with warning.
- Tests: simulate missing numba, simulate compilation failure.

---

## Phase 10: Polars adapter (depends on 8.x) ✅ DONE

### 10.1 — `flowbars.adapters.polars` module ✅
- Mirror functions for every `compute_*_bars()`.
- numpy bridge: zero-copy where possible → core function → wrap back to polars.
- Batch-only (no streaming equivalent for polars).
- Tests: equivalence with pandas path (identical output, field-by-field, float
  tolerance).

---

## Phase 11: Bar quality report (depends on 8.x) ✅ DONE

### 11.1 — `bar_quality_report(bars_df) -> dict` ✅
- Return autocorrelation (Ljung-Box).
- Normality of returns (Jarque-Bera).
- Bar-count stability across days/sessions (coefficient of variation of bar counts).
- Run-bar fragmentation flag (fraction of run bars with ≤2 ticks).
- Tests: known synthetic data → expected statistics.

---

## Phase 12: Synthetic sample data (depends on 8.x) ✅ DONE

### 12.1 — Bundled sample ✅
- A few hundred rows of synthetic tick data.
- In `flowbars/sample_data/` or similar.
- Loaded via `flowbars.load_sample_data()`.
- Used by README quickstart and unit tests.
- Tests: quickstart example in README runs copy-paste without error.

---

## Phase 13: Documentation ✅ DONE

### 13.1 — README ✅
- What it does, honest comparison to mlfinlab/mlfinpy.
- Exact data sourcing/reproduction instructions (Binance, date range, file placement).
- Install, quickstart, stated assumptions (UTC, no auto schema detection, ordering
  stance, thread safety, float64).
- Benchmark numbers (from phase 9).

### 13.2 — Docstrings ✅
- Every public function/class has a docstring: signature, *why*, assumptions.
- Internal helpers: docstring where non-obvious.

---

## Phase 14: Comparison notebooks (depends on 12.x)

### 14.1 — flowbars vs. mlfinlab/mlfinpy (correctness + performance)
- Same input data, compare output bars.
- Show where they diverge and why.
- Benchmark: flowbars Python, flowbars numba, mlfinlab, mlfinpy.

### 14.2 — Tick-built vs. minute-approximated bars
- Same period, bars built from ticks vs. bars approximated from minute OHLCV.
- Quantify divergence (not just show both exist).
- Include run bar fragmentation analysis.

### 14.3 — Seeded vs. unseeded EWMA (warm-up effect)
- First N bars from unseeded EWMA vs. calibrated-seed EWMA.
- Quantify convergence rate, show `bar_quality_report` early-instability flag.

---

## Phase 15: QA-TESTER.md final pass

### 15.1 — Run the full QA-TESTER.md checklist
- Every item must be PASS. No FAIL, no UNCERTAIN.
- Human sign-off required (per STANDARDS.md §4) — not self-cleared.

### 15.2 — Final checks
- `pip install flowbars` from clean venv → quickstart works.
- `pip install flowbars[numba]` → numba path works.
- `pip install flowbars[polars]` → polars adapter works.
- Import has no side effects.
- All CI gates green.

---

## Dependency graph (simplified)

```
0.1 Schema ──┐
0.2 Core    ──┤
              ├── 1.1 Tick rule ────────────────────────────────┐
              ├── 2.1 Base accumulator ──┬── 2.2-2.7 Variants ─┤
              ├── 3.1 Threshold ABC     ──┬── 3.2-3.4 Impls ───┤
              └── 4.1 Calendar ABC      ──┬── 4.2-4.3 Impls ───┤
                                          │                      │
                                  5.1 BaseBarConstructor ────────┤
                                          │                      │
                              ┌───────────┴───────────┐          │
                              │                       │          │
                         6.x Standard bars      7.x Info bars ──┘
                              │                       │
                              └───────────┬───────────┘
                                          │
                                    8.x Public API
                                          │
                        ┌─────────────────┼─────────────────┐
                        │                 │                 │
                  9.x numba        10.x Polars       11.x Quality
                       │                 │
                       └─────────┬───────┘
                                 │
                          12.x Sample data
                                 │
                          13.x README/docs
                                 │
                          14.x Notebooks
                                 │
                          15.x QA pass
```

## Key rules

- **Base before variants.** Phase 6 (standard bars) before Phase 7 (information bars)
  because imbalance/run bars depend on tick-rule and EWMA threshold logic that
  standard bars don't need, but both share the base constructor from Phase 5.
- **Tests alongside code, not after.** Every phase includes its own tests. No "test
  everything at the end" phase.
- **One bar type at a time.** Each bar type (6.1-6.4, 7.1-7.6) is implemented and
  tested fully before the next. Don't batch them.
- **No mid-build scope additions.** Per STANDARDS.md §2. New ideas go in "future work"
  list in the spec, not into the current build.
- **Commit after each phase.** Clean, atomic commits.

## Time estimate (rough — for planning only)

| Phase | Est. effort |
|-------|-------------|
| 0-1: Foundation + Tick rule | Small |
| 2: Accumulators (7 variants) | Medium |
| 3: Threshold estimation | Medium |
| 4: Calendars | Small |
| 5: Base constructor + state | Large |
| 6: Standard bars (4 types) | Medium |
| 7: Info-driven bars (6 types) | Large |
| 8: Public API + registry | Small |
| 9: numba backend + benchmarks | Medium |
| 10: Polars adapter | Small |
| 11: Bar quality report | Small |
| 12: Sample data | Small |
| 13: Documentation | Medium |
| 14: Comparison notebooks | Large |
| 15: QA final pass | Medium |
