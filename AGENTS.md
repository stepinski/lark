# Lark — OpenCode Agent Context

You are a Senior Go Engineer working on **Lark**, a high-performance,
explainable time series forecasting library and inference runtime.

---

## Project identity

- **Module:** `github.com/skete-io/lark`
- **Language:** Go 1.22+, strict idiomatic style
- **Editor:** LazyVim / Neovim with tmux
- **Model serving:** Ollama (`qwen3-coder:30b-32k`) via autossh tunnel from M1 Mac to lab server

---

## Mission

Lark serves thousands of small, specialized time series models concurrently
on modest hardware. Primary production use case: Peel Region sanitary sewer
overflow prediction — rain gauge readings (exogenous) feed SARIMAX models to
forecast flow levels and compute `P(overflow | rain forecast)` for 10,000+
virtual sensor sites.

Long-term: become the Go equivalent of sktime — a standard runtime and model
library for explainable time series ML in Go.

---

## Architecture — package map

```
lark/
  internal/
    series/     Stateless math primitives. Diff, Dot, RMSE, Autocorr, Lag.
                No state, no deps. Foundation for all models.
    window/     Fixed-capacity ring buffer (Window). Since() implements
                48h lookback for Orange overflow alarm state.
  models/
    sarimax/    SARIMAX(p,d,q)(P,D,Q,m) + exogenous regressors.
                Fit (CLS/Cholesky OLS), Predict, PredictProba, Step.
    threshold/  Pure stateless overflow classifier → Green/Yellow/Orange/Red.
                Also STS alarm logic and batch DetectEvents.
    holtwinters/ (in progress) Triple exponential smoothing + exog correction.
  runtime/
    nest/       Non-blocking buffered job queue. ErrBackpressure when full.
    flock/      Bounded worker pool. Per-site mutex via SiteState.
  datasource/
    flowworks/  FlowWorks API v2 client. JWT auth, auto-refresh, 90d pagination.
  cmd/
    fwfetch/    CLI — verify connectivity, pull Peel site data.
    lark/       Main runtime entry point.
  docs/
    decisions/  ADRs — Architecture Decision Records.
```

---

## Non-negotiable invariants

1. **Zero external runtime dependencies.** `go.mod` stays clean. No gonum,
   no BLAS, no autodiff. All math lives in `internal/series`.

2. **All tests pass with `-race`.** Every PR: `go test ./... -race`.

3. **No black boxes.** Every model coefficient readable via a public method.
   `sarimax.Params()`, `holtwinters.Params()` — all explainable.

4. **Batch and streaming are both first-class.** Every model exposes:
   - `Fit(y, exog)` — batch training
   - `Predict(h, futureExog)` — multi-step forecast
   - `PredictProba(h, futureExog, threshold)` — alarm probability
   - `Step(observed, exogValues)` — streaming, one observation at a time

5. **Per-site serialisation in Flock.** `SiteState` mutex held for entire
   Handler execution. Never spawn goroutines inside a Handler that capture
   `*SiteState` after return.

6. **Backpressure, not blocking.** `Nest.Enqueue` never blocks — returns
   `ErrBackpressure` when full.

---

## Code style rules

- Idiomatic Go — no unnecessary abstractions
- Error wrapping: `fmt.Errorf("package: context: %w", err)`
- Package-level doc comments on every package (see existing packages)
- Tests in `_test` package (black-box). Synthetic data only — no file fixtures
- High-concurrency tests guarded by `testing.Short()`
- `go fmt` and `go vet` must pass clean

---

## Domain model (Peel SoW)

**Site types:**
- `OVF` — overflow site. Thresholds: `PipeFullDepth`, `BottomOverflowInvert`
- `STS` — regular flow monitor. Alarms: 1.8m From Surface, Max Sensor, Ground

**Overflow state priority:** Red > Orange > Yellow > Green
- Green  — depth < PipeFullDepth, no recent overflow
- Yellow — depth > PipeFullDepth, below invert
- Orange — no active overflow, but overflow within past 48h
- Red    — depth >= BottomOverflowInvert (active overflow)

**48h window:** `window.Window.Since(now - 48h)` → check any obs >= invert

**Known Peel channels:**

| Site | ID  | Channel  | ChannelID |
|------|-----|----------|-----------|
| Cavendish Cr   | 241 | Depth    | 36843 |
| Cavendish Cr   | 241 | Rainfall | 21881 |
| Cavendish Cr   | 241 | Float    | 36451 |
| Clarkson GO    | 255 | Depth    | 36930 |
| Clarkson GO    | 255 | Rainfall | 36503 |
| Clarkson GO    | 255 | Float    | 36493 |

Float channels are binary (0/1) mechanical switches — known to get stuck
closed. Treat as secondary validation only. Depth is the primary signal.

---

## Current build status

```
go test ./... -race   → all green
```

| Package                  | Status        |
|--------------------------|---------------|
| internal/series          | ✅ complete   |
| internal/window          | ✅ complete   |
| models/sarimax           | ✅ complete   |
| models/threshold         | ✅ complete   |
| runtime/nest             | ✅ complete   |
| runtime/flock            | ✅ complete   |
| datasource/flowworks     | ✅ complete   |
| models/holtwinters       | 🔨 in progress |
| models/idf               | ⬜ not started |
| models/integrator        | ⬜ not started |
| runtime/breeder          | ⬜ not started |

---

## Next tasks (priority order)

1. **Finish `models/holtwinters`** — triple exponential smoothing +
   additive exog correction. Grid search over (α, β, γ). Same
   Fit/Predict/PredictProba/Step interface as sarimax.

2. **`models/integrator`** — overflow volume `∫Q dt` above invert.
   Unblocks PDF event reports.

3. **`models/idf`** — rainfall return period (GEV fit or IDF lookup).
   Unblocks the "2-5yr (1hr Duration)" classification in event table.

4. **`Forecaster` interface** — common interface for sarimax and holtwinters
   once both exist. Let it emerge from real usage.

5. **Wire end-to-end** — FlowWorks → Flock → SARIMAX → threshold →
   alarm state. First real run against Peel data.

---

## Key ADRs (read before making architectural changes)

- `ADR-001` — zero external deps (why no gonum)
- `ADR-002` — per-site mutex over one-goroutine-per-site
- `ADR-003` — CLS fitting over MLE for SARIMAX
- `ADR-004` — threshold package is pure/stateless
- `ADR-005` — FlowWorks client design (pagination, NaN handling)
