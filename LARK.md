# Lark — AI Session Context

> Paste this file at the start of any new session to restore full project context.
> Keep it updated as the project evolves. Last updated: 2025-10.

---

## Mission

Lark is a high-performance, explainable time series forecasting library and
inference runtime written in Go. It is designed to serve thousands of small,
specialized models (time series, physics-based, statistical) with high
concurrency on modest hardware — including edge devices.

**Primary production use case (v0.1):** Peel Region Overflow Alert System.
Lark powers the prediction layer of a real-time sanitary sewer overflow
detection system: rain gauge readings (exogenous) are fed into SARIMAX models
to forecast flow levels and compute P(overflow | rain forecast) for 10,000+
virtual sensor sites.

**Long-term ambition:** Become the Go equivalent of sktime — a standard runtime
and model library for time series ML in Go. Every model must be explainable:
no black boxes, all parameters physically grounded and inspectable.

---

## Project name and metaphor system

The Lark metaphor maps cleanly to the architecture:
- **Nest** — the job queue (buffered channel, backpressure)
- **Flock** — the worker pool (bounded goroutines)
- **Breeder** — polyglot model retraining (not yet built; Python/Julia IPC)

The name is short, Go-ecosystem appropriate, and domain-agnostic.
GitHub: `github.com/skete-io/lark`

---

## Package map

```
lark/
  internal/
    series/     Stateless math primitives: Diff, Dot, RMSE, Autocorr, Lag, etc.
                No state, no deps. Every model builds on this.
    window/     Fixed-capacity ring buffer for streaming observations.
                Window.Since() implements the 48h lookback for Orange alarm state.
  models/
    sarimax/    SARIMAX(p,d,q)(P,D,Q,m) with exogenous regressors.
                Fit (CLS/Cholesky OLS), Predict (multi-step), PredictProba
                (Gaussian alarm probability), Step (streaming, one obs at a time).
    threshold/  Pure stateless overflow state classifier (Green/Yellow/Orange/Red).
                Also handles STS site alarm logic and batch event detection.
                Directly encodes Peel Region SoW threshold logic.
  runtime/
    nest/       Non-blocking buffered job queue. Returns ErrBackpressure when
                full instead of blocking — 503-style load shedding.
    flock/      Bounded worker pool. Per-site mutex via SiteState ensures no two
                workers ever race on the same sensor. Handler func is user-supplied.
  docs/
    decisions/  Architecture Decision Records (ADRs). One per significant decision.
```

---

## Invariants — must not be broken

1. **Zero external dependencies.** `go.mod` must stay clean. All math is
   implemented from scratch in `internal/series`. No matrix libraries,
   no autodiff, no BLAS bindings.

2. **All tests pass with `-race`.** The race detector is the primary
   correctness guarantee for the concurrent runtime. Every PR must pass
   `go test ./... -race`.

3. **No black boxes.** Every model coefficient must be readable via a
   public method (e.g. `sarimax.Params()`). Operators must be able to
   explain why the model flagged an overflow.

4. **Streaming and batch are first-class.** Every model must expose both
   `Fit/Predict` (batch) and `Step` (streaming, one observation at a time).
   The streaming path is what Flock calls.

5. **Per-site serialisation in Flock.** The `SiteState` mutex must be held
   for the entire duration of `Handler` execution. Never spawn goroutines
   inside a Handler that capture `*SiteState`.

6. **Backpressure, not blocking.** `Nest.Enqueue` must never block. If the
   queue is full it returns `ErrBackpressure`. Callers decide how to handle it.

---

## Domain model (Peel SoW)

Two site types:
- **OVF** — overflow site. Has a `BottomOverflowInvert` depth threshold.
  Classified as Green / Yellow / Orange / Red.
- **STS** — regular flow monitor. Only visible on map during active alarm
  or within 48h of one. Alarm types: 1.8m From Surface, Max Sensor, Ground Alarm.

Overflow state priority: **Red > Orange > Yellow > Green**

The 48h Orange window is computed via `window.Window.Since(now - 48h)` —
check if any observation in that window had depth >= BottomOverflowInvert.

Key data per event (for PDF reports): Start, End, Duration, Volume (∫Q dt),
Rainfall (mm), Rainfall Return Period (IDF classification).

---

## What is built (v0.1 progress)

| Package | Status | Notes |
|---|---|---|
| `internal/series` | ✅ done | All primitives, fully tested |
| `internal/window` | ✅ done | Ring buffer, Since(), Max(), Sum() |
| `models/sarimax` | ✅ done | Fit, Predict, PredictProba, Step |
| `models/threshold` | ✅ done | OVF + STS classification, DetectEvents |
| `runtime/nest` | ✅ done | Queue, backpressure, Drain |
| `runtime/flock` | ✅ done | Worker pool, Registry, per-site mutex |
| `models/holtwinters` | ⬜ not started | Next after threshold |
| `models/idf` | ⬜ not started | Rainfall return period (GEV fit) |
| `models/integrator` | ⬜ not started | Overflow volume ∫Q dt |
| `runtime/breeder` | ⬜ not started | Python/Julia IPC for ONNX retraining |

---

## Open questions / design decisions pending

- **Holt-Winters exogenous term:** HW doesn't natively support regressors.
  Decision needed: additive correction term post-forecast, or a proper
  transfer-function wrapper? Leaning toward additive correction for
  explainability.

- **IDF model:** GEV (Generalized Extreme Value) fit vs empirical lookup table.
  Peel will supply historical IDF data — may just be a lookup with interpolation
  rather than a full GEV fit. Confirm with client before implementing.

- **Forecaster interface:** Should `sarimax.Model` and `holtwinters.Model`
  implement a common `Forecaster` interface? Deferred until both exist so the
  interface emerges from real usage rather than speculation.

- **ONNX / Breeder:** Out of scope for v0.1. Will revisit once Go-native models
  are validated against SWMM benchmarks.

- **CLI runner:** A `cmd/lark` binary for running simulations from `.inp` files
  is planned but not started. Will wrap Flock + a config loader.

---

## Testing conventions

- TDD: tests written alongside (or before) implementation.
- Every test file is in `_test` package (black-box testing).
- Synthetic data used for model tests — no file fixtures.
- High-concurrency stress tests guarded by `testing.Short()`.
- Run: `go test ./... -race`

---

## Tech stack

- Go 1.22+
- LazyVim / Neovim as primary IDE
- GitHub Actions CI (planned): `go test ./... -race` + `go vet ./...`
- No external runtime dependencies (test-only deps acceptable if needed)
