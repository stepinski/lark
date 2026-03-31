# ADR-002: Per-Site Mutex in Flock Worker Pool

**Status:** Accepted  
**Date:** 2025-10

## Context

At 10,000+ sensor sites, the Flock worker pool must process concurrent
observations without racing on per-site model state (SARIMAX history,
window buffer, fitted coefficients).

Two strategies were considered:

1. **One goroutine per site** — each site owns its goroutine and a dedicated
   channel. Simple, no locking. Cost: ~2KB stack × 10,000 = ~20MB baseline,
   plus scheduler overhead for 10k goroutines.

2. **Bounded worker pool + per-site mutex** — W workers (configurable ceiling)
   share work via a queue. Per-site mutex ensures only one worker touches a
   given site at a time. Cost: W goroutines total.

## Decision

Bounded worker pool with per-site mutex (`SiteState.mu sync.Mutex`).

The mutex is acquired by Flock before calling the user Handler and released
after. Handler implementations must not spawn goroutines that outlive the
Handler call.

Workers = configurable via `flock.Config.Workers`. Default suggested value
is `runtime.NumCPU()` for compute-bound workloads.

## Consequences

- Operationally controllable: worker count can be tuned per deployment
  without code changes.
- Graceful shutdown via `context.Context` cancellation + `Nest.Drain()`.
- Handler authors have a clear contract: single-threaded within a site,
  no need for internal locking.
- Thundering herd risk: if all 10k sites tick simultaneously, the Nest
  queue absorbs the burst and applies backpressure if it overflows.
- `SiteState.Custom interface{}` allows arbitrary model state without
  Flock knowing anything about forecasting models — clean separation.
