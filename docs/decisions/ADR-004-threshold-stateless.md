# ADR-004: Threshold Package Is Pure and Stateless

**Status:** Accepted  
**Date:** 2025-10

## Context

The overflow state classifier (Green/Yellow/Orange/Red) could be designed as:
1. A stateful struct that tracks its own 48h history internally.
2. A pure function that takes current depth + a `recentOverflow bool` flag
   computed by the caller from the `window` package.

## Decision

Option 2: pure, stateless functions. `threshold.Classify(depth, cfg, recentOverflow)`.

The caller (Flock Handler) is responsible for:
- Maintaining the `window.Window` for the site.
- Computing `recentOverflow` via `window.Since(now - 48h)` and checking
  whether any observation in that slice was >= `BottomOverflowInvert`.

## Consequences

- `threshold` package has no imports from `window` or `runtime` — zero
  coupling, easy to test in isolation with plain floats.
- The `DetectEvents` batch function also stays stateless: takes `[]int64`
  timestamps and `[]float64` depths, returns `[]OverflowEvent`.
- The 48h recentOverflow flag computation is explicit in the Handler — it's
  visible in the code where it matters, not hidden inside the classifier.
- STS site logic (`ClassifySTS`) follows the same pattern, returning both
  the alarm type and a visibility bool — the two outputs the Peel map UI needs.
