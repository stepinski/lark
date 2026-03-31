# ADR-001: Zero External Runtime Dependencies

**Status:** Accepted  
**Date:** 2025-10

## Context

Lark targets edge devices and single-server deployments serving 10,000+
virtual sensors. Every external dependency is a supply chain risk, a
compilation cost, and a potential source of CGO complications on ARM/edge
hardware.

The primary alternative was using gonum for linear algebra (OLS, matrix
operations) and a statistics library for distributions.

## Decision

Lark has zero external runtime dependencies. All math — matrix operations,
OLS via Cholesky decomposition, normal CDF approximation, differencing,
autocorrelation — is implemented in `internal/series` or inline in model
packages.

## Consequences

- `go.mod` contains only the stdlib. Auditable in seconds.
- Slightly more implementation work per model (Cholesky OLS vs gonum.mat).
- No CGO, no platform-specific build tags — compiles cleanly for Linux/ARM.
- All algorithms are directly readable in the codebase — supports the
  "no black boxes" explainability contract.
- Test-only dependencies (e.g. testify) are acceptable if genuinely needed,
  but have not been required so far.
