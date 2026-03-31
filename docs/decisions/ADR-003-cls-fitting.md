# ADR-003: Conditional Least Squares for SARIMAX Fitting

**Status:** Accepted  
**Date:** 2025-10

## Context

SARIMAX can be fitted via several methods:
- **MLE (Maximum Likelihood Estimation)** — statistically optimal, but
  requires numerical optimisation (gradient descent / L-BFGS), which needs
  either autodiff or hand-coded gradients.
- **Conditional Least Squares (CLS)** — treats the first `burnIn()`
  observations as fixed and minimises sum of squared residuals. Reduces
  to a linear regression solvable by OLS.
- **Hannan-Rissanen** — two-stage: AR fit then MA correction. Simple but
  less accurate for combined ARMA models.

## Decision

Conditional Least Squares with OLS solved via Cholesky decomposition on
the normal equations (X'X β = X'y).

Rationale:
1. **No autodiff needed** — keeps the zero-external-dependency invariant.
2. **Fully deterministic** — same data always produces same coefficients.
3. **Explainable** — each coefficient has a direct linear interpretation.
4. **Fast for small feature counts** — Cholesky on a 10×10 matrix is
   microseconds; SARIMAX orders in practice are rarely > (3,1,3)(1,1,1,24).
5. **Sufficient accuracy** — CLS vs MLE difference is negligible for the
   alarm prediction use case; we care about P(overflow) ordering, not
   exact likelihood.

## Consequences

- MA coefficient estimates are slightly biased vs MLE for short series.
  Acceptable given typical sensor history length (months to years).
- Initial residuals are set to zero (the "conditional" part). This
  introduces a burn-in period of `max(p, q, P*m, Q*m)` observations.
- Auto-calibration (nudging physical parameters to minimise RMSE) is a
  future layer on top of CLS — planned for `models/calibrator`.
