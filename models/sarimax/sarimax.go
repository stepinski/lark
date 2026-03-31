// Package sarimax implements a Seasonal AutoRegressive Integrated Moving
// Average model with eXogenous regressors (SARIMAX).
//
// Model order notation: (p,d,q)(P,D,Q,m)
//
//	p  = non-seasonal AR order
//	d  = non-seasonal differencing order
//	q  = non-seasonal MA order
//	P  = seasonal AR order
//	D  = seasonal differencing order
//	Q  = seasonal MA order
//	m  = seasonal period (e.g. 24 for hourly data with daily seasonality)
//
// Exogenous regressors (rain gauge readings, temperature, etc.) are supplied
// as a [][]float64 matrix aligned with the training series. Each regressor
// column is multiplied by a fitted coefficient β_k and added to the forecast.
//
// Fitting uses Conditional Least Squares — the first max(p, q, P*m, Q*m)
// observations are used as burn-in and excluded from the loss. This keeps the
// implementation dependency-free and every coefficient physically interpretable.
//
// Thread safety: a fitted *Model is safe for concurrent reads (Predict,
// PredictProba). Fit and Step mutate state and must not be called concurrently.
package sarimax

import (
	"errors"
	"fmt"
	"math"

	"github.com/stepinski/lark/internal/series"
)

// Order specifies the SARIMAX model order.
type Order struct {
	P, D, Q int // non-seasonal: AR, differencing, MA
	SP, SD, SQ, M int // seasonal: AR, differencing, MA, period
}

// String returns a compact human-readable label, e.g. "(1,1,1)(1,1,0,24)".
func (o Order) String() string {
	return fmt.Sprintf("(%d,%d,%d)(%d,%d,%d,%d)", o.P, o.D, o.Q, o.SP, o.SD, o.SQ, o.M)
}

// validate returns an error if the order is structurally invalid.
func (o Order) validate() error {
	if o.P < 0 || o.D < 0 || o.Q < 0 {
		return errors.New("sarimax: non-seasonal orders must be >= 0")
	}
	if o.SP < 0 || o.SD < 0 || o.SQ < 0 {
		return errors.New("sarimax: seasonal orders must be >= 0")
	}
	if (o.SP > 0 || o.SD > 0 || o.SQ > 0) && o.M < 2 {
		return errors.New("sarimax: seasonal period M must be >= 2 when seasonal orders > 0")
	}
	return nil
}

// burnIn returns the minimum number of observations consumed by the model's
// lag structure before the first residual can be computed.
func (o Order) burnIn() int {
	b := o.P
	if o.Q > b {
		b = o.Q
	}
	if s := o.SP * o.M; s > b {
		b = s
	}
	if s := o.SQ * o.M; s > b {
		b = s
	}
	return b
}

// Params holds all fitted coefficients. Fields are exported so callers can
// inspect, log, or manually override them — central to Lark's explainability
// contract.
type Params struct {
	// AR holds non-seasonal autoregressive coefficients φ_1 … φ_p.
	AR []float64
	// SAR holds seasonal AR coefficients Φ_1 … Φ_P (at lags m, 2m, …).
	SAR []float64
	// MA holds non-seasonal moving-average coefficients θ_1 … θ_q.
	MA []float64
	// SMA holds seasonal MA coefficients Θ_1 … Θ_Q.
	SMA []float64
	// Exog holds regression coefficients β_1 … β_k, one per exogenous column.
	Exog []float64
	// Intercept (constant) term μ.
	Intercept float64
}

// Model is the SARIMAX estimator. Zero value is unusable; construct via New.
type Model struct {
	order  Order
	params Params

	// history of the *differenced* series, kept for streaming Step() calls.
	diffHistory []float64
	// residual history for MA terms.
	residHistory []float64

	// seeds needed to invert differencing (first d raw observations).
	rawSeeds        []float64
	seasonalSeeds   []float64

	fitted bool
}

// New constructs an unfitted SARIMAX model with the given order.
func New(o Order) (*Model, error) {
	if err := o.validate(); err != nil {
		return nil, err
	}
	return &Model{order: o}, nil
}

// Params returns a copy of the fitted coefficients.
// Returns zero Params if the model has not been fitted.
func (m *Model) Params() Params {
	p := m.params
	p.AR = cloneSlice(m.params.AR)
	p.SAR = cloneSlice(m.params.SAR)
	p.MA = cloneSlice(m.params.MA)
	p.SMA = cloneSlice(m.params.SMA)
	p.Exog = cloneSlice(m.params.Exog)
	return p
}

// Fit estimates model parameters from y (endogenous) and exog (optional
// exogenous regressors, may be nil). Each column of exog must have the same
// length as y.
//
// Algorithm: Conditional Least Squares via a single-pass OLS on the feature
// matrix built from lagged values of the differenced series and lagged
// residuals.
func (m *Model) Fit(y []float64, exog [][]float64) error {
	o := m.order
	nExog := len(exog)

	// --- 1. validate exog alignment ---
	for i, col := range exog {
		if len(col) != len(y) {
			return fmt.Errorf("sarimax: exog column %d length %d != y length %d", i, len(col), len(y))
		}
	}

	// --- 2. seasonal differencing first, then regular differencing ---
	yd := series.SeasonalDiff(y, o.SD, o.M)
	m.seasonalSeeds = y[:o.SD*o.M]
	yd = series.Diff(yd, o.D)
	m.rawSeeds = y[:o.D] // seeds for inversion

	// diff exog columns in the same way
	exogD := make([][]float64, nExog)
	for i, col := range exog {
		tmp := series.SeasonalDiff(col, o.SD, o.M)
		exogD[i] = series.Diff(tmp, o.D)
	}

	n := len(yd)
	burn := o.burnIn()
	if n-burn < 2 {
		return fmt.Errorf("sarimax: insufficient observations (%d) for order %s (burn-in %d)", len(y), o, burn)
	}

	// --- 3. build feature matrix (rows = observations after burn-in) ---
	// Features per row:
	//   1 (intercept) + p (AR lags) + P (seasonal AR lags) +
	//   q (MA lags, using zero residuals initially) +
	//   Q (seasonal MA lags) + nExog
	nFeatures := 1 + o.P + o.SP + o.Q + o.SQ + nExog
	nObs := n - burn

	X := make([][]float64, nObs)
	for i := range X {
		X[i] = make([]float64, nFeatures)
	}
	yTrain := make([]float64, nObs)

	residuals := make([]float64, n) // zero-initialised — CLS assumption

	for t := burn; t < n; t++ {
		row := t - burn
		col := 0

		// intercept
		X[t-burn][col] = 1.0
		col++

		// non-seasonal AR: yd[t-1] … yd[t-p]
		for lag := 1; lag <= o.P; lag++ {
			X[t-burn][col] = yd[t-lag]
			col++
		}

		// seasonal AR: yd[t-m] … yd[t-P*m]
		for s := 1; s <= o.SP; s++ {
			X[t-burn][col] = yd[t-s*o.M]
			col++
		}

		// non-seasonal MA: residuals[t-1] … residuals[t-q]
		for lag := 1; lag <= o.Q; lag++ {
			if t-lag >= 0 {
				X[t-burn][col] = residuals[t-lag]
			}
			col++
		}

		// seasonal MA
		for s := 1; s <= o.SQ; s++ {
			if t-s*o.M >= 0 {
				X[t-burn][col] = residuals[t-s*o.M]
			}
			col++
		}

		// exogenous regressors (aligned to differenced length)
		for k := 0; k < nExog; k++ {
			if t < len(exogD[k]) {
				X[t-burn][col] = exogD[k][t]
			}
			col++
		}

		yTrain[row] = yd[t]

		// update residuals using current (zero) coefficients — will iterate
		_ = row
	}

	// --- 4. OLS: β = (X'X)^{-1} X'y ---
	coeffs, err := olsQR(X, yTrain, nFeatures)
	if err != nil {
		return fmt.Errorf("sarimax: OLS failed: %w", err)
	}

	// --- 5. unpack coefficients into Params ---
	col := 0
	m.params.Intercept = coeffs[col]; col++
	m.params.AR = make([]float64, o.P)
	for i := range m.params.AR { m.params.AR[i] = coeffs[col]; col++ }
	m.params.SAR = make([]float64, o.SP)
	for i := range m.params.SAR { m.params.SAR[i] = coeffs[col]; col++ }
	m.params.MA = make([]float64, o.Q)
	for i := range m.params.MA { m.params.MA[i] = coeffs[col]; col++ }
	m.params.SMA = make([]float64, o.SQ)
	for i := range m.params.SMA { m.params.SMA[i] = coeffs[col]; col++ }
	m.params.Exog = make([]float64, nExog)
	for i := range m.params.Exog { m.params.Exog[i] = coeffs[col]; col++ }

	// --- 6. store history for streaming ---
	m.diffHistory = make([]float64, len(yd))
	copy(m.diffHistory, yd)
	m.residHistory = residuals
	m.fitted = true

	return nil
}

// Predict returns h-step-ahead point forecasts on the differenced scale,
// then inverts differencing to return the original scale.
// futureExog must have len h if exogenous regressors were used in Fit.
func (m *Model) Predict(h int, futureExog [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, errors.New("sarimax: model not fitted")
	}
	o := m.order
	nExog := len(m.params.Exog)
	if nExog > 0 && len(futureExog) != nExog {
		return nil, fmt.Errorf("sarimax: expected %d exog columns, got %d", nExog, len(futureExog))
	}
	for i, col := range futureExog {
		if len(col) < h {
			return nil, fmt.Errorf("sarimax: futureExog[%d] has %d rows, need %d", i, len(col), h)
		}
	}

	// extend differenced history with forecasts
	hist := make([]float64, len(m.diffHistory)+h)
	copy(hist, m.diffHistory)
	resid := make([]float64, len(m.residHistory)+h) // future residuals = 0
	copy(resid, m.residHistory)

	n0 := len(m.diffHistory)
	forecasts := make([]float64, h)

	for step := 0; step < h; step++ {
		t := n0 + step
		yhat := m.params.Intercept

		for lag, phi := range m.params.AR {
			if t-lag-1 >= 0 {
				yhat += phi * hist[t-lag-1]
			}
		}
		for s, phi := range m.params.SAR {
			lag := (s + 1) * o.M
			if t-lag >= 0 {
				yhat += phi * hist[t-lag]
			}
		}
		for lag, theta := range m.params.MA {
			if t-lag-1 >= 0 {
				yhat += theta * resid[t-lag-1]
			}
		}
		for s, theta := range m.params.SMA {
			lag := (s + 1) * o.M
			if t-lag >= 0 {
				yhat += theta * resid[t-lag]
			}
		}
		for k, beta := range m.params.Exog {
			yhat += beta * futureExog[k][step]
		}

		hist[t] = yhat
		forecasts[step] = yhat
	}

	// invert regular differencing (skip if d=0)
	result := forecasts
	if o.D > 0 && len(m.rawSeeds) > 0 {
		result = series.InvertDiff(forecasts, m.rawSeeds)
		// trim seeds prefix
		if len(result) > h {
			result = result[len(result)-h:]
		}
	}

	return result, nil
}

// PredictProba returns, for each forecast horizon step, the probability that
// the predicted value exceeds threshold. It uses a Gaussian approximation
// over the in-sample residuals.
//
// This is Lark's primary alarm signal: P(flow > alarm_level | rain forecast).
func (m *Model) PredictProba(h int, futureExog [][]float64, threshold float64) ([]float64, error) {
	forecasts, err := m.Predict(h, futureExog)
	if err != nil {
		return nil, err
	}

	// estimate residual standard deviation from training residuals
	sigma := math.Sqrt(series.Variance(m.residHistory))
	if sigma == 0 {
		sigma = 1e-6
	}

	probs := make([]float64, h)
	for i, yhat := range forecasts {
		// P(Y > threshold) = P(Z > (threshold - yhat) / sigma)
		z := (threshold - yhat) / sigma
		probs[i] = 1.0 - normalCDF(z)
	}
	return probs, nil
}

// Step ingests one new observation and optional exogenous values, updates
// internal state, and returns the one-step-ahead forecast.
// This is the streaming API — call it at each sensor tick.
func (m *Model) Step(observed float64, exogValues []float64) (float64, error) {
	if !m.fitted {
		return 0, errors.New("sarimax: model not fitted")
	}
	o := m.order

	// difference the new observation: apply same transforms as Fit
	// simplified: track the rolling window for differencing
	raw := append(m.rawSeeds, observed)
	diffed := series.Diff(raw, o.D)
	newVal := diffed[len(diffed)-1]

	m.diffHistory = append(m.diffHistory, newVal)

	// compute one-step forecast using updated history
	t := len(m.diffHistory) - 1
	yhat := m.params.Intercept

	for lag, phi := range m.params.AR {
		if t-lag-1 >= 0 {
			yhat += phi * m.diffHistory[t-lag-1]
		}
	}
	for s, phi := range m.params.SAR {
		lag := (s + 1) * o.M
		if t-lag >= 0 {
			yhat += phi * m.diffHistory[t-lag]
		}
	}
	for lag, theta := range m.params.MA {
		if t-lag-1 >= 0 && t-lag-1 < len(m.residHistory) {
			yhat += theta * m.residHistory[t-lag-1]
		}
	}
	for k, beta := range m.params.Exog {
		if k < len(exogValues) {
			yhat += beta * exogValues[k]
		}
	}

	// update residuals
	residual := newVal - yhat
	m.residHistory = append(m.residHistory, residual)

	// slide raw seeds window for next Step call
	if o.D > 0 {
		m.rawSeeds = append(m.rawSeeds[1:], observed)
	}

	return yhat, nil
}

// --- OLS via normal equations (X'X β = X'y) with Cholesky ---
// For small feature counts (< 50) this is fast and allocation-minimal.
func olsQR(X [][]float64, y []float64, nFeatures int) ([]float64, error) {
	n := len(X)
	// XtX = X' * X  (nFeatures × nFeatures)
	XtX := make([]float64, nFeatures*nFeatures)
	Xty := make([]float64, nFeatures)

	for i := 0; i < n; i++ {
		for j := 0; j < nFeatures; j++ {
			Xty[j] += X[i][j] * y[i]
			for k := 0; k <= j; k++ {
				XtX[j*nFeatures+k] += X[i][j] * X[i][k]
			}
		}
	}
	// fill upper triangle (symmetric)
	for j := 0; j < nFeatures; j++ {
		for k := j + 1; k < nFeatures; k++ {
			XtX[j*nFeatures+k] = XtX[k*nFeatures+j]
		}
	}

	// Cholesky decomposition in-place
	L := make([]float64, nFeatures*nFeatures)
	for i := 0; i < nFeatures; i++ {
		for j := 0; j <= i; j++ {
			sum := XtX[i*nFeatures+j]
			for k := 0; k < j; k++ {
				sum -= L[i*nFeatures+k] * L[j*nFeatures+k]
			}
			if i == j {
				if sum <= 0 {
					return nil, fmt.Errorf("matrix not positive definite at diagonal %d (sum=%.6g)", i, sum)
				}
				L[i*nFeatures+j] = math.Sqrt(sum)
			} else {
				L[i*nFeatures+j] = sum / L[j*nFeatures+j]
			}
		}
	}

	// forward substitution: L z = X'y
	z := make([]float64, nFeatures)
	for i := 0; i < nFeatures; i++ {
		sum := Xty[i]
		for k := 0; k < i; k++ {
			sum -= L[i*nFeatures+k] * z[k]
		}
		z[i] = sum / L[i*nFeatures+i]
	}

	// backward substitution: L' β = z
	beta := make([]float64, nFeatures)
	for i := nFeatures - 1; i >= 0; i-- {
		sum := z[i]
		for k := i + 1; k < nFeatures; k++ {
			sum -= L[k*nFeatures+i] * beta[k]
		}
		beta[i] = sum / L[i*nFeatures+i]
	}

	return beta, nil
}

// normalCDF returns the CDF of the standard normal at x (Abramowitz & Stegun
// approximation, max error 7.5e-8).
func normalCDF(x float64) float64 {
	if x > 8 {
		return 1
	}
	if x < -8 {
		return 0
	}
	t := 1.0 / (1.0 + 0.2316419*math.Abs(x))
	poly := t * (0.319381530 + t*(-0.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))))
	p := 1.0 - (1.0/math.Sqrt(2*math.Pi))*math.Exp(-0.5*x*x)*poly
	if x < 0 {
		return 1 - p
	}
	return p
}

func cloneSlice(s []float64) []float64 {
	if s == nil {
		return nil
	}
	out := make([]float64, len(s))
	copy(out, s)
	return out
}
