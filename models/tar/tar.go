// Package tar implements a Threshold AutoRegressive model with eXogenous
// regressors (TAR-X).
//
// The model partitions observations into two regimes based on whether a
// delayed value of the endogenous series y(t-d) is above or below a
// threshold r:
//
//	LOW  regime (y(t-d) <= r):  y(t) = φ₁₀ + Σ φ₁ᵢ·y(t-i) + Σ β₁ₖ·x(t-k) + ε₁
//	HIGH regime (y(t-d) >  r):  y(t) = φ₂₀ + Σ φ₂ᵢ·y(t-i) + Σ β₂ₖ·x(t-k) + ε₂
//
// Each regime is fitted independently by Conditional Least Squares (OLS via
// Cholesky decomposition). The threshold r and delay d are selected by grid
// search over candidate values minimising total in-sample SSE.
//
// This model is particularly suited to hydraulic depth series where the
// system exhibits a nonlinear tipping point: normal rain produces small
// linear depth responses (LOW regime), but above some hydraulic threshold
// the system surges (HIGH regime). The regime coefficients make this
// physically interpretable to operators.
//
// Thread safety: a fitted *Model is safe for concurrent reads (Predict,
// PredictProba, RegimeAt). Fit and Step mutate state.
package tar

import (
	"errors"
	"fmt"
	"math"
)

// RegimeID identifies which regime an observation belongs to.
type RegimeID int

const (
	LowRegime  RegimeID = 0
	HighRegime RegimeID = 1
)

func (r RegimeID) String() string {
	if r == LowRegime {
		return "LOW"
	}
	return "HIGH"
}

// RegimeParams holds the fitted coefficients for one regime.
// All fields are exported for full explainability.
type RegimeParams struct {
	// Intercept is the constant term φ₀.
	Intercept float64
	// AR holds autoregressive coefficients φ₁..φ_p.
	AR []float64
	// Exog holds exogenous regressor coefficients β₁..β_k.
	Exog []float64
	// N is the number of training observations assigned to this regime.
	N int
	// SSE is the in-sample sum of squared errors for this regime.
	SSE float64
	// Sigma is the residual standard deviation (used for PredictProba).
	Sigma float64
}

// Params holds all fitted model parameters.
type Params struct {
	// Threshold is the fitted regime boundary r.
	Threshold float64
	// Delay is the fitted threshold lag d (y(t-d) determines regime).
	Delay int
	// Low holds coefficients for the LOW regime (y(t-d) <= Threshold).
	Low RegimeParams
	// High holds coefficients for the HIGH regime (y(t-d) > Threshold).
	High RegimeParams
}

// Summary returns a human-readable model description.
func (p Params) Summary(label string) string {
	return fmt.Sprintf(
		"%s TAR(p=%d, d=%d, r=%.1f)\n"+
			"  LOW  (y(t-%d) <= %.1f): n=%d  intercept=%.4f  AR=%v  exog=%v  σ=%.3f\n"+
			"  HIGH (y(t-%d) >  %.1f): n=%d  intercept=%.4f  AR=%v  exog=%v  σ=%.3f",
		label,
		len(p.Low.AR), p.Delay, p.Threshold,
		p.Delay, p.Threshold, p.Low.N, p.Low.Intercept, fmtSlice(p.Low.AR), fmtSlice(p.Low.Exog), p.Low.Sigma,
		p.Delay, p.Threshold, p.High.N, p.High.Intercept, fmtSlice(p.High.AR), fmtSlice(p.High.Exog), p.High.Sigma,
	)
}

// Config controls model construction and grid search.
type Config struct {
	// P is the autoregressive order (number of lags of y).
	P int
	// ExogLags specifies which lags of each exogenous column to include.
	// ExogLags[k] is the list of lags for exog column k.
	// Example: [][]int{{5}} uses rain at lag 5 (25min at 5-min resolution).
	ExogLags [][]int
	// ThresholdCandidates are the values of r to try in the grid search.
	// If nil, candidates are auto-generated from the 10th–90th percentile
	// of y in steps of 5 percentile points.
	ThresholdCandidates []float64
	// DelayCandidates are the values of d to try (default: [1, 2, 3]).
	DelayCandidates []int
	// MinRegimeSize is the minimum number of observations required in each
	// regime for a valid split. Default: max(P+1, 10).
	MinRegimeSize int
}

func (c *Config) delayCandidates() []int {
	if len(c.DelayCandidates) > 0 {
		return c.DelayCandidates
	}
	return []int{1, 2, 3}
}

func (c *Config) minRegime(p int) int {
	if c.MinRegimeSize > 0 {
		return c.MinRegimeSize
	}
	if p+1 > 10 {
		return p + 1
	}
	return 10
}

// Model is the TAR-X estimator.
type Model struct {
	cfg    Config
	params Params

	// history of raw observations for streaming Step().
	history []float64
	// exog history for streaming.
	exogHistory [][]float64

	fitted bool
}

// New constructs an unfitted TAR model.
func New(cfg Config) (*Model, error) {
	if cfg.P < 1 {
		return nil, fmt.Errorf("tar: P must be >= 1, got %d", cfg.P)
	}
	return &Model{cfg: cfg}, nil
}

// Params returns a copy of the fitted parameters.
func (m *Model) Params() Params { return m.params }

// Fit estimates the threshold, delay, and per-regime coefficients.
// y is the endogenous series; exog is optional (nil or len == len(ExogLags)).
func (m *Model) Fit(y []float64, exog [][]float64) error {
	cfg := m.cfg
	nExogCols := len(exog)
	if nExogCols != len(cfg.ExogLags) {
		return fmt.Errorf("tar: %d exog columns but %d ExogLags entries",
			nExogCols, len(cfg.ExogLags))
	}
	for i, col := range exog {
		if len(col) != len(y) {
			return fmt.Errorf("tar: exog[%d] length %d != y length %d",
				i, len(col), len(y))
		}
	}

	n := len(y)
	minSize := cfg.minRegime(cfg.P)
	if n < 2*minSize+cfg.P {
		return fmt.Errorf("tar: insufficient observations (%d)", n)
	}

	// generate threshold candidates if not provided
	candidates := cfg.ThresholdCandidates
	if len(candidates) == 0 {
		candidates = percentileCandidates(y, 10, 90, 20)
	}

	// grid search over (threshold, delay)
	bestSSE := math.Inf(1)
	bestR := candidates[0]
	bestD := 1

	for _, d := range cfg.delayCandidates() {
		for _, r := range candidates {
			lowIdx, highIdx := splitByThreshold(y, d, r, cfg.P)
			if len(lowIdx) < minSize || len(highIdx) < minSize {
				continue
			}

			lowSSE, err1 := regimeSSE(y, exog, cfg.ExogLags, lowIdx, cfg.P)
			highSSE, err2 := regimeSSE(y, exog, cfg.ExogLags, highIdx, cfg.P)
			if err1 != nil || err2 != nil {
				continue
			}

			totalSSE := lowSSE + highSSE
			if totalSSE < bestSSE {
				bestSSE = totalSSE
				bestR = r
				bestD = d
			}
		}
	}

	// fit final model with best (r, d)
	lowIdx, highIdx := splitByThreshold(y, bestD, bestR, cfg.P)

	lowP, err := fitRegime(y, exog, cfg.ExogLags, lowIdx, cfg.P)
	if err != nil {
		return fmt.Errorf("tar: fit LOW regime: %w", err)
	}
	highP, err := fitRegime(y, exog, cfg.ExogLags, highIdx, cfg.P)
	if err != nil {
		return fmt.Errorf("tar: fit HIGH regime: %w", err)
	}

	m.params = Params{
		Threshold: bestR,
		Delay:     bestD,
		Low:       lowP,
		High:      highP,
	}

	m.history = make([]float64, len(y))
	copy(m.history, y)
	m.exogHistory = make([][]float64, nExogCols)
	for k := range exog {
		m.exogHistory[k] = make([]float64, len(y))
		copy(m.exogHistory[k], exog[k])
	}
	m.fitted = true
	return nil
}

// RegimeAt returns which regime applies at time t given y(t-d).
// Returns LowRegime if the delayed value is <= Threshold, HighRegime otherwise.
func (m *Model) RegimeAt(t int) RegimeID {
	d := m.params.Delay
	if t-d < 0 || t-d >= len(m.history) {
		return LowRegime
	}
	if m.history[t-d] <= m.params.Threshold {
		return LowRegime
	}
	return HighRegime
}

// Predict returns h one-step-ahead forecasts using the appropriate regime
// at each step. futureExog must be provided if exog was used in Fit.
func (m *Model) Predict(h int, futureExog [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, errors.New("tar: model not fitted")
	}
	if err := m.validateFutureExog(h, futureExog); err != nil {
		return nil, err
	}

	hist := make([]float64, len(m.history)+h)
	copy(hist, m.history)

	exogHist := make([][]float64, len(m.exogHistory))
	for k := range m.exogHistory {
		exogHist[k] = make([]float64, len(m.exogHistory[k])+h)
		copy(exogHist[k], m.exogHistory[k])
		for step := 0; step < h; step++ {
			if k < len(futureExog) && step < len(futureExog[k]) {
				exogHist[k][len(m.exogHistory[k])+step] = futureExog[k][step]
			}
		}
	}

	forecasts := make([]float64, h)
	n0 := len(m.history)

	for step := 0; step < h; step++ {
		t := n0 + step
		regime := m.regimeAtHist(hist, t)
		yhat := m.predictOne(hist, exogHist, t, regime)
		hist[t] = yhat
		forecasts[step] = yhat
	}

	return forecasts, nil
}

// PredictProba returns P(y(t+h) > threshold) for each step using a Gaussian
// approximation over the regime's residuals.
func (m *Model) PredictProba(h int, futureExog [][]float64, threshold float64) ([]float64, error) {
	forecasts, err := m.Predict(h, futureExog)
	if err != nil {
		return nil, err
	}

	probs := make([]float64, h)
	hist := m.history

	for step, yhat := range forecasts {
		// determine which regime applies at this step
		t := len(hist) + step
		regime := m.regimeAtHist(hist, t)
		sigma := m.params.Low.Sigma
		if regime == HighRegime {
			sigma = m.params.High.Sigma
		}
		if sigma == 0 {
			sigma = 1e-6
		}
		z := (threshold - yhat) / sigma
		probs[step] = 1.0 - normalCDF(z)
	}
	return probs, nil
}

// Step ingests one new observation, updates state, returns one-step forecast.
// This is the streaming API for Flock integration.
func (m *Model) Step(observed float64, exogValues []float64) (float64, error) {
	if !m.fitted {
		return 0, errors.New("tar: model not fitted")
	}

	m.history = append(m.history, observed)
	t := len(m.history) - 1

	for k := range m.exogHistory {
		val := 0.0
		if k < len(exogValues) {
			val = exogValues[k]
		}
		m.exogHistory[k] = append(m.exogHistory[k], val)
	}

	// forecast next step
	regime := m.regimeAtHist(m.history, t+1)
	yhat := m.predictOne(m.history, m.exogHistory, t+1, regime)

	return yhat, nil
}

// --- internal helpers ---

func (m *Model) regimeAtHist(hist []float64, t int) RegimeID {
	d := m.params.Delay
	idx := t - d
	if idx < 0 || idx >= len(hist) {
		return LowRegime
	}
	if hist[idx] <= m.params.Threshold {
		return LowRegime
	}
	return HighRegime
}

func (m *Model) predictOne(hist []float64, exogHist [][]float64, t int, regime RegimeID) float64 {
	p := m.cfg.P
	rp := m.params.Low
	if regime == HighRegime {
		rp = m.params.High
	}

	yhat := rp.Intercept
	for i, phi := range rp.AR {
		lag := i + 1
		if t-lag >= 0 {
			yhat += phi * hist[t-lag]
		}
	}

	coeffIdx := 0
	for k, lags := range m.cfg.ExogLags {
		for _, lag := range lags {
			if coeffIdx < len(rp.Exog) && t-lag >= 0 && t-lag < len(exogHist[k]) {
				yhat += rp.Exog[coeffIdx] * exogHist[k][t-lag]
			}
			coeffIdx++
		}
	}
	_ = p
	return yhat
}

func (m *Model) validateFutureExog(h int, futureExog [][]float64) error {
	nExog := len(m.cfg.ExogLags)
	if nExog == 0 {
		return nil
	}
	if len(futureExog) != nExog {
		return fmt.Errorf("tar: expected %d exog columns, got %d", nExog, len(futureExog))
	}
	for i, col := range futureExog {
		if len(col) < h {
			return fmt.Errorf("tar: futureExog[%d] has %d rows, need %d", i, len(col), h)
		}
	}
	return nil
}

// splitByThreshold returns indices of observations in LOW and HIGH regimes.
// Observation t is assigned to HIGH if y(t-d) > r, to LOW otherwise.
// Only observations t >= max(p, d) are included (burn-in excluded).
func splitByThreshold(y []float64, d int, r float64, p int) (low, high []int) {
	burn := p
	if d > burn {
		burn = d
	}
	for t := burn; t < len(y); t++ {
		if y[t-d] > r {
			high = append(high, t)
		} else {
			low = append(low, t)
		}
	}
	return
}

// regimeSSE computes the SSE for a single regime given observation indices.
func regimeSSE(y []float64, exog [][]float64, exogLags [][]int, idx []int, p int) (float64, error) {
	rp, err := fitRegime(y, exog, exogLags, idx, p)
	if err != nil {
		return math.Inf(1), err
	}
	return rp.SSE, nil
}

// fitRegime fits OLS coefficients for a single regime.
func fitRegime(y []float64, exog [][]float64, exogLags [][]int, idx []int, p int) (RegimeParams, error) {
	n := len(idx)

	// count exog features
	nExogFeatures := 0
	for _, lags := range exogLags {
		nExogFeatures += len(lags)
	}
	nFeatures := 1 + p + nExogFeatures // intercept + AR + exog

	X := make([][]float64, n)
	yVec := make([]float64, n)

	for i, t := range idx {
		X[i] = make([]float64, nFeatures)
		col := 0
		X[i][col] = 1.0 // intercept
		col++
		for lag := 1; lag <= p; lag++ {
			if t-lag >= 0 {
				X[i][col] = y[t-lag]
			}
			col++
		}
		for k, lags := range exogLags {
			for _, lag := range lags {
				if t-lag >= 0 && k < len(exog) {
					X[i][col] = exog[k][t-lag]
				}
				col++
			}
		}
		yVec[i] = y[t]
	}

	coeffs, err := olsCholesky(X, yVec, nFeatures)
	if err != nil {
		return RegimeParams{}, err
	}

	// compute SSE and sigma
	var sse float64
	for i, t := range idx {
		pred := coeffs[0]
		col := 1
		for lag := 1; lag <= p; lag++ {
			if t-lag >= 0 {
				pred += coeffs[col] * y[t-lag]
			}
			col++
		}
		for k, lags := range exogLags {
			for _, lag := range lags {
				if t-lag >= 0 && k < len(exog) {
					pred += coeffs[col] * exog[k][t-lag]
				}
				col++
			}
		}
		d := yVec[i] - pred
		sse += d * d
	}

	sigma := 0.0
	if n > nFeatures {
		sigma = math.Sqrt(sse / float64(n-nFeatures))
	}

	rp := RegimeParams{
		Intercept: coeffs[0],
		AR:        make([]float64, p),
		N:         n,
		SSE:       sse,
		Sigma:     sigma,
	}
	copy(rp.AR, coeffs[1:1+p])
	if nExogFeatures > 0 {
		rp.Exog = make([]float64, nExogFeatures)
		copy(rp.Exog, coeffs[1+p:])
	}
	return rp, nil
}

// percentileCandidates generates threshold candidates from percentiles of y.
func percentileCandidates(y []float64, pctLow, pctHigh, nSteps int) []float64 {
	sorted := make([]float64, len(y))
	copy(sorted, y)
	sortFloats(sorted)

	lo := sorted[len(sorted)*pctLow/100]
	hi := sorted[len(sorted)*pctHigh/100]
	step := (hi - lo) / float64(nSteps)

	if step <= 0 {
		return []float64{lo}
	}

	candidates := make([]float64, 0, nSteps)
	for v := lo; v <= hi; v += step {
		candidates = append(candidates, v)
	}
	return candidates
}

func sortFloats(s []float64) {
	// insertion sort — fine for percentile calculation
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// olsCholesky solves the normal equations X'Xβ = X'y via Cholesky.
func olsCholesky(X [][]float64, y []float64, nFeatures int) ([]float64, error) {
	n := len(X)
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
	for j := 0; j < nFeatures; j++ {
		for k := j + 1; k < nFeatures; k++ {
			XtX[j*nFeatures+k] = XtX[k*nFeatures+j]
		}
	}

	// small ridge regularisation (λ=1e-8) to handle near-singular cases
	// in regimes with few observations or low-variance columns
	const ridge = 1e-8
	for i := 0; i < nFeatures; i++ {
		XtX[i*nFeatures+i] += ridge
	}

	L := make([]float64, nFeatures*nFeatures)
	for i := 0; i < nFeatures; i++ {
		for j := 0; j <= i; j++ {
			sum := XtX[i*nFeatures+j]
			for k := 0; k < j; k++ {
				sum -= L[i*nFeatures+k] * L[j*nFeatures+k]
			}
			if i == j {
				if sum <= 0 {
					return nil, fmt.Errorf("matrix not positive definite at diagonal %d", i)
				}
				L[i*nFeatures+j] = math.Sqrt(sum)
			} else {
				L[i*nFeatures+j] = sum / L[j*nFeatures+j]
			}
		}
	}

	z := make([]float64, nFeatures)
	for i := 0; i < nFeatures; i++ {
		sum := Xty[i]
		for k := 0; k < i; k++ {
			sum -= L[i*nFeatures+k] * z[k]
		}
		z[i] = sum / L[i*nFeatures+i]
	}

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

// normalCDF — Abramowitz & Stegun approximation, max error 7.5e-8.
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

func fmtSlice(v []float64) string {
	if len(v) == 0 {
		return "[]"
	}
	s := "["
	for i, f := range v {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%.4f", f)
	}
	return s + "]"
}
