package sarimax_test

import (
	"math"
	"testing"

	"github.com/stepinski/lark/models/sarimax"
)

// syntheticARSeries generates y(t) = phi*y(t-1) + noise-free, starting from seed.
// Used to verify the AR coefficient recovery.
func syntheticARSeries(n int, phi, seed float64) []float64 {
	y := make([]float64, n)
	y[0] = seed
	for i := 1; i < n; i++ {
		y[i] = phi*y[i-1]
	}
	return y
}

// syntheticARWithExog generates y(t) = phi*y(t-1) + beta*x(t) + 0 noise.
func syntheticARWithExog(n int, phi, beta, seed float64, x []float64) []float64 {
	y := make([]float64, n)
	y[0] = seed
	for i := 1; i < n; i++ {
		y[i] = phi*y[i-1] + beta*x[i]
	}
	return y
}

// --- Order ---

func TestOrder_String(t *testing.T) {
	o := sarimax.Order{P: 1, D: 1, Q: 1, SP: 1, SD: 0, SQ: 0, M: 24}
	got := o.String()
	want := "(1,1,1)(1,0,0,24)"
	if got != want {
		t.Errorf("Order.String() = %q, want %q", got, want)
	}
}

func TestNew_InvalidOrder(t *testing.T) {
	_, err := sarimax.New(sarimax.Order{P: -1})
	if err == nil {
		t.Error("expected error for negative P, got nil")
	}
}

func TestNew_SeasonalWithoutM(t *testing.T) {
	_, err := sarimax.New(sarimax.Order{SP: 1, M: 0})
	if err == nil {
		t.Error("expected error for seasonal order without M")
	}
}

// --- Fit ---

func TestFit_AR1_RecoversPhi(t *testing.T) {
	// Pure AR(1) with φ=0.8, no noise — OLS should recover φ exactly.
	truePhi := 0.8
	y := syntheticARSeries(200, truePhi, 1.0)

	m, _ := sarimax.New(sarimax.Order{P: 1})
	if err := m.Fit(y, nil); err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	p := m.Params()
	if len(p.AR) != 1 {
		t.Fatalf("expected 1 AR coeff, got %d", len(p.AR))
	}
	if math.Abs(p.AR[0]-truePhi) > 0.01 {
		t.Errorf("AR[0] = %.4f, want %.4f (±0.01)", p.AR[0], truePhi)
	}
}

func TestFit_ExogCoeffRecovered(t *testing.T) {
	// y(t) = 0.5*y(t-1) + 2.0*x(t)
	n := 300
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i%10) * 0.1 // simple periodic exog
	}
	y := syntheticARWithExog(n, 0.5, 2.0, 0.5, x)

	m, _ := sarimax.New(sarimax.Order{P: 1})
	if err := m.Fit(y, [][]float64{x}); err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	p := m.Params()
	if len(p.Exog) != 1 {
		t.Fatalf("expected 1 exog coeff, got %d", len(p.Exog))
	}
	// Looser tolerance due to collinearity with AR term
	if math.Abs(p.Exog[0]-2.0) > 0.2 {
		t.Errorf("Exog[0] = %.4f, want ~2.0 (±0.2)", p.Exog[0])
	}
}

func TestFit_InsufficientData(t *testing.T) {
	m, _ := sarimax.New(sarimax.Order{P: 5, Q: 5})
	err := m.Fit([]float64{1, 2, 3}, nil)
	if err == nil {
		t.Error("expected error for insufficient data")
	}
}

func TestFit_ExogLengthMismatch(t *testing.T) {
	m, _ := sarimax.New(sarimax.Order{P: 1})
	y := make([]float64, 50)
	x := make([]float64, 30) // wrong length
	err := m.Fit(y, [][]float64{x})
	if err == nil {
		t.Error("expected error for exog length mismatch")
	}
}

// --- Predict ---

func TestPredict_ReturnsHForecasts(t *testing.T) {
	y := syntheticARSeries(100, 0.7, 2.0)
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_ = m.Fit(y, nil)

	forecasts, err := m.Predict(5, nil)
	if err != nil {
		t.Fatalf("Predict error: %v", err)
	}
	if len(forecasts) != 5 {
		t.Errorf("len(forecasts) = %d, want 5", len(forecasts))
	}
}

func TestPredict_StationaryARBounded(t *testing.T) {
	// For stationary AR(1) with |φ| < 1, multi-step forecasts must remain
	// finite and bounded — they must not diverge.
	y := syntheticARSeries(200, 0.6, 10.0)
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_ = m.Fit(y, nil)

	forecasts, _ := m.Predict(50, nil)
	for i, f := range forecasts {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Errorf("forecasts[%d] = %v (non-finite)", i, f)
		}
		if math.Abs(f) > 1e6 {
			t.Errorf("forecasts[%d] = %.4f diverged", i, f)
		}
	}
}

func TestPredict_NotFitted(t *testing.T) {
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_, err := m.Predict(3, nil)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestPredict_MissingFutureExog(t *testing.T) {
	n := 100
	x := make([]float64, n)
	y := syntheticARWithExog(n, 0.5, 1.0, 1.0, x)
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_ = m.Fit(y, [][]float64{x})

	// omit futureExog → should error
	_, err := m.Predict(3, nil)
	if err == nil {
		t.Error("expected error for missing futureExog")
	}
}

// --- PredictProba ---

func TestPredictProba_RangeValid(t *testing.T) {
	y := syntheticARSeries(150, 0.7, 5.0)
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_ = m.Fit(y, nil)

	probs, err := m.PredictProba(5, nil, 3.0)
	if err != nil {
		t.Fatalf("PredictProba error: %v", err)
	}
	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("probs[%d] = %.4f, want in [0,1]", i, p)
		}
	}
}

func TestPredictProba_HighThresholdLowProb(t *testing.T) {
	y := syntheticARSeries(150, 0.5, 1.0) // series ~bounded near 0
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_ = m.Fit(y, nil)

	// threshold far above any forecast → probability should be near 0
	probs, _ := m.PredictProba(3, nil, 1e9)
	for i, p := range probs {
		if p > 0.01 {
			t.Errorf("probs[%d] = %.4f, want ~0 for extreme threshold", i, p)
		}
	}
}

// --- Step (streaming) ---

func TestStep_ReturnsValue(t *testing.T) {
	y := syntheticARSeries(100, 0.7, 2.0)
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_ = m.Fit(y, nil)

	_, err := m.Step(y[99], nil)
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}
}

func TestStep_NotFitted(t *testing.T) {
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_, err := m.Step(1.0, nil)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestStep_MultipleStepsStable(t *testing.T) {
	y := syntheticARSeries(150, 0.6, 3.0)
	m, _ := sarimax.New(sarimax.Order{P: 1})
	_ = m.Fit(y[:100], nil)

	// stream the held-out observations — must not panic or return NaN
	for i := 100; i < 150; i++ {
		forecast, err := m.Step(y[i], nil)
		if err != nil {
			t.Fatalf("Step[%d] error: %v", i, err)
		}
		if math.IsNaN(forecast) || math.IsInf(forecast, 0) {
			t.Fatalf("Step[%d] returned non-finite forecast: %v", i, forecast)
		}
	}
}
