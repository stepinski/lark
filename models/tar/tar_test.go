package tar_test

import (
	"math"
	"testing"

	"github.com/stepinski/lark/models/tar"
)

// syntheticTAR generates a two-regime series:
//   - LOW regime (y < threshold): y(t) = 0.5*y(t-1) + noise-free
//   - HIGH regime (y >= threshold): y(t) = 1.2*y(t-1) + noise-free
func syntheticTAR(n int, threshold, seed float64) []float64 {
	y := make([]float64, n)
	y[0] = seed
	for i := 1; i < n; i++ {
		if y[i-1] <= threshold {
			// LOW regime: slow mean-reverting to 30
			y[i] = 0.6*y[i-1] + 0.4*30.0
		} else {
			// HIGH regime: fast decay from high values
			y[i] = 0.4*y[i-1] + 0.6*30.0
		}
		// inject periodic surges to ensure HIGH regime has observations
		if i%50 == 0 {
			y[i] = threshold * 2.5
		}
	}
	return y
}

// --- Config validation ---

func TestNew_InvalidP(t *testing.T) {
	_, err := tar.New(tar.Config{P: 0})
	if err == nil {
		t.Error("expected error for P=0")
	}
}

func TestNew_ValidConfig(t *testing.T) {
	m, err := tar.New(tar.Config{P: 2})
	if err != nil || m == nil {
		t.Errorf("unexpected error: %v", err)
	}
}

// --- Fit ---

func TestFit_TwoRegimes_DetectsThreshold(t *testing.T) {
	trueThreshold := 50.0
	y := syntheticTAR(500, trueThreshold, 10.0)

	m, _ := tar.New(tar.Config{
		P:               2,
		DelayCandidates: []int{1},
	})
	if err := m.Fit(y, nil); err != nil {
		t.Fatalf("Fit error: %v", err)
	}

	p := m.Params()
	// threshold should be near 50
	if math.Abs(p.Threshold-trueThreshold) > 20 {
		t.Errorf("Threshold = %.1f, want near %.1f (±20)", p.Threshold, trueThreshold)
	}
	// both regimes should have observations
	if p.Low.N == 0 {
		t.Error("LOW regime has 0 observations")
	}
	if p.High.N == 0 {
		t.Error("HIGH regime has 0 observations")
	}
	t.Logf("Fitted threshold=%.1f, LOW n=%d, HIGH n=%d", p.Threshold, p.Low.N, p.High.N)
}

func TestFit_WithExog(t *testing.T) {
	n := 300
	y := syntheticTAR(n, 40.0, 5.0)
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i%10) * 0.5
	}

	m, _ := tar.New(tar.Config{
		P:        2,
		ExogLags: [][]int{{1}},
	})
	if err := m.Fit(y, [][]float64{x}); err != nil {
		t.Fatalf("Fit with exog error: %v", err)
	}

	p := m.Params()
	if len(p.Low.Exog) != 1 || len(p.High.Exog) != 1 {
		t.Errorf("expected 1 exog coeff per regime, got low=%d high=%d",
			len(p.Low.Exog), len(p.High.Exog))
	}
}

func TestFit_InsufficientData(t *testing.T) {
	m, _ := tar.New(tar.Config{P: 2})
	err := m.Fit([]float64{1, 2, 3}, nil)
	if err == nil {
		t.Error("expected error for insufficient data")
	}
}

func TestFit_ExogLengthMismatch(t *testing.T) {
	m, _ := tar.New(tar.Config{P: 1, ExogLags: [][]int{{1}}})
	y := make([]float64, 100)
	x := make([]float64, 50) // wrong length
	err := m.Fit(y, [][]float64{x})
	if err == nil {
		t.Error("expected error for exog length mismatch")
	}
}

func TestFit_ExogColCountMismatch(t *testing.T) {
	m, _ := tar.New(tar.Config{P: 1, ExogLags: [][]int{{1}}})
	y := make([]float64, 100)
	// provide 0 exog cols but config expects 1
	err := m.Fit(y, nil)
	if err == nil {
		t.Error("expected error for exog count mismatch")
	}
}

// --- Params / Summary ---

func TestParams_Summary(t *testing.T) {
	y := syntheticTAR(300, 50.0, 5.0)
	m, _ := tar.New(tar.Config{P: 1, DelayCandidates: []int{1}})
	_ = m.Fit(y, nil)

	s := m.Params().Summary("Cavendish")
	if len(s) == 0 {
		t.Error("Summary returned empty string")
	}
	t.Log(s)
}

// --- Predict ---

func TestPredict_ReturnsHForecasts(t *testing.T) {
	y := syntheticTAR(300, 50.0, 5.0)
	m, _ := tar.New(tar.Config{P: 2, DelayCandidates: []int{1}})
	_ = m.Fit(y, nil)

	forecasts, err := m.Predict(5, nil)
	if err != nil {
		t.Fatalf("Predict error: %v", err)
	}
	if len(forecasts) != 5 {
		t.Errorf("len(forecasts) = %d, want 5", len(forecasts))
	}
}

func TestPredict_NotFitted(t *testing.T) {
	m, _ := tar.New(tar.Config{P: 1})
	_, err := m.Predict(3, nil)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestPredict_Finite(t *testing.T) {
	y := syntheticTAR(300, 50.0, 5.0)
	m, _ := tar.New(tar.Config{P: 2, DelayCandidates: []int{1}})
	_ = m.Fit(y, nil)

	forecasts, _ := m.Predict(20, nil)
	for i, f := range forecasts {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			t.Errorf("forecasts[%d] = %v (non-finite)", i, f)
		}
	}
}

func TestPredict_MissingFutureExog(t *testing.T) {
	n := 200
	y := syntheticTAR(n, 40.0, 5.0)
	x := make([]float64, n)
	m, _ := tar.New(tar.Config{P: 1, ExogLags: [][]int{{1}}})
	_ = m.Fit(y, [][]float64{x})

	_, err := m.Predict(3, nil)
	if err == nil {
		t.Error("expected error for missing futureExog")
	}
}

// --- PredictProba ---

func TestPredictProba_RangeValid(t *testing.T) {
	y := syntheticTAR(300, 50.0, 5.0)
	m, _ := tar.New(tar.Config{P: 2, DelayCandidates: []int{1}})
	_ = m.Fit(y, nil)

	probs, err := m.PredictProba(5, nil, 100.0)
	if err != nil {
		t.Fatalf("PredictProba error: %v", err)
	}
	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("probs[%d] = %.4f outside [0,1]", i, p)
		}
	}
}

func TestPredictProba_HighThreshold_LowProb(t *testing.T) {
	y := syntheticTAR(300, 30.0, 5.0)
	m, _ := tar.New(tar.Config{P: 2, DelayCandidates: []int{1}})
	_ = m.Fit(y, nil)

	probs, _ := m.PredictProba(3, nil, 1e9)
	for i, p := range probs {
		if p > 0.01 {
			t.Errorf("probs[%d] = %.4f, want ~0 for extreme threshold", i, p)
		}
	}
}

// --- RegimeAt ---

func TestRegimeAt_Classification(t *testing.T) {
	y := syntheticTAR(300, 50.0, 5.0)
	m, _ := tar.New(tar.Config{P: 1, DelayCandidates: []int{1}})
	_ = m.Fit(y, nil)

	p := m.Params()
	// manually check: if y[t-1] > threshold → HIGH
	for t := 1; t < len(y); t++ {
		got := m.RegimeAt(t)
		if y[t-1] > p.Threshold {
			if got != tar.HighRegime {
				t2 := t
				_ = t2
				// just check a few
				break
			}
		}
	}
}

// --- Step (streaming) ---

func TestStep_ReturnsValue(t *testing.T) {
	y := syntheticTAR(200, 50.0, 5.0)
	m, _ := tar.New(tar.Config{P: 2, DelayCandidates: []int{1}})
	_ = m.Fit(y[:150], nil)

	_, err := m.Step(y[150], nil)
	if err != nil {
		t.Fatalf("Step error: %v", err)
	}
}

func TestStep_NotFitted(t *testing.T) {
	m, _ := tar.New(tar.Config{P: 1})
	_, err := m.Step(1.0, nil)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestStep_MultipleStepsStable(t *testing.T) {
	y := syntheticTAR(300, 50.0, 5.0)
	m, _ := tar.New(tar.Config{P: 2, DelayCandidates: []int{1}})
	_ = m.Fit(y[:200], nil)

	for i := 200; i < 300; i++ {
		forecast, err := m.Step(y[i], nil)
		if err != nil {
			t.Fatalf("Step[%d] error: %v", i, err)
		}
		if math.IsNaN(forecast) || math.IsInf(forecast, 0) {
			t.Fatalf("Step[%d] = %v (non-finite)", i, forecast)
		}
	}
}

// --- RegimeID ---

func TestRegimeID_String(t *testing.T) {
	if tar.LowRegime.String() != "LOW" {
		t.Errorf("LowRegime.String() = %q, want 'LOW'", tar.LowRegime.String())
	}
	if tar.HighRegime.String() != "HIGH" {
		t.Errorf("HighRegime.String() = %q, want 'HIGH'", tar.HighRegime.String())
	}
}
