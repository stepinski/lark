package logreg_test

import (
	"math"
	"testing"

	"github.com/stepinski/lark/models/logreg"
)

// linearSeparable generates a trivially separable dataset:
// x > threshold → y=1, else y=0
func linearSeparable(n int, threshold float64) ([][]float64, []float64) {
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		v := float64(i) / float64(n) * 2.0 // 0 to 2
		X[i] = []float64{v}
		if v > threshold {
			y[i] = 1
		}
	}
	return X, y
}

// --- construction ---

func TestNew_DefaultConfig(t *testing.T) {
	m := logreg.New(logreg.Config{})
	if m == nil {
		t.Error("expected non-nil model")
	}
}

// --- Fit ---

func TestFit_LinearSeparable(t *testing.T) {
	X, y := linearSeparable(200, 1.0)
	m := logreg.New(logreg.Config{
		L2Lambda: 0.01, MaxIter: 2000,
	})
	if err := m.Fit(X, y); err != nil {
		t.Fatalf("Fit error: %v", err)
	}
	p := m.Params()
	// weight should be positive (higher x → higher P(y=1))
	if p.Weights[0] <= 0 {
		t.Errorf("weight = %.4f, expected positive for separable data", p.Weights[0])
	}
}

func TestFit_EmptyX(t *testing.T) {
	m := logreg.New(logreg.Config{})
	err := m.Fit([][]float64{}, []float64{})
	if err == nil {
		t.Error("expected error for empty X")
	}
}

func TestFit_YLengthMismatch(t *testing.T) {
	m := logreg.New(logreg.Config{})
	X := [][]float64{{1.0}, {2.0}}
	y := []float64{1.0} // wrong length
	if err := m.Fit(X, y); err == nil {
		t.Error("expected error for y length mismatch")
	}
}

func TestFit_FeatureCountMismatch(t *testing.T) {
	m := logreg.New(logreg.Config{})
	X := [][]float64{{1.0, 2.0}, {3.0}} // row 1 has wrong length
	y := []float64{1.0, 0.0}
	if err := m.Fit(X, y); err == nil {
		t.Error("expected error for inconsistent feature count")
	}
}

// --- Predict ---

func TestPredict_NotFitted(t *testing.T) {
	m := logreg.New(logreg.Config{})
	_, err := m.Predict([][]float64{{1.0}})
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestPredict_RangeValid(t *testing.T) {
	X, y := linearSeparable(100, 0.8)
	m := logreg.New(logreg.Config{L2Lambda: 0.1})
	_ = m.Fit(X, y)

	probs, err := m.Predict(X)
	if err != nil {
		t.Fatalf("Predict error: %v", err)
	}
	for i, p := range probs {
		if p < 0 || p > 1 {
			t.Errorf("probs[%d] = %.4f outside [0,1]", i, p)
		}
	}
}

func TestPredict_DirectionCorrect(t *testing.T) {
	// after fitting: P(y=1|x=1.8) > P(y=1|x=0.2)
	X, y := linearSeparable(200, 1.0)
	m := logreg.New(logreg.Config{L2Lambda: 0.01, MaxIter: 2000})
	_ = m.Fit(X, y)

	pHigh, _ := m.PredictOne([]float64{1.8})
	pLow, _ := m.PredictOne([]float64{0.2})
	if pHigh <= pLow {
		t.Errorf("P(x=1.8)=%.4f should be > P(x=0.2)=%.4f", pHigh, pLow)
	}
}

func TestPredictOne_NotFitted(t *testing.T) {
	m := logreg.New(logreg.Config{})
	_, err := m.PredictOne([]float64{1.0})
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestPredictOne_FeatureMismatch(t *testing.T) {
	X, y := linearSeparable(50, 0.5)
	m := logreg.New(logreg.Config{})
	_ = m.Fit(X, y)
	_, err := m.PredictOne([]float64{1.0, 2.0}) // wrong number of features
	if err == nil {
		t.Error("expected error for feature count mismatch")
	}
}

// --- class imbalance ---

func TestFit_ImbalancedData_PosWeight(t *testing.T) {
	// 190 negatives, 10 positives — without reweighting model may predict all 0
	n := 200
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		v := float64(i) / float64(n)
		X[i] = []float64{v}
		if i >= 190 { // only last 10 are positive
			y[i] = 1
		}
	}

	// with class weight = 19 (190/10), model should still give high P for positives
	posWeight := 19.0
	m := logreg.New(logreg.Config{
		L2Lambda:  0.01,
		MaxIter:   2000,
		PosWeight: posWeight,
	})
	_ = m.Fit(X, y)

	// check that high x values get higher probability than low x values
	pHigh, _ := m.PredictOne([]float64{0.99})
	pLow, _ := m.PredictOne([]float64{0.01})
	if pHigh <= pLow {
		t.Errorf("P(x=0.99)=%.4f should be > P(x=0.01)=%.4f with pos reweighting", pHigh, pLow)
	}
}

// --- multiple features ---

func TestFit_MultiFeature(t *testing.T) {
	// y=1 if x1 > 0.5 AND x2 > 0.5
	n := 400
	X := make([][]float64, n)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		x1 := float64(i%20) / 20.0
		x2 := float64(i/20) / 20.0
		X[i] = []float64{x1, x2}
		if x1 > 0.5 && x2 > 0.5 {
			y[i] = 1
		}
	}
	m := logreg.New(logreg.Config{L2Lambda: 0.01, MaxIter: 2000})
	if err := m.Fit(X, y); err != nil {
		t.Fatalf("Fit error: %v", err)
	}
	p := m.Params()
	// both weights should be positive
	if p.Weights[0] <= 0 || p.Weights[1] <= 0 {
		t.Errorf("both weights should be positive, got %v", p.Weights)
	}
}

// --- Params.Summary ---

func TestParams_Summary(t *testing.T) {
	X, y := linearSeparable(100, 0.5)
	m := logreg.New(logreg.Config{
		FeatureNames: []string{"rain_6hr"},
	})
	_ = m.Fit(X, y)
	s := m.Params().Summary()
	if len(s) == 0 {
		t.Error("Summary returned empty string")
	}
	t.Log(s)
}

// --- TrainingStats ---

func TestTrainingStats(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}, {4}, {5}}
	y := []float64{0, 0, 0, 1, 1}
	m := logreg.New(logreg.Config{})
	_ = m.Fit(X, y)
	nPos, nNeg := m.TrainingStats()
	if nPos != 2 || nNeg != 3 {
		t.Errorf("TrainingStats = (%d, %d), want (2, 3)", nPos, nNeg)
	}
}

// --- numerical stability ---

func TestPredict_NumericalStability_ExtremeValues(t *testing.T) {
	X := [][]float64{{1e6}, {-1e6}}
	y := []float64{1, 0}
	m := logreg.New(logreg.Config{L2Lambda: 1.0})
	_ = m.Fit(X, y)

	probs, _ := m.Predict([][]float64{{1e6}, {-1e6}})
	for i, p := range probs {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Errorf("probs[%d] = %v (non-finite)", i, p)
		}
		if p < 0 || p > 1 {
			t.Errorf("probs[%d] = %.6f outside [0,1]", i, p)
		}
	}
}
