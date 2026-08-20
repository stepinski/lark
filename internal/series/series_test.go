package series_test

import (
	"math"
	"testing"

	"github.com/stepinski/lark/internal/series"
)

func TestDiff_FirstOrder(t *testing.T) {
	src := []float64{1, 3, 6, 10, 15}
	got := series.Diff(src, 1)
	want := []float64{2, 3, 4, 5}
	assertSliceEqual(t, got, want, 1e-9)
}

func TestDiff_SecondOrder(t *testing.T) {
	src := []float64{1, 3, 6, 10, 15}
	got := series.Diff(src, 2)
	want := []float64{1, 1, 1}
	assertSliceEqual(t, got, want, 1e-9)
}

func TestSeasonalDiff(t *testing.T) {
	// period=4, D=1: y[t] - y[t-4]
	src := []float64{10, 20, 30, 40, 12, 22, 32, 42}
	got := series.SeasonalDiff(src, 1, 4)
	want := []float64{2, 2, 2, 2}
	assertSliceEqual(t, got, want, 1e-9)
}

func TestInvertDiff_RoundTrip(t *testing.T) {
	original := []float64{5, 8, 12, 17, 23}
	diffed := series.Diff(original, 1)
	recovered := series.InvertDiff(diffed, original[:1])
	assertSliceEqual(t, recovered, original, 1e-9)
}

func TestDot(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	got := series.Dot(a, b)
	want := 32.0
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("Dot = %v, want %v", got, want)
	}
}

func TestRMSE(t *testing.T) {
	pred := []float64{2, 4, 6}
	obs := []float64{1, 4, 7}
	// errors: 1, 0, -1 → MSE = 2/3 → RMSE = sqrt(2/3)
	want := math.Sqrt(2.0 / 3.0)
	got := series.RMSE(pred, obs)
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("RMSE = %v, want %v", got, want)
	}
}

func TestMean(t *testing.T) {
	got := series.Mean([]float64{2, 4, 6, 8})
	if math.Abs(got-5.0) > 1e-9 {
		t.Errorf("Mean = %v, want 5.0", got)
	}
}

func TestAutocorr_Lag0IsOne(t *testing.T) {
	v := []float64{1, 2, 3, 4, 5}
	got := series.Autocorr(v, 0)
	if math.Abs(got-1.0) > 1e-9 {
		t.Errorf("Autocorr lag0 = %v, want 1.0", got)
	}
}

func TestLag(t *testing.T) {
	v := []float64{1, 2, 3, 4}
	got := series.Lag(v, 2)
	want := []float64{0, 0, 1, 2}
	assertSliceEqual(t, got, want, 1e-9)
}

// --- helpers ---

func assertSliceEqual(t *testing.T, got, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > tol {
			t.Errorf("[%d] got %v, want %v", i, got[i], want[i])
		}
	}
}
