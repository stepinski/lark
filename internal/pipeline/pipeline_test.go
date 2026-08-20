package pipeline

import (
	"math"
	"testing"
	"time"

	"github.com/stepinski/lark/datasource/flowworks"
)

func TestBuildDataset_BasicShape(t *testing.T) {
	// Generate 3000 observations (enough for 288 warmup + horizon)
	rows := make([]Obs, 3000)
	for i := range rows {
		rows[i] = Obs{
			T:       time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).Add(time.Duration(i) * 5 * time.Minute),
			DepthMM: 800.0,
			Rain6hr: 0.0,
		}
	}

	invert := 900.0
	horizon := 12
	X, y, ts := BuildDataset(rows, invert, horizon, false, false)

	// Expected: 3000 - 288 - 12 = 2700 samples
	wantN := 2700
	if len(X) != wantN {
		t.Fatalf("expected %d samples, got %d", wantN, len(X))
	}
	if len(y) != wantN {
		t.Fatalf("expected %d labels, got %d", wantN, len(y))
	}
	if len(ts) != wantN {
		t.Fatalf("expected %d timestamps, got %d", wantN, len(ts))
	}

	// Each feature vector has 5 elements: excess, amc, rain1hr, mSin, mCos
	if len(X[0]) != 5 {
		t.Fatalf("expected 5 features, got %d", len(X[0]))
	}
}

func TestBuildDataset_AMCNonZeroAfterRain(t *testing.T) {
	// Generate observations with rain in the last 2016 readings
	rows := make([]Obs, 3000)
	for i := range rows {
		rows[i] = Obs{
			T:       time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).Add(time.Duration(i) * 5 * time.Minute),
			DepthMM: 800.0,
			Rain6hr: 0.0,
		}
	}

	// Add rain to the last 100 observations
	for i := 2900; i < 3000; i++ {
		rows[i].Rain6hr = 10.0 // 10mm per 5-min interval
	}

	invert := 900.0
	horizon := 12
	X, _, _ := BuildDataset(rows, invert, horizon, false, false)

	// AMC (index 1) should be non-zero for samples after rain started
	var nonZeroAMC int
	for _, row := range X {
		if row[1] > 0.001 {
			nonZeroAMC++
		}
	}

	if nonZeroAMC == 0 {
		t.Fatal("expected non-zero AMC after rain, got all zeros")
	}

	t.Logf("non-zero AMC samples: %d / %d", nonZeroAMC, len(X))
}

func TestBuildDataset_ExcessProportionalToInvert(t *testing.T) {
	// Generate observations with constant depth
	rows := make([]Obs, 3000)
	depth := 850.0
	for i := range rows {
		rows[i] = Obs{
			T:       time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).Add(time.Duration(i) * 5 * time.Minute),
			DepthMM: depth,
			Rain6hr: 0.0,
		}
	}

	horizon := 12

	// Test with different invert values
	for _, invert := range []float64{900.0, 800.0, 700.0} {
		X, _, _ := BuildDataset(rows, invert, horizon, false, false)
		wantExcess := depth - invert

		for i, row := range X {
			if math.Abs(row[0]-wantExcess) > 0.001 {
				t.Errorf("invert=%.0f: sample %d excess=%.2f, want %.2f",
					invert, i, row[0], wantExcess)
			}
		}
	}
}

func TestBuildDataset_LabelCorrect(t *testing.T) {
	// Generate observations where depth crosses invert at a known point
	rows := make([]Obs, 3000)
	invert := 900.0
	for i := range rows {
		rows[i] = Obs{
			T:       time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC).Add(time.Duration(i) * 5 * time.Minute),
			DepthMM: 800.0,
			Rain6hr: 0.0,
		}
	}

	// Set depth >= invert at index 2900
	rows[2900].DepthMM = 950.0

	horizon := 12
	_, y, _ := BuildDataset(rows, invert, horizon, false, false)

	// Sample at index 2900 - 288 = 2612 should have label 1.0
	// (overflow occurs within horizon steps)
	sampleIdx := 2900 - 288
	if sampleIdx >= len(y) {
		t.Fatalf("sample index %d out of range", sampleIdx)
	}

	if y[sampleIdx] != 1.0 {
		t.Errorf("expected label 1.0 at sample %d, got %.2f", sampleIdx, y[sampleIdx])
	}

	// Sample well before the overflow should have label 0.0
	if y[0] != 0.0 {
		t.Errorf("expected label 0.0 at sample 0, got %.2f", y[0])
	}
}

func TestAlignChannels(t *testing.T) {
	data := map[int][]flowworks.DataPoint{
		1: {
			{Time: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC), Value: 800.0},
			{Time: time.Date(2024, 1, 1, 0, 5, 0, 0, time.UTC), Value: 810.0},
		},
		2: {
			{Time: time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC), Value: 5.0},
			{Time: time.Date(2024, 1, 1, 0, 5, 0, 0, time.UTC), Value: 10.0},
		},
	}

	rows := AlignChannels(data, 1, 2)
	if len(rows) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(rows))
	}

	if rows[0].DepthMM != 800.0 {
		t.Errorf("expected depth 800.0, got %.1f", rows[0].DepthMM)
	}
	if rows[0].Rain6hr != 5.0 {
		t.Errorf("expected rain 5.0, got %.1f", rows[0].Rain6hr)
	}
}
