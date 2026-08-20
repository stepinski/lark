package pipeline

import (
	"testing"
	"time"
)

func TestEvaluateProbs_FalseAlarms(t *testing.T) {
	// Scenario: model fires alarms (p > 0.5) on 3 samples,
	// but no overflow events exist at all.
	// Every alarm should be counted as a false alarm.
	rows := make([]Obs, 100)
	for i := range rows {
		rows[i] = Obs{
			T:       time.Date(2025, 4, 1, 0, 0, 0, 0, time.UTC).Add(time.Duration(i) * 5 * time.Minute),
			DepthMM: 850.0, // above excessGate (900 - 100 = 800)
		}
	}

	invert := 900.0
	horizon := 12
	excessGate := invert - 100.0

	// 3 samples with p > 0.5 and depth >= excessGate
	X := make([][]float64, 100)
	probs := make([]float64, 100)
	for i := range X {
		X[i] = []float64{850.0} // depth feature
		probs[i] = 0.1
	}
	// Fire alarms at indices 10, 30, 50
	probs[10] = 0.9
	probs[30] = 0.9
	probs[50] = 0.9

	// No overflow events — all alarms are false
	events := []OverflowEvent{}

	got := EvaluateProbs(probs, X, rows, events, invert, horizon, false)

	if got.TruePositives != 0 {
		t.Errorf("expected 0 true positives, got %d", got.TruePositives)
	}
	if got.FalseAlarms != 3 {
		t.Errorf("expected 3 false alarms, got %d", got.FalseAlarms)
	}
	if len(got.LeadTimes) != 0 {
		t.Errorf("expected 0 lead times, got %d", len(got.LeadTimes))
	}
	if got.AvgLeadTime != 0 {
		t.Errorf("expected 0 avg lead time, got %.1f", got.AvgLeadTime)
	}

	// Sanity: excessGate was computed correctly
	if excessGate != 800.0 {
		t.Fatalf("excessGate = %.1f, want 800.0", excessGate)
	}
}

func TestEvaluateProbs_TruePositives(t *testing.T) {
	// Scenario: alarm fires, overflow event occurs within horizon.
	rows := make([]Obs, 200)
	for i := range rows {
		rows[i] = Obs{
			T:       time.Date(2025, 4, 1, 0, 0, 0, 0, time.UTC).Add(time.Duration(i) * 5 * time.Minute),
			DepthMM: 850.0,
		}
	}

	invert := 900.0
	horizon := 12

	X := make([][]float64, 200)
	probs := make([]float64, 200)
	for i := range X {
		X[i] = []float64{850.0}
		probs[i] = 0.1
	}
	// Alarm at index 10 (time = 50min)
	probs[10] = 0.9

	// Overflow event starts at index 20 (time = 100min) = 50min after alarm
	// horizon*5 = 60min, so this is within horizon
	events := []OverflowEvent{{Start: 20, End: 25}}

	got := EvaluateProbs(probs, X, rows, events, invert, horizon, false)

	if got.TruePositives != 1 {
		t.Errorf("expected 1 true positive, got %d", got.TruePositives)
	}
	if got.FalseAlarms != 0 {
		t.Errorf("expected 0 false alarms, got %d", got.FalseAlarms)
	}
	if len(got.LeadTimes) != 1 {
		t.Fatalf("expected 1 lead time, got %d", len(got.LeadTimes))
	}
	if got.LeadTimes[0] != 50 {
		t.Errorf("expected lead time 50min, got %dmin", got.LeadTimes[0])
	}
	if got.AvgLeadTime != 50.0 {
		t.Errorf("expected avg lead time 50.0, got %.1f", got.AvgLeadTime)
	}
}

func TestEvaluateProbs_BelowGateIgnored(t *testing.T) {
	// Scenario: p > 0.5 but depth below excessGate — should be ignored.
	rows := make([]Obs, 50)
	for i := range rows {
		rows[i] = Obs{
			T:       time.Date(2025, 4, 1, 0, 0, 0, 0, time.UTC).Add(time.Duration(i) * 5 * time.Minute),
			DepthMM: 700.0, // below excessGate (900 - 100 = 800)
		}
	}

	invert := 900.0
	horizon := 12

	X := make([][]float64, 50)
	probs := make([]float64, 50)
	for i := range X {
		X[i] = []float64{700.0}
		probs[i] = 0.9
	}

	events := []OverflowEvent{}

	got := EvaluateProbs(probs, X, rows, events, invert, horizon, false)

	if got.TruePositives != 0 {
		t.Errorf("expected 0 true positives, got %d", got.TruePositives)
	}
	if got.FalseAlarms != 0 {
		t.Errorf("expected 0 false alarms (below gate), got %d", got.FalseAlarms)
	}
}
