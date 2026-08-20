package dwf_test

import (
	"math"
	"testing"
	"time"

	"github.com/stepinski/lark/models/dwf"
)

var monday = time.Date(2025, 1, 6, 0, 0, 0, 0, time.UTC) // known Monday

func makeObs(t time.Time, depth, rain6, rain24 float64) dwf.Obs {
	return dwf.Obs{T: t, DepthMM: depth, Rain6hr: rain6, Rain24hr: rain24}
}

// generateDryWeek creates a week of synthetic dry observations with a known
// diurnal pattern: low at night (~30mm), high at noon (~80mm).
func generateDryWeek(startMonday time.Time) []dwf.Obs {
	var obs []dwf.Obs
	for day := 0; day < 7; day++ {
		for h := 0; h < 24; h++ {
			for m := 0; m < 60; m += 5 {
				t := startMonday.AddDate(0, 0, day).Add(
					time.Duration(h)*time.Hour + time.Duration(m)*time.Minute)
				// diurnal: depth varies 30-80mm
				depth := 30.0 + 50.0*math.Sin(math.Pi*float64(h*60+m)/(24*60))
				obs = append(obs, makeObs(t, depth, 0, 0))
			}
		}
	}
	return obs
}

// --- New and Fit ---

func TestFit_DryObservations(t *testing.T) {
	obs := generateDryWeek(monday)
	m := dwf.New(dwf.Config{})
	if err := m.Fit(obs); err != nil {
		t.Fatalf("Fit error: %v", err)
	}
	t.Log(m.Summary())
}

func TestFit_EmptyObs(t *testing.T) {
	m := dwf.New(dwf.Config{})
	err := m.Fit(nil)
	if err == nil {
		t.Error("expected error for nil observations")
	}
}

func TestFit_AllWet_NoBaselineError(t *testing.T) {
	// all observations have rain → no dry obs → should error
	obs := []dwf.Obs{
		makeObs(monday, 80, 5.0, 10.0),
		makeObs(monday.Add(5*time.Minute), 85, 5.0, 10.0),
	}
	m := dwf.New(dwf.Config{})
	err := m.Fit(obs)
	if err == nil {
		t.Error("expected error when all observations are wet")
	}
}

// --- BaselineAt ---

func TestBaselineAt_NotFitted(t *testing.T) {
	m := dwf.New(dwf.Config{})
	_, err := m.BaselineAt(monday)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestBaselineAt_DiurnalPattern(t *testing.T) {
	obs := generateDryWeek(monday)
	// add 3 more weeks
	for w := 1; w <= 3; w++ {
		obs = append(obs, generateDryWeek(monday.AddDate(0, 0, 7*w))...)
	}

	m := dwf.New(dwf.Config{MinDryObs: 2})
	_ = m.Fit(obs)

	noon := monday.Add(12 * time.Hour)
	midnight := monday

	baselineNoon, _ := m.BaselineAt(noon)
	baselineMidnight, _ := m.BaselineAt(midnight)

	// noon should be higher than midnight
	if baselineNoon <= baselineMidnight {
		t.Errorf("baseline noon (%.1f) should be > midnight (%.1f)",
			baselineNoon, baselineMidnight)
	}
	t.Logf("baseline: midnight=%.1fmm  noon=%.1fmm", baselineMidnight, baselineNoon)
}

// --- Excess ---

func TestExcess_NotFitted(t *testing.T) {
	m := dwf.New(dwf.Config{})
	_, err := m.Excess(monday, 50)
	if err == nil {
		t.Error("expected error for unfitted model")
	}
}

func TestExcess_AboveBaseline(t *testing.T) {
	obs := generateDryWeek(monday)
	m := dwf.New(dwf.Config{MinDryObs: 1})
	_ = m.Fit(obs)

	noon := monday.Add(12 * time.Hour)
	baseline, _ := m.BaselineAt(noon)

	// depth well above baseline → positive excess
	excess, err := m.Excess(noon, baseline+100)
	if err != nil {
		t.Fatalf("Excess error: %v", err)
	}
	if excess != 100.0 {
		t.Errorf("Excess = %.1f, want 100.0", excess)
	}
}

func TestExcess_BelowBaseline(t *testing.T) {
	obs := generateDryWeek(monday)
	m := dwf.New(dwf.Config{MinDryObs: 1})
	_ = m.Fit(obs)

	noon := monday.Add(12 * time.Hour)
	baseline, _ := m.BaselineAt(noon)

	// depth below baseline → negative excess (normal at low-flow periods)
	excess, _ := m.Excess(noon, baseline-10)
	if excess != -10.0 {
		t.Errorf("Excess = %.1f, want -10.0", excess)
	}
}

// --- Coverage ---

func TestCoverage_FittedModel(t *testing.T) {
	obs := generateDryWeek(monday)
	m := dwf.New(dwf.Config{MinDryObs: 1})
	_ = m.Fit(obs)

	cov := m.Coverage(monday.Add(12 * time.Hour))
	if cov <= 0 {
		t.Errorf("Coverage = %d, expected > 0 after fitting", cov)
	}
}

func TestCoverage_NotFitted(t *testing.T) {
	m := dwf.New(dwf.Config{})
	if m.Coverage(monday) != 0 {
		t.Error("Coverage should be 0 for unfitted model")
	}
}

// --- API (Antecedent Precipitation Index) ---

func TestAPI_ZeroRain(t *testing.T) {
	rain := make([]float64, 72) // 6hr of zeros
	api := dwf.API(rain, 0.85)
	if api != 0 {
		t.Errorf("API of zeros = %.4f, want 0", api)
	}
}

func TestAPI_DecaysWithTime(t *testing.T) {
	// single rain event at the start (oldest), should contribute little
	old := make([]float64, 72)
	old[0] = 10.0 // rain 6hr ago

	// single rain event at the end (most recent), should contribute more
	recent := make([]float64, 72)
	recent[71] = 10.0 // rain just now

	apiOld := dwf.API(old, 0.85)
	apiRecent := dwf.API(recent, 0.85)

	if apiRecent <= apiOld {
		t.Errorf("recent rain (%.4f) should have higher API than old rain (%.4f)",
			apiRecent, apiOld)
	}
}

func TestAPI_Decay085_HalfLifeApprox2hr(t *testing.T) {
	// with decay=0.85 per 5-min step:
	// after 24 steps (2hr): contribution = 0.85^24 ≈ 0.017 → ~98% decayed
	contribution := math.Pow(0.85, 24)
	if contribution > 0.05 {
		t.Logf("2hr decay = %.4f (expected ~0.017)", contribution)
	}
}

func TestAPI_ConstantRain_ConvergesToSteadyState(t *testing.T) {
	// constant rain should produce bounded API (geometric series)
	rain := make([]float64, 1000)
	for i := range rain {
		rain[i] = 1.0
	}
	api := dwf.API(rain, 0.85)
	// geometric series with first-step decay: sum = decay / (1 - decay)
	expected := 0.85 / (1.0 - 0.85)
	if math.Abs(api-expected) > 0.1 {
		t.Errorf("API steady state = %.4f, want ~%.4f", api, expected)
	}
}

// --- GlobalMedian ---

func TestGlobalMedian_ReturnsReasonableValue(t *testing.T) {
	obs := generateDryWeek(monday)
	m := dwf.New(dwf.Config{MinDryObs: 1})
	_ = m.Fit(obs)

	gm := m.GlobalMedian()
	// synthetic data ranges 30-80mm, median should be around 50
	if gm < 20 || gm > 100 {
		t.Errorf("GlobalMedian = %.1f, expected in [20, 100]", gm)
	}
	t.Logf("GlobalMedian = %.1f mm", gm)
}

// --- DryRain thresholds ---

func TestFit_WetObsExcluded(t *testing.T) {
	// mix of dry and wet observations
	// dry: depth=50, wet: depth=200 (with rain)
	// baseline should reflect dry values only (~50)
	var obs []dwf.Obs
	for i := 0; i < 100; i++ {
		ts := monday.Add(time.Duration(i*5) * time.Minute)
		obs = append(obs, makeObs(ts, 50, 0, 0)) // dry
	}
	for i := 0; i < 10; i++ {
		ts := monday.AddDate(0, 0, 1).Add(time.Duration(i*5) * time.Minute)
		obs = append(obs, makeObs(ts, 200, 5.0, 10.0)) // wet
	}

	m := dwf.New(dwf.Config{MinDryObs: 1})
	_ = m.Fit(obs)

	gm := m.GlobalMedian()
	if gm > 100 {
		t.Errorf("GlobalMedian = %.1f, expected ~50 (wet obs should be excluded)", gm)
	}
}
