// Package dwf implements a Dry Weather Flow (DWF) baseline model.
//
// The DWF baseline captures the typical diurnal and weekly depth pattern
// during dry conditions. It is the foundation of hydrological wet weather
// analysis: the "excess" above DWF is what rainfall drives.
//
// Baseline construction:
//  1. Select dry observations: rain_6hr == 0 AND rain_24hr < threshold
//  2. Group by (weekday, 5-min interval of day)
//  3. For each group, compute median depth (robust to outliers)
//
// Usage:
//
//	m := dwf.New(dwf.Config{IntervalMinutes: 5})
//	m.Fit(observations)
//	excess := m.Excess(t, currentDepth)
//	api := dwf.API(recentRain, 0.85)  // antecedent precipitation index
package dwf

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

const (
	// IntervalsPerDay is the number of 5-min intervals in a day.
	IntervalsPerDay = 288
	// DaysPerWeek is always 7.
	DaysPerWeek = 7
)

// Obs is a single timestamped observation used for DWF fitting.
type Obs struct {
	T        time.Time
	DepthMM  float64
	Rain6hr  float64 // rolling 6hr rainfall sum (mm)
	Rain24hr float64 // rolling 24hr rainfall sum (mm)
}

// Config controls DWF model construction.
type Config struct {
	// IntervalMinutes is the observation interval (default: 5).
	IntervalMinutes int
	// DryRain6hrThreshold: observations with Rain6hr > this are excluded.
	// Default: 0.0 (strictly dry).
	DryRain6hrThreshold float64
	// DryRain24hrThreshold: observations with Rain24hr > this are excluded.
	// Default: 2.0 mm (allows very light antecedent rain).
	DryRain24hrThreshold float64
	// MinDryObs is the minimum number of dry observations required per
	// (weekday, interval) cell to compute a reliable baseline.
	// Default: 4.
	MinDryObs int
}

func (c Config) intervalMin() int {
	if c.IntervalMinutes <= 0 {
		return 5
	}
	return c.IntervalMinutes
}

func (c Config) dryRain6() float64 {
	return c.DryRain6hrThreshold // zero is valid
}

func (c Config) dryRain24() float64 {
	if c.DryRain24hrThreshold <= 0 {
		return 2.0
	}
	return c.DryRain24hrThreshold
}

func (c Config) minObs() int {
	if c.MinDryObs <= 0 {
		return 4
	}
	return c.MinDryObs
}

// Model holds the fitted DWF baseline.
// baseline[weekday][interval] = median dry depth (mm)
// coverage[weekday][interval] = number of dry observations used
type Model struct {
	cfg      Config
	baseline [DaysPerWeek][IntervalsPerDay]float64
	coverage [DaysPerWeek][IntervalsPerDay]int
	fitted   bool
	// global median used as fallback when coverage is insufficient
	globalMedian float64
}

// New creates an unfitted DWF model.
func New(cfg Config) *Model {
	return &Model{cfg: cfg}
}

// Fit builds the baseline from a set of observations.
// Observations should span at least 4 weeks for reliable baselines.
func (m *Model) Fit(obs []Obs) error {
	if len(obs) == 0 {
		return errors.New("dwf: empty observation set")
	}

	cfg := m.cfg
	intervalsPerDay := IntervalsPerDay
	if cfg.intervalMin() != 5 {
		intervalsPerDay = 24 * 60 / cfg.intervalMin()
	}

	// group by (weekday, interval)
	type cell = []float64
	groups := make([][]cell, DaysPerWeek)
	for d := range groups {
		groups[d] = make([]cell, intervalsPerDay)
	}

	var allDry []float64

	for _, o := range obs {
		// skip wet observations
		if o.Rain6hr > cfg.dryRain6() || o.Rain24hr > cfg.dryRain24() {
			continue
		}
		if math.IsNaN(o.DepthMM) || o.DepthMM < 0 {
			continue
		}

		wd := int(o.T.Weekday())
		interval := intervalOfDay(o.T, cfg.intervalMin())
		groups[wd][interval] = append(groups[wd][interval], o.DepthMM)
		allDry = append(allDry, o.DepthMM)
	}

	if len(allDry) == 0 {
		return errors.New("dwf: no dry observations found — check rain thresholds")
	}

	m.globalMedian = median(allDry)

	// compute medians per cell
	minObs := cfg.minObs()
	for d := 0; d < DaysPerWeek; d++ {
		for i := 0; i < intervalsPerDay; i++ {
			vals := groups[d][i]
			m.coverage[d][i] = len(vals)
			if len(vals) >= minObs {
				m.baseline[d][i] = median(vals)
			} else {
				// fallback to global median
				m.baseline[d][i] = m.globalMedian
			}
		}
	}

	m.fitted = true
	return nil
}

// BaselineAt returns the expected dry weather depth at time t.
func (m *Model) BaselineAt(t time.Time) (float64, error) {
	if !m.fitted {
		return 0, errors.New("dwf: model not fitted")
	}
	wd := int(t.Weekday())
	interval := intervalOfDay(t, m.cfg.intervalMin())
	return m.baseline[wd][interval], nil
}

// Excess returns the depth above the DWF baseline at time t.
// Negative values mean depth is below the baseline (common at night).
// Returns 0 and error if model is not fitted.
func (m *Model) Excess(t time.Time, depthMM float64) (float64, error) {
	baseline, err := m.BaselineAt(t)
	if err != nil {
		return 0, err
	}
	return depthMM - baseline, nil
}

// Coverage returns the number of dry observations used for the baseline
// at time t. Low coverage means the baseline at that time is less reliable.
func (m *Model) Coverage(t time.Time) int {
	if !m.fitted {
		return 0
	}
	wd := int(t.Weekday())
	interval := intervalOfDay(t, m.cfg.intervalMin())
	return m.coverage[wd][interval]
}

// GlobalMedian returns the global dry weather median depth, used as fallback
// when per-cell coverage is insufficient.
func (m *Model) GlobalMedian() float64 { return m.globalMedian }

// Summary returns a human-readable description of the baseline.
func (m *Model) Summary() string {
	if !m.fitted {
		return "DWF model not fitted"
	}
	minCov, maxCov := math.MaxInt32, 0
	for d := 0; d < DaysPerWeek; d++ {
		for i := 0; i < IntervalsPerDay; i++ {
			c := m.coverage[d][i]
			if c < minCov {
				minCov = c
			}
			if c > maxCov {
				maxCov = c
			}
		}
	}
	return fmt.Sprintf("DWF baseline: global_median=%.1fmm  coverage=[%d,%d] obs/cell",
		m.globalMedian, minCov, maxCov)
}

// --- Antecedent Precipitation Index ---

// API computes the Antecedent Precipitation Index from a slice of recent
// rainfall observations (most recent last) and a decay factor per step.
//
// API(t) = Σ rain(t-k) * decay^k   for k = 1..len(rain)
//
// A decay of 0.85 per 5-min step corresponds to roughly 50% soil drainage
// over 2 hours — appropriate for urban sewer catchments.
//
// Higher API = wetter antecedent conditions = catchment more responsive.
func API(recentRain []float64, decayPerStep float64) float64 {
	var api float64
	decay := 1.0
	for i := len(recentRain) - 1; i >= 0; i-- {
		decay *= decayPerStep
		api += recentRain[i] * decay
	}
	return api
}

// APIFromObs computes the API from a slice of Obs, using only the Rain6hr
// incremental values. Steps are assumed to be at cfg.IntervalMinutes spacing.
func APIFromObs(obs []Obs, decay float64) float64 {
	rain := make([]float64, len(obs))
	for i, o := range obs {
		rain[i] = o.Rain6hr // use 6hr increment as per-step rain
	}
	return API(rain, decay)
}

// --- helpers ---

// intervalOfDay returns the 0-based interval index within the day for time t.
func intervalOfDay(t time.Time, intervalMin int) int {
	minuteOfDay := t.Hour()*60 + t.Minute()
	idx := minuteOfDay / intervalMin
	maxIntervals := 24 * 60 / intervalMin
	if idx >= maxIntervals {
		idx = maxIntervals - 1
	}
	return idx
}

// median returns the median of a float64 slice (makes a copy).
func median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sorted := make([]float64, len(vals))
	copy(sorted, vals)
	sort.Float64s(sorted)
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2.0
	}
	return sorted[n/2]
}
