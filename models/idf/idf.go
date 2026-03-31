// Package idf implements Intensity-Duration-Frequency (IDF) curve lookup
// for rainfall return period classification.
//
// IDF curves relate rainfall intensity (mm/hr), duration (minutes), and
// return period (years). For a given storm event (total mm over a duration),
// the return period answers: "how often does a storm this intense occur?"
//
// The Peel Region SoW requires classifying each overflow event with a
// rainfall return period (e.g. "2-5yr (1hr Duration)") based on historical
// IDF data provided by the Region.
//
// Two usage modes:
//
//  1. Lookup table — Peel provides IDF data as a table of
//     (duration_min, return_period_yr) → intensity_mm_hr.
//     Use NewFromTable to load this.
//
//  2. Power-law fit — if only a few IDF points are available, fit
//     the standard power-law model: i = a / (d + b)^c
//     where i=intensity, d=duration. Use NewFromFit.
//
// Once constructed, ReturnPeriod(totalMM, durationMin) returns the
// estimated return period for a given storm.
package idf

import (
	"fmt"
	"math"
	"sort"
)

// ReturnPeriodLabel formats a return period in years as a human-readable
// string matching the Peel SoW format (e.g. "2-5yr (1hr Duration)").
type ReturnPeriodLabel struct {
	// Lo is the lower bound of the return period bracket (years).
	Lo float64
	// Hi is the upper bound (0 = "100+" or "extreme").
	Hi float64
	// DurationMin is the storm duration in minutes.
	DurationMin int
}

// String returns the label in Peel SoW format.
func (r ReturnPeriodLabel) String() string {
	dur := formatDuration(r.DurationMin)
	if r.Hi == 0 {
		return fmt.Sprintf(">100yr (%s Duration)", dur)
	}
	if r.Lo == r.Hi {
		return fmt.Sprintf("%.0fyr (%s Duration)", r.Lo, dur)
	}
	return fmt.Sprintf("%.0f-%.0fyr (%s Duration)", r.Lo, r.Hi, dur)
}

func formatDuration(min int) string {
	if min < 60 {
		return fmt.Sprintf("%dmin", min)
	}
	if min%60 == 0 {
		hr := min / 60
		if hr == 1 {
			return "1hr"
		}
		return fmt.Sprintf("%dhr", hr)
	}
	return fmt.Sprintf("%dmin", min)
}

// IDFPoint is a single entry in an IDF table.
type IDFPoint struct {
	DurationMin    int     // storm duration in minutes
	ReturnPeriodYr float64 // return period in years
	IntensityMMHr  float64 // rainfall intensity in mm/hr
}

// Curve holds a complete IDF dataset for a single gauge or region.
type Curve struct {
	points []IDFPoint
	// sorted duration values for interpolation
	durations []int
	// sorted return period values
	periods []float64
	// grid: [durationIdx][periodIdx] = intensity
	grid [][]float64
}

// NewFromTable builds an IDF Curve from a flat list of IDFPoints.
// The table must contain at least one point per duration/period combination.
// Peel Region will supply this data per Assumption 6 in the SoW.
func NewFromTable(points []IDFPoint) (*Curve, error) {
	if len(points) == 0 {
		return nil, fmt.Errorf("idf: empty point set")
	}

	// collect unique durations and periods
	durSet := make(map[int]struct{})
	perSet := make(map[float64]struct{})
	for _, p := range points {
		if p.DurationMin <= 0 {
			return nil, fmt.Errorf("idf: DurationMin must be > 0, got %d", p.DurationMin)
		}
		if p.ReturnPeriodYr <= 0 {
			return nil, fmt.Errorf("idf: ReturnPeriodYr must be > 0, got %.1f", p.ReturnPeriodYr)
		}
		if p.IntensityMMHr < 0 {
			return nil, fmt.Errorf("idf: IntensityMMHr must be >= 0")
		}
		durSet[p.DurationMin] = struct{}{}
		perSet[p.ReturnPeriodYr] = struct{}{}
	}

	durations := sortedInts(durSet)
	periods := sortedFloats(perSet)

	// build intensity grid
	durIdx := make(map[int]int, len(durations))
	for i, d := range durations {
		durIdx[d] = i
	}
	perIdx := make(map[float64]int, len(periods))
	for i, p := range periods {
		perIdx[p] = i
	}

	grid := make([][]float64, len(durations))
	for i := range grid {
		grid[i] = make([]float64, len(periods))
	}
	for _, p := range points {
		grid[durIdx[p.DurationMin]][perIdx[p.ReturnPeriodYr]] = p.IntensityMMHr
	}

	return &Curve{
		points:    points,
		durations: durations,
		periods:   periods,
		grid:      grid,
	}, nil
}

// NewPeelDefault returns a Curve pre-loaded with approximate IDF values
// representative of the Peel Region (southern Ontario). These are placeholder
// values based on Environment Canada Atlas of Canada IDF data for Toronto area.
//
// Peel Region will supply their actual IDF data per SoW Assumption 6.
// Replace this with NewFromTable(actualPeelData) when available.
func NewPeelDefault() *Curve {
	// intensities in mm/hr for durations (minutes) × return periods (years)
	// Source: Environment Canada IDF curves, Toronto/Pearson area
	points := []IDFPoint{
		// 5 min
		{5, 2, 87.6}, {5, 5, 114.0}, {5, 10, 132.0}, {5, 25, 156.0}, {5, 50, 174.0}, {5, 100, 192.0},
		// 10 min
		{10, 2, 65.4}, {10, 5, 85.2}, {10, 10, 99.0}, {10, 25, 117.0}, {10, 50, 130.8}, {10, 100, 144.0},
		// 15 min
		{15, 2, 53.2}, {15, 5, 69.2}, {15, 10, 80.4}, {15, 25, 95.2}, {15, 50, 106.4}, {15, 100, 117.2},
		// 30 min
		{30, 2, 35.6}, {30, 5, 46.4}, {30, 10, 54.0}, {30, 25, 64.0}, {30, 50, 71.6}, {30, 100, 79.2},
		// 60 min (1hr)
		{60, 2, 22.2}, {60, 5, 29.0}, {60, 10, 33.8}, {60, 25, 40.2}, {60, 50, 44.8}, {60, 100, 49.8},
		// 120 min (2hr)
		{120, 2, 14.2}, {120, 5, 18.6}, {120, 10, 21.6}, {120, 25, 25.8}, {120, 50, 28.8}, {120, 100, 31.8},
		// 360 min (6hr)
		{360, 2, 6.6}, {360, 5, 8.6}, {360, 10, 10.0}, {360, 25, 11.8}, {360, 50, 13.2}, {360, 100, 14.6},
		// 720 min (12hr)
		{720, 2, 3.8}, {720, 5, 5.0}, {720, 10, 5.8}, {720, 25, 6.8}, {720, 50, 7.6}, {720, 100, 8.4},
		// 1440 min (24hr)
		{1440, 2, 2.2}, {1440, 5, 2.8}, {1440, 10, 3.2}, {1440, 25, 3.8}, {1440, 50, 4.4}, {1440, 100, 4.8},
	}
	c, _ := NewFromTable(points)
	return c
}

// ReturnPeriod estimates the return period for a storm event.
//
// totalMM is the total rainfall in mm over the event.
// durationMin is the storm duration in minutes.
//
// The function finds the best-matching duration in the IDF table, computes
// the equivalent intensity, and returns the bracketing return period.
func (c *Curve) ReturnPeriod(totalMM float64, durationMin int) ReturnPeriodLabel {
	if totalMM <= 0 || durationMin <= 0 {
		return ReturnPeriodLabel{Lo: 0, Hi: 2, DurationMin: durationMin}
	}

	// intensity in mm/hr
	intensityMMHr := totalMM / float64(durationMin) * 60.0

	// find nearest duration bracket
	durIdx := nearestDuration(c.durations, durationMin)
	matchedDur := c.durations[durIdx]

	// find which return period bracket this intensity falls into
	row := c.grid[durIdx]

	// row is sorted by increasing period → increasing intensity
	// find where our intensity falls
	if intensityMMHr <= row[0] {
		return ReturnPeriodLabel{Lo: 0, Hi: c.periods[0], DurationMin: matchedDur}
	}
	if intensityMMHr >= row[len(row)-1] {
		return ReturnPeriodLabel{Lo: c.periods[len(c.periods)-1], Hi: 0, DurationMin: matchedDur}
	}

	for i := 0; i < len(row)-1; i++ {
		if intensityMMHr >= row[i] && intensityMMHr < row[i+1] {
			return ReturnPeriodLabel{
				Lo:          c.periods[i],
				Hi:          c.periods[i+1],
				DurationMin: matchedDur,
			}
		}
	}

	return ReturnPeriodLabel{Lo: c.periods[len(c.periods)-1], Hi: 0, DurationMin: matchedDur}
}

// BestDuration finds the storm duration (from available IDF durations) that
// maximises the return period for a given event.
//
// This is useful when the exact storm duration is uncertain — it finds the
// most conservative (highest return period) interpretation.
func (c *Curve) BestDuration(totalMM float64, candidateDurations []int) ReturnPeriodLabel {
	var best ReturnPeriodLabel
	bestLo := -1.0

	for _, d := range candidateDurations {
		rp := c.ReturnPeriod(totalMM, d)
		if rp.Lo > bestLo {
			bestLo = rp.Lo
			best = rp
		}
	}
	return best
}

// Intensities returns the intensity table for a given return period (years).
// Returns nil if the return period is not in the table.
func (c *Curve) Intensities(returnPeriodYr float64) []IDFPoint {
	perIdx := -1
	for i, p := range c.periods {
		if p == returnPeriodYr {
			perIdx = i
			break
		}
	}
	if perIdx < 0 {
		return nil
	}

	out := make([]IDFPoint, len(c.durations))
	for i, d := range c.durations {
		out[i] = IDFPoint{
			DurationMin:    d,
			ReturnPeriodYr: returnPeriodYr,
			IntensityMMHr:  c.grid[i][perIdx],
		}
	}
	return out
}

// --- helpers ---

func nearestDuration(durations []int, target int) int {
	best := 0
	bestDiff := abs(durations[0] - target)
	for i, d := range durations {
		diff := abs(d - target)
		if diff < bestDiff {
			bestDiff = diff
			best = i
		}
	}
	return best
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func sortedInts(m map[int]struct{}) []int {
	s := make([]int, 0, len(m))
	for k := range m {
		s = append(s, k)
	}
	sort.Ints(s)
	return s
}

func sortedFloats(m map[float64]struct{}) []float64 {
	s := make([]float64, 0, len(m))
	for k := range m {
		s = append(s, k)
	}
	sort.Float64s(s)
	return s
}

// --- power-law fit (alternative to table lookup) ---

// PowerLawParams holds coefficients for i = a / (d + b)^c
// where i = intensity (mm/hr), d = duration (minutes).
type PowerLawParams struct {
	A, B, C float64
}

// Intensity returns the rainfall intensity for a given duration.
func (p PowerLawParams) Intensity(durationMin float64) float64 {
	return p.A / math.Pow(durationMin+p.B, p.C)
}
