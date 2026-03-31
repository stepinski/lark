// cmd/predict fits a logistic regression model to predict P(overflow in next
// H hours) from antecedent rainfall and current depth conditions, then
// produces 48-hour forecasts under multiple rain scenarios.
//
// This replaces the failed recursive TAR depth forecasting approach.
// The key insight: the April 3 2025 overflow was triggered by extreme
// rainfall intensity, not predictable from depth autoregression.
//
// Usage — historical validation:
//
//	go run ./cmd/predict \
//	  -csv ./data/cavendish_2yr.csv \
//	  -depth 36843 -rain 21881 \
//	  -invert 940 -horizon 12 \
//	  -as-of "2025-04-02T18:00:00Z" \
//	  -out hindcast.csv
//
// Usage — live forecast:
//
//	go run ./cmd/predict \
//	  -csv ./data/cavendish_2yr.csv \
//	  -depth 36843 -rain 21881 \
//	  -invert 940 -horizon 48 \
//	  -lat 43.55 -lon -79.65 \
//	  -out forecast.csv
package main

import (
	"context"
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/stepinski/lark/datasource/openmeteo"
	"github.com/stepinski/lark/models/logreg"
)

type obs struct {
	T     time.Time
	Depth float64
	Rain  float64
}

type forecastRow struct {
	T        time.Time
	Scenario string
	RainMM   float64 // rain in this window
	PAlarm   float64 // P(overflow in horizon)
}

func main() {
	csvPath  := flag.String("csv", "", "FlowWorks CSV export (required)")
	depthCol := flag.String("depth", "36843", "Depth channel ID")
	rainCol  := flag.String("rain", "21881", "Rainfall channel ID")
	invert   := flag.Float64("invert", 940.0, "Overflow invert depth (mm)")
	horizon  := flag.Int("horizon", 12, "Event prediction horizon in hours")
	lat      := flag.Float64("lat", 43.55, "Site latitude")
	lon      := flag.Float64("lon", -79.65, "Site longitude")
	asOf     := flag.String("as-of", "", "Simulate as-of this UTC time (RFC3339). Empty = now.")
	outCSV   := flag.String("out", "forecast.csv", "Output CSV path")
	flag.Parse()

	if *csvPath == "" {
		log.Fatal("-csv required")
	}

	// --- determine origin time ---
	var originTime time.Time
	if *asOf != "" {
		var err error
		originTime, err = time.Parse(time.RFC3339, *asOf)
		if err != nil {
			log.Fatalf("invalid -as-of: %v", err)
		}
	} else {
		originTime = time.Now().UTC()
	}
	fmt.Printf("→ forecast origin: %s\n", originTime.Format("2006-01-02 15:04 UTC"))
	fmt.Printf("  event horizon: %dhr\n\n", *horizon)

	// --- load data ---
	fmt.Println("→ loading data...")
	rows, err := loadCSV(*csvPath, "ch_"+*depthCol, "ch_"+*rainCol)
	if err != nil {
		log.Fatalf("load: %v", err)
	}

	// split at origin
	var trainRows, futureRows []obs
	for _, r := range rows {
		if !r.T.After(originTime) {
			trainRows = append(trainRows, r)
		} else {
			futureRows = append(futureRows, r)
		}
	}
	fmt.Printf("  %d training observations (up to %s)\n",
		len(trainRows), trainRows[len(trainRows)-1].T.Format("2006-01-02 15:04"))

	horizonSteps := *horizon * 12 // 5-min steps

	// --- build training dataset ---
	fmt.Println("→ building event prediction dataset...")
	X, y, timestamps := buildEventDataset(rows, originTime, *invert, horizonSteps)

	nPos := 0
	for _, label := range y {
		if label > 0.5 {
			nPos++
		}
	}
	nNeg := len(y) - nPos
	fmt.Printf("  %d samples: %d overflow windows, %d non-overflow windows\n",
		len(y), nPos, nNeg)

	if nPos == 0 {
		log.Fatal("no overflow events in training data — cannot fit model")
	}

	// class weight: ratio of negative to positive
	posWeight := float64(nNeg) / float64(nPos)
	fmt.Printf("  class weight for positives: %.1f\n", posWeight)

	// --- fit logistic regression ---
	fmt.Println("→ fitting logistic regression...")
	m := logreg.New(logreg.Config{
		L2Lambda:  1.0, // strong regularisation: only 1 overflow event
		MaxIter:   5000,
		PosWeight: posWeight,
		FeatureNames: []string{
			"rain_6hr_mm",
			"rain_24hr_mm",
			"depth_mm",
			"month_sin",
			"month_cos",
		},
	})
	if err := m.Fit(X, y); err != nil {
		log.Fatalf("Fit: %v", err)
	}

	fmt.Println(m.Params().Summary())

	// --- validate on training data ---
	fmt.Println("→ in-sample validation (looking for overflow windows)...")
	probs, _ := m.Predict(X)
	validateInSample(timestamps, y, probs, trainRows, *invert)

	// --- fetch actual rain forecast ---
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	omClient := openmeteo.NewClient()
	var actualRainPts []openmeteo.HourlyPoint

	if *asOf != "" {
		fmt.Printf("\n→ fetching hindcast for %s...\n", originTime.Format("2006-01-02 15:04 UTC"))
		pts, err := omClient.HindcastForecast(ctx, *lat, *lon, originTime, *horizon+2)
		if err != nil {
			fmt.Printf("  ⚠ hindcast unavailable: %v\n", err)
		} else {
			actualRainPts = pts
			totalRain := 0.0
			for _, p := range pts {
				totalRain += p.PrecipMM
			}
			fmt.Printf("  ✓ %d hourly points, total %.1fmm\n", len(pts), totalRain)
		}
	} else {
		fmt.Printf("\n→ fetching live forecast (%.4f°N %.4f°E)...\n", *lat, *lon)
		forecastDays := (*horizon / 24) + 1
		if forecastDays > 2 {
			forecastDays = 2
		}
		pts, err := omClient.HourlyForecast(ctx, *lat, *lon, forecastDays)
		if err != nil {
			fmt.Printf("  ⚠ forecast unavailable: %v\n", err)
		} else {
			actualRainPts = pts
			totalRain := 0.0
			for _, p := range pts {
				totalRain += p.PrecipMM
			}
			fmt.Printf("  ✓ %d hourly points, total %.1fmm\n", len(pts), totalRain)
		}
	}

	// --- run scenario predictions ---
	fmt.Println("\n→ running scenario predictions...")

	currentDepth := trainRows[len(trainRows)-1].Depth
	rain6hr := rollingRainSum(trainRows, originTime, 6*12)
	rain24hr := rollingRainSum(trainRows, originTime, 24*12)
	monthSin, monthCos := monthEncoding(originTime)

	fmt.Printf("  current state: depth=%.1fmm  rain_6hr=%.1fmm  rain_24hr=%.1fmm\n",
		currentDepth, rain6hr, rain24hr)

	scenarios := openmeteo.StandardScenarios()

	type scenarioResult struct {
		name     string
		rainMM   float64
		pAlarm   float64
	}
	var results []scenarioResult

	for _, sc := range scenarios {
		forecastRainMM := sc.MMPerHr * float64(*horizon)
		features := []float64{
			rain6hr + forecastRainMM*0.25, // blend antecedent + forecast
			rain24hr + forecastRainMM,
			currentDepth,
			monthSin,
			monthCos,
		}
		p, _ := m.PredictOne(features)
		results = append(results, scenarioResult{sc.Name, forecastRainMM, p})
		fmt.Printf("  %-10s  forecast_rain=%.1fmm  P(overflow)=%.3f\n",
			sc.Name, forecastRainMM, p)
	}

	// actual forecast scenario
	if len(actualRainPts) > 0 {
		totalForecastRain := 0.0
		for _, p := range actualRainPts {
			totalForecastRain += p.PrecipMM
		}
		features := []float64{
			rain6hr + totalForecastRain*0.25,
			rain24hr + totalForecastRain,
			currentDepth,
			monthSin,
			monthCos,
		}
		p, _ := m.PredictOne(features)
		results = append(results, scenarioResult{"forecast", totalForecastRain, p})
		fmt.Printf("  %-10s  forecast_rain=%.1fmm  P(overflow)=%.3f\n",
			"forecast", totalForecastRain, p)
	}

	// --- summary ---
	fmt.Printf("\n  %-12s %12s %10s\n", "Scenario", "Rain (mm)", "P(overflow)")
	fmt.Println("  " + strings.Repeat("-", 38))
	for _, r := range results {
		alarm := ""
		if r.pAlarm > 0.5 {
			alarm = " ⚠ ALARM"
		} else if r.pAlarm > 0.2 {
			alarm = " ! elevated"
		}
		fmt.Printf("  %-12s %12.1f %10.3f%s\n", r.name, r.rainMM, r.pAlarm, alarm)
	}

	// --- write CSV ---
	var csvRows []forecastRow
	for _, r := range results {
		csvRows = append(csvRows, forecastRow{
			T:        originTime,
			Scenario: r.name,
			RainMM:   r.rainMM,
			PAlarm:   r.pAlarm,
		})
	}
	if err := writeCSV(*outCSV, csvRows); err != nil {
		log.Fatalf("write: %v", err)
	}
	fmt.Printf("\n✓ forecast written to %s\n", *outCSV)
}

// buildEventDataset creates (features, label) pairs for each 5-min timestep
// up to originTime. Labels look forward in the full rows slice so that
// timesteps before origin can be labelled 1 if overflow occurs in their
// horizon even if that overflow is after origin.
// Features: rain_6hr, rain_24hr, current_depth, month_sin, month_cos.
func buildEventDataset(rows []obs, origin time.Time, invert float64, horizonSteps int) ([][]float64, []float64, []time.Time) {
	n := len(rows)
	var X [][]float64
	var y []float64
	var ts []time.Time

	for i := 0; i+horizonSteps < n; i++ {
		// only use timesteps up to origin as feature rows
		if rows[i].T.After(origin) {
			break
		}
		// check if overflow occurs in next horizonSteps
		label := 0.0
		for j := i + 1; j <= i+horizonSteps && j < n; j++ {
			if rows[j].Depth >= invert {
				label = 1.0
				break
			}
		}

		// features
		rain6hr := rollingSum(rows, i, 6*12)
		rain24hr := rollingSum(rows, i, 24*12)
		depth := rows[i].Depth
		if depth < 0 {
			depth = 0
		}
		mSin, mCos := monthEncoding(rows[i].T)

		X = append(X, []float64{rain6hr, rain24hr, depth, mSin, mCos})
		y = append(y, label)
		ts = append(ts, rows[i].T)
	}
	return X, y, ts
}

func rollingSum(rows []obs, idx, steps int) float64 {
	var s float64
	start := idx - steps
	if start < 0 {
		start = 0
	}
	for i := start; i < idx; i++ {
		s += rows[i].Rain
	}
	return s
}

func rollingRainSum(rows []obs, before time.Time, steps int) float64 {
	var s float64
	count := 0
	for i := len(rows) - 1; i >= 0 && count < steps; i-- {
		if rows[i].T.Before(before) {
			s += rows[i].Rain
			count++
		}
	}
	return s
}

func monthEncoding(t time.Time) (sin, cos float64) {
	m := float64(t.Month()-1) / 12.0 * 2 * math.Pi
	return math.Sin(m), math.Cos(m)
}

func validateInSample(timestamps []time.Time, y, probs []float64, rows []obs, invert float64) {
	// find overflow windows
	for i, label := range y {
		if label > 0.5 {
			fmt.Printf("  overflow window at %s: P(alarm)=%.3f\n",
				timestamps[i].Format("2006-01-02 15:04"), probs[i])
		}
	}
	// find maximum P in the 6 hours before the overflow
	overflowT := time.Time{}
	for _, r := range rows {
		if r.Depth >= invert {
			overflowT = r.T
			break
		}
	}
	if !overflowT.IsZero() {
		window := overflowT.Add(-6 * time.Hour)
		maxP := 0.0
		maxT := time.Time{}
		for i, ts := range timestamps {
			if ts.After(window) && ts.Before(overflowT) {
				if probs[i] > maxP {
					maxP = probs[i]
					maxT = ts
				}
			}
		}
		if !maxT.IsZero() {
			lead := int(overflowT.Sub(maxT).Minutes())
			fmt.Printf("  max P(alarm) in 6hr before overflow: %.3f at %s (%dmin lead)\n",
				maxP, maxT.Format("15:04"), lead)
		}
	}
}

func writeCSV(path string, rows []forecastRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{"timestamp", "scenario", "forecast_rain_mm", "p_overflow"})
	for _, r := range rows {
		_ = w.Write([]string{
			r.T.Format("2006-01-02T15:04:05Z"),
			r.Scenario,
			strconv.FormatFloat(r.RainMM, 'f', 2, 64),
			strconv.FormatFloat(r.PAlarm, 'f', 4, 64),
		})
	}
	w.Flush()
	return w.Error()
}

func loadCSV(path, depthCol, rainCol string) ([]obs, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	headers, err := r.Read()
	if err != nil {
		return nil, err
	}
	depthIdx, rainIdx := -1, -1
	for i, h := range headers {
		if h == depthCol {
			depthIdx = i
		}
		if h == rainCol {
			rainIdx = i
		}
	}
	if depthIdx < 0 {
		return nil, fmt.Errorf("column %q not found", depthCol)
	}
	if rainIdx < 0 {
		return nil, fmt.Errorf("column %q not found", rainCol)
	}
	records, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	var rows []obs
	for _, rec := range records {
		ts, err := time.Parse("2006-01-02T15:04:05Z", rec[0])
		if err != nil {
			continue
		}
		if rec[depthIdx] == "" {
			continue
		}
		dv, err := strconv.ParseFloat(rec[depthIdx], 64)
		if err != nil {
			continue
		}
		rv := 0.0
		if rec[rainIdx] != "" {
			if v, err := strconv.ParseFloat(rec[rainIdx], 64); err == nil {
				rv = v
			}
		}
		rows = append(rows, obs{T: ts, Depth: dv, Rain: rv})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].T.Before(rows[j].T) })
	return rows, nil
}
