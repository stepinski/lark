// cmd/predict fits a logistic regression model using hydrologically-correct
// features (DWF excess + AMC) to predict P(overflow in next H hours).
//
// Hydrological approach per Peel SoW and standard practice:
//   - DWF baseline: expected dry weather depth by hour/weekday
//   - Excess: depth above DWF (what rain drives)
//   - AMC: Antecedent Moisture Condition (API with decay)
//   - Event prediction: P(overflow | current_excess, AMC, forecast_rain)
//
// Usage:
//
//	go run ./cmd/predict \
//	  -csv ./data/cavendish_2yr.csv \
//	  -depth 36843 -rain 21881 \
//	  -invert 940 -horizon 12 \
//	  -as-of "2025-04-02T18:00:00Z"
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
	"github.com/stepinski/lark/models/dwf"
	"github.com/stepinski/lark/models/logreg"
)

type obs struct {
	T        time.Time
	DepthMM  float64
	RainMM   float64
}

func main() {
	csvPath  := flag.String("csv", "", "FlowWorks CSV export (required)")
	depthCol := flag.String("depth", "36843", "Depth channel ID")
	rainCol  := flag.String("rain", "21881", "Rainfall channel ID")
	invert   := flag.Float64("invert", 940.0, "Overflow invert depth (mm)")
	horizon  := flag.Int("horizon", 12, "Event prediction horizon in hours")
	lat      := flag.Float64("lat", 43.55, "Site latitude")
	lon      := flag.Float64("lon", -79.65, "Site longitude")
	asOf     := flag.String("as-of", "", "Simulate as-of this UTC time (RFC3339)")
	outCSV   := flag.String("out", "forecast.csv", "Output CSV path")
	flag.Parse()

	if *csvPath == "" {
		log.Fatal("-csv required")
	}

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

	var trainRows []obs
	for _, r := range rows {
		if !r.T.After(originTime) {
			trainRows = append(trainRows, r)
		}
	}
	fmt.Printf("  %d observations up to %s\n",
		len(trainRows), trainRows[len(trainRows)-1].T.Format("2006-01-02 15:04"))

	// --- fit DWF baseline ---
	fmt.Println("→ fitting DWF baseline...")
	dwfObs := toDWFObs(trainRows)
	dwfModel := dwf.New(dwf.Config{
		DryRain6hrThreshold:  0.0,
		DryRain24hrThreshold: 2.0,
		MinDryObs:            2,
	})
	if err := dwfModel.Fit(dwfObs); err != nil {
		log.Fatalf("DWF fit: %v", err)
	}
	fmt.Println(" ", dwfModel.Summary())

	// --- build event prediction dataset ---
	fmt.Println("→ building event dataset (DWF excess + AMC features)...")
	horizonSteps := *horizon * 12
	X, y, timestamps := buildDataset(rows, originTime, dwfModel, *invert, horizonSteps)

	nPos := 0
	for _, label := range y {
		if label > 0.5 {
			nPos++
		}
	}
	nNeg := len(y) - nPos
	fmt.Printf("  %d samples: %d overflow, %d non-overflow  (ratio 1:%.0f)\n",
		len(y), nPos, nNeg, float64(nNeg)/math.Max(float64(nPos), 1))

	if nPos == 0 {
		log.Fatal("no overflow events in training data")
	}

	// --- fit logistic regression ---
	fmt.Println("→ fitting logistic regression...")
	posWeight := float64(nNeg) / float64(nPos)
	m := logreg.New(logreg.Config{
		L2Lambda:  1.0,
		MaxIter:   5000,
		PosWeight: posWeight,
		FeatureNames: []string{
			"excess_mm",      // depth above DWF baseline
			"amc_api",        // antecedent moisture (API)
			"rain_1hr_mm",    // recent 1hr rainfall
			"month_sin",
			"month_cos",
		},
	})
	if err := m.Fit(X, y); err != nil {
		log.Fatalf("logreg Fit: %v", err)
	}
	fmt.Println(m.Params().Summary())

	// --- in-sample validation ---
	fmt.Println("→ in-sample validation...")
	probs, _ := m.Predict(X)
	maxP, maxT := 0.0, time.Time{}
	overflowT := findFirstOverflow(rows, *invert)
	for i, label := range y {
		if label > 0.5 && probs[i] > maxP {
			maxP = probs[i]
			maxT = timestamps[i]
		}
	}
	if !maxT.IsZero() {
		lead := 0
		if !overflowT.IsZero() {
			lead = int(overflowT.Sub(maxT).Minutes())
		}
		fmt.Printf("  peak P(alarm) in overflow windows: %.3f at %s (%dmin lead)\n",
			maxP, maxT.Format("2006-01-02 15:04"), lead)
	}

	// --- fetch rain forecast ---
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	omClient := openmeteo.NewClient()
	var forecastRainMM float64

	if *asOf != "" {
		pts, err := omClient.HindcastForecast(ctx, *lat, *lon, originTime, *horizon+2)
		if err != nil {
			fmt.Printf("  ⚠ hindcast unavailable: %v\n", err)
		} else {
			for _, p := range pts {
				forecastRainMM += p.PrecipMM
			}
			fmt.Printf("  hindcast total rain: %.1fmm over %dhr\n", forecastRainMM, *horizon)
		}
	} else {
		forecastDays := (*horizon / 24) + 1
		if forecastDays > 2 {
			forecastDays = 2
		}
		pts, err := omClient.HourlyForecast(ctx, *lat, *lon, forecastDays)
		if err != nil {
			fmt.Printf("  ⚠ forecast unavailable: %v\n", err)
		} else {
			steps := *horizon * 12
			fiveMins := openmeteo.ResampleTo5Min(pts)
			if len(fiveMins) > steps {
				fiveMins = fiveMins[:steps]
			}
			for _, p := range fiveMins {
				forecastRainMM += p.PrecipMM
			}
			fmt.Printf("  forecast total rain: %.1fmm over %dhr\n", forecastRainMM, *horizon)
		}
	}

	// --- scenario predictions ---
	fmt.Println("\n→ running scenario predictions...")

	lastObs := trainRows[len(trainRows)-1]
	excess, _ := dwfModel.Excess(lastObs.T, lastObs.DepthMM)
	amc := computeAMC(trainRows, originTime, 288) // 24hr API
	rain1hr := rollingSum(trainRows, originTime, 12) // 1hr
	mSin, mCos := monthEncoding(originTime)

	fmt.Printf("  current state: depth=%.1fmm  excess=%.1fmm  AMC=%.3f  rain_1hr=%.1fmm\n",
		lastObs.DepthMM, excess, amc, rain1hr)

	scenarios := openmeteo.StandardScenarios()

	type result struct {
		name   string
		rainMM float64
		pAlarm float64
	}
	var results []result

	for _, sc := range scenarios {
		rainMM := sc.MMPerHr * float64(*horizon)
		// blend forecast rain into AMC
		projectedAMC := amc + rainMM*0.1
		features := []float64{excess, projectedAMC, rain1hr + rainMM*0.1, mSin, mCos}
		p, _ := m.PredictOne(features)
		results = append(results, result{sc.Name, rainMM, p})
	}

	// actual forecast
	if forecastRainMM > 0 {
		projectedAMC := amc + forecastRainMM*0.1
		features := []float64{excess, projectedAMC, rain1hr + forecastRainMM*0.1, mSin, mCos}
		p, _ := m.PredictOne(features)
		results = append(results, result{"forecast", forecastRainMM, p})
	}

	fmt.Printf("\n  %-12s %10s %12s\n", "Scenario", "Rain (mm)", "P(overflow)")
	fmt.Println("  " + strings.Repeat("-", 38))
	for _, r := range results {
		alarm := ""
		if r.pAlarm > 0.5 {
			alarm = " ⚠ ALARM"
		} else if r.pAlarm > 0.2 {
			alarm = " ! elevated"
		}
		fmt.Printf("  %-12s %10.1f %12.3f%s\n", r.name, r.rainMM, r.pAlarm, alarm)
	}

	// --- write CSV ---
	f, _ := os.Create(*outCSV)
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{"timestamp", "scenario", "forecast_rain_mm", "p_overflow"})
	for _, r := range results {
		_ = w.Write([]string{
			originTime.Format("2006-01-02T15:04:05Z"),
			r.name,
			strconv.FormatFloat(r.rainMM, 'f', 2, 64),
			strconv.FormatFloat(r.pAlarm, 'f', 4, 64),
		})
	}
	w.Flush()
	fmt.Printf("\n✓ forecast written to %s\n", *outCSV)
}

// buildDataset creates (features, label) pairs using hydrological features.
func buildDataset(rows []obs, origin time.Time, dwfModel *dwf.Model,
	invert float64, horizonSteps int) ([][]float64, []float64, []time.Time) {

	n := len(rows)
	var X [][]float64
	var y []float64
	var ts []time.Time

	for i := 0; i < n; i++ {
		if rows[i].T.After(origin) {
			break
		}
		if i+horizonSteps >= n {
			continue
		}

		// label: overflow in next horizonSteps
		label := 0.0
		for j := i + 1; j <= i+horizonSteps && j < n; j++ {
			if rows[j].DepthMM >= invert {
				label = 1.0
				break
			}
		}

		// features
		excess, _ := dwfModel.Excess(rows[i].T, rows[i].DepthMM)
		amc := computeAMC(rows[:i+1], rows[i].T, 288)
		rain1hr := rollingSum(rows[:i+1], rows[i].T, 12)
		mSin, mCos := monthEncoding(rows[i].T)

		X = append(X, []float64{excess, amc, rain1hr, mSin, mCos})
		y = append(y, label)
		ts = append(ts, rows[i].T)
	}
	return X, y, ts
}

func computeAMC(rows []obs, before time.Time, steps int) float64 {
	rain := make([]float64, 0, steps)
	for i := len(rows) - 1; i >= 0 && len(rain) < steps; i-- {
		if rows[i].T.Before(before) || rows[i].T.Equal(before) {
			rain = append(rain, rows[i].RainMM)
		}
	}
	// reverse to chronological order
	for i, j := 0, len(rain)-1; i < j; i, j = i+1, j-1 {
		rain[i], rain[j] = rain[j], rain[i]
	}
	return dwf.API(rain, 0.85)
}

func rollingSum(rows []obs, before time.Time, steps int) float64 {
	var s float64
	count := 0
	for i := len(rows) - 1; i >= 0 && count < steps; i-- {
		if rows[i].T.Before(before) || rows[i].T.Equal(before) {
			s += rows[i].RainMM
			count++
		}
	}
	return s
}

func monthEncoding(t time.Time) (sin, cos float64) {
	m := float64(t.Month()-1) / 12.0 * 2 * math.Pi
	return math.Sin(m), math.Cos(m)
}

func findFirstOverflow(rows []obs, invert float64) time.Time {
	for _, r := range rows {
		if r.DepthMM >= invert {
			return r.T
		}
	}
	return time.Time{}
}

func toDWFObs(rows []obs) []dwf.Obs {
	// compute rolling rain sums for DWF fitting
	out := make([]dwf.Obs, len(rows))
	for i, r := range rows {
		rain6 := 0.0
		rain24 := 0.0
		for j := i - 1; j >= 0 && j >= i-72; j-- {
			rain6 += rows[j].RainMM
		}
		for j := i - 1; j >= 0 && j >= i-288; j-- {
			rain24 += rows[j].RainMM
		}
		out[i] = dwf.Obs{T: r.T, DepthMM: r.DepthMM, Rain6hr: rain6, Rain24hr: rain24}
	}
	return out
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
		rows = append(rows, obs{T: ts, DepthMM: dv, RainMM: rv})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].T.Before(rows[j].T) })
	return rows, nil
}
