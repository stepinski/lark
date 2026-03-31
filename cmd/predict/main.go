// cmd/predict loads a trained TAR model from a CSV, fetches rain forecast
// from Open-Meteo, and produces 48-hour overflow probability forecasts
// under multiple rain scenarios.
//
// Usage — live forecast:
//
//	go run ./cmd/predict \
//	  -csv cavendish_2yr.csv \
//	  -depth 36843 -rain 21881 \
//	  -invert 940 -fullpipe 250 \
//	  -lat 43.55 -lon -79.65 \
//	  -horizon 48 \
//	  -out forecast.csv
//
// Usage — historical validation (what would we have predicted on Apr 2?):
//
//	go run ./cmd/predict \
//	  -csv cavendish_2yr.csv \
//	  -depth 36843 -rain 21881 \
//	  -invert 940 -fullpipe 250 \
//	  -lat 43.55 -lon -79.65 \
//	  -horizon 12 \
//	  -as-of "2025-04-02T18:00:00Z" \
//	  -out hindcast.csv
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
	"github.com/stepinski/lark/models/tar"
	"github.com/stepinski/lark/models/threshold"
)

// obs is a single aligned depth+rain observation.
type obs struct {
	T     time.Time
	Depth float64
	Rain  float64
}

// forecastRow is one timestep in the output.
type forecastRow struct {
	T        time.Time
	Horizon  int // minutes ahead
	Scenario string
	RainMM   float64
	Forecast float64
	PAlarm   float64
	State    threshold.State
}

func main() {
	csvPath  := flag.String("csv", "", "FlowWorks CSV export (required)")
	depthCol := flag.String("depth", "36843", "Depth channel ID")
	rainCol  := flag.String("rain", "21881", "Rainfall channel ID")
	invert   := flag.Float64("invert", 940.0, "Overflow invert depth (mm)")
	fullpipe := flag.Float64("fullpipe", 250.0, "Full pipe depth (mm)")
	lat      := flag.Float64("lat", 43.55, "Site latitude")
	lon      := flag.Float64("lon", -79.65, "Site longitude")
	horizon  := flag.Int("horizon", 48, "Forecast horizon in hours (max 48)")
	asOf     := flag.String("as-of", "", "Simulate forecast as of this UTC time (e.g. 2025-04-02T18:00:00Z). Empty = now.")
	outCSV   := flag.String("out", "forecast.csv", "Output CSV path")
	flag.Parse()

	if *csvPath == "" {
		log.Fatal("-csv required")
	}
	if *horizon < 1 || *horizon > 48 {
		log.Fatal("-horizon must be 1-48")
	}

	// --- determine forecast origin time ---
	var originTime time.Time
	if *asOf != "" {
		var err error
		originTime, err = time.Parse(time.RFC3339, *asOf)
		if err != nil {
			log.Fatalf("invalid -as-of: %v (use RFC3339 e.g. 2025-04-02T18:00:00Z)", err)
		}
	} else {
		originTime = time.Now().UTC()
	}

	fmt.Printf("→ forecast origin: %s\n", originTime.Format("2006-01-02 15:04 UTC"))
	fmt.Printf("  horizon: %dhr (%d steps at 5-min)\n", *horizon, *horizon*12)

	// --- load and fit model on data up to origin time ---
	fmt.Println("\n→ loading and fitting model...")
	rows, err := loadCSV(*csvPath, "ch_"+*depthCol, "ch_"+*rainCol)
	if err != nil {
		log.Fatalf("load: %v", err)
	}

	// use only data up to origin time for fitting
	var trainRows []obs
	for _, r := range rows {
		if !r.T.After(originTime) {
			trainRows = append(trainRows, r)
		}
	}
	if len(trainRows) == 0 {
		log.Fatal("no training data before origin time")
	}

	fmt.Printf("  training on %d observations (up to %s)\n",
		len(trainRows), trainRows[len(trainRows)-1].T.Format("2006-01-02 15:04"))

	rainLag := []int{5}
	trainY, trainExog := buildFeatures(trainRows, rainLag)

	m, err := tar.New(tar.Config{
		P:               2,
		ExogLags:        [][]int{rainLag},
		DelayCandidates: []int{1, 2, 3},
	})
	if err != nil {
		log.Fatalf("tar.New: %v", err)
	}
	if err := m.Fit(trainY, trainExog); err != nil {
		log.Fatalf("tar.Fit: %v", err)
	}

	tp := m.Params()
	fmt.Println(tp.Summary("  Site"))

	// current depth state
	lastObs := trainRows[len(trainRows)-1]
	fmt.Printf("\n  current depth: %.1f mm (at %s)\n",
		lastObs.Depth, lastObs.T.Format("15:04 UTC"))

	siteCfg := threshold.SiteConfig{
		PipeFullDepth:        *fullpipe,
		BottomOverflowInvert: *invert,
	}

	horizonSteps := *horizon * 12 // 5-min steps

	// --- fetch rain forecasts ---
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	fmt.Println("\n→ fetching rain forecasts...")

	// scenarios: always run these
	scenarios := openmeteo.StandardScenarios()

	// add actual forecast from Open-Meteo
	var actualRain []openmeteo.HourlyPoint
	omClient := openmeteo.NewClient()

	if *asOf != "" {
		// historical validation mode
		fmt.Printf("  fetching hindcast for %s...\n", originTime.Format("2006-01-02 15:04 UTC"))
		pts, err := omClient.HindcastForecast(ctx, *lat, *lon, originTime, *horizon+2)
		if err != nil {
			fmt.Printf("  ⚠ hindcast unavailable: %v\n  using scenarios only\n", err)
		} else {
			actualRain = openmeteo.ResampleTo5Min(pts)
			fmt.Printf("  ✓ hindcast: %d hourly points → %d 5-min points\n",
				len(pts), len(actualRain))
		}
	} else {
		// live forecast mode
		fmt.Printf("  fetching live forecast for %.4f°N %.4f°E...\n", *lat, *lon)
		forecastDays := (*horizon / 24) + 1
		if forecastDays > 2 {
			forecastDays = 2
		}
		pts, err := omClient.HourlyForecast(ctx, *lat, *lon, forecastDays)
		if err != nil {
			fmt.Printf("  ⚠ forecast unavailable: %v\n  using scenarios only\n", err)
		} else {
			actualRain = openmeteo.ResampleTo5Min(pts)
			// trim to horizon
			if len(actualRain) > horizonSteps {
				actualRain = actualRain[:horizonSteps]
			}
			fmt.Printf("  ✓ forecast: %d hourly points → %d 5-min points\n",
				len(pts), len(actualRain))
		}
	}

	// --- run scenario forecasts ---
	fmt.Println("\n→ running scenario forecasts...")

	var allRows []forecastRow

	// synthetic scenarios
	for _, sc := range scenarios {
		rain := openmeteo.SyntheticForecast(originTime, horizonSteps, sc.MMPerHr)
		rows := runScenario(m, siteCfg, originTime, rain, sc.Name, *invert, horizonSteps)
		allRows = append(allRows, rows...)
		firstAlarm := firstAlarmHour(rows, 0.5)
		fmt.Printf("  %-10s (%.1fmm/hr): %s\n",
			sc.Name, sc.MMPerHr, alarmSummary(firstAlarm))
	}

	// actual forecast scenario
	if len(actualRain) > 0 {
		rows := runScenario(m, siteCfg, originTime, actualRain, "forecast", *invert, horizonSteps)
		allRows = append(allRows, rows...)
		firstAlarm := firstAlarmHour(rows, 0.5)
		totalRain := 0.0
		for _, p := range actualRain {
			totalRain += p.PrecipMM
		}
		fmt.Printf("  %-10s (%.1fmm total): %s\n",
			"forecast", totalRain, alarmSummary(firstAlarm))
	}

	// --- print summary table ---
	fmt.Printf("\n  Overflow probability summary (threshold=940mm, P>50%% = alarm):\n")
	fmt.Printf("  %-12s %6s %6s %8s %8s %8s %8s\n",
		"scenario", "6hr", "12hr", "18hr", "24hr", "36hr", "48hr")
	fmt.Println("  " + strings.Repeat("-", 68))

	scenarioNames := []string{"none", "light", "moderate", "heavy"}
	if len(actualRain) > 0 {
		scenarioNames = append(scenarioNames, "forecast")
	}

	byScenario := groupByScenario(allRows)
	for _, name := range scenarioNames {
		rows := byScenario[name]
		fmt.Printf("  %-12s %6.3f %6.3f %8.3f %8.3f %8.3f %8.3f\n",
			name,
			maxPAlarmAtHorizon(rows, 6*12),
			maxPAlarmAtHorizon(rows, 12*12),
			maxPAlarmAtHorizon(rows, 18*12),
			maxPAlarmAtHorizon(rows, 24*12),
			maxPAlarmAtHorizon(rows, 36*12),
			maxPAlarmAtHorizon(rows, 48*12),
		)
	}

	// --- write CSV ---
	if err := writeCSV(*outCSV, allRows); err != nil {
		log.Fatalf("write: %v", err)
	}
	fmt.Printf("\n✓ forecast written to %s\n", *outCSV)
}

// runScenario runs the TAR model recursively over horizonSteps using the
// provided rain points as exogenous input.
func runScenario(m *tar.Model, cfg threshold.SiteConfig, origin time.Time,
	rain []openmeteo.HourlyPoint, name string, invert float64, horizonSteps int) []forecastRow {

	rows := make([]forecastRow, 0, horizonSteps)

	// build rain slice aligned to 5-min steps
	rainVals := make([]float64, horizonSteps)
	for i, p := range rain {
		if i < horizonSteps {
			rainVals[i] = p.PrecipMM
		}
	}

	// recursive multi-step: feed forecast back as input
	// futureExog for each step uses the lagged rain value
	hist := make([]float64, 0, horizonSteps)

	// we need the last few history points for AR lags
	// clone the model state by running predictions step by step
	prevForecast := 0.0

	for step := 0; step < horizonSteps; step++ {
		// rain at lag 5 for this step
		rainAtLag5 := 0.0
		if step-5 >= 0 {
			rainAtLag5 = rainVals[step-5]
		}

		futureExog := [][]float64{{rainAtLag5}}

		fc, err := m.Predict(1, futureExog)
		forecast := prevForecast
		if err == nil && len(fc) > 0 {
			forecast = fc[0]
			if forecast < 0 {
				forecast = 0
			}
		}

		probs, _ := m.PredictProba(1, futureExog, invert)
		pAlarm := 0.0
		if len(probs) > 0 {
			pAlarm = probs[0]
		}

		// determine state using forecast depth
		recentOverflow := false
		for _, h := range hist {
			if h >= invert {
				recentOverflow = true
				break
			}
		}
		state := threshold.Classify(forecast, cfg, recentOverflow)

		t := origin.Add(time.Duration(step*5) * time.Minute)
		rows = append(rows, forecastRow{
			T:        t,
			Horizon:  (step + 1) * 5,
			Scenario: name,
			RainMM:   rainVals[step],
			Forecast: forecast,
			PAlarm:   pAlarm,
			State:    state,
		})

		// feed forecast back into model for next step
		_, _ = m.Step(forecast, []float64{rainAtLag5})
		hist = append(hist, forecast)
		prevForecast = forecast
	}

	return rows
}

// --- helpers ---

func firstAlarmHour(rows []forecastRow, threshold float64) int {
	for _, r := range rows {
		if r.PAlarm >= threshold {
			return r.Horizon / 60
		}
	}
	return -1
}

func alarmSummary(hour int) string {
	if hour < 0 {
		return "no alarm in horizon"
	}
	return fmt.Sprintf("⚠ P(overflow)>50%% at +%dhr", hour)
}

func maxPAlarmAtHorizon(rows []forecastRow, maxStep int) float64 {
	var max float64
	for _, r := range rows {
		if r.Horizon <= maxStep*5 {
			if r.PAlarm > max {
				max = r.PAlarm
			}
		}
	}
	return max
}

func groupByScenario(rows []forecastRow) map[string][]forecastRow {
	m := make(map[string][]forecastRow)
	for _, r := range rows {
		m[r.Scenario] = append(m[r.Scenario], r)
	}
	return m
}

func writeCSV(path string, rows []forecastRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	_ = w.Write([]string{
		"timestamp", "horizon_min", "scenario",
		"rain_mm", "forecast_mm", "p_alarm", "state",
	})
	for _, r := range rows {
		_ = w.Write([]string{
			r.T.Format("2006-01-02T15:04:05Z"),
			strconv.Itoa(r.Horizon),
			r.Scenario,
			ff(r.RainMM),
			ff(r.Forecast),
			ff(r.PAlarm),
			r.State.String(),
		})
	}
	w.Flush()
	return w.Error()
}

func ff(v float64) string {
	if math.IsNaN(v) {
		return ""
	}
	return strconv.FormatFloat(v, 'f', 4, 64)
}

// --- data loading (same as cmd/fit) ---

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

func buildFeatures(rows []obs, rainLags []int) (y []float64, exog [][]float64) {
	n := len(rows)
	y = make([]float64, n)
	exog = make([][]float64, len(rainLags))
	for i := range exog {
		exog[i] = make([]float64, n)
	}
	for i, r := range rows {
		y[i] = r.Depth
		if r.Depth < 0 {
			y[i] = 0
		}
		for j, lag := range rainLags {
			if i-lag >= 0 {
				exog[j][i] = rows[i-lag].Rain
			}
		}
	}
	return
}
