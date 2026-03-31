// cmd/fit loads a FlowWorks CSV export, preprocesses depth and rainfall,
// fits both SARIMAX and TAR models, and compares them on a held-out test
// period containing known overflow events.
//
// Usage:
//
//	go run ./cmd/fit \
//	  -csv cavendish_2yr.csv \
//	  -depth 36843 \
//	  -rain 21881 \
//	  -invert 940 \
//	  -fullpipe 250 \
//	  -test-from 2025-04-01 \
//	  -out results.csv
package main

import (
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

	"github.com/stepinski/lark/models/sarimax"
	"github.com/stepinski/lark/models/tar"
	"github.com/stepinski/lark/models/threshold"
)

// --- data types ---

type obs struct {
	T     time.Time
	Depth float64
	Rain  float64
}

type predRow struct {
	T            time.Time
	Actual       float64
	SARIMAXFcast float64
	SARIMAXAlarm float64
	TARFcast     float64
	TARAlarm     float64
	TARRegime    tar.RegimeID
	Rain         float64
	RainRule     bool
	State        threshold.State
}

func main() {
	csvPath  := flag.String("csv", "", "Path to FlowWorks CSV export (required)")
	depthCol := flag.String("depth", "36843", "Depth channel ID (ch_XXXXX)")
	rainCol  := flag.String("rain", "21881", "Rainfall channel ID (ch_XXXXX)")
	invert   := flag.Float64("invert", 940.0, "Bottom overflow invert depth (mm)")
	fullpipe := flag.Float64("fullpipe", 250.0, "Full pipe depth (mm)")
	testFrom := flag.String("test-from", "2025-04-01", "Start of test period (yyyy-MM-dd)")
	outCSV   := flag.String("out", "results.csv", "Output CSV path")
	maxGap   := flag.Duration("max-gap", 4*time.Hour, "Max gap before segment split")
	flag.Parse()

	if *csvPath == "" {
		log.Fatal("-csv required")
	}

	// --- 1. load and preprocess ---
	fmt.Println("→ loading CSV...")
	raw, err := loadCSV(*csvPath, "ch_"+*depthCol, "ch_"+*rainCol)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	fmt.Printf("  %d rows loaded\n", len(raw))

	clean := preprocess(raw, *maxGap)
	fmt.Printf("  %d rows after preprocessing\n", len(clean))

	// --- 2. train/test split ---
	splitTime, err := time.Parse("2006-01-02", *testFrom)
	if err != nil {
		log.Fatalf("invalid -test-from: %v", err)
	}
	splitTime = splitTime.UTC()
	train, test := splitAt(clean, splitTime)
	fmt.Printf("  train: %d rows (%s → %s)\n",
		len(train), train[0].T.Format("2006-01-02"), train[len(train)-1].T.Format("2006-01-02"))
	fmt.Printf("  test:  %d rows (%s → %s)\n",
		len(test), test[0].T.Format("2006-01-02"), test[len(test)-1].T.Format("2006-01-02"))

	// --- 3. features ---
	rainLag := []int{5} // 25min lag
	trainY, trainExog := buildFeatures(train, rainLag)
	_, testExog := buildFeatures(test, rainLag)

	siteCfg := threshold.SiteConfig{
		PipeFullDepth:        *fullpipe,
		BottomOverflowInvert: *invert,
	}

	// --- 4. fit SARIMAX(2,1,0) ---
	fmt.Println("\n→ fitting SARIMAX(2,1,0)...")
	sm, err := sarimax.New(sarimax.Order{P: 2, D: 1, Q: 0})
	if err != nil {
		log.Fatalf("sarimax.New: %v", err)
	}
	if err := sm.Fit(trainY, trainExog); err != nil {
		log.Fatalf("sarimax.Fit: %v", err)
	}
	sp := sm.Params()
	fmt.Printf("  AR=%v  Exog=%v  Intercept=%.4f\n",
		fmtFloats(sp.AR), fmtFloats(sp.Exog), sp.Intercept)

	// --- 5. fit TAR(2, exog lag=5) ---
	fmt.Println("→ fitting TAR(p=2, exog lag=5)...")
	tm, err := tar.New(tar.Config{
		P:               2,
		ExogLags:        [][]int{rainLag},
		DelayCandidates: []int{1, 2, 3},
	})
	if err != nil {
		log.Fatalf("tar.New: %v", err)
	}
	if err := tm.Fit(trainY, trainExog); err != nil {
		log.Fatalf("tar.Fit: %v", err)
	}
	tp := tm.Params()
	fmt.Println(tp.Summary("  Cavendish"))

	// --- 6. walk-forward validation ---
	fmt.Println("\n→ validating on test set...")
	results := walkForward(sm, tm, test, testExog, siteCfg, *invert, rainLag)

	sarimaxRMSE := computeRMSE(results, false)
	tarRMSE := computeRMSE(results, true)
	sarimaxLead := findAlarmLeadTime(results, *invert, false)
	tarLead := findAlarmLeadTime(results, *invert, true)

	fmt.Printf("\n  %-20s %8s %8s\n", "Metric", "SARIMAX", "TAR")
	fmt.Println("  " + strings.Repeat("-", 38))
	fmt.Printf("  %-20s %8.2f %8.2f  mm\n", "RMSE", sarimaxRMSE, tarRMSE)
	fmt.Printf("  %-20s %8d %8d  min\n", "Alarm lead time", sarimaxLead, tarLead)

	// --- 7. print overflow event window ---
	fmt.Println("\n  Overflow event window (2025-04-03 00:00–03:00 UTC):")
	fmt.Printf("  %-20s %8s %8s %6s %8s %6s %6s %6s %s\n",
		"timestamp", "actual", "sar_fc", "P(sar)", "tar_fc", "P(tar)", "regime", "rain", "state")
	fmt.Println("  " + strings.Repeat("-", 100))

	eventStart, _ := time.Parse(time.RFC3339, "2025-04-03T00:00:00Z")
	eventEnd, _   := time.Parse(time.RFC3339, "2025-04-03T03:00:00Z")
	for _, r := range results {
		if r.T.Before(eventStart) || r.T.After(eventEnd) {
			continue
		}
		overflow := ""
		if r.Actual >= *invert {
			overflow = "*** OVF"
		}
		rule := " "
		if r.RainRule {
			rule = "⚠"
		}
		fmt.Printf("  %-20s %8.1f %8.1f %6.3f %8.1f %6.3f %6s %6.1f %s %s\n",
			r.T.Format("2006-01-02 15:04"),
			r.Actual, r.SARIMAXFcast, r.SARIMAXAlarm,
			r.TARFcast, r.TARAlarm, r.TARRegime,
			r.Rain, rule, overflow)
	}

	// --- 8. write CSV ---
	if err := writeResults(*outCSV, results); err != nil {
		log.Fatalf("write: %v", err)
	}
	fmt.Printf("\n✓ results written to %s\n", *outCSV)
}

// --- walk-forward ---

func walkForward(sm *sarimax.Model, tm *tar.Model, test []obs,
	testExog [][]float64, cfg threshold.SiteConfig, invert float64,
	rainLag []int) []predRow {

	const ruleRainWindow = 12   // 1hr at 5-min
	const ruleRainThreshold = 3.0
	const ruleDepthThreshold = 100.0

	results := make([]predRow, 0, len(test))

	for i, o := range test {
		futureExog := make([][]float64, len(testExog))
		for k := range testExog {
			futureExog[k] = []float64{testExog[k][i]}
		}

		// SARIMAX forecast
		sarFcast, sarAlarm := math.NaN(), 0.0
		if fc, err := sm.Predict(1, futureExog); err == nil && len(fc) > 0 {
			sarFcast = fc[0]
		}
		if pr, err := sm.PredictProba(1, futureExog, invert); err == nil && len(pr) > 0 {
			sarAlarm = pr[0]
		}

		// TAR forecast
		tarFcast, tarAlarm := math.NaN(), 0.0
		if fc, err := tm.Predict(1, futureExog); err == nil && len(fc) > 0 {
			tarFcast = fc[0]
		}
		if pr, err := tm.PredictProba(1, futureExog, invert); err == nil && len(pr) > 0 {
			tarAlarm = pr[0]
		}

		// TAR regime
		tarRegime := tm.RegimeAt(len(test) + i)

		// rain rule
		rainSum := 0.0
		for k := i - ruleRainWindow; k < i; k++ {
			if k >= 0 {
				rainSum += test[k].Rain
			}
		}
		ruleAlarm := rainSum >= ruleRainThreshold && o.Depth >= ruleDepthThreshold

		// recent overflow for orange state
		recentOverflow := false
		cutoff := o.T.Add(-48 * time.Hour)
		for k := i - 1; k >= 0; k-- {
			if test[k].T.Before(cutoff) {
				break
			}
			if test[k].Depth >= invert {
				recentOverflow = true
				break
			}
		}
		state := threshold.Classify(o.Depth, cfg, recentOverflow)

		results = append(results, predRow{
			T:            o.T,
			Actual:       o.Depth,
			SARIMAXFcast: sarFcast,
			SARIMAXAlarm: sarAlarm,
			TARFcast:     tarFcast,
			TARAlarm:     tarAlarm,
			TARRegime:    tarRegime,
			Rain:         o.Rain,
			RainRule:     ruleAlarm,
			State:        state,
		})

		// update both models with actual observation
		exogVals := make([]float64, len(testExog))
		for k := range testExog {
			exogVals[k] = testExog[k][i]
		}
		_, _ = sm.Step(o.Depth, exogVals)
		_, _ = tm.Step(o.Depth, exogVals)
	}

	return results
}

// --- metrics ---

func computeRMSE(results []predRow, useTAR bool) float64 {
	var sum float64
	var n int
	for _, r := range results {
		fc := r.SARIMAXFcast
		if useTAR {
			fc = r.TARFcast
		}
		if math.IsNaN(fc) {
			continue
		}
		d := r.Actual - fc
		sum += d * d
		n++
	}
	if n == 0 {
		return math.NaN()
	}
	return math.Sqrt(sum / float64(n))
}

func findAlarmLeadTime(results []predRow, invert float64, useTAR bool) int {
	overflowT := time.Time{}
	for _, r := range results {
		if r.Actual >= invert {
			overflowT = r.T
			break
		}
	}
	if overflowT.IsZero() {
		return 0
	}
	for _, r := range results {
		if r.T.After(overflowT) {
			break
		}
		alarm := r.SARIMAXAlarm
		if useTAR {
			alarm = r.TARAlarm
		}
		if alarm > 0.5 {
			return int(overflowT.Sub(r.T).Minutes())
		}
	}
	return 0
}

// --- I/O ---

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

func preprocess(rows []obs, _ time.Duration) []obs {
	out := make([]obs, len(rows))
	copy(out, rows)
	for i := range out {
		if out[i].Depth < 0 {
			out[i].Depth = 0
		}
	}
	return out
}

func splitAt(rows []obs, t time.Time) (train, test []obs) {
	for _, r := range rows {
		if r.T.Before(t) {
			train = append(train, r)
		} else {
			test = append(test, r)
		}
	}
	return
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
		for j, lag := range rainLags {
			if i-lag >= 0 {
				exog[j][i] = rows[i-lag].Rain
			}
		}
	}
	return
}

func writeResults(path string, results []predRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	_ = w.Write([]string{
		"timestamp", "actual_mm",
		"sarimax_forecast_mm", "sarimax_p_alarm",
		"tar_forecast_mm", "tar_p_alarm", "tar_regime",
		"rain_mm", "rain_rule", "state",
	})
	for _, r := range results {
		rule := "false"
		if r.RainRule {
			rule = "true"
		}
		_ = w.Write([]string{
			r.T.Format("2006-01-02T15:04:05Z"),
			ff(r.Actual), ff(r.SARIMAXFcast), ff(r.SARIMAXAlarm),
			ff(r.TARFcast), ff(r.TARAlarm), r.TARRegime.String(),
			ff(r.Rain), rule, r.State.String(),
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

func fmtFloats(v []float64) string {
	parts := make([]string, len(v))
	for i, f := range v {
		parts[i] = fmt.Sprintf("%.4f", f)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
