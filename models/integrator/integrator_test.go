package integrator_test

import (
	"testing"
	"time"

	"github.com/stepinski/lark/models/integrator"
)

var epoch = time.Date(2025, 4, 3, 1, 0, 0, 0, time.UTC)

func t(offsetMin int) time.Time {
	return epoch.Add(time.Duration(offsetMin) * time.Minute)
}

func pt(offsetMin int, depth, flow float64) integrator.Point {
	return integrator.Point{T: t(offsetMin), Depth: depth, Flow: flow}
}

var cfg = integrator.Config{
	InvertDepthMM:       940.0,
	MinEventDurationSec: 60,
	MaxGapSec:           600,
}

// --- DetectEvents ---

func TestDetectEvents_SingleEvent(t *testing.T) {
	points := []integrator.Point{
		pt(0, 800, 50),
		pt(5, 950, 100),  // overflow starts
		pt(10, 1000, 150),
		pt(15, 970, 120),
		pt(20, 930, 80),  // overflow ends
		pt(25, 800, 50),
	}
	events := integrator.DetectEvents(points, cfg)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	e := events[0]
	if e.PeakDepthMM != 1000 {
		t.Errorf("PeakDepth = %.1f, want 1000", e.PeakDepthMM)
	}
	if e.Active() {
		t.Error("event should be closed")
	}
}

func TestDetectEvents_ActiveEvent(t *testing.T) {
	points := []integrator.Point{
		pt(0, 800, 50),
		pt(5, 960, 110),
		pt(10, 1050, 180),
		pt(15, 1072, 200), // still overflowing at end of series
	}
	events := integrator.DetectEvents(points, cfg)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if !events[0].Active() {
		t.Error("expected active event (overflow still ongoing)")
	}
}

func TestDetectEvents_NoOverflow(t *testing.T) {
	points := []integrator.Point{
		pt(0, 100, 10),
		pt(5, 120, 12),
		pt(10, 90, 9),
	}
	events := integrator.DetectEvents(points, cfg)
	if len(events) != 0 {
		t.Errorf("got %d events, want 0", len(events))
	}
}

func TestDetectEvents_MultipleEvents(t *testing.T) {
	points := []integrator.Point{
		pt(0, 800, 50),
		pt(5, 960, 100),   // event 1 start
		pt(10, 970, 110),
		pt(15, 950, 90),
		pt(20, 800, 50),   // event 1 end
		pt(25, 700, 40),
		pt(35, 960, 110),  // event 2 start (gap = 15min = 900s >= maxGap 600s → split)
		pt(40, 980, 130),
		pt(45, 950, 120),
		pt(50, 800, 60),   // event 2 end
	}
	events := integrator.DetectEvents(points, cfg)
	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}
}

// --- Volume calculation ---

func TestDetectEvents_VolumeCalculation(t *testing.T) {
	// constant flow of 100 l/s for 10 minutes = 60,000 litres
	points := []integrator.Point{
		pt(0, 950, 100),
		pt(5, 960, 100),
		pt(10, 970, 100),
		pt(15, 800, 0),  // end
	}
	events := integrator.DetectEvents(points, cfg)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	// 2 intervals × 100 l/s × 300s = 60,000 L
	expectedL := 60000.0
	if events[0].VolumeL < expectedL*0.9 || events[0].VolumeL > expectedL*1.1 {
		t.Errorf("VolumeL = %.0f, want ~%.0f (±10%%)", events[0].VolumeL, expectedL)
	}
}

func TestEvent_VolumeKL(t *testing.T) {
	e := integrator.Event{VolumeL: 5000}
	if e.VolumeKL() != 5.0 {
		t.Errorf("VolumeKL = %.2f, want 5.0", e.VolumeKL())
	}
}

// --- Duration ---

func TestEvent_DurationString(t *testing.T) {
	e := integrator.Event{
		Start:           epoch,
		End:             epoch.Add(110 * time.Minute),
		DurationSeconds: 110 * 60,
	}
	s := e.DurationString()
	if s == "" || s == "active" {
		t.Errorf("DurationString = %q, expected formatted duration", s)
	}
	t.Logf("DurationString = %s", s)
}

func TestEvent_Active(t *testing.T) {
	active := integrator.Event{Start: epoch}
	closed := integrator.Event{Start: epoch, End: epoch.Add(time.Hour)}
	if !active.Active() {
		t.Error("expected Active() = true for zero End")
	}
	if closed.Active() {
		t.Error("expected Active() = false for non-zero End")
	}
}

// --- Rainfall window ---

func TestRainfallWindow_SoWDefinition(t *testing.T) {
	// event starts at epoch (01:05)
	// rainfall window: epoch-24hr to epoch+24hr
	event := integrator.Event{Start: epoch}

	rainPoints := []integrator.RainPoint{
		{T: epoch.Add(-25 * time.Hour), RainMM: 99.0}, // outside window
		{T: epoch.Add(-23 * time.Hour), RainMM: 2.0},  // inside
		{T: epoch.Add(-1 * time.Hour), RainMM: 5.0},   // inside
		{T: epoch, RainMM: 3.0},                        // inside (at start)
		{T: epoch.Add(6 * time.Hour), RainMM: 1.0},    // inside
		{T: epoch.Add(25 * time.Hour), RainMM: 99.0},  // outside window
	}

	total := integrator.RainfallWindow(event, rainPoints)
	want := 2.0 + 5.0 + 3.0 + 1.0
	if total != want {
		t.Errorf("RainfallWindow = %.1f, want %.1f", total, want)
	}
}

// --- EventsInRange ---

func TestEventsInRange(t *testing.T) {
	events := []integrator.Event{
		{Start: epoch.Add(-48 * time.Hour), End: epoch.Add(-47 * time.Hour)},
		{Start: epoch.Add(-2 * time.Hour), End: epoch.Add(-1 * time.Hour)},
		{Start: epoch, End: epoch.Add(time.Hour)},
		{Start: epoch.Add(48 * time.Hour), End: epoch.Add(49 * time.Hour)},
	}

	rangeStart := epoch.Add(-3 * time.Hour)
	rangeEnd := epoch.Add(2 * time.Hour)
	got := integrator.EventsInRange(events, rangeStart, rangeEnd)
	if len(got) != 2 {
		t.Errorf("EventsInRange = %d events, want 2", len(got))
	}
}

// --- TotalVolume ---

func TestTotalVolume(t *testing.T) {
	events := []integrator.Event{
		{VolumeL: 1000},
		{VolumeL: 2500},
		{VolumeL: 500},
	}
	got := integrator.TotalVolume(events)
	if got != 4000 {
		t.Errorf("TotalVolume = %.0f, want 4000", got)
	}
}

// --- April 3 2025 real event regression test ---

func TestApril3Event_Detection(t *testing.T) {
	// simplified version of the real event profile
	// overflow starts at t=65min (01:05 UTC = epoch+65min relative to 00:00)
	origin := time.Date(2025, 4, 3, 0, 0, 0, 0, time.UTC)
	tp := func(min int, depth, flow float64) integrator.Point {
		return integrator.Point{
			T:     origin.Add(time.Duration(min) * time.Minute),
			Depth: depth,
			Flow:  flow,
		}
	}

	points := []integrator.Point{
		tp(0, 84, 5), tp(5, 84, 5), tp(10, 83, 5),
		tp(35, 91, 8), tp(40, 102, 12), tp(45, 130, 20),
		tp(50, 238, 45), tp(55, 512, 90), tp(60, 906, 180),
		tp(65, 968, 200),  // overflow start
		tp(70, 1016, 210),
		tp(75, 1048, 215),
		tp(80, 1066, 218),
		tp(85, 1072, 220), // peak
		tp(90, 1069, 218),
		tp(110, 1000, 200),
		tp(115, 982, 195),
		tp(120, 964, 185),
		tp(125, 938, 170), // just below invert — event ends
		tp(130, 900, 150),
	}

	// Use larger maxGap for this test to handle the 25min gap in the profile
	aprilCfg := integrator.Config{
		InvertDepthMM:       940.0,
		MinEventDurationSec: 60,
		MaxGapSec:           1800, // 30min to handle gaps in the real profile
	}
	events := integrator.DetectEvents(points, aprilCfg)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1 for April 3 profile", len(events))
	}

	e := events[0]
	t.Logf("April 3 event: start=%s duration=%s peak=%.0fmm volume=%.0fL",
		e.Start.Format("15:04"), e.DurationString(), e.PeakDepthMM, e.VolumeL)

	if e.PeakDepthMM < 1000 {
		t.Errorf("PeakDepth = %.1f, expected ~1072", e.PeakDepthMM)
	}
	if e.VolumeL <= 0 {
		t.Error("VolumeL should be > 0")
	}
}
