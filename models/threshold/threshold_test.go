package threshold_test

import (
	"testing"

	"github.com/stepinski/lark/models/threshold"
)

// shared config matching Peel site geometry (depths in metres)
var stdConfig = threshold.SiteConfig{
	PipeFullDepth:        1.0, // 100% full at 1.0m
	BottomOverflowInvert: 1.5, // overflow begins at 1.5m
}

// --- State.String and IsAlarm ---

func TestState_String(t *testing.T) {
	cases := []struct {
		s    threshold.State
		want string
	}{
		{threshold.Green, "Green"},
		{threshold.Yellow, "Yellow"},
		{threshold.Orange, "Orange"},
		{threshold.Red, "Red"},
	}
	for _, tc := range cases {
		if got := tc.s.String(); got != tc.want {
			t.Errorf("State(%d).String() = %q, want %q", tc.s, got, tc.want)
		}
	}
}

func TestState_IsAlarm(t *testing.T) {
	if threshold.Green.IsAlarm() {
		t.Error("Green should not be alarm")
	}
	if threshold.Yellow.IsAlarm() {
		t.Error("Yellow should not be alarm")
	}
	if !threshold.Orange.IsAlarm() {
		t.Error("Orange should be alarm")
	}
	if !threshold.Red.IsAlarm() {
		t.Error("Red should be alarm")
	}
}

// --- SiteConfig.Validate ---

func TestSiteConfig_Validate_Valid(t *testing.T) {
	if err := stdConfig.Validate(); err != nil {
		t.Errorf("unexpected validation error: %v", err)
	}
}

func TestSiteConfig_Validate_ZeroPipeFullDepth(t *testing.T) {
	cfg := threshold.SiteConfig{PipeFullDepth: 0, BottomOverflowInvert: 1.5}
	if err := cfg.Validate(); err == nil {
		t.Error("expected error for zero PipeFullDepth")
	}
}

func TestSiteConfig_Validate_InvertBelowPipeFull(t *testing.T) {
	cfg := threshold.SiteConfig{PipeFullDepth: 1.5, BottomOverflowInvert: 1.0}
	if err := cfg.Validate(); err == nil {
		t.Error("expected error when BottomOverflowInvert <= PipeFullDepth")
	}
}

// --- Classify: OVF site state transitions ---

func TestClassify_Green(t *testing.T) {
	// depth below 100% full, no recent overflow
	got := threshold.Classify(0.8, stdConfig, false)
	if got != threshold.Green {
		t.Errorf("got %s, want Green", got)
	}
}

func TestClassify_Green_AtExactlyPipeFull(t *testing.T) {
	// depth == PipeFullDepth is NOT yellow per SoW (must be strictly above)
	got := threshold.Classify(1.0, stdConfig, false)
	if got != threshold.Green {
		t.Errorf("got %s, want Green at exactly PipeFullDepth", got)
	}
}

func TestClassify_Yellow(t *testing.T) {
	// depth above 100% but below overflow invert
	got := threshold.Classify(1.2, stdConfig, false)
	if got != threshold.Yellow {
		t.Errorf("got %s, want Yellow", got)
	}
}

func TestClassify_Yellow_JustBelowInvert(t *testing.T) {
	got := threshold.Classify(1.499, stdConfig, false)
	if got != threshold.Yellow {
		t.Errorf("got %s, want Yellow just below invert", got)
	}
}

func TestClassify_Red_AtInvert(t *testing.T) {
	// depth exactly at BottomOverflowInvert → Red
	got := threshold.Classify(1.5, stdConfig, false)
	if got != threshold.Red {
		t.Errorf("got %s, want Red at BottomOverflowInvert", got)
	}
}

func TestClassify_Red_AboveInvert(t *testing.T) {
	got := threshold.Classify(2.1, stdConfig, false)
	if got != threshold.Red {
		t.Errorf("got %s, want Red above invert", got)
	}
}

func TestClassify_Orange_RecentOverflow_CurrentlyNormal(t *testing.T) {
	// depth below pipe full, but recentOverflow flag set → Orange
	got := threshold.Classify(0.5, stdConfig, true)
	if got != threshold.Orange {
		t.Errorf("got %s, want Orange", got)
	}
}

func TestClassify_Orange_RecentOverflow_CurrentlySurcharging(t *testing.T) {
	// depth above pipe full but below invert, recent overflow → Orange wins over Yellow
	got := threshold.Classify(1.2, stdConfig, true)
	if got != threshold.Orange {
		t.Errorf("got %s, want Orange (not Yellow) when recent overflow", got)
	}
}

func TestClassify_Red_WinsOver_Orange(t *testing.T) {
	// Red must take priority even when recentOverflow is also true
	got := threshold.Classify(1.5, stdConfig, true)
	if got != threshold.Red {
		t.Errorf("got %s, want Red to win over Orange", got)
	}
}

// --- PipeFillRatio ---

func TestPipeFillRatio_Normal(t *testing.T) {
	got := threshold.PipeFillRatio(0.75, stdConfig)
	want := 0.75
	if got != want {
		t.Errorf("PipeFillRatio = %.4f, want %.4f", got, want)
	}
}

func TestPipeFillRatio_AtFull(t *testing.T) {
	got := threshold.PipeFillRatio(1.0, stdConfig)
	if got != 1.0 {
		t.Errorf("PipeFillRatio at full = %.4f, want 1.0", got)
	}
}

func TestPipeFillRatio_Surcharge(t *testing.T) {
	got := threshold.PipeFillRatio(1.3, stdConfig)
	if got <= 1.0 {
		t.Errorf("PipeFillRatio should be > 1.0 during surcharge, got %.4f", got)
	}
}

func TestPipeFillRatio_ZeroDepth(t *testing.T) {
	got := threshold.PipeFillRatio(0, stdConfig)
	if got != 0 {
		t.Errorf("PipeFillRatio at zero depth = %.4f, want 0", got)
	}
}

// --- STSAlarmType ---

func TestSTSAlarmType_String(t *testing.T) {
	cases := []struct {
		a    threshold.STSAlarmType
		want string
	}{
		{threshold.STSNoAlarm, "No Alarm"},
		{threshold.STSDepthAlarm, "1.8m From Surface"},
		{threshold.STSMaxSensor, "Max Sensor Depth"},
		{threshold.STSGroundAlarm, "Ground Alarm"},
	}
	for _, tc := range cases {
		if got := tc.a.String(); got != tc.want {
			t.Errorf("STSAlarmType.String() = %q, want %q", got, tc.want)
		}
	}
}

var stsConfig = threshold.STSConfig{
	SurfaceElevation: 4.0,  // 4m to surface
	MaxSensorDepth:   3.5,  // sensor maxes at 3.5m
	GroundAlarmDepth: 3.8,  // ground alarm at 3.8m
}

func TestClassifySTS_NoAlarm_NotVisible(t *testing.T) {
	alarm, visible := threshold.ClassifySTS(1.0, stsConfig, false)
	if alarm != threshold.STSNoAlarm || visible {
		t.Errorf("got alarm=%s visible=%v, want NoAlarm/false", alarm, visible)
	}
}

func TestClassifySTS_DepthAlarm(t *testing.T) {
	// surface=4.0, depth=2.1 → 4.0-2.1 = 1.9m from surface > 1.8 → no alarm
	alarm, _ := threshold.ClassifySTS(2.1, stsConfig, false)
	if alarm != threshold.STSNoAlarm {
		t.Errorf("got %s, want NoAlarm (1.9m from surface)", alarm)
	}
	// depth=2.3 → 4.0-2.3 = 1.7m from surface ≤ 1.8 → DepthAlarm
	alarm, visible := threshold.ClassifySTS(2.3, stsConfig, false)
	if alarm != threshold.STSDepthAlarm || !visible {
		t.Errorf("got alarm=%s visible=%v, want DepthAlarm/true", alarm, visible)
	}
}

func TestClassifySTS_MaxSensor(t *testing.T) {
	alarm, visible := threshold.ClassifySTS(3.5, stsConfig, false)
	if alarm != threshold.STSMaxSensor || !visible {
		t.Errorf("got alarm=%s visible=%v, want MaxSensor/true", alarm, visible)
	}
}

func TestClassifySTS_GroundAlarm(t *testing.T) {
	alarm, visible := threshold.ClassifySTS(3.8, stsConfig, false)
	if alarm != threshold.STSGroundAlarm || !visible {
		t.Errorf("got alarm=%s visible=%v, want GroundAlarm/true", alarm, visible)
	}
}

func TestClassifySTS_RecentAlarm_MakesVisible(t *testing.T) {
	// depth well below any threshold, but recentAlarm=true → visible, no alarm
	alarm, visible := threshold.ClassifySTS(0.5, stsConfig, true)
	if alarm != threshold.STSNoAlarm || !visible {
		t.Errorf("got alarm=%s visible=%v, want NoAlarm/true (48h window)", alarm, visible)
	}
}

func TestClassifySTS_GroundAlarm_WinsOverMaxSensor(t *testing.T) {
	// depth above both MaxSensor and GroundAlarm thresholds → GroundAlarm wins
	alarm, _ := threshold.ClassifySTS(3.9, stsConfig, false)
	if alarm != threshold.STSGroundAlarm {
		t.Errorf("got %s, want GroundAlarm to win", alarm)
	}
}

// --- DetectEvents ---

func TestDetectEvents_SingleEvent(t *testing.T) {
	ts := []int64{100, 200, 300, 400, 500}
	depths := []float64{0.5, 1.6, 1.7, 1.5, 0.4} // overflow at t=200,300,400
	events := threshold.DetectEvents(ts, depths, 1.5)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if events[0].Start != 200 {
		t.Errorf("event Start = %d, want 200", events[0].Start)
	}
	if events[0].End != 400 {
		t.Errorf("event End = %d, want 400", events[0].End)
	}
	if events[0].Active() {
		t.Error("expected event to be closed")
	}
}

func TestDetectEvents_MultipleEvents(t *testing.T) {
	ts := []int64{1, 2, 3, 4, 5, 6, 7}
	depths := []float64{1.6, 0.5, 0.5, 1.8, 1.9, 0.3, 0.3}
	events := threshold.DetectEvents(ts, depths, 1.5)
	if len(events) != 2 {
		t.Fatalf("got %d events, want 2", len(events))
	}
}

func TestDetectEvents_ActiveEvent_NoEnd(t *testing.T) {
	ts := []int64{1, 2, 3}
	depths := []float64{0.5, 1.6, 1.8} // overflow starts and never ends
	events := threshold.DetectEvents(ts, depths, 1.5)
	if len(events) != 1 {
		t.Fatalf("got %d events, want 1", len(events))
	}
	if !events[0].Active() {
		t.Error("expected event to be active (End == 0)")
	}
	if events[0].DurationSeconds() != 0 {
		t.Error("active event DurationSeconds should be 0")
	}
}

func TestDetectEvents_NoEvents(t *testing.T) {
	ts := []int64{1, 2, 3}
	depths := []float64{0.1, 0.2, 0.3}
	events := threshold.DetectEvents(ts, depths, 1.5)
	if len(events) != 0 {
		t.Errorf("got %d events, want 0", len(events))
	}
}

func TestDetectEvents_LengthMismatch_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on length mismatch")
		}
	}()
	threshold.DetectEvents([]int64{1, 2}, []float64{1.0}, 1.5)
}

func TestOverflowEvent_DurationSeconds(t *testing.T) {
	e := threshold.OverflowEvent{Start: 1000, End: 1600}
	if e.DurationSeconds() != 600 {
		t.Errorf("DurationSeconds = %d, want 600", e.DurationSeconds())
	}
}
