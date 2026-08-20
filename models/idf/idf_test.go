package idf_test

import (
	"strings"
	"testing"

	"github.com/stepinski/lark/models/idf"
)

func TestNewFromTable_Valid(t *testing.T) {
	pts := []idf.IDFPoint{
		{60, 2, 22.2}, {60, 5, 29.0}, {60, 10, 33.8},
		{120, 2, 14.2}, {120, 5, 18.6}, {120, 10, 21.6},
	}
	c, err := idf.NewFromTable(pts)
	if err != nil || c == nil {
		t.Fatalf("NewFromTable error: %v", err)
	}
}

func TestNewFromTable_Empty(t *testing.T) {
	_, err := idf.NewFromTable(nil)
	if err == nil {
		t.Error("expected error for empty points")
	}
}

func TestNewFromTable_InvalidDuration(t *testing.T) {
	_, err := idf.NewFromTable([]idf.IDFPoint{{DurationMin: 0, ReturnPeriodYr: 2, IntensityMMHr: 10}})
	if err == nil {
		t.Error("expected error for zero duration")
	}
}

func TestReturnPeriod_WithinTable(t *testing.T) {
	c := idf.NewPeelDefault()

	// 1hr storm with 22.2mm → intensity = 22.2mm/hr → exactly 2yr return period
	rp := c.ReturnPeriod(22.2, 60)
	if rp.Lo > 2 || rp.Hi > 5 {
		t.Errorf("ReturnPeriod(22.2mm, 60min) = %s, expected 2yr bracket", rp)
	}
	t.Logf("ReturnPeriod(22.2mm, 60min) = %s", rp)
}

func TestReturnPeriod_HighIntensity(t *testing.T) {
	c := idf.NewPeelDefault()
	// extreme event: 50mm in 1hr = 50mm/hr → should be >100yr
	rp := c.ReturnPeriod(50.0, 60)
	if rp.Hi != 0 { // Hi=0 means ">100yr"
		t.Errorf("Expected >100yr for extreme event, got %s", rp)
	}
	t.Logf("ReturnPeriod(50mm, 60min) = %s", rp)
}

func TestReturnPeriod_LowIntensity(t *testing.T) {
	c := idf.NewPeelDefault()
	// very light rain: 1mm in 1hr → below 2yr
	rp := c.ReturnPeriod(1.0, 60)
	t.Logf("ReturnPeriod(1mm, 60min) = %s", rp)
	// should be below the lowest return period in table
	if rp.Lo > 2 {
		t.Errorf("Expected sub-2yr, got %s", rp)
	}
}

func TestReturnPeriodLabel_String(t *testing.T) {
	cases := []struct {
		label idf.ReturnPeriodLabel
		want  string
	}{
		{idf.ReturnPeriodLabel{Lo: 2, Hi: 5, DurationMin: 60}, "2-5yr (1hr Duration)"},
		{idf.ReturnPeriodLabel{Lo: 100, Hi: 0, DurationMin: 60}, ">100yr (1hr Duration)"},
		{idf.ReturnPeriodLabel{Lo: 10, Hi: 10, DurationMin: 30}, "10yr (30min Duration)"},
		{idf.ReturnPeriodLabel{Lo: 5, Hi: 10, DurationMin: 120}, "5-10yr (2hr Duration)"},
	}
	for _, tc := range cases {
		got := tc.label.String()
		if got != tc.want {
			t.Errorf("String() = %q, want %q", got, tc.want)
		}
	}
}

func TestReturnPeriod_AprilEvent(t *testing.T) {
	// April 3 2025: ~8mm in 25min → 8/25*60 = 19.2mm/hr
	// Note: the *burst* was 6.2mm in 5min = 74mm/hr which is extreme.
	// Using total storm accumulation gives a lower bound return period.
	c := idf.NewPeelDefault()
	rp := c.ReturnPeriod(8.0, 25)
	t.Logf("April 3 event (8mm/25min): %s", rp)
	// just verify it produces a valid label
	if rp.String() == "" {
		t.Error("expected non-empty return period label")
	}
}

func TestIntensities_ReturnsTable(t *testing.T) {
	c := idf.NewPeelDefault()
	pts := c.Intensities(2.0)
	if len(pts) == 0 {
		t.Error("expected non-empty intensity table for 2yr")
	}
	for _, p := range pts {
		if p.ReturnPeriodYr != 2.0 {
			t.Errorf("expected ReturnPeriodYr=2, got %.1f", p.ReturnPeriodYr)
		}
	}
}

func TestIntensities_UnknownPeriod(t *testing.T) {
	c := idf.NewPeelDefault()
	pts := c.Intensities(99.9)
	if pts != nil {
		t.Error("expected nil for unknown return period")
	}
}

func TestNewPeelDefault_Smoke(t *testing.T) {
	c := idf.NewPeelDefault()
	if c == nil {
		t.Fatal("NewPeelDefault returned nil")
	}
	// spot check: 1hr 5yr intensity should be > 2yr intensity
	pts2 := c.Intensities(2.0)
	pts5 := c.Intensities(5.0)
	var i2, i5 float64
	for _, p := range pts2 {
		if p.DurationMin == 60 {
			i2 = p.IntensityMMHr
		}
	}
	for _, p := range pts5 {
		if p.DurationMin == 60 {
			i5 = p.IntensityMMHr
		}
	}
	if i5 <= i2 {
		t.Errorf("5yr intensity (%.1f) should be > 2yr (%.1f) at 1hr", i5, i2)
	}
}

func TestReturnPeriodLabel_DurationFormatting(t *testing.T) {
	cases := []struct {
		min  int
		want string
	}{
		{5, "5min"},
		{30, "30min"},
		{60, "1hr"},
		{120, "2hr"},
		{1440, "24hr"},
		{45, "45min"},
	}
	for _, tc := range cases {
		rp := idf.ReturnPeriodLabel{Lo: 2, Hi: 5, DurationMin: tc.min}
		s := rp.String()
		if !strings.Contains(s, tc.want) {
			t.Errorf("duration %dmin: got %q, expected to contain %q", tc.min, s, tc.want)
		}
	}
}
