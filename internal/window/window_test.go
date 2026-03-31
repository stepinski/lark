package window_test

import (
	"testing"
	"time"

	"github.com/stepinski/lark/internal/window"
)

var epoch = time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)

func t(offsetMinutes int) time.Time {
	return epoch.Add(time.Duration(offsetMinutes) * time.Minute)
}

func obs(offsetMinutes int, val float64) window.Obs {
	return window.Obs{T: t(offsetMinutes), Val: val}
}

// --- capacity and basic push ---

func TestNew_PanicOnZeroCap(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for cap=0")
		}
	}()
	window.New(0)
}

func TestPush_LenGrows(t *testing.T) {
	w := window.New(5)
	for i := 0; i < 3; i++ {
		w.Push(obs(i, float64(i)))
	}
	if w.Len() != 3 {
		t.Errorf("Len = %d, want 3", w.Len())
	}
}

func TestPush_CapNotExceeded(t *testing.T) {
	w := window.New(3)
	for i := 0; i < 10; i++ {
		w.Push(obs(i, float64(i)))
	}
	if w.Len() != 3 {
		t.Errorf("Len = %d after overflow, want 3", w.Len())
	}
	if !w.Full() {
		t.Error("expected Full() == true")
	}
}

// --- ring buffer correctness ---

func TestSlice_ChronologicalOrder(t *testing.T) {
	w := window.New(4)
	for i := 0; i < 4; i++ {
		w.Push(obs(i, float64(i+1)))
	}
	got := w.Slice()
	for i, o := range got {
		want := float64(i + 1)
		if o.Val != want {
			t.Errorf("Slice[%d].Val = %.1f, want %.1f", i, o.Val, want)
		}
	}
}

func TestSlice_AfterOverwrite_OldestDropped(t *testing.T) {
	w := window.New(3)
	// push 4 observations into cap-3 window
	for i := 1; i <= 4; i++ {
		w.Push(obs(i, float64(i)))
	}
	got := w.Slice()
	// should hold observations 2, 3, 4
	if len(got) != 3 {
		t.Fatalf("len = %d, want 3", len(got))
	}
	if got[0].Val != 2 || got[1].Val != 3 || got[2].Val != 4 {
		t.Errorf("wrong values after overwrite: %v", got)
	}
}

func TestValues_ReturnsFloat64Slice(t *testing.T) {
	w := window.New(3)
	w.Push(obs(0, 1.5))
	w.Push(obs(1, 2.5))
	vals := w.Values()
	if len(vals) != 2 || vals[0] != 1.5 || vals[1] != 2.5 {
		t.Errorf("Values = %v", vals)
	}
}

// --- Last ---

func TestLast_EmptyError(t *testing.T) {
	w := window.New(3)
	_, err := w.Last()
	if err == nil {
		t.Error("expected error on empty window")
	}
}

func TestLast_ReturnsMostRecent(t *testing.T) {
	w := window.New(5)
	w.Push(obs(0, 10))
	w.Push(obs(1, 20))
	w.Push(obs(2, 30))
	got, err := w.Last()
	if err != nil {
		t.Fatalf("Last error: %v", err)
	}
	if got.Val != 30 {
		t.Errorf("Last.Val = %.1f, want 30", got.Val)
	}
}

// --- Since (48h lookback for Peel overflow logic) ---

func TestSince_FiltersCorrectly(t *testing.T) {
	w := window.New(10)
	// push observations at t=0, 10, 20, 30, 40, 50 minutes
	for i := 0; i < 6; i++ {
		w.Push(obs(i*10, float64(i)))
	}
	// request observations since t=25min — should get t=30, 40, 50
	got := w.Since(epoch.Add(25 * time.Minute))
	if len(got) != 3 {
		t.Fatalf("Since len = %d, want 3", len(got))
	}
	if got[0].Val != 3 || got[1].Val != 4 || got[2].Val != 5 {
		t.Errorf("Since values = %v", got)
	}
}

func TestSince_AllBeforeCutoff_ReturnsEmpty(t *testing.T) {
	w := window.New(5)
	for i := 0; i < 5; i++ {
		w.Push(obs(i, float64(i)))
	}
	future := epoch.Add(1 * time.Hour)
	got := w.Since(future)
	if len(got) != 0 {
		t.Errorf("expected empty slice, got %v", got)
	}
}

func TestSince_AllAfterCutoff_ReturnsAll(t *testing.T) {
	w := window.New(5)
	for i := 1; i <= 5; i++ {
		w.Push(obs(i*10, float64(i)))
	}
	got := w.Since(epoch) // cutoff before everything
	if len(got) != 5 {
		t.Errorf("Since len = %d, want 5", len(got))
	}
}

// --- aggregates ---

func TestMax(t *testing.T) {
	w := window.New(5)
	for _, v := range []float64{3, 1, 4, 1, 5} {
		w.Push(window.Obs{Val: v})
	}
	got, err := w.Max()
	if err != nil || got != 5 {
		t.Errorf("Max = %.1f, err = %v; want 5, nil", got, err)
	}
}

func TestMax_EmptyError(t *testing.T) {
	w := window.New(3)
	_, err := w.Max()
	if err == nil {
		t.Error("expected error on empty window")
	}
}

func TestSum_RainfallAccumulation(t *testing.T) {
	w := window.New(6)
	for _, v := range []float64{1.0, 2.0, 3.0} {
		w.Push(window.Obs{Val: v})
	}
	if w.Sum() != 6.0 {
		t.Errorf("Sum = %.1f, want 6.0", w.Sum())
	}
}

// --- Reset ---

func TestReset_ClearsWindow(t *testing.T) {
	w := window.New(4)
	w.Push(obs(0, 99))
	w.Push(obs(1, 88))
	w.Reset()
	if w.Len() != 0 {
		t.Errorf("Len after Reset = %d, want 0", w.Len())
	}
	if _, err := w.Last(); err == nil {
		t.Error("expected error after Reset")
	}
}

// --- large capacity ring correctness ---

func TestRingCorrectness_LargeN(t *testing.T) {
	const cap = 100
	w := window.New(cap)
	// push 250 values — window should hold last 100
	for i := 0; i < 250; i++ {
		w.Push(window.Obs{Val: float64(i)})
	}
	vals := w.Values()
	if len(vals) != cap {
		t.Fatalf("len = %d, want %d", len(vals), cap)
	}
	// first value in window should be 250-100 = 150
	if vals[0] != 150 {
		t.Errorf("vals[0] = %.0f, want 150", vals[0])
	}
	if vals[99] != 249 {
		t.Errorf("vals[99] = %.0f, want 249", vals[99])
	}
}
