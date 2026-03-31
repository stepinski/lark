package flock_test

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stepinski/lark/runtime/flock"
	"github.com/stepinski/lark/runtime/nest"
)

// --- Nest tests ---

func TestNest_EnqueueDequeue(t *testing.T) {
	n := nest.New(5)
	job := nest.Job{SiteID: "site-1", Value: 1.23}
	if err := n.Enqueue(job); err != nil {
		t.Fatalf("Enqueue error: %v", err)
	}
	if n.Len() != 1 {
		t.Errorf("Len = %d, want 1", n.Len())
	}
}

func TestNest_Backpressure(t *testing.T) {
	n := nest.New(3)
	for i := 0; i < 3; i++ {
		if err := n.Enqueue(nest.Job{SiteID: "s"}); err != nil {
			t.Fatalf("unexpected error on enqueue %d: %v", i, err)
		}
	}
	// 4th enqueue on full queue must return ErrBackpressure
	err := n.Enqueue(nest.Job{SiteID: "s"})
	if err != nest.ErrBackpressure {
		t.Errorf("expected ErrBackpressure, got %v", err)
	}
}

func TestNest_AfterDrain_ReturnsErrClosed(t *testing.T) {
	n := nest.New(5)
	n.Drain()
	err := n.Enqueue(nest.Job{SiteID: "s"})
	if err != nest.ErrClosed {
		t.Errorf("expected ErrClosed after Drain, got %v", err)
	}
}

func TestNest_DrainIdempotent(t *testing.T) {
	n := nest.New(5)
	// calling Drain twice must not panic
	n.Drain()
	n.Drain()
}

func TestNest_Utilisation(t *testing.T) {
	n := nest.New(4)
	_ = n.Enqueue(nest.Job{SiteID: "s"})
	_ = n.Enqueue(nest.Job{SiteID: "s"})
	got := n.Utilisation()
	want := 0.5
	if got != want {
		t.Errorf("Utilisation = %.2f, want %.2f", got, want)
	}
}

// --- Registry tests ---

func TestRegistry_RegisterAndGet(t *testing.T) {
	r := flock.NewRegistry()
	r.Register("site-A", 100, nil)
	state := r.Get("site-A")
	if state == nil {
		t.Fatal("expected non-nil SiteState")
	}
	if state.Window == nil {
		t.Error("expected non-nil Window")
	}
}

func TestRegistry_UnknownSite(t *testing.T) {
	r := flock.NewRegistry()
	if r.Get("nonexistent") != nil {
		t.Error("expected nil for unregistered site")
	}
}

func TestRegistry_Len(t *testing.T) {
	r := flock.NewRegistry()
	r.Register("a", 10, nil)
	r.Register("b", 10, nil)
	if r.Len() != 2 {
		t.Errorf("Len = %d, want 2", r.Len())
	}
}

// --- Flock integration tests ---

// echoHandler is a trivial handler: returns the job value as forecast.
func echoHandler(job nest.Job, state *flock.SiteState) nest.Result {
	return nest.Result{
		SiteID:   job.SiteID,
		ObsTime:  job.ObsTime,
		Forecast: job.Value,
	}
}

func TestFlock_ProcessesAllJobs(t *testing.T) {
	const numJobs = 50

	r := flock.NewRegistry()
	r.Register("site-1", 100, nil)

	n := nest.New(numJobs)
	for i := 0; i < numJobs; i++ {
		_ = n.Enqueue(nest.Job{SiteID: "site-1", Value: float64(i)})
	}
	n.Drain()

	f := flock.New(flock.Config{Workers: 4}, r, echoHandler)
	ctx := context.Background()

	var got int
	go f.Run(ctx, n)
	for range f.Results() {
		got++
	}

	if got != numJobs {
		t.Errorf("processed %d jobs, want %d", got, numJobs)
	}
}

func TestFlock_UnknownSiteEmitsError(t *testing.T) {
	r := flock.NewRegistry()
	// do NOT register "ghost-site"

	n := nest.New(5)
	_ = n.Enqueue(nest.Job{SiteID: "ghost-site", Value: 1.0})
	n.Drain()

	f := flock.New(flock.Config{Workers: 1}, r, echoHandler)
	go f.Run(context.Background(), n)

	var errResult *nest.Result
	for res := range f.Results() {
		r := res
		errResult = &r
	}
	if errResult == nil || errResult.Err == nil {
		t.Error("expected error result for unknown site")
	}
}

func TestFlock_GracefulShutdown_ContextCancel(t *testing.T) {
	r := flock.NewRegistry()
	r.Register("site-x", 50, nil)

	n := nest.New(200)
	for i := 0; i < 100; i++ {
		_ = n.Enqueue(nest.Job{SiteID: "site-x", Value: float64(i)})
	}

	ctx, cancel := context.WithCancel(context.Background())
	f := flock.New(flock.Config{Workers: 2}, r, echoHandler)

	done := make(chan struct{})
	go func() {
		f.Run(ctx, n)
		close(done)
	}()

	// cancel after a short delay — workers should exit cleanly
	cancel()
	n.Drain()

	select {
	case <-done:
		// clean exit
	case <-time.After(2 * time.Second):
		t.Error("Flock did not shut down within 2s after context cancel")
	}
}

func TestFlock_ResultsClosedAfterRun(t *testing.T) {
	r := flock.NewRegistry()
	n := nest.New(5)
	n.Drain() // empty, immediately closed

	f := flock.New(flock.Config{Workers: 2}, r, echoHandler)
	go f.Run(context.Background(), n)

	// ranging over Results() must terminate (channel closed after workers exit)
	done := make(chan struct{})
	go func() {
		for range f.Results() {
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Error("Results channel not closed after empty run")
	}
}

// TestFlock_PerSiteSerialisation verifies that concurrent workers never
// process the same site simultaneously (no race on SiteState).
// The race detector will catch any violation even if the assertion passes.
func TestFlock_PerSiteSerialisation(t *testing.T) {
	const numSites = 10
	const jobsPerSite = 20

	r := flock.NewRegistry()
	for i := 0; i < numSites; i++ {
		r.Register(fmt.Sprintf("site-%d", i), 100, nil)
	}

	// counter handler: increments a per-site counter stored in Custom.
	// If two workers ran for the same site concurrently, the race detector
	// would fire on the non-atomic increment.
	type siteCounter struct{ n int }
	for i := 0; i < numSites; i++ {
		state := r.Get(fmt.Sprintf("site-%d", i))
		state.Custom = &siteCounter{}
	}

	handler := func(job nest.Job, state *flock.SiteState) nest.Result {
		c := state.Custom.(*siteCounter)
		c.n++ // intentionally non-atomic — safe only because mutex is held
		return nest.Result{SiteID: job.SiteID, Forecast: float64(c.n)}
	}

	n := nest.New(numSites * jobsPerSite)
	for i := 0; i < numSites; i++ {
		for j := 0; j < jobsPerSite; j++ {
			_ = n.Enqueue(nest.Job{
				SiteID: fmt.Sprintf("site-%d", i),
				Value:  float64(j),
			})
		}
	}
	n.Drain()

	f := flock.New(flock.Config{Workers: 8}, r, handler)
	go f.Run(context.Background(), n)

	var total int
	for range f.Results() {
		total++
	}

	if total != numSites*jobsPerSite {
		t.Errorf("total results = %d, want %d", total, numSites*jobsPerSite)
	}

	// verify each site's counter reached jobsPerSite
	for i := 0; i < numSites; i++ {
		state := r.Get(fmt.Sprintf("site-%d", i))
		c := state.Custom.(*siteCounter)
		if c.n != jobsPerSite {
			t.Errorf("site-%d counter = %d, want %d", i, c.n, jobsPerSite)
		}
	}
}

// TestFlock_HighConcurrency_NoPanic stress-tests 10k sites with many workers.
// Primary goal: no panics, no deadlocks, race detector clean.
func TestFlock_HighConcurrency_NoPanic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping high-concurrency test in short mode")
	}

	const numSites = 1000
	const workers = 32

	r := flock.NewRegistry()
	for i := 0; i < numSites; i++ {
		r.Register(fmt.Sprintf("s%d", i), 50, nil)
	}

	var processed atomic.Int64
	handler := func(job nest.Job, state *flock.SiteState) nest.Result {
		processed.Add(1)
		return nest.Result{SiteID: job.SiteID, Forecast: job.Value * 1.1}
	}

	n := nest.New(numSites * 5)
	var wg sync.WaitGroup
	wg.Add(numSites)
	// concurrent producers — one per site
	for i := 0; i < numSites; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 5; j++ {
				_ = n.Enqueue(nest.Job{
					SiteID:  fmt.Sprintf("s%d", id),
					Value:   float64(j),
					ObsTime: time.Now(),
				})
			}
		}(i)
	}

	wg.Wait()
	n.Drain()

	f := flock.New(flock.Config{Workers: workers}, r, handler)
	go f.Run(context.Background(), n)
	for range f.Results() {
	}

	want := int64(numSites * 5)
	if got := processed.Load(); got != want {
		t.Errorf("processed %d, want %d", got, want)
	}
}
