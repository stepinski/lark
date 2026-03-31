// Package flock implements a bounded worker pool for streaming sensor
// observations through registered site models.
//
// Architecture:
//
//	Nest (job queue) ──► Flock (W workers) ──► Results chan
//
// Each worker dequeues a Job, looks up the site's SiteState in the Registry,
// acquires the site-level mutex, calls the user-supplied Handler, and emits
// a Result. Site state is never shared between workers simultaneously —
// only one worker processes a given site at any moment.
//
// Flock respects context cancellation for graceful shutdown: in-flight jobs
// finish, then workers exit and Results is closed.
package flock

import (
	"context"
	"sync"

	"github.com/stepinski/lark/internal/window"
	"github.com/stepinski/lark/runtime/nest"
)

// Handler is the user-supplied function called for each job.
// It receives the job and the site's mutable state, and must return a Result.
// Handler is called with the site mutex already held — do not re-acquire it.
type Handler func(job nest.Job, state *SiteState) nest.Result

// SiteState holds per-site mutable runtime state.
// Access is serialised by Flock — Handler implementations must not spawn
// goroutines that capture and mutate SiteState after the Handler returns.
type SiteState struct {
	mu      sync.Mutex
	Window  *window.Window
	// Custom holds arbitrary model state (e.g. *sarimax.Model).
	// Type-assert in your Handler.
	Custom interface{}
}

// Registry maps site IDs to their runtime state.
// Zero value is ready to use.
type Registry struct {
	mu    sync.RWMutex
	sites map[string]*SiteState
}

// NewRegistry allocates an empty Registry.
func NewRegistry() *Registry {
	return &Registry{sites: make(map[string]*SiteState)}
}

// Register adds or replaces a site's state. Safe to call at any time,
// including after Flock has started. Existing in-flight jobs for the site
// will complete before the new state is visible to subsequent jobs.
func (r *Registry) Register(siteID string, windowCap int, custom interface{}) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.sites[siteID] = &SiteState{
		Window: window.New(windowCap),
		Custom: custom,
	}
}

// Get returns the SiteState for id, or nil if not registered.
func (r *Registry) Get(siteID string) *SiteState {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.sites[siteID]
}

// Len returns the number of registered sites.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.sites)
}

// Config controls Flock behaviour.
type Config struct {
	// Workers is the number of goroutines in the pool.
	// Defaults to 1 if <= 0.
	Workers int
	// ResultsBuffer is the capacity of the results channel.
	// Defaults to Workers * 10 if <= 0.
	ResultsBuffer int
}

func (c *Config) workers() int {
	if c.Workers <= 0 {
		return 1
	}
	return c.Workers
}

func (c *Config) resultsBuffer() int {
	if c.ResultsBuffer <= 0 {
		return c.workers() * 10
	}
	return c.ResultsBuffer
}

// Flock is the worker pool. Construct with New and start with Run.
type Flock struct {
	cfg      Config
	registry *Registry
	handler  Handler
	results  chan nest.Result
}

// New constructs a Flock. Does not start workers.
func New(cfg Config, registry *Registry, handler Handler) *Flock {
	return &Flock{
		cfg:      cfg,
		registry: registry,
		handler:  handler,
		results:  make(chan nest.Result, cfg.resultsBuffer()),
	}
}

// Results returns the channel on which workers emit processed Results.
// The channel is closed after all workers have exited (i.e. after ctx is
// cancelled and the Nest is drained).
func (f *Flock) Results() <-chan nest.Result {
	return f.results
}

// Run starts the worker pool and blocks until all workers exit.
// Workers exit when ctx is cancelled AND the Nest's job channel is closed.
// Call Run in a goroutine; read from Results() concurrently.
//
//	go flock.Run(ctx, n)
//	for result := range flock.Results() { ... }
func (f *Flock) Run(ctx context.Context, n *nest.Nest) {
	w := f.cfg.workers()
	var wg sync.WaitGroup
	wg.Add(w)

	jobs := n.Dequeue()

	for i := 0; i < w; i++ {
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					// drain remaining jobs before exiting so producers
					// don't block on a closed results channel
					return
				case job, ok := <-jobs:
					if !ok {
						return // Nest was drained
					}
					f.process(job)
				}
			}
		}()
	}

	wg.Wait()
	close(f.results)
}

// process dispatches a single job to the handler under the site mutex.
func (f *Flock) process(job nest.Job) {
	state := f.registry.Get(job.SiteID)
	if state == nil {
		// unknown site — emit error result, do not panic
		f.results <- nest.Result{
			SiteID:  job.SiteID,
			ObsTime: job.ObsTime,
			Err:     errUnknownSite(job.SiteID),
		}
		return
	}

	state.mu.Lock()
	// push observation into the sliding window before calling handler
	state.Window.Push(window.Obs{T: job.ObsTime, Val: job.Value})
	result := f.handler(job, state)
	state.mu.Unlock()

	f.results <- result
}

// errUnknownSite returns a typed error for an unregistered site.
type unknownSiteError string

func errUnknownSite(id string) error { return unknownSiteError(id) }
func (e unknownSiteError) Error() string {
	return "flock: unknown site: " + string(e)
}
