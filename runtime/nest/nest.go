// Package nest implements a non-blocking, buffered job queue with backpressure.
//
// The queue has a fixed capacity. When full, Enqueue returns ErrBackpressure
// immediately (503-style) rather than blocking, allowing callers to shed load
// or apply retry logic without risking goroutine pile-up under burst traffic.
//
// Nest is safe for concurrent use by multiple producers and consumers.
package nest

import (
	"errors"
	"time"
)

// ErrBackpressure is returned by Enqueue when the queue is at capacity.
var ErrBackpressure = errors.New("nest: queue full — backpressure applied")

// ErrClosed is returned when operating on a closed Nest.
var ErrClosed = errors.New("nest: queue closed")

// Job represents a single unit of work: one sensor observation to be
// processed by a Flock worker.
type Job struct {
	// SiteID is the unique identifier for the sensor site.
	SiteID string
	// ObsTime is the timestamp of the observation.
	ObsTime time.Time
	// Value is the sensor reading (depth, flow, rainfall, etc.).
	Value float64
	// Exog holds optional exogenous values (e.g. rain gauge reading)
	// aligned to this timestep. May be nil.
	Exog []float64
}

// Result is the output produced by a Flock worker after processing a Job.
type Result struct {
	// SiteID echoes the originating job's SiteID.
	SiteID string
	// ObsTime echoes the originating job's ObsTime.
	ObsTime time.Time
	// Forecast is the model's one-step-ahead prediction.
	Forecast float64
	// Err is non-nil if the worker failed to process the job.
	Err error
}

// Nest is the job queue. Zero value is unusable; construct with New.
type Nest struct {
	ch     chan Job
	cap_   int
	closed chan struct{}
}

// New creates a Nest with the given buffer capacity.
// Panics if cap <= 0.
func New(cap int) *Nest {
	if cap <= 0 {
		panic("nest: capacity must be > 0")
	}
	return &Nest{
		ch:     make(chan Job, cap),
		cap_:   cap,
		closed: make(chan struct{}),
	}
}

// Enqueue attempts to add a job to the queue without blocking.
// Returns ErrBackpressure if the queue is full.
// Returns ErrClosed if the Nest has been closed.
func (n *Nest) Enqueue(j Job) error {
	select {
	case <-n.closed:
		return ErrClosed
	default:
	}
	select {
	case n.ch <- j:
		return nil
	default:
		return ErrBackpressure
	}
}

// Dequeue returns the internal channel for consumers (Flock workers) to
// range over. The channel is closed when Drain is called.
func (n *Nest) Dequeue() <-chan Job {
	return n.ch
}

// Len returns the current number of jobs waiting in the queue.
func (n *Nest) Len() int {
	return len(n.ch)
}

// Cap returns the maximum queue capacity.
func (n *Nest) Cap() int {
	return n.cap_
}

// Utilisation returns the fraction of capacity currently in use [0.0, 1.0].
func (n *Nest) Utilisation() float64 {
	return float64(len(n.ch)) / float64(n.cap_)
}

// Drain signals producers that no new jobs should be enqueued, then closes
// the underlying channel so Flock workers drain and exit cleanly.
// Safe to call multiple times (idempotent).
func (n *Nest) Drain() {
	select {
	case <-n.closed:
		// already closed — no-op
	default:
		close(n.closed)
		close(n.ch)
	}
}
