// Package window provides a fixed-capacity ring buffer for streaming
// time series observations.
//
// Window is NOT safe for concurrent use — callers must synchronise externally.
// This is intentional: in Flock, each site owns its own Window and is only
// ever touched by one worker goroutine at a time.
package window

import (
	"errors"
	"time"
)

// Obs is a single timestamped observation.
type Obs struct {
	T   time.Time
	Val float64
}

// Window is a fixed-capacity circular buffer of Obs values.
// Once full, the oldest observation is silently overwritten.
type Window struct {
	buf  []Obs
	cap_ int
	head int // index of the next write position
	size int // current number of valid entries
}

// New allocates a Window with the given capacity.
// Panics if cap <= 0.
func New(cap int) *Window {
	if cap <= 0 {
		panic("window: capacity must be > 0")
	}
	return &Window{
		buf:  make([]Obs, cap),
		cap_: cap,
	}
}

// Push appends a new observation, overwriting the oldest if full.
func (w *Window) Push(o Obs) {
	w.buf[w.head] = o
	w.head = (w.head + 1) % w.cap_
	if w.size < w.cap_ {
		w.size++
	}
}

// Len returns the number of valid observations currently held.
func (w *Window) Len() int { return w.size }

// Cap returns the maximum capacity of the window.
func (w *Window) Cap() int { return w.cap_ }

// Full reports whether the window has reached capacity.
func (w *Window) Full() bool { return w.size == w.cap_ }

// Slice returns a copy of current observations in chronological order
// (oldest first). The returned slice has length w.Len().
func (w *Window) Slice() []Obs {
	out := make([]Obs, w.size)
	start := (w.head - w.size + w.cap_) % w.cap_
	for i := 0; i < w.size; i++ {
		out[i] = w.buf[(start+i)%w.cap_]
	}
	return out
}

// Values returns only the float64 values in chronological order.
// Useful for passing directly to model Step() or Fit().
func (w *Window) Values() []float64 {
	obs := w.Slice()
	out := make([]float64, len(obs))
	for i, o := range obs {
		out[i] = o.Val
	}
	return out
}

// Last returns the most recently pushed observation.
// Returns an error if the window is empty.
func (w *Window) Last() (Obs, error) {
	if w.size == 0 {
		return Obs{}, errors.New("window: empty")
	}
	idx := (w.head - 1 + w.cap_) % w.cap_
	return w.buf[idx], nil
}

// Since returns all observations with T >= cutoff, in chronological order.
// Useful for the 48-hour lookback required by the Peel overflow alert logic.
func (w *Window) Since(cutoff time.Time) []Obs {
	all := w.Slice()
	// all is sorted oldest→newest; find first entry >= cutoff
	lo := 0
	for lo < len(all) && all[lo].T.Before(cutoff) {
		lo++
	}
	out := make([]Obs, len(all)-lo)
	copy(out, all[lo:])
	return out
}

// Max returns the maximum value in the window.
// Returns 0 and an error if empty.
func (w *Window) Max() (float64, error) {
	if w.size == 0 {
		return 0, errors.New("window: empty")
	}
	max := w.buf[(w.head-w.size+w.cap_)%w.cap_].Val
	start := (w.head - w.size + w.cap_) % w.cap_
	for i := 1; i < w.size; i++ {
		v := w.buf[(start+i)%w.cap_].Val
		if v > max {
			max = v
		}
	}
	return max, nil
}

// Sum returns the sum of all values in the window.
// Useful for accumulating rainfall totals over a time range.
func (w *Window) Sum() float64 {
	var s float64
	start := (w.head - w.size + w.cap_) % w.cap_
	for i := 0; i < w.size; i++ {
		s += w.buf[(start+i)%w.cap_].Val
	}
	return s
}

// Reset clears all observations without reallocating.
func (w *Window) Reset() {
	w.head = 0
	w.size = 0
}
