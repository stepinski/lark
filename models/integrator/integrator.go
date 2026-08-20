// Package integrator computes overflow volume by integrating flow above the
// overflow invert threshold over time.
//
// Per the Peel SoW, overflow volume is calculated from flow channel data:
//
//	V = ∫ Q(t) dt   where Q(t) > 0 only when depth(t) >= invertDepth
//
// The SoW specifies: use Final Flow channel when available, else Raw flow.
// Volume is reported in litres (L) or kilolitres (kL).
//
// This package also detects overflow events (contiguous periods where
// depth >= invert) and computes per-event statistics required for the
// 30-day event table and PDF reports.
package integrator

import (
	"fmt"
	"math"
	"time"
)

// Point is a single timestamped flow/depth observation.
type Point struct {
	T     time.Time
	Depth float64 // mm
	Flow  float64 // l/s (litres per second)
}

// Event represents a single overflow event as required by the Peel SoW
// 30-day event table.
type Event struct {
	// Start is the first timestamp where depth >= invert.
	Start time.Time
	// End is the last timestamp where depth >= invert.
	// Zero if the event is still active.
	End time.Time
	// DurationSeconds is the event duration.
	DurationSeconds int64
	// VolumeL is the total overflow volume in litres.
	// Computed as ∫ flow * dt for all timesteps during the event.
	VolumeL float64
	// PeakDepthMM is the maximum depth observed during the event.
	PeakDepthMM float64
	// PeakFlowLS is the maximum flow observed during the event.
	PeakFlowLS float64
	// PointCount is the number of observations in the event.
	PointCount int
}

// Active reports whether the overflow event is still ongoing.
func (e Event) Active() bool { return e.End.IsZero() }

// DurationString returns the event duration as a human-readable string.
func (e Event) DurationString() string {
	if e.Active() {
		return "active"
	}
	d := time.Duration(e.DurationSeconds) * time.Second
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	if h > 0 {
		return fmt.Sprintf("%dh%02dm", h, m)
	}
	return fmt.Sprintf("%dm", m)
}

// VolumeKL returns the event volume in kilolitres.
func (e Event) VolumeKL() float64 { return e.VolumeL / 1000.0 }

// Config controls event detection and volume calculation.
type Config struct {
	// InvertDepthMM is the overflow invert elevation (mm).
	// Overflow occurs when depth >= InvertDepthMM.
	InvertDepthMM float64

	// MinEventDurationSec is the minimum duration (seconds) for an event
	// to be reported. Shorter events are considered noise. Default: 60s.
	MinEventDurationSec int64

	// MaxGapSec is the maximum gap between observations before splitting
	// an event into two separate events. Default: 600s (10 min).
	MaxGapSec int64
}

func (c Config) minDuration() int64 {
	if c.MinEventDurationSec <= 0 {
		return 60
	}
	return c.MinEventDurationSec
}

func (c Config) maxGap() int64 {
	if c.MaxGapSec <= 0 {
		return 600
	}
	return c.MaxGapSec
}

// DetectEvents scans a time series of flow/depth observations and returns
// all overflow events. Points must be in chronological order.
//
// Volume is computed using the trapezoidal rule: for each pair of consecutive
// points within an event, V += (Q1 + Q2) / 2 * dt.
func DetectEvents(points []Point, cfg Config) []Event {
	if len(points) == 0 {
		return nil
	}

	var events []Event
	var current *Event
	var prevPoint *Point

	for i := range points {
		p := &points[i]
		overflowing := p.Depth >= cfg.InvertDepthMM && !math.IsNaN(p.Depth)

		if overflowing {
			if current == nil {
				// start new event
				e := Event{
					Start:       p.T,
					PeakDepthMM: p.Depth,
					PeakFlowLS:  p.Flow,
					PointCount:  1,
				}
				current = &e
			} else {
				// check if gap since last point is too large (split event)
				if prevPoint != nil {
					gap := p.T.Unix() - prevPoint.T.Unix()
					if gap >= cfg.maxGap() {
						// close current event and start new one
						current.End = prevPoint.T
						current.DurationSeconds = current.End.Unix() - current.Start.Unix()
						if current.DurationSeconds >= cfg.minDuration() {
							events = append(events, *current)
						}
						e := Event{
							Start:       p.T,
							PeakDepthMM: p.Depth,
							PeakFlowLS:  p.Flow,
							PointCount:  1,
						}
						current = &e
						prevPoint = p
						continue
					}

					// add volume via trapezoidal rule
					dt := float64(p.T.Unix() - prevPoint.T.Unix()) // seconds
					avgFlow := (prevPoint.Flow + p.Flow) / 2.0
					if avgFlow > 0 {
						current.VolumeL += avgFlow * dt
					}
				}

				// update peak stats
				if p.Depth > current.PeakDepthMM {
					current.PeakDepthMM = p.Depth
				}
				if p.Flow > current.PeakFlowLS {
					current.PeakFlowLS = p.Flow
				}
				current.PointCount++
			}
		} else {
			if current != nil && prevPoint != nil {
				// close event
				current.End = prevPoint.T
				current.DurationSeconds = current.End.Unix() - current.Start.Unix()
				if current.DurationSeconds >= cfg.minDuration() {
					events = append(events, *current)
				}
				current = nil
			}
		}

		prevPoint = p
	}

	// close any still-active event
	if current != nil {
		if prevPoint != nil && prevPoint.T != current.Start {
			current.End = time.Time{} // mark as active
			current.DurationSeconds = 0
			events = append(events, *current)
		}
	}

	return events
}

// TotalVolume computes the total overflow volume across all events in litres.
func TotalVolume(events []Event) float64 {
	var total float64
	for _, e := range events {
		total += e.VolumeL
	}
	return total
}

// EventsInRange returns events that overlap the given time range.
func EventsInRange(events []Event, start, end time.Time) []Event {
	var out []Event
	for _, e := range events {
		// event overlaps range if it starts before end AND (ends after start OR is active)
		if e.Start.Before(end) {
			if e.Active() || e.End.After(start) {
				out = append(out, e)
			}
		}
	}
	return out
}

// RainfallWindow computes the total rainfall associated with an overflow event
// per the Peel SoW definition:
//
//	"sum of all rainfall from 24 hours prior to the start of the overflow
//	 to 24 hours after the start of the overflow"
//
// rainPoints must be in chronological order. Returns total rainfall in mm.
func RainfallWindow(event Event, rainPoints []RainPoint) float64 {
	windowStart := event.Start.Add(-24 * time.Hour)
	windowEnd := event.Start.Add(24 * time.Hour)

	var total float64
	for _, rp := range rainPoints {
		if !rp.T.Before(windowStart) && !rp.T.After(windowEnd) {
			total += rp.RainMM
		}
	}
	return total
}

// RainPoint is a single timestamped rainfall observation.
type RainPoint struct {
	T      time.Time
	RainMM float64
}
