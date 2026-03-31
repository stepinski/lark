// Package threshold implements the overflow state classifier for Lark.
//
// It encodes the exact threshold logic from the Peel Region Overflow Alert
// System SoW:
//
//	Green  — no overflow, mainline below 100% full
//	Yellow — no overflow, mainline above 100% but below overflow invert
//	Red    — active overflow (level >= bottom overflow invert)
//	Orange — no current overflow, but overflow occurred within past 48 hours
//
// All functions are pure and stateless. State (the 48h lookback) is owned
// by the caller via the window package — this package only classifies.
package threshold

import "fmt"

// State represents the current overflow condition of a monitored site.
type State int

const (
	// Green — normal operation. Mainline below 100% full, no recent overflow.
	Green State = iota
	// Yellow — elevated. Mainline surcharging (>100%) but not yet overflowing.
	Yellow
	// Orange — recent overflow. An overflow occurred within the past 48 hours
	// but is not currently active.
	Orange
	// Red — active overflow. Level is at or above the outlet invert.
	Red
)

// String returns the human-readable state label used in the Peel UI.
func (s State) String() string {
	switch s {
	case Green:
		return "Green"
	case Yellow:
		return "Yellow"
	case Orange:
		return "Orange"
	case Red:
		return "Red"
	default:
		return fmt.Sprintf("State(%d)", int(s))
	}
}

// IsAlarm reports whether the state requires operator attention.
// Red and Orange are both alarm states per the SoW.
func (s State) IsAlarm() bool {
	return s == Red || s == Orange
}

// SiteConfig holds the physical thresholds for a single OVF monitoring site.
// All depth values are in the same unit (typically metres).
type SiteConfig struct {
	// PipeFullDepth is the depth at which the mainline is 100% full.
	// Yellow threshold: depth > PipeFullDepth.
	PipeFullDepth float64

	// BottomOverflowInvert is the invert elevation of the lowest overflow
	// outlet. Red threshold: depth >= BottomOverflowInvert.
	BottomOverflowInvert float64
}

// Validate returns an error if the config is physically inconsistent.
func (c SiteConfig) Validate() error {
	if c.PipeFullDepth <= 0 {
		return fmt.Errorf("threshold: PipeFullDepth must be > 0, got %.4g", c.PipeFullDepth)
	}
	if c.BottomOverflowInvert <= c.PipeFullDepth {
		return fmt.Errorf(
			"threshold: BottomOverflowInvert (%.4g) must be > PipeFullDepth (%.4g)",
			c.BottomOverflowInvert, c.PipeFullDepth,
		)
	}
	return nil
}

// Classify returns the overflow State for a given instantaneous depth reading,
// and whether an overflow was observed within the lookback window.
//
// Parameters:
//
//	depth           — current sensor depth reading
//	cfg             — site physical thresholds
//	recentOverflow  — true if any depth reading in the past 48 hours
//	                  was >= cfg.BottomOverflowInvert (caller computes
//	                  this via window.Since + any value >= threshold)
//
// Priority order (Red > Orange > Yellow > Green) matches the SoW.
func Classify(depth float64, cfg SiteConfig, recentOverflow bool) State {
	// Red: currently overflowing
	if depth >= cfg.BottomOverflowInvert {
		return Red
	}
	// Orange: was overflowing recently (but not right now)
	if recentOverflow {
		return Orange
	}
	// Yellow: mainline surcharging
	if depth > cfg.PipeFullDepth {
		return Yellow
	}
	return Green
}

// PipeFillRatio returns the fraction of pipe capacity currently used [0, ∞).
// Values > 1.0 indicate surcharge (mainline above 100% full).
// Returns 0 if PipeFullDepth is zero (guards against division by zero).
func PipeFillRatio(depth float64, cfg SiteConfig) float64 {
	if cfg.PipeFullDepth == 0 {
		return 0
	}
	return depth / cfg.PipeFullDepth
}

// STSAlarmType represents the alarm conditions for STS (regular flow monitor)
// sites, which have different trigger logic to OVF sites per the SoW.
type STSAlarmType int

const (
	STSNoAlarm      STSAlarmType = iota
	STSDepthAlarm                // 1.8m from surface
	STSMaxSensor                 // max sensor depth reached
	STSGroundAlarm               // ground level alarm
)

// String returns the alarm label shown in the Peel drilldown view.
func (a STSAlarmType) String() string {
	switch a {
	case STSNoAlarm:
		return "No Alarm"
	case STSDepthAlarm:
		return "1.8m From Surface"
	case STSMaxSensor:
		return "Max Sensor Depth"
	case STSGroundAlarm:
		return "Ground Alarm"
	default:
		return fmt.Sprintf("STSAlarmType(%d)", int(a))
	}
}

// STSConfig holds the physical thresholds for an STS flow monitoring site.
type STSConfig struct {
	// SurfaceElevation is the ground surface depth reference (metres).
	SurfaceElevation float64
	// MaxSensorDepth is the maximum measurable depth of the sensor.
	MaxSensorDepth float64
	// GroundAlarmDepth is the depth at which the ground alarm triggers.
	GroundAlarmDepth float64
}

// ClassifySTS evaluates which (if any) alarm condition is active for an STS
// site. Returns the highest-priority alarm and whether the site should be
// visible on the map (per SoW: only visible during active or recent alarm).
//
// Priority: GroundAlarm > MaxSensor > DepthAlarm > NoAlarm.
func ClassifySTS(depth float64, cfg STSConfig, recentAlarm bool) (alarm STSAlarmType, visible bool) {
	if depth >= cfg.GroundAlarmDepth && cfg.GroundAlarmDepth > 0 {
		return STSGroundAlarm, true
	}
	if depth >= cfg.MaxSensorDepth && cfg.MaxSensorDepth > 0 {
		return STSMaxSensor, true
	}
	if cfg.SurfaceElevation > 0 && (cfg.SurfaceElevation-depth) <= 1.8 {
		return STSDepthAlarm, true
	}
	// no active alarm — still visible if alarm occurred within 48h
	if recentAlarm {
		return STSNoAlarm, true
	}
	return STSNoAlarm, false
}

// OverflowEvent marks the start and end of a detected overflow period.
// Used by the event detector to build the 30-day event table in the SoW.
type OverflowEvent struct {
	// Start is the first timestamp at which depth >= BottomOverflowInvert.
	Start int64 // Unix seconds
	// End is the last timestamp at which depth >= BottomOverflowInvert.
	// Zero means the overflow is still active.
	End int64 // Unix seconds
}

// Active reports whether the overflow event is still ongoing.
func (e OverflowEvent) Active() bool { return e.End == 0 }

// DurationSeconds returns the event duration. Returns 0 for active events.
func (e OverflowEvent) DurationSeconds() int64 {
	if e.Active() {
		return 0
	}
	return e.End - e.Start
}

// DetectEvents scans a time series of (timestamp, depth) pairs and returns
// all overflow events where depth >= invertDepth.
//
// This is the batch version used for the 30-day event table and PDF reports.
// The streaming equivalent is handled by the Flock handler tracking state
// transitions via Classify().
func DetectEvents(timestamps []int64, depths []float64, invertDepth float64) []OverflowEvent {
	if len(timestamps) != len(depths) {
		panic("threshold: DetectEvents: timestamps and depths length mismatch")
	}

	var events []OverflowEvent
	var current *OverflowEvent

	for i, d := range depths {
		overflowing := d >= invertDepth
		if overflowing && current == nil {
			// start new event
			e := OverflowEvent{Start: timestamps[i]}
			current = &e
		} else if !overflowing && current != nil {
			// close event on the previous sample
			current.End = timestamps[i-1]
			events = append(events, *current)
			current = nil
		}
	}

	// close any still-active event at end of series
	if current != nil {
		current.End = 0 // active
		events = append(events, *current)
	}

	return events
}
