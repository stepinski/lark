package pipeline

import "time"

// OverflowEvent marks a contiguous overflow period by observation indices.
type OverflowEvent struct {
	Start int
	End   int
}

// EvalResult holds the confusion matrix counts and lead times
// for a single site evaluation.
type EvalResult struct {
	TruePositives int
	FalseAlarms   int
	LeadTimes     []int
	AvgLeadTime   float64
}

// EvaluateProbs scores predicted probabilities against observed overflow events.
// It counts an alarm when p > 0.5 and depth exceeds the excess gate (invert - 100mm).
// If floatBased is true, the excess gate check is skipped (float channel is ground truth).
// If an overflow event occurs within the horizon, it's a true positive.
// If no overflow occurs within the horizon, it's a false alarm.
func EvaluateProbs(probs []float64, X [][]float64, rows []Obs, events []OverflowEvent, invert float64, horizon int, floatBased bool) EvalResult {
	excessGate := invert - 100.0
	truePositives := 0
	falseAlarms := 0
	var leadTimes []int

	for i, p := range probs {
		if p > 0.5 && (!floatBased && rows[i].DepthMM >= excessGate || floatBased) {
			alarmTime := rows[i].T
			found := false

			for _, ev := range events {
				evTime := rows[ev.Start].T
				if evTime.After(alarmTime) || evTime.Equal(alarmTime) {
					lead := int(evTime.Sub(alarmTime).Minutes())
					if lead >= 0 && lead <= horizon*5 {
						truePositives++
						leadTimes = append(leadTimes, lead)
						found = true
						break
					}
				}
			}

			if !found {
				falseAlarms++
			}
		}
	}

	var avgLeadTime float64
	if len(leadTimes) > 0 {
		sum := 0
		for _, lt := range leadTimes {
			sum += lt
		}
		avgLeadTime = float64(sum) / float64(len(leadTimes))
	}

	return EvalResult{
		TruePositives: truePositives,
		FalseAlarms:   falseAlarms,
		LeadTimes:     leadTimes,
		AvgLeadTime:   avgLeadTime,
	}
}

// Dummy reference to time to avoid unused import
var _ = time.Minute
