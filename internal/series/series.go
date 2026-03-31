// Package series provides stateless numeric primitives for time series
// manipulation. All functions operate on plain []float64 slices and allocate
// minimally — callers are expected to pre-allocate output buffers where
// performance matters.
package series

import "math"

// Diff returns the d-th order difference of src.
// len(result) == len(src) - d.
// Panics if d >= len(src).
func Diff(src []float64, d int) []float64 {
	out := make([]float64, len(src))
	copy(out, src)
	for i := 0; i < d; i++ {
		for j := len(out) - 1; j >= 1; j-- {
			out[j] = out[j] - out[j-1]
		}
		out = out[1:]
	}
	return out
}

// SeasonalDiff returns the D-th order seasonal difference of src at period m.
// len(result) == len(src) - D*m.
func SeasonalDiff(src []float64, D, m int) []float64 {
	out := make([]float64, len(src))
	copy(out, src)
	for i := 0; i < D; i++ {
		n := len(out)
		next := make([]float64, n-m)
		for j := m; j < n; j++ {
			next[j-m] = out[j] - out[j-m]
		}
		out = next
	}
	return out
}

// InvertDiff reconstructs the original series from a differenced series and
// the d seed values (the first d observations of the original series).
// len(seeds) must equal d.
func InvertDiff(diffed []float64, seeds []float64) []float64 {
	d := len(seeds)
	out := make([]float64, d+len(diffed))
	copy(out, seeds)
	for i, v := range diffed {
		out[d+i] = out[d+i-1] + v
	}
	return out
}

// Dot returns the inner product of a and b. Panics if lengths differ.
func Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("series: Dot length mismatch")
	}
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Mean returns the arithmetic mean of v. Returns 0 for empty slice.
func Mean(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	var s float64
	for _, x := range v {
		s += x
	}
	return s / float64(len(v))
}

// Variance returns the sample variance of v (Bessel-corrected).
func Variance(v []float64) float64 {
	if len(v) < 2 {
		return 0
	}
	mu := Mean(v)
	var s float64
	for _, x := range v {
		d := x - mu
		s += d * d
	}
	return s / float64(len(v)-1)
}

// Autocorr returns the autocorrelation of v at lag k.
func Autocorr(v []float64, k int) float64 {
	n := len(v)
	if k >= n {
		return 0
	}
	mu := Mean(v)
	var num, den float64
	for i := k; i < n; i++ {
		num += (v[i] - mu) * (v[i-k] - mu)
	}
	for i := 0; i < n; i++ {
		d := v[i] - mu
		den += d * d
	}
	if den == 0 {
		return 0
	}
	return num / den
}

// RMSE returns the root mean squared error between predicted and observed.
// Panics if lengths differ or slice is empty.
func RMSE(predicted, observed []float64) float64 {
	if len(predicted) != len(observed) {
		panic("series: RMSE length mismatch")
	}
	var s float64
	for i := range predicted {
		d := predicted[i] - observed[i]
		s += d * d
	}
	return math.Sqrt(s / float64(len(predicted)))
}

// Lag returns a copy of v shifted right by k positions, zero-padded on the
// left. Useful for constructing regressor matrices.
func Lag(v []float64, k int) []float64 {
	out := make([]float64, len(v))
	for i := k; i < len(v); i++ {
		out[i] = v[i-k]
	}
	return out
}
