// Package logreg implements L2-regularised logistic regression for binary
// event prediction.
//
// Designed for imbalanced datasets (rare overflow events) with few positive
// examples. Key features:
//   - L2 regularisation (ridge) to prevent overfitting on sparse positive class
//   - Class weights to handle severe class imbalance
//   - Gradient descent with configurable learning rate and iterations
//   - Calibrated probability output via sigmoid
//   - Full coefficient explainability
//
// Intended use: predict P(overflow event in next H hours) from antecedent
// conditions and weather forecast features.
package logreg

import (
	"errors"
	"fmt"
	"math"
)

// Params holds fitted model coefficients.
// All fields exported for explainability.
type Params struct {
	// Weights holds one coefficient per feature.
	Weights []float64
	// Bias is the intercept term.
	Bias float64
	// FeatureNames optionally labels each weight for human-readable output.
	FeatureNames []string
}

// Summary returns a human-readable coefficient table.
func (p Params) Summary() string {
	s := fmt.Sprintf("Logistic Regression (bias=%.4f)\n", p.Bias)
	for i, w := range p.Weights {
		name := fmt.Sprintf("x%d", i)
		if i < len(p.FeatureNames) {
			name = p.FeatureNames[i]
		}
		s += fmt.Sprintf("  %-28s %.6f\n", name, w)
	}
	return s
}

// Config controls model fitting.
type Config struct {
	// L2Lambda is the L2 regularisation strength. Higher = more shrinkage.
	// Default: 0.1. Use higher values (1.0-10.0) when positive examples < 10.
	L2Lambda float64
	// LearningRate for gradient descent. Default: 0.01.
	LearningRate float64
	// MaxIter is the maximum number of gradient descent iterations. Default: 1000.
	MaxIter int
	// Tol is the convergence tolerance on loss change. Default: 1e-6.
	Tol float64
	// PosWeight is the weight applied to positive examples to handle class
	// imbalance. Set to (n_negative / n_positive) for balanced treatment.
	// Default: 1.0 (no reweighting).
	PosWeight float64
	// FeatureNames optionally labels coefficients for human-readable output.
	FeatureNames []string
}

func (c *Config) l2() float64 {
	if c.L2Lambda <= 0 {
		return 0.1
	}
	return c.L2Lambda
}

func (c *Config) lr() float64 {
	if c.LearningRate <= 0 {
		return 0.01
	}
	return c.LearningRate
}

func (c *Config) maxIter() int {
	if c.MaxIter <= 0 {
		return 1000
	}
	return c.MaxIter
}

func (c *Config) tol() float64 {
	if c.Tol <= 0 {
		return 1e-6
	}
	return c.Tol
}

func (c *Config) posWeight() float64 {
	if c.PosWeight <= 0 {
		return 1.0
	}
	return c.PosWeight
}

// Model is the logistic regression estimator.
type Model struct {
	cfg    Config
	params Params
	fitted bool
	// training stats for calibration reporting
	nPos int
	nNeg int
}

// New creates an unfitted model.
func New(cfg Config) *Model {
	return &Model{cfg: cfg}
}

// Params returns the fitted coefficients.
func (m *Model) Params() Params { return m.params }

// Fit trains the model using gradient descent.
// X is an n×p feature matrix (n observations, p features).
// y is a binary label vector (0 or 1) of length n.
func (m *Model) Fit(X [][]float64, y []float64) error {
	if len(X) == 0 {
		return errors.New("logreg: empty training set")
	}
	n := len(X)
	if len(y) != n {
		return fmt.Errorf("logreg: X has %d rows but y has %d", n, len(y))
	}
	p := len(X[0])
	for i, row := range X {
		if len(row) != p {
			return fmt.Errorf("logreg: row %d has %d features, want %d", i, len(row), p)
		}
	}

	// count class balance
	for _, label := range y {
		if label > 0.5 {
			m.nPos++
		} else {
			m.nNeg++
		}
	}

	// sample weights
	weights := make([]float64, n)
	posW := m.cfg.posWeight()
	for i, label := range y {
		if label > 0.5 {
			weights[i] = posW
		} else {
			weights[i] = 1.0
		}
	}

	// initialise parameters
	w := make([]float64, p)
	b := 0.0

	lr := m.cfg.lr()
	lambda := m.cfg.l2()
	prevLoss := math.Inf(1)

	for iter := 0; iter < m.cfg.maxIter(); iter++ {
		// forward pass + compute gradients
		dw := make([]float64, p)
		db := 0.0
		loss := 0.0

		for i, row := range X {
			yhat := sigmoid(dot(w, row) + b)
			err := yhat - y[i]
			wi := weights[i]

			for j := range w {
				dw[j] += wi * err * row[j]
			}
			db += wi * err

			// weighted cross-entropy loss
			eps := 1e-12
			if y[i] > 0.5 {
				loss -= wi * math.Log(yhat+eps)
			} else {
				loss -= wi * math.Log(1-yhat+eps)
			}
		}

		// L2 regularisation on weights (not bias)
		for j := range w {
			dw[j] = (dw[j] + lambda*w[j]) / float64(n)
		}
		db /= float64(n)
		loss = loss/float64(n) + 0.5*lambda*l2norm(w)/float64(n)

		// update
		for j := range w {
			w[j] -= lr * dw[j]
		}
		b -= lr * db

		// convergence check
		if math.Abs(prevLoss-loss) < m.cfg.tol() {
			break
		}
		prevLoss = loss
	}

	m.params = Params{
		Weights:      w,
		Bias:         b,
		FeatureNames: m.cfg.FeatureNames,
	}
	m.fitted = true
	return nil
}

// Predict returns P(y=1) for each row in X.
func (m *Model) Predict(X [][]float64) ([]float64, error) {
	if !m.fitted {
		return nil, errors.New("logreg: model not fitted")
	}
	out := make([]float64, len(X))
	for i, row := range X {
		if len(row) != len(m.params.Weights) {
			return nil, fmt.Errorf("logreg: row %d has %d features, want %d",
				i, len(row), len(m.params.Weights))
		}
		out[i] = sigmoid(dot(m.params.Weights, row) + m.params.Bias)
	}
	return out, nil
}

// PredictOne returns P(y=1) for a single feature vector.
func (m *Model) PredictOne(x []float64) (float64, error) {
	if !m.fitted {
		return 0, errors.New("logreg: model not fitted")
	}
	if len(x) != len(m.params.Weights) {
		return 0, fmt.Errorf("logreg: x has %d features, want %d",
			len(x), len(m.params.Weights))
	}
	return sigmoid(dot(m.params.Weights, x) + m.params.Bias), nil
}

// TrainingStats returns the number of positive and negative training examples.
func (m *Model) TrainingStats() (nPos, nNeg int) {
	return m.nPos, m.nNeg
}

// --- math helpers ---

func sigmoid(z float64) float64 {
	if z > 500 {
		return 1.0
	}
	if z < -500 {
		return 0.0
	}
	return 1.0 / (1.0 + math.Exp(-z))
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func l2norm(w []float64) float64 {
	var s float64
	for _, v := range w {
		s += v * v
	}
	return s
}
