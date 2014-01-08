package sgd

import (
        "math"
)

// step size update rules
func eta_inverse(K float64, τ float64, κ float64) float64 {
        return 1 / K
}
func eta_bottou(K float64, τ float64, κ float64) float64 {
        return math.Pow((τ + K), κ)
}

// Link Functions
func identity(x float64) float64 {
        return x
}

func logit(x float64) float64 {
        return 1 / (1 + math.Exp(-x))
}

// Loss Functions
func grad_linear_loss(x []float64, y float64, θ []float64) []float64 {
        grad := make([]float64, len(θ))
        y_est := 0.0
        for i, θi := range θ {
                y_est += x[i] * θi
        }
        for i, xi := range x {
                grad[i] = (y - y_est) * xi
        }
        return grad
}

func grad_logistic_loss(x []float64, y float64, θ []float64) []float64 {
        // grad = ( y - 1 / math.Exp(-(x * theta)) ) * x
        grad := make([]float64, len(θ))
        y_est := 0.0
        for i, θi := range θ {
                y_est += x[i] * θi
        }

        for i, xi := range x {
                grad[i] = (y - logit(y_est)) * xi
        }
        return grad
}
