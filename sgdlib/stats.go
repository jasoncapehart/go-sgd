/*
Stat Functions for SGD
======================

1. Link Functions
    - Identity
    - Logit
2. Loss Function
    - Linear
    - Logistic
3. Model RNG
    - GLM

*/

package stats

import (
    "math"
    "math/rand"
    "time"
)

// Link Functions
//===================================
type link_func func(x float64) (y float64)

var link_map = map[string]link_func {
    "identity":identity,
    "logit":logit,
}

func identity(x float64) (y float64) {
    y = x
    return y
}

func logit(x float64) (y float64) {
    y = 1 / (1 + math.Exp(-x))
    return y
}


// Loss Functions
//====================================

type loss_func func(y float64, x []float64, theta []float64) (grad []float64)

var loss_map = map[string]loss_func {
    "linear":grad_linear_loss,
    "logistic":grad_logistic_loss,
}

func grad_linear_loss(y float64, x []float64, theta []float64) (grad []float64) {
    // grad = x_i * (y_est(theta, x_i) - y_i)
    var y_est float64
    grad = make([]float64, len(theta))

    for i := 0; i < len(theta); i++ {
	   y_est = y_est + x[i] * theta[i]
       grad = (y - y_est) * x[i]
    }

	return grad
}

func grad_logistic_loss(y float64, x []float64, theta []float64) (grad []float64) {
    // grad = ( y - 1 / math.Exp(-(x * theta)) ) * x
    var y_est float64
    grad = make([]float64, len(theta))

    for i := 0; i < len(theta); i++ {
        y_est = y_est + x[i] * theta[i]
        grad[i] = (y - logit(y_est)) * x[i]
    }

    return grad
}

// Data Generators
//===============================

// Lin reg RNG
func Lin_reg_gen(n int, betas []float64, beta0 float64) (x [][]float64, y []float64) {
    // TODO: The size of x is known beforehand, so allocate a fixed 2d array 
    x = make([][]float64, n)
    y = make([]float64, n)

    for i := 0; i < n; i++ {
        y[i] = beta0
        x[i] = make([]float64, len(betas))
        for j := 0; j < len(betas); j++ {
            x[i][j] = rand.Float64()
            y[i] = y[i] + betas[j] * x[i][j] + rand.NormFloat64()
        }
    }
    return x, y
}

// Log reg RNG
func Log_reg_gen(n int, betas []float64, beta0 float64) (x [][]float64, y []float64) {
    x = make([][]float64, n)
    y = make([]float64, n)

    for i := 0; i < n; i++ {
        y[i] = beta0
        x[i] = make([]float64, len(betas))
        for j := 0; j < len(betas); j++ {
            x[i][j] = rand.Float64()
            y[i] = y[i] + betas[j] * x[i][j] + rand.NormFloat64()
        }
        y[i] = logit(y[i])
    }

    return x, y
}

// GLM generation
type Glm_gen struct {
    Betas []float64
    Beta0 float64
    Link_func string
}

type Obs {
    Y float64
    X []float64
}

// TODO: Add mean and sd parameter for noise

func Glm_rng(params Glm_gen, out chan Obs) {
    for {
        select {
        case params:
            n := len(params.Betas)
            data := Obs{}
            data.X := make([]float64, n)
            data.Y := model.Beta0

            for i := 0; i < n; i ++ {
                data.X[i] = rand.Float64()
                data.Y = data.Y + params.Betas[i] * data.X[i]
            }

            data.Y = link_map[params.Link_func](data.Y + rand.NormFloat64())
            out <-data
        }
    }
}

