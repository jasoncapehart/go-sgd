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

package sgdlib

import (
	"fmt"
	"math"
	"math/rand"
)

// Link Functions
//===================================
type link_func func(x float64) (y float64)

var link_map = map[string]link_func{
	"identity": identity,
	"logit":    logit,
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

var loss_map = map[string]loss_func{
	"linear":   grad_linear_loss,
	"logistic": grad_logistic_loss,
}

func grad_linear_loss(y float64, x []float64, theta []float64) (grad []float64) {
	// grad = x_i * (y_est(theta, x_i) - y_i)
	var y_est float64
	grad = make([]float64, len(theta))

	for i := 0; i < len(theta); i++ {
		y_est = y_est + x[i]*theta[i]
		grad[i] = (y - y_est) * x[i]
	}

	return grad
}

func grad_logistic_loss(y float64, x []float64, theta []float64) (grad []float64) {
	// grad = ( y - 1 / math.Exp(-(x * theta)) ) * x
	var y_est float64
	grad = make([]float64, len(theta))

	for i := 0; i < len(theta); i++ {
		y_est = y_est + x[i]*theta[i]
		grad[i] = (y - logit(y_est)) * x[i]
	}

	return grad
}

// GLM generation

type Err struct {
	Mean  float64
	StDev float64
}

type Glm_gen struct {
	Betas     []float64
	Beta0     float64
	Link_func string
	Error     Err
}

// TODO: Add mean and sd parameter for noise

func glm_rng(model_params chan Glm_gen, out chan Obs) {
	for {
		select {
		case params := <-model_params:
			n := len(params.Betas)
			data := Obs{}
			data.X = make([]float64, n)
			data.Y = params.Beta0

			for i := 0; i < n; i++ {
				data.X[i] = rand.Float64()
				data.Y = data.Y + params.Betas[i]*data.X[i] + (rand.NormFloat64()*params.Error.StDev + params.Error.Mean)
			}

			data.Y = link_map[params.Link_func](data.Y)
			fmt.Printf("Obs: %v \n", data)
			out <- data
		}
	}
}

func Lin_reg_gen(β []float64) Obs {
	Y := 0.0
	X := make([]float64, len(β))
	for i, _ := range β {
		X[i] = rand.Float64()
		Y += β[i] * X[i]
	}
	return Obs{
		X: X,
		Y: Y,
	}
}
