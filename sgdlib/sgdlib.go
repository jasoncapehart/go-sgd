/*
Stochastic Gradient Descent (SGD)
=================================

Input
    theta_0 : initial array of parameters
    eta : initial step size
    loss_func : type of loss function to use "linear", "logistic", etc.

Output
    theta_hat : array of parameter estimates
    eta : current step size

Ref "Maching Learning from a Probabilistic Perspective" by Murphy pg. 264

Pseudo-code
-----------
Initialize theta, eta
repeat
    Randomly permute data
    for i = 1:N do
        g = grad(f(theta, (y_i, x_i)))
        theta_i+1 = theta_i - eta * g
        Update eta
until converged

SGD w/ Step Size
Complexity O(n)
Changes
    x  - Each theta update needs to be taken wrt all parameters
    x  - Need to figure out how to specify intercept for lin and log reg
            - Specify that data must be centered
    eta needs to be a list to handle per-parameter step sizes (e.g. adagrad)
    Step size needs to be a parameter in the function with a dict data structure

*/

package sgdlib

import (
	// "fmt"
    "math"
    "math/rand"
    "time"
)

// TODO: Add map{} for step_size functions

// Learning Rate Schedule 
//=======================

type eta_func func(k int) (eta float64)

var eta_map = map[string]eta_func {
    "inverse":eta_inverse,
}

func eta_inverse(k int) (eta float64) {
    eta = 1 / float64(k)
    return eta
}

// Channel Types
//======================
type model struct {
    Y float64
    X []float64
    Theta0 []float64
    Loss_func string
    Eta int
}

type theta_hat []float64

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
    // g = x_i * (y_est(theta, x_i) - y_i)
    var y_est float64
    grad = make([]float64, len(theta))

    // y_est = theta * x_i
    for i := 0; i < len(theta); i++ {
	   y_est = y_est + x[i] * theta[i]
    }

    // grad = (y - y_est) * x_i
    for i := 0; i < len(theta); i++ {
        grad[i] = (y - y_est) * x[i]
    }

	return grad
}

func grad_logistic_loss(y float64, x []float64, theta []float64) (grad []float64) {
    // grad = ( y - 1 / math.Exp(-(x * theta)) ) * x
    var y_est float64
    grad = make([]float64, len(theta))

    // y_est = theta * x_i
    for i := 0; i < len(theta); i++ {
        y_est = y_est + x[i] * theta[i]
    }

    // grad = (y - y_est) * x_i
    for i := 0; i < len(theta); i++ {
        grad[i] = (y - logit(y_est)) * x[i]
    }

    return grad
}

// Data Generators
//===============================
// TODO: Lin and Log reg could be same func by passing in a "linear" or "logit" func

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

// TODO: Add mean and sd parameters

func Gen_lin_model_rng(betas []float64, beta0 float64, link_func string, out chan []float64) {
    time := time.NewTicker(time.Duration(1) * time.Second)
    for {
        select {
        case <- time.C:
            n := len(betas)
            x := make([]float64, n)
            model_rnd := make([]float64, n + 1)
            y := beta0
            for i := 0; i < n; i ++ {
                x[i] = rand.Float64()
                model_rnd[i + 1] = x[i]
                y = y + betas[i] * x[i]
            }
            y = link_map[link_func](y + rand.NormFloat64())
            model_rnd[0] = y
            out <- model_rnd
        }
    }
}

// SGD Kernel
//=============================

func Sgd(y float64, x []float64, theta0 []float64, loss_func string, eta int) (theta_hat []float64) {
    theta_hat = make([]float64, len(theta0))
    step_size := (1/(float64(eta) + 1))

    grad := loss_map[loss_func](y, x, theta0)

    for i := 0; i < len(theta0); i++ {
        theta_hat[i] = theta0[i] + step_size * grad[i]
    }

    return theta_hat
}

func Sgd_online(input chan model, output chan theta_hat) {
    for {
        select {
        case model := <-input:
            theta_est := make([]float64, len(model.Theta0))

            step_size := (1/(float64(model.Eta) + 1))
            grad := loss_map[model.Loss_func](model.Y, model.X, model.Theta0)

            for i := 0; i < len(model.Theta0); i++ {
                theta_est[i] = model.Theta0[i] + step_size * grad[i]
            }
            output <-theta_est
        }
    }
}

