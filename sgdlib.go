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

*/

package sgdlib

import (
	"fmt"
    "math"
    "math/rand"
)

// TODO: Add map{} for step_size functions

type loss_func func(y float64, x float64, theta0 float64) (theta1 float64)

var loss_map = map[string]loss_func {
    "linear":grad_linear_loss,
    "logistic":grad_logistic_loss,
}

func grad_linear_loss(y float64, x float64, theta float64) (grad float64) {
	grad = (y - x * theta) * x
	return grad
}

func grad_logistic_loss(y float64, x float64, theta float64) (grad float64) {
    grad = ( y - 1 / math.Exp(-(x * theta)) ) * x
    return grad
}

// SGD w/ step size
// First reasonable approximation of SGD
func sgd3(y float64, x []float64, theta0 []float64, loss_func string, eta int) (theta_hat []float64) {
    param_count := len(theta0)
    theta_hat = make([]float64, param_count)
    theta_hat = theta0
    step_size := (1/(float64(eta) + 1))
    fmt.Printf("Step Size: %v \n", step_size)

    for i := 0; i < param_count; i++ {
        theta_est := loss_map[loss_func](y, x[i], theta0[i])
        theta_hat[i] = theta_hat[i] + step_size * theta_est
    }
    return theta_hat
}

// 2d lin reg RNG
// TODO: Add intercept
func lin_reg_rng(n int, slope float64) (x [][]float64, y []float64) {
    x = make([][]float64, n)
    y = make([]float64, n)
    
    // TODO: More efficient to allocate 2d slice as a big 1d slice
    //      Example: http://golang.org/doc/effective_go.html
    for i := 0; i < n; i++ {
       var rnd = []float64{rand.Float64()}
       x[i] = rnd
       y[i] = rnd[0]*slope + rand.NormFloat64()
    }
    return x, y
}
