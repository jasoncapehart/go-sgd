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
    // "math"
    "math/rand"
)

// TODO: Add map{} for step_size functions

type loss_func func(y float64, x []float64, theta []float64) (grad []float64)

type lossFuns map[string]loss_func

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
    grad = make([]float64, 1)
    return grad
}

/*
SGD w/ Step Size
Complexity O(n)
Changes
    x  - Each theta update needs to be taken wrt all parameters
    x  - Need to figure out how to specify intercept for lin and log reg
            - Specify that data must be centered
    eta needs to be a list to handle per-parameter step sizes (e.g. adagrad)
    Step size needs to be a parameter in the function with a dict data structure
*/

func Sgd(y float64, x []float64, theta0 []float64, loss_func string, eta int) (theta_hat []float64) {
    theta_hat = make([]float64, len(theta0))
    step_size := (1/(float64(eta) + 1))

    grad := loss_map[loss_func](y, x, theta0)

    for i := 0; i < len(theta0); i++ {
        theta_hat[i] = theta0[i] + step_size * grad[i]
        fmt.Printf("%v \n", theta_hat)
    }

    return theta_hat
}

// Lin reg RNG
func Lin_reg_gen(n int, betas []float64, beta0 float64) (x [][]float64, y []float64) {
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

/*
// 2d lin reg RNG
func Lin_reg_rng(n int, slope float64, intcp float64) (x [][]float64, y []float64) {
    x = make([][]float64, n)
    y = make([]float64, n)

    for i := 0; i < n; i++ {
       var rnd = []float64{rand.Float64()}
       x[i] = rnd
       y[i] = rnd[0]*slope + intcp + rand.NormFloat64()
    }
    return x, y
}
*/


