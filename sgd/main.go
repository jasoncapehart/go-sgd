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

package main

import (
	"fmt"
    "github.com/jasoncapehart/go-sgd/sgdlib"
)

func main() {
    // Lin reg test
    betas = []float64{2, 2}
    x, y := sgdlib.Lin_reg_gen(100, betas, 0)
    theta_est := []float64{1}
    for i := 1; i < 20; i ++ {
        theta_est := sgdlib.Sgd(y[i], x[i], theta_est, "linear", i)
        fmt.Printf("Loop: %v , Est: %v \n", i, theta_est)
    }

}

