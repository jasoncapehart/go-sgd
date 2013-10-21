/*
Stochastic Gradient Descent (SGD)

Input
    theta_0 : initial array of parameters
    eta : initial step size
    loss_func : type of loss function to use "linear", "logistic", etc.

Output
    theta_hat : array
    eta : current step size

Psuedo-Code from Murphy pg. 264
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

// Not quite SGD
func sgd1(y []float64, x []float64, theta0 []float64) (theta1 []float64) {
	param_count := len(theta0)
    theta1 = make([]float64, param_count)

	for i := 0; i < param_count; i++ {
		theta1[i] = grad_linear_loss(y[i], x[i], theta0[i])
	}
	return theta1
}


// SGD w/ no step size
func sgd2(y float64, x []float64, theta0 []float64, loss_func string) (theta1 []float64) {
    param_count := len(theta0)
    theta1 = make([]float64, param_count)

    for i := 0; i < param_count; i++ {
        theta1[i] = loss_map[loss_func](y, x[i], theta0[i])
    }
    return theta1
}

// SGD w/ step size
// First reasonable approximation of SGD
func sgd3(y float64, x []float64, theta0 []float64, loss_func string, step_n int) (theta_hat []float64) {
    param_count := len(theta0)
    theta_hat = make([]float64, param_count)
    theta_hat = theta0
    step_size := (1/(float64(step_n) + 1))
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

func main() {
    // Do Anything
	fmt.Printf("hello world \n")

    // Test the loss func
    test := grad_linear_loss(2, 1, 2)
    fmt.Printf("Loss Func: %v \n", test)

    // Pass variable len slices to a func
    var y = []float64{2, 4, 6, 8}
    var x = []float64{1, 2, 3, 4}
    var theta0 = []float64{2, 2, 2, 2}

	test2 := sgd1(y, x, theta0)
    fmt.Printf("SGD Func: %v \n", test2)

    // Pass loss function to sgd
    test3 := sgd2(y[0], x, theta0, "linear")
    fmt.Printf("SGD Func: %v \n", test3)

    // Pass logistic loss function to sgd
    var a = []float64{0, 0.25, 0.5, 1}
    var b = []float64{1, 2, 3, 4}
    var c = []float64{1, 1, 1 ,1}

    test4 := sgd2(a[0], b, c, "logistic")
    fmt.Printf("SGD Func: %v \n", test4)

    // 1d lin reg test w/ no step size
    fmt.Printf("SGD w/ no step size \n")
    x1, y1 := lin_reg_rng(100, 2)
    theta_est := []float64{1}
    for i := 1; i < 10; i ++ {
        theta_est := sgd2(y1[i], x1[i], theta_est, "linear")
        fmt.Printf("Loop: %v , Est: %v \n", i, theta_est)
    }

    // 1d lin reg test w/ step size
    fmt.Printf("SGD step size \n")
    x2, y2 := lin_reg_rng(100, 2)
    theta_est1 := []float64{1}
    for i := 1; i < 20; i ++ {
        theta_est1 := sgd3(y2[i], x2[i], theta_est1, "linear", i)
        fmt.Printf("Loop: %v , Est: %v \n", i, theta_est1)
    }

}

