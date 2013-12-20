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
    "stats"
)

// TODO: Add map{} for step_size functions

// Channel Types
//======================
type Obs struct {
    Y float64
    X []float64
}

type Model struct {
    Theta0 []float64
    Loss_func string
    Learn_rate Rate
    Eta_func string
    Theta_hat []float64
    N int
}

// Learning Rate Schedule 
//=======================
// TODO: Create AdaGrad func

type Rate struct {
    K int
    Tau_0 float64
    Kappa float64
}

type eta_func func(learn_rate Rate) (eta float64)

var eta_map = map[string]eta_func {
    "inverse":eta_inverse,
    "bottou":bottou,
}

func eta_inverse(learn_rate Rate) (eta float64) {
    eta = 1 / float64(Rate.K)
    return eta
}

func bottou(learn_rate Rate) (eta float64) {
    eta = mat.Pow((Rate.Tau_0 + Rate.K), -Rate.Kappa)
    return eta
}


// SGD Kernel
//===========

//TODO: Include a check for convergence

func Sgd(data chan Obs, sgd_params chan Model, state chan Model, poll chan bool, quit chan bool) {
    var curr_state Model
    var learn_rate Rate
    for {
        select {
        // poll state
        case msg := <-poll:
            if poll == 0:
                state <-curr_state
        // initialize SGD process
        case params := <-sgd_params:
            // Set the process variables
            theta0 := params.Theta0
            loss_func := params.Loss_func
            learn_rate := params.Learn_rate
            eta_func := params.Eta_func
            n := params.N
            // Set the state for the unchanging Model vars
            curr_state.Theta0 = theta0
            curr_state.Loss_func = loss_func
            curr_state.Eta_func = eta_func
        // update state
        case obs := <-data:
            y := obs.Y
            x := obs.X
            n = n + 1
            curr_state.N = n
            curr_state.Learn_rate.K = n

            eta := eta_map[eta_func](learn_rate)
            grad := stats.loss_map[loss_func](y, x, theta0)

            for i:=0; i < len(theta0); i++ {
                theta_est[i] = theta_est[i] + eta * grad[i]
            }

            fmt.Printf("theta_hat: %v \n", theta_est)
            curr_state.Theta_hat = theta_est
        case terminate := <-quit:
            if quit == true:
               return
        }
    }
}

/*

func Sgd(y float64, x []float64, theta0 []float64, loss_func string, eta int) (theta_hat []float64) {
    theta_hat = make([]float64, len(theta0))
    step_size := (1/(float64(eta) + 1))

    grad := loss_map[loss_func](y, x, theta0)

    for i := 0; i < len(theta0); i++ {
        theta_hat[i] = theta0[i] + step_size * grad[i]
    }

    return theta_hat
}

// TODO: This interface still doesn't feel right ...

func Sgd_online(data chan Obs, sgd_params chan Model) {
    for {
        select {
        // Initialize parameters 
        case params := <-sgd_params && sgd_params.N == 0:
            theta_est := params.Theta0
            loss_func := params.Loss_func
            eta := params.Eta
        // Update state
        case obs := <-data:
            y := obs.Y
            x := obs.X
            step_size := (1/(float64(eta) + 1))
            grad := loss_map[loss_func](y, x, theta0)

            for i := 0; i < len(theta0); i++ {
                theta_est[i] = theta_est[i] + step_size * grad[i]
            }
            // Update N, theta_hat, etc.
        }
        // Poll state
        // TODO: Send sgd_params back through the channel
    }
}
*/
