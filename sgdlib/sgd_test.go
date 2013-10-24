/*
Tests for SGD

Ref:
http://golang.org/doc/code.html#Testing
http://www.golang-book.com/12

*/

package sgdlib

import (
    "testing"
    "fmt"
)

func TestLinear(t *testing.T) {
    betas := []float64{1, 2, 3}
    x, y := Lin_reg_gen(100, betas, 0)
    theta_est := []float64{1, 2, 3}

    for i := 0; i < 100; i++ {
        fmt.Printf("%v \n", theta_est)
        theta_est = Sgd(y[i], x[i], theta_est, "linear", i)
    }

    if theta_est[0] < 0.90 || theta_est[0] > 1.10 {
        t.Errorf("Did not converge!")
    }
}

func TestLogistic(t *testing.T) {
    betas := []float64{1, 2, 3}
    x, y := Log_reg_gen(100, betas, 0)
    theta_est := []float64{1, 2, 3}

    for i := 0; i < 100; i++ {
        fmt.Printf("%v \n", theta_est)
        theta_est = Sgd(y[i], x[i], theta_est, "logistic", i)
    }

    if theta_est[0] < 0.90 || theta_est[0] > 1.10 {
        t.Errorf("Did not converge!")
    }
}
