/*
Tests for SGD

http://golang.org/doc/code.html#Testing
http://www.golang-book.com/12

Example:

package newmath

import "testing"

func TestSqrt(t *testing.T) {
    const in, out = 4, 2
    if x := Sqrt(in); x != out {
        t.Errorf("Sqrt(%v) = %v, want %v", in, x, out)
    }
}

*/

package sgdlib

import (
    "testing"
    "fmt"
)

func Test(t *testing.T) {
    /*
    x, y := Lin_reg_rng(100, 2, 0)
    theta_est := make([]float64, 1)

    for i := 0; i < 20; i++ {
        fmt.Printf("%v \n", theta_est)
        theta_est = Sgd(y[i], x[i], theta_est, "linear", i)
    }

    if theta_est[0] < 1.90 || theta_est[0] > 2.10 {
        t.Errorf("Did not converge!")
    }
    */

    betas := []float64{2, 2}
    x, y := Lin_reg_gen(100, betas, 0)
    theta_est := make([]float64, 2)

    for i := 0; i < 20; i++ {
        fmt.Printf("%v \n", theta_est)
        theta_est = Sgd(y[i], x[i], theta_est, "linear", i)
    }

    if theta_est[0] < 1.90 || theta_est[0] > 2.10 {
        t.Errorf("Did not converge!")
    }

}
