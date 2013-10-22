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
    //"github.com/jasoncapehart/go-sgd/sgdlib"
)

func Test(t *testing.T) {
    x, y := Lin_reg_rng(100, 2)
    theta_est := []float64{1}

    for i := 1; i < 20; i++ {
        theta_est = Sgd(y[i], x[i], theta_est, "linear", i)
    }

    if theta_est[0] < 1.90 || theta_est[0] > 2.10 {
        t.Errorf("Did not converge!")
    }
}
