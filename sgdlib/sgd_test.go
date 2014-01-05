/*
Tests for SGD

Ref:
http://golang.org/doc/code.html#Testing
http://www.golang-book.com/12

*/

package sgdlib

import (
	"fmt"
	"testing"
)

func TestLinear(t *testing.T) {

	// create a generator of test data
	β := []float64{1, 2, 3}
	getTestDataChan := make(chan Obs)
	go Lin_reg_gen(β, getTestDataChan)

	// make all the channels
	dataChan := make(chan Obs)
	paramsChan := make(chan Model)
	stateChan := make(chan chan Model)
	quitChan := make(chan bool)
	responseChan := make(chan Model)

	// set the sgd kernel going
	go Sgd(dataChan, paramsChan, stateChan, quitChan)

	// initialise the kernel
	// TODO does this really need to be done through a channel?
	rate := Rate{}
	θ := []float64{1, 2, 3}
	paramsChan <- Model{
		Theta0:     θ,
		Loss_func:  "linear",
		Learn_rate: rate,
		Eta_func:   "inverse",
	}

	// run the sgd
	for i := 0; i < 100; i++ {
		// send through a data point
		o := <-getTestDataChan
		dataChan <- o
		// get the state
		stateChan <- responseChan
		// print it
		fmt.Println(<-responseChan)
	}

	// now the sgd has seen a whole bunch of data points, where's it at?
	finalState := <-responseChan
	// hopefully we converged on the right answer..
	if finalState.Theta_hat[0] < 0.90 || finalState.Theta_hat[0] > 1.10 {
		t.Errorf("Did not converge!")
	}
}

/*

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

func TestLinearConc(t *testing.T) {
	data := make(chan []float64)
	kill := make(chan bool)

	for i := 0; i < 100; i++ {
		// Draw from GLM
		// Transform RNG to SGD input
		// Send to SGD
	}

	// Check convergence
}
*/
