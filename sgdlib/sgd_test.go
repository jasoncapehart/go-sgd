/*
Tests for SGD

Ref:
http://golang.org/doc/code.html#Testing
http://www.golang-book.com/12

*/

package sgdlib

import (
	"fmt"
	"log"
	"testing"
)

func TestLinear(t *testing.T) {

	// true params
	β := []float64{1, 2, 3}

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
	θ := []float64{1, 1, 1}
	paramsChan <- Model{
		Theta0:     θ,
		Loss_func:  "linear",
		Learn_rate: rate,
		Eta_func:   "inverse",
	}

	// run the sgd
	var state Model
	for i := 0; i < 500; i++ {
		// send through a data point
		o := Lin_reg_gen(β)
		dataChan <- o
		// get the state
		stateChan <- responseChan
		state = <-responseChan
		// print it
		fmt.Printf("%+v\n", state)
	}

	quitChan <- true

	// now the sgd has seen a whole bunch of data points, where's it at?
	// hopefully we converged on the right answer..
	log.Println(state.Theta_hat)
	if state.Theta_hat[0] < 0.90 || state.Theta_hat[0] > 1.10 {
		t.Errorf("Did not converge!")
	}
	log.Println("done")
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
