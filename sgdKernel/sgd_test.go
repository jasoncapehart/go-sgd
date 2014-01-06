package sgd

import (
	"log"
	"math/rand"
	"testing"
)

func linearModel(β []float64, getChan chan chan obs, quitChan chan bool) {
	for {
		select {
		case resChan := <-getChan:
			y := 0.0
			x := make([]float64, len(β))
			for i, βi := range β {
				x[i] = rand.Float64()
				y += βi * x[i]
			}
			resChan <- obs{
				x: x,
				y: y,
			}
		case <-quitChan:
			return
		}
	}
}

func TestSgd(t *testing.T) {

	// model
	β := []float64{1, 2, 3}
	getChan := make(chan chan obs)
	modelQuitChan := make(chan bool)
	go linearModel(β, getChan, modelQuitChan)

	// sgdkernel
	dataChan := make(chan obs)
	paramChan := make(chan params)
	stateChan := make(chan chan []float64)
	kernelQuitChan := make(chan bool)
	θ_0 := []float64{1, 1, 1}
	go SgdKernel(dataChan, paramChan, stateChan, kernelQuitChan,
		grad_linear_loss, eta_inverse, θ_0)

	// test
	var θ []float64
	modelRespChan := make(chan obs)
	kernelRespChan := make(chan []float64)
	for i := 0; i < 100; i++ {
		// get data
		getChan <- modelRespChan
		obs := <-modelRespChan
		// send to kernel
		dataChan <- obs
		// get state from kernel
		stateChan <- kernelRespChan
		// ... in order to print it
		θ = <-kernelRespChan
		log.Println(θ)
	}

	if θ[0] < 0.90 || θ[0] > 1.10 {
		t.Errorf("Did not converge!")
	}

}
