package sgd

import (
        "log"
        "math/rand"
        "testing"
)

func linearModel(β []float64, getChan chan chan obs, quitChan chan bool) {
        σ := 0.3
        for {
                select {
                case resChan := <-getChan:
                        y := 0.0
                        x := make([]float64, len(β))
                        for i, βi := range β {
                                x[i] = rand.NormFloat64() * float64(i+1)
                                y += βi * x[i]
                        }
                        resChan <- obs{
                                x: x,
                                y: y + rand.NormFloat64()*σ,
                        }
                case <-quitChan:
                        return
                }
        }
}

func logisticModel(β []float64, getChan chan chan obs, quitChan chan bool) {
        for {
                select {
                case resChan := <-getChan:
                        y := 0.0
                        x := make([]float64, len(β))
                        for i, βi := range β {
                                x[i] = rand.NormFloat64() * float64(i+1)
                                y += βi * x[i]
                        }
                        if logit(y) >= 0.5 {
                                y = 1
                        } else {
                                y = 0
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

func TestSgdLinear(t *testing.T) {

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
        θ_0 := []float64{2, 1, 1}
        go SgdKernel(dataChan, paramChan, stateChan, kernelQuitChan,
                grad_linear_loss, eta_inverse, θ_0)

        // test
        var θ []float64
        modelRespChan := make(chan obs)
        kernelRespChan := make(chan []float64)
        for i := 0; i < 5000; i++ {
                // get data
                getChan <- modelRespChan
                obs := <-modelRespChan
                // send to kernel
                dataChan <- obs
        }
        // get state from kernel
        stateChan <- kernelRespChan
        // ... in order to print it
        θ = <-kernelRespChan

        log.Println(θ)

        if !((0.9 < θ[0]) && (θ[0] < 1.1)) {
                t.Errorf("Failed to converge on correct θ_0")
        }
        if !((1.9 < θ[1]) && (θ[1] < 2.1)) {
                t.Errorf("Failed to converge on correct θ_1")
        }
        if !((2.9 < θ[2]) && (θ[2] < 3.1)) {
                t.Errorf("Failed to converge on correct θ_2")
        }

}

func TestSgdLogistic(t *testing.T) {

        // model
        β := []float64{1, 2, 3}
        getChan := make(chan chan obs)
        modelQuitChan := make(chan bool)
        go logisticModel(β, getChan, modelQuitChan)

        // sgdkernel
        dataChan := make(chan obs)
        paramChan := make(chan params)
        stateChan := make(chan chan []float64)
        kernelQuitChan := make(chan bool)
        θ_0 := []float64{1, 1, 1}
        go SgdKernel(dataChan, paramChan, stateChan, kernelQuitChan,
                grad_logistic_loss, eta_inverse, θ_0)

        // test
        var θ []float64
        modelRespChan := make(chan obs)
        kernelRespChan := make(chan []float64)
        for i := 0; i < 50000; i++ {
                // get data
                getChan <- modelRespChan
                obs := <-modelRespChan
                // send to kernel
                dataChan <- obs
        }
        // get state from kernel
        stateChan <- kernelRespChan
        θ = <-kernelRespChan
        log.Println(θ)

        if !((0.9 < θ[0]) && (θ[0] < 1.1)) {
                t.Errorf("Failed to converge on correct θ_0")
        }
        if !((1.9 < θ[1]) && (θ[1] < 2.1)) {
                t.Errorf("Failed to converge on correct θ_1")
        }
        if !((2.9 < θ[2]) && (θ[2] < 3.1)) {
                t.Errorf("Failed to converge on correct θ_2")
        }
}
