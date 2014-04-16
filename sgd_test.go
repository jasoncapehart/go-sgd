package sgd

import (
    "fmt"
    "math/rand"
    "testing"
)

func linearModel(β []float64, getChan chan chan Obs, quitChan chan bool) {
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
            resChan <- Obs{
                X: x,
                Y: y + rand.NormFloat64()*σ,
            }
        case <-quitChan:
            return
        }
    }
}

func logisticModel(β []float64, getChan chan chan Obs, quitChan chan bool) {
    for {
        select {
        case resChan := <-getChan:
            μ := 0.0
            x := make([]float64, len(β))
            for i, βi := range β {
                x[i] = rand.NormFloat64() * float64(i+1)
                μ += βi * x[i]
            }
            var y float64
            if rand.Float64() <= logit(μ) {
                y = 1
            } else {
                y = 0
            }
            resChan <- Obs{
                X: x,
                Y: y,
            }
        case <-quitChan:
            return
        }
    }
}

func TestSgdLinear(t *testing.T) {

    // model
    β := []float64{1, 2, 3}
    getChan := make(chan chan Obs)
    modelQuitChan := make(chan bool)
    go linearModel(β, getChan, modelQuitChan)

    // sgdkernel
    dataChan := make(chan Obs)
    paramChan := make(chan Params)
    stateChan := make(chan chan []float64)
    kernelQuitChan := make(chan bool)
    θ_0 := []float64{2, 1, 1}
    go SgdKernel(dataChan, paramChan, stateChan, kernelQuitChan, GradLinearLoss, EtaInverse, θ_0)

    // test
    var θ []float64
    modelRespChan := make(chan Obs)
    kernelRespChan := make(chan []float64)
    for i := 0; i < 200; i++ {
        // get data
        getChan <- modelRespChan
        obs := <-modelRespChan
        // send to kernel
        dataChan <- obs
        stateChan <- kernelRespChan
        // ... in order to print it
        θ = <-kernelRespChan
    }
    // get state from kernel
    stateChan <- kernelRespChan
    // ... in order to print it
    θ = <-kernelRespChan

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
    β := []float64{30}
    getChan := make(chan chan Obs)
    modelQuitChan := make(chan bool)
    go logisticModel(β, getChan, modelQuitChan)

    // sgdkernel
    dataChan := make(chan Obs)
    paramChan := make(chan Params)
    stateChan := make(chan chan []float64)
    kernelQuitChan := make(chan bool)
    θ_0 := []float64{1.5}

    /*
        x := []float64{0.5}
        y := 1.0
        θhat := []float64{0.0}
        for i := 0; i < 80; i++ {
            gi := GradLogisticLoss(x, y, θhat)[0]
            θhat[0] += 0.3
            fmt.Printf("%.2f %.2f\n", θhat, gi)
        }
    */

    go SgdKernel(dataChan, paramChan, stateChan, kernelQuitChan,
        GradLogisticLoss, EtaInverse, θ_0)

    // test
    var θ []float64
    modelRespChan := make(chan Obs)
    kernelRespChan := make(chan []float64)
    for i := 0; i < 500000; i++ {
        // get data
        getChan <- modelRespChan
        obs := <-modelRespChan
        // send to kernel
        dataChan <- obs
        stateChan <- kernelRespChan
        θ = <-kernelRespChan
        fmt.Println(θ[0])
    }
    // get state from kernel
    stateChan <- kernelRespChan
    θ = <-kernelRespChan

    if !((0.9 < θ[0]) && (θ[0] < 1.1)) {
        t.Errorf("Failed to converge on correct θ_0")
    }
    /*
        if !((1.9 < θ[0]) && (θ[0] < 2.1)) {
            t.Errorf("Failed to converge on correct θ_1")
        }
        if !((2.9 < θ[2]) && (θ[2] < 3.1)) {
            t.Errorf("Failed to converge on correct θ_2")
        }
    */
}
