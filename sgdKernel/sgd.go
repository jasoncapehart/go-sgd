package sgd

import (
	//	"log"
	"math"
)

type lossFunc func(x []float64, y float64, θ []float64) []float64
type stepFunc func(K float64, τ float64, κ float64) float64

type params struct {
	τ float64
	κ float64
}

type obs struct {
	x []float64
	y float64
}

func SgdKernel(dataChan chan obs, paramChan chan params, stateChan chan chan []float64, quitChan chan bool,
	J lossFunc, getStepSize stepFunc, θ_0 []float64) {
	// initialise the state
	θ := θ_0
	// initialise the counter
	n := 0
	// initialise the params
	var τ, κ float64
	for {
		select {
		case o := <-dataChan:
			// calculate the local gradient
			x := o.x
			y := o.y
			grad := J(x, y, θ)
			// update the step size
			n += 1
			K := float64(n)
			η := getStepSize(K, τ, κ)
			// update the new state
			for i, _ := range θ {
				θ[i] += η * grad[i]
			}
		case params := <-paramChan:
			τ = params.τ
			κ = params.κ
		case resChan := <-stateChan:
			resChan <- θ
		case <-quitChan:
			return
		}
	}
}

// step size update rules
func eta_inverse(K float64, τ float64, κ float64) float64 {
	return 1 / K
}
func eta_bottou(K float64, τ float64, κ float64) float64 {
	return math.Pow((τ + K), κ)
}

// Link Functions
func identity(x float64) float64 {
	return x
}

func logit(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Loss Functions
func grad_linear_loss(x []float64, y float64, θ []float64) []float64 {
	grad := make([]float64, len(θ))
	y_est := 0.0
	for i, θi := range θ {
		y_est += x[i] * θi
	}
	for i, xi := range x {
		grad[i] = (y - y_est) * xi
	}
	return grad
}

func grad_logistic_loss(x []float64, y float64, θ []float64) []float64 {
	// grad = ( y - 1 / math.Exp(-(x * theta)) ) * x
	grad := make([]float64, len(θ))
	yEst := 0.0
	for i, θi := range θ {
		yEst += x[i] * θi
	}
	err := (y - logit(yEst))
	for i, xi := range x {
		grad[i] = err * xi
	}
	//log.Println("x", x, "y", y, "yEst", logit(yEst), "err", err, "Δ", grad)
	return grad
}
