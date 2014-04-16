package sgd

import (
	"log"
	"math"
)

type LossFunc func(x []float64, y float64, θ []float64) []float64
type StepFunc func(K float64, τ float64, κ float64) float64

type Params struct {
	τ float64
	κ float64
}

type Obs struct {
	X []float64
	Y float64
}

func SgdKernel(dataChan chan Obs, paramChan chan Params, stateChan chan chan []float64, quitChan chan bool,
	J LossFunc, getStepSize StepFunc, θ_0 []float64) {
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
			x := o.X
			y := o.Y
			grad := J(x, y, θ)
			// update the step size
			n += 1
			K := float64(n)
			η := getStepSize(K, τ, κ)
			log.Println("η", η)
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
func EtaInverse(K float64, τ float64, κ float64) float64 {
	return 1 / K
}
func EtaBottou(K float64, τ float64, κ float64) float64 {
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
func GradLinearLoss(x []float64, y float64, θ []float64) []float64 {
	grad := make([]float64, len(θ))
	yEst := 0.0
	for i, θi := range θ {
		yEst += x[i] * θi
	}
	err := (y - yEst)
	for i, xi := range x {
		grad[i] = err * xi
	}
	log.Println("x", x, "y", y, "yEst", yEst, "err", err, "Δ", grad)
	return grad
}

func GradLogisticLoss(x []float64, y float64, θ []float64) []float64 {
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

// Regularization Functions
func Lasso_Regularization(θ []float64, λ float64) []float64 {
	penalty := 0.0
	for i, θi := range θ {
		penalty += math.Abs(θi)
	}
	penalty = λ * penalty
	return penalty
}

func Ridge_Regularization(θ []float64, λ float64) []float64 {
	penalty := 0.0
	for i, θi := range θ {
		penalty += math.Pow(θi, 2)
	}
	penalty = λ * penalty
	return penalty
}

func Elastic_Regularization(θ []float64, λ1 float64, λ2 float64) []float64 {
	penalty := 0.0
	for i, θi := range θ {
		penalty += math.Pow(θi, 2)
	}
	penalty = λ * penalty
	return penalty
}
