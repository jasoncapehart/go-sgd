package sgd

import (
	//	"fmt"
	"log"
	"math"
)

type LossFunc func(x []float64, y float64, θ []float64) []float64
type StepFunc func(K float64, τ float64, κ float64) float64

type Params struct {
	τ  float64
	κ  float64
	λ1 float64
	λ2 float64
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
	var τ, κ, λ1, λ2 float64
	// initialise the adagrad step size modifier
	s := make([]float64, len(θ))
	for {
		select {
		case o := <-dataChan:
			// calculate the local gradient
			x := o.X
			y := o.Y
			grad := J(x, y, θ)
			// update the step size
			n += 1
			K := float64(n) // K is just the number of observations seen so far
			η := getStepSize(K, τ, κ)
			// update the new state
			for i, _ := range θ {
				s[i] += math.Pow(grad[i], 2)
				θ[i] += η * (grad[i] - λ1 - 2*λ2*θ[i]) / (τ + math.Sqrt(s[i]))
				//fmt.Printf("\t η:%.2f Δ:%.2f θ:%.2f\n", η, grad[i], θ[i])
				//fmt.Printf("\t θ:%.2f\n", θ[i])
			}
		case params := <-paramChan:
			τ = params.τ
			κ = params.κ
			λ1 = params.λ1
			λ2 = params.λ2
		case resChan := <-stateChan:
			resChan <- θ
		case <-quitChan:
			return
		}
	}
}

// step size update rules
func EtaConstant(K, τ, κ float64) float64 {
	return 0.1
}
func EtaInverse(K float64, τ float64, κ float64) float64 {
	return 1 / K
}
func EtaBottou(K float64, τ float64, κ float64) float64 {
	return math.Pow((τ + K), -κ)
}

// Link Functions
func identity(x float64) float64 {
	return x
}

func logit(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Loss Functions
func LinearLoss(x, θ []float64, y, σ float64) float64 {
	// log((1/(2*pi*sigma**2))**0.5*exp(-1/(2*σ**2)*(y-w*x)**2))
	σ2 := math.Pow(σ, 2)
	z := math.Pow(1.0/(2*math.Pi*σ2), 0.5)
	yest := 0.0
	for i, xi := range x {
		yest += θ[i] * xi
	}
	return -math.Log(z * math.Exp(-1/(2*σ2)*math.Pow((y-yest), 2)))
}

func GradLinearLoss(x []float64, y float64, θ []float64) []float64 {
	grad := make([]float64, len(θ))
	yEst := 0.0
	for i, θi := range θ {
		yEst += x[i] * θi
	}
	err := y - yEst
	for i, xi := range x {
		grad[i] = err * xi
	}
	//log.Println("x", x, "y", y, "yEst", yEst, "err", err, "Δ", grad)
	return grad
}

func LogisticLoss(x, θ []float64, y float64) float64 {
	yest := 0.0
	for i, θi := range θ {
		yest += θi * x[i]
	}
	μ := logit(yest)
	if μ < 0 {
		log.Println("μ cannot be less than zero")
	}
	if μ > 1 {
		log.Println("μ cannot be greater than one")
	}
	ll := -(y*math.Log(μ) + (1-y)*math.Log(1-μ))
	if math.IsNaN(ll) {
		log.Println("NaN log likelihood!", y, μ, θ, x)
	}
	return ll
}

func GradLogisticLoss(x []float64, y float64, θ []float64) []float64 {
	grad := make([]float64, len(θ))
	yest := 0.0
	for i, θi := range θ {
		yest += θi * x[i]
	}
	//err := (logit(yest) - y)
	err := y - logit(yest)
	for i, xi := range x {
		grad[i] = err * xi
		//fmt.Printf("x:%+.2f y:%.0f, yest:%+.2f, logit(yest): %.2f err:%+.2f, Δ: %v\n", xi, y, yest, logit(yest), err, grad)
	}
	/*
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
	*/
	return grad
}
