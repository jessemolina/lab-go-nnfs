package ann

import "math/rand"

type Neuron struct {
	Weights []float64
	Bias    float64
}

// pass by value neuron - it can make a copy, we don't need to modify it's core value
// returns scalar value for dot product with bias (w^{T} * x + b)
func (n *Neuron) Predict(x []float64) []float64 {
	z := 0.0
	for i := range x {
		z += x[i] * n.Weights[i]
	}
	return []float64{z + n.Bias}
}

// create a neuron with zero bias and random weights in d dimmension
func NewRandomNeuron(d int) Neuron {
	var w []float64
	for i := 0; i < d; i++ {
		w = append(w, rand.Float64())
	}
	return Neuron{w, 0}
}
