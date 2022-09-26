package ann

import "math/rand"

// TODO consider using pointer semantics - when to pass by value vs pass by address

// ================================================================
// FUNCTIONS

// create a neuron with random weights and bias
// inputs defines the size of weights
func NewRandomNeuron(inputs int) Neuron {
	var w []float64
	for i := 0; i < inputs; i++ {
		w = append(w, rand.Float64())
	}
	return Neuron{w, 0}
}

// num of nuerons, num of dimmensions (len of inputs)
func NewRandomLayer(inputs int, outputs int) Layer {
	ns := make([]Neuron, outputs)
	for i := 0; i < outputs; i++ {
		ns[i] = NewRandomNeuron(inputs)
	}
	return Layer{ns}
}

// ================================================================
// TYPES - Neuron

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


// ================================================================
// TYPES - Layer

type Layer struct {
	Neurons []Neuron
}
