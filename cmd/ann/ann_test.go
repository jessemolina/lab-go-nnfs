package ann_test

import (
	"testing"

	"github.com/jessemolina/lab-go-nnfs/cmd/ann"
)

// ================================================================
// TEST VARIABLES
// TODO write table driven testing - create variations of n to test against

// inputs - 4 input features
var x = []float64{1.0, 2.0, 3.0, 2.5}

// weights - 3 output neurons
var w = [][]float64{
	{0.2, 0.8, -0.5, 1},
	{0.5, -0.91, 0.26, -0.5},
	{-0.26, -0.27, 0.17, 0.87},
}

// bias - 3 weight biases
var b = []float64{2, 3, 0.5}

// neuron - 4 inputs features
var n = ann.Neuron{w[0], b[0]}

// layer - 3 output neurons
var l = ann.Layer{
	Neurons: make([]*ann.Neuron, len(w)),
}

// ================================================================
// TEST FUNCTIONS

// test NewRandonNeuron(d int)
func TestNewRandomNeuron(t *testing.T) {
	n := ann.NewRandomNeuron(len(x))
	results := len(n.Weights)
	expected := len(x)

	if results != expected {
		t.Errorf("TestNewRandomNeuron\nexpected:%d\nresults:%d\n", expected, results)
	}
}

// TODO test NewRandomLayer(outputs int)

// ================================================================
// TEST METHODS


// test (*Neuron).Predict()
func TestPredict(t *testing.T) {
	results := n.Predict(x)
	expected := []float64{4.8}
	if results[0] != expected[0] {
		t.Errorf("TestPredict\nexpected:%f\nresults:%f\n", expected, results)
	}
}
