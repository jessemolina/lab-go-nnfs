package ann_test

import (
	"reflect"
	"testing"

	"github.com/jessemolina/lab-go-nnfs/cmd/ann"
)

// ================================================================
// TEST VARIABLES
// TODO write table driven testing - create variations of n to test against

// inputs - 4 input features, per vector in matrix
var x = [][]float64{
	{1.0, 2.0, 3.0, 2.5},
	{2.0, 5.0, -1.0, 2.0},
	{-1.5, 2.7, 3.3, -0.8},
}

// weights - 3 output neurons
var w = [][]float64{
	{0.2, 0.8, -0.5, 1},
	{0.5, -0.91, 0.26, -0.5},
	{-0.26, -0.27, 0.17, 0.87},
}

// bias - 3 weight biases
var b = []float64{2, 3, 0.5}

// layer - 3 output neurons
var l = ann.Layer{
	Neurons: []ann.Neuron{
		{w[0], b[0]},
		{w[1], b[1]},
		{w[2], b[2]},
	},
}

// ================================================================
// TEST FUNCTIONS

// test NewRandonNeuron(d int) Neuron
// create a neuron with len(x[0]) inputs
// len(n.Weights) should match the number of input features (x)
// if size of weights array != size of input array
func TestNewRandomNeuron(t *testing.T) {
	n := ann.NewRandomNeuron(len(x[0]))
	results := len(n.Weights)
	expected := len(x[0])

	if results != expected {
		t.Errorf("TestNewRandomNeuron\nexpected:%d\nresults:%d\n", expected, results)
	}
}

// test NewRandomLayer(outputs int) Layer
// number of output is the number of neurons expectedj
// the number of input features (x) should match the number of weight arrays (w{})
func TestNewRandomLayer(t *testing.T) {
	inputs := len(x[0])
	outputs := len(w)
	l := ann.NewRandomLayer(inputs, outputs)

	// number of weights should match number if inputs; number of neurons should match outputs
	expected := inputs == len(l.Neurons[0].Weights) && outputs == len(l.Neurons)
	if expected != true {
		t.Errorf("TestNewRandomLayer\nexpected:%v\nresults:%v\n", true, expected)
	}
}

// ================================================================
// TEST METHODS - Neuron

// test (*Neuron).Predict(x []float64) []float64
func TestNeuronPredict(t *testing.T) {
	results := l.Neurons[0].Predict(x[0])
	expected := []float64{4.8}
	if results[0] != expected[0] {
		t.Errorf("TestPredict\nexpected:%f\nresults:%f\n", expected, results)
	}
}

// ================================================================
// TEST METHODS - Layer

// test (*Layer).Predict()
func TestLayerPredict(t *testing.T) {
	results := l.Predict(x[0])
	expected := []float64{4.8, 1.21, 2.385}
	if !reflect.DeepEqual(results, expected){
		t.Errorf("TestPredict\nexpected:%f\nresults:%f\n", expected, results)
	}
}
