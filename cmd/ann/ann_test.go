package ann_test

import (
	"testing"

	"github.com/jessemolina/lab-go-nnfs/cmd/ann"
)

// TODO write table driven testing - create variations of n to test against

var n = ann.Neuron{[]float64{1, 2, 3},2}

// test (*Neuron).Predict()
func TestPredict(t *testing.T) {
	results := n.Predict([]float64{0.2, 0.8, -0.5})
	expected := []float64{2.4}
	if results[0] != expected[0] {
		t.Errorf("TestPredict\nexpected:%f\nresults:%f\n", expected, results)
	}
}

// test NewRandonNeurdon(d int)
func TestNewRandomNeuron(t *testing.T) {
	d := 3
	n := ann.NewRandomNeuron(d)
	results := len(n.Weights)
	expected := d

	if results != expected {
		t.Errorf("TestNewRandomNeuron\nexpected:%d\nresults:%d\n", expected, results)
	}
}
