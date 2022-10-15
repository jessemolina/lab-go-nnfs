package main

import (
	"fmt"
	"log"
	"gorgonia.org/gorgonia"
)

func main() {

	// zero value initilization
	var a, b, c *gorgonia.Node
	var err error

	// create a new graph
	g := gorgonia.NewGraph()

	// assign values to the nodes
	a = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("a"))
	b = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("b"))

	// assign operation to the node
	c, err = gorgonia.Add(a, b)
	if err != nil {
		log.Fatal(err)
	}

	// create a new turing machine
	machine := gorgonia.NewTapeMachine(g)

	gorgonia.Let(a, 1.0)
	gorgonia.Let(b, 2.0)

	if machine.RunAll() != nil {
		log.Fatal(err)
	}

	fmt.Println(c.Value())

}
