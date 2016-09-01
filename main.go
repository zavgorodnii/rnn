package main

import (
	"rnn/baseNN"
	"rnn/common"
	"rnn/simple"
)

func main() {
	// testSimpleRNN()
	baseNN.Run()
}

func testSimpleRNN() {
	args := &simple.Args{
		NumInput:      5,
		NumHidden:     3,
		NumOutput:     5,
		BackPropDepth: 5,
	}
	rnn := simple.NewRNN(args)
	input := common.GetRandomDense(20, 5)
	// expected := common.GetRandomDense(20, 5)
	// outputs, hiddens := rnn.Forward(input)
	rnn.Forward(input)
	// fmt.Println("Outputs:")
	// common.PrintDense(outputs)
	// fmt.Println("Hiddens:")
	// common.PrintDense(hiddens)
	// rnn.Print()
	// rnn.Backward(input, expected)
}
