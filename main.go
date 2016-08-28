package main

import (
	"fmt"
	"rnn/common"
	"rnn/simple"
)

func main() {
	testSimpleRNN()
}

func testSimpleRNN() {
	args := &simple.Args{
		NumInput:  5,
		NumHidden: 3,
		NumOutput: 5,
	}
	rnn := simple.NewRNN(args)
	input := common.GetRandomSamples(10, 5)
	expected := common.GetRandomSamples(10, 5)
	outputs, hiddens := rnn.Forward(input)
	fmt.Println("Outputs:")
	common.PrintDense(outputs)
	fmt.Println("Hiddens:")
	common.PrintDense(hiddens)
	rnn.Print()
}
