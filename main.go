package main

import (
	"fmt"
	"os"
	"rnn/basicNN"
	"rnn/common"
	"rnn/simple"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		return
	}
	switch os.Args[1] {
	case "--basicNN":
		basicNN.Test()
	default:
		fmt.Printf("Unknown training mode: %s\n", os.Args[1])
		printUsage()
		return
	}
}

func printUsage() {
	modes := "--basicNN | --SimpleRNN | --Elman-- | --Jordan | --LSTM"
	fmt.Printf("Please provide the training mode: %s\n", modes)
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
