package main

import (
	"fmt"
	"os"
	"rnn/basicNN"
	"rnn/vanillaRNN"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		return
	}
	switch os.Args[1] {
	case "--basicNN":
		basicNN.Test()
	case "--vanillaRNN":
		vanillaRNN.Test()
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
