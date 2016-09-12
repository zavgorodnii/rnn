package elman

import (
	"fmt"
	"math/rand"
	"rnn/common"
)

// Test runs @numEpochs iterations of introducing the Abstract Time Series
// data to an Elman RNN and prints the network's output after the last
// iteration.
func Test() {
	rand.Seed(0)
	var (
		numEpochs = 4001
		numInput  = 4
		numHidden = 6 // May be changed to see how the network behaves
		numOutput = 4
	)
	fmt.Println("====================================================")
	fmt.Println("Testing basic Vanilla RNN on sample series dataset:")
	fmt.Println("====================================================")
	args := &Args{
		Eta:    0.025, // May be changed to see how the network behaves
		NumInp: numInput,
		NumHid: numHidden,
		NumOut: numOutput,
		Depth:  3,
	}
	nn := NewElman(args)
	input, expected := common.GetAbstractTimeSeries()
	nn.RunEpochs(numEpochs, input, expected)
}
