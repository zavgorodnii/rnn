package vanillaRNN

import (
	"fmt"
	"math/rand"
	"rnn/common"
)

func Test() {
	rand.Seed(0)
	var (
		numEpochs = 100000
		numInput  = 4
		numHidden = 3 // May be changed to see how the network behaves
		numOutput = 4
	)
	fmt.Println("====================================================")
	fmt.Println("Testing basic Vanilla RNN on sample series dataset:")
	fmt.Println("====================================================")
	args := &Args{
		Eta:    0.001, // May be changed to see how the network behaves
		NumInp: numInput,
		NumHid: numHidden,
		NumOut: numOutput,
		Depth:  2,
	}
	nn := NewRNN(args)
	input, expected := common.GetAbstractTimeSeries()
	nn.RunEpochs(numEpochs, input, expected)
}
