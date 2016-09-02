package baseNN

import (
	"math/rand"
	"rnn/common"
)

// Test tests the basic NN on the Iris dataset.
func Test() {
	rand.Seed(0)
	var (
		numEpochs = 3001
		numInput  = 4
		numHidden = 4
		numOutput = 3
	)
	fmt.Println("Testing basic ")
	args := &Args{
		Eta:    0.001,
		NumInp: numInput,
		NumHid: numHidden,
		NumOut: numOutput,
	}
	nn := NewNN(args)
	input, expected := common.GetIris()
	nn.RunEpochs(numEpochs, input, expected)
}
