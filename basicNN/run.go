package basicNN

import (
	"fmt"
	"math/rand"
	"rnn/common"
)

// Test tests the basic NN on the Iris dataset.
func Test() {
	rand.Seed(0)
	var (
		numEpochs = 3001
		numInput  = 4
		numHidden = 4 // May be changed to see how the network behaves
		numOutput = 3
	)
	fmt.Println("================================================")
	fmt.Println("Testing basic feed-forward NN on Iris dataset:")
	fmt.Println("================================================")
	args := &Args{
		Eta:    0.001, // May be changed to see how the network behaves
		NumInp: numInput,
		NumHid: numHidden,
		NumOut: numOutput,
	}
	nn := NewNN(args)
	input, expected := common.GetIris()
	nn.RunEpochs(numEpochs, input, expected)
}
