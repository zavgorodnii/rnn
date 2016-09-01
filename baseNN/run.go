package baseNN

import (
	"math/rand"
	"rnn/common"
	"time"
)

// Run fits an NN.
func Run() {
	rand.Seed(time.Now().UTC().UnixNano())
	var (
		numIterations = 100000
		numInput      = 4
		numHidden     = 5
		numOutput     = 3
	)
	args := &Args{
		Eta:    0.001,
		NumInp: numInput,
		NumHid: numHidden,
		NumOut: numOutput,
	}
	nn := NewNN(args)
	input, expected := common.GetIris()
	nn.Test(input, expected, numIterations)
}
