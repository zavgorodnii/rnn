package baseNN

import "rnn/common"

// Run fits an NN.
func Run() {
	var (
		numIterations = 1000
		numSamples    = 100
		numInput      = 7
		numHidden     = 5
		numOutput     = 2
	)
	args := &Args{
		Eta:    0.001,
		NumInp: numInput,
		NumHid: numHidden,
		NumOut: numOutput,
	}
	nn := NewNN(args)
	input, expected := common.GetTwoClass(numSamples, numInput)
	nn.Fit(input, expected, numIterations)
}
