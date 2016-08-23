package simple

import (
	"math"
	"rnn/common"

	"github.com/gonum/matrix/mat64"
)

// Args holds names parameters to the NewRNN() constructor.
type Args struct {
	NumInput  int
	NumHidden int
	NumOutput int
}

// RNN is a simple Recursive Neuron Network.
type RNN struct {
	inputDim       int
	hiddenDim      int
	outputDim      int
	inputToHidden  *mat64.Dense
	hiddenToHidden *mat64.Dense
	hiddenToOutput *mat64.Dense
}

// NewRNN is a constructor for SimpleRNN. Initializes weight matrices.
func NewRNN(args *Args) *RNN {
	out := &RNN{
		inputDim:  args.NumInput,
		hiddenDim: args.NumHidden,
		outputDim: args.NumOutput,
	}
	out.init()
	return out
}

func (s *RNN) init() {
	// Initialize matrix of weights from input to hidden layer
	s.inputToHidden = mat64.NewDense(s.hiddenDim, s.inputDim, nil)
	s.hiddenToHidden = mat64.NewDense(s.hiddenDim, s.hiddenDim, nil)
	s.hiddenToOutput = mat64.NewDense(s.outputDim, s.hiddenDim, nil)
	// As in (Glorot & Bengio, 2010):
	// Positive upper boundary for random init weights for inputToHidden
	inputToHidden := math.Sqrt(1. / float64(s.inputDim))
	// Positive upper boundary for random init weights for hiddenToAny
	hiddenToAny := math.Sqrt(1. / float64(s.hiddenDim))
	common.RandomDense(-inputToHidden, inputToHidden, s.inputToHidden)
	common.RandomDense(-hiddenToAny, hiddenToAny, s.hiddenToHidden)
	common.RandomDense(-hiddenToAny, hiddenToAny, s.hiddenToOutput)
}
