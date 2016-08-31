package simple

import (
	"fmt"
	"math"
	"rnn/common"

	"github.com/gonum/matrix/mat64"
)

// Args holds names parameters to the NewRNN() constructor.
type Args struct {
	NumInput      int
	NumHidden     int
	NumOutput     int
	BackPropDepth int
}

// RNN is a vanilla Recursive Neuron Network. All fields are exported for
// easier inspection.
type RNN struct {
	InputDim       int
	HiddenDim      int
	OutputDim      int
	BackPropDepth  int
	InputToHidden  *mat64.Dense
	HiddenToHidden *mat64.Dense
	HiddenToOutput *mat64.Dense
}

// NewRNN is a constructor for SimpleRNN. Initializes weight matrices.
func NewRNN(args *Args) *RNN {
	out := &RNN{
		InputDim:      args.NumInput,
		HiddenDim:     args.NumHidden,
		OutputDim:     args.NumOutput,
		BackPropDepth: args.BackPropDepth,
	}
	out.init()
	return out
}

// Forward returns all outputs and all corresponding hidden states for and
// @input. @input can be a matrix representation of a single sentence where
// rows are one-hot encoded words.
func (n *RNN) Forward(input *mat64.Dense) (outs, hids *mat64.Dense) {
	numSteps, _ := input.Dims()
	// Allocate space for all hidden states that we get while propagating
	hiddens := mat64.NewDense(numSteps, n.HiddenDim, nil)
	// Allocate space for all outputs that we get while propagating
	outputs := mat64.NewDense(numSteps, n.OutputDim, nil)
	// At first time step we have no previous hidden state, so we explicitly
	// create it with a "zero mock" t-1 hidden state and the first element
	// of the @samples
	initialHidden := n.getHidden(
		mat64.NewVector(n.HiddenDim, nil), input.RowView(0),
	)
	hiddens.SetRow(0, initialHidden.RawVector().Data)
	outputs.SetRow(0, n.getOutputSlice(initialHidden))
	// For each time step
	for t := 1; t < numSteps; t++ {
		var (
			currSample = input.RowView(t)
			currHidden = hiddens.RowView(t)
			prevHidden = hiddens.RowView(t - 1)
		)
		hiddens.SetRow(t, n.getHiddenSlice(prevHidden, currSample))
		outputs.SetRow(t, n.getOutputSlice(currHidden))
	}
	return outputs, hiddens
}

// Backward returns three derivatives of the loss:
func (n *RNN) Backward(input, expected *mat64.Dense) (i, h, o *mat64.Dense) {
	var (
		// numSteps, _      = input.Dims()
		// outs, hids       = n.Forward(input) // Outputs and hidden layers
		inRows, inCols   = n.InputToHidden.Dims()
		hidRows, hidCols = n.HiddenToHidden.Dims()
		outRows, outCols = n.HiddenToOutput.Dims()
		dLossdIn         = mat64.NewDense(inRows, inCols, nil)
		dLossdHid        = mat64.NewDense(hidRows, hidCols, nil)
		dLossdOut        = mat64.NewDense(outRows, outCols, nil)
	)
	return dLossdIn, dLossdHid, dLossdOut
}

// Print the NN's internals.
func (n *RNN) Print() {
	fmt.Printf("Input to hidden:\n")
	common.PrintDense(n.InputToHidden)
	fmt.Printf("Hidden to hidden:\n")
	common.PrintDense(n.HiddenToHidden)
	fmt.Printf("Hidden to output:\n")
	common.PrintDense(n.HiddenToOutput)
}

func (n *RNN) init() {
	// Initialize matrix of weights from input to hidden layer
	n.InputToHidden = mat64.NewDense(n.HiddenDim, n.InputDim, nil)
	n.HiddenToHidden = mat64.NewDense(n.HiddenDim, n.HiddenDim, nil)
	n.HiddenToOutput = mat64.NewDense(n.OutputDim, n.HiddenDim, nil)
	// Initialize at random as in (Glorot & Bengio, 2010):
	// Positive upper boundary for random init weights for inputToHidden
	inputToHidden := math.Sqrt(1. / float64(n.InputDim))
	// Positive upper boundary for random init weights for hiddenToAny
	hiddenToAny := math.Sqrt(1. / float64(n.HiddenDim))
	common.RandomDense(-inputToHidden, inputToHidden, n.InputToHidden)
	common.RandomDense(-hiddenToAny, hiddenToAny, n.HiddenToHidden)
	common.RandomDense(-hiddenToAny, hiddenToAny, n.HiddenToOutput)
}

// getHiddenSlice is a wrapper around getHidden() which returns a slice of
// float64.
func (n *RNN) getHiddenSlice(prevHidden, sample *mat64.Vector) []float64 {
	return n.getHidden(prevHidden, sample).RawVector().Data
}

// getHidden calculates current hidden state as follows:
//	1. 	Multiplies inputToHidden matrix by the input sample (same as getting
//		a weighted sum of inputs for each hidden neuron);
// 	2.	Multiplies hiddenToHidden matrix by previous hidden layer (same as
//		getting a weighted sum of inputs for each hidden neuron);
//  3.  Sums the results from steps 1, 2 and applies activation function
//		(hyperbolic tanhent in this case).
func (n *RNN) getHidden(prevHidden, sample *mat64.Vector) *mat64.Vector {
	var (
		out        = mat64.NewVector(n.HiddenDim, nil)
		fromInput  = mat64.NewVector(n.HiddenDim, nil)
		fromHidden = mat64.NewVector(n.HiddenDim, nil)
	)
	fromInput.MulVec(n.InputToHidden, sample)
	fromHidden.MulVec(n.HiddenToHidden, prevHidden)
	out.AddVec(fromInput, fromHidden)
	return common.GetVectorTanh(out)
}

// getOutputSlice is a wrapper around getOutput() which returns a slice of
// float64.
func (n *RNN) getOutputSlice(currHidden *mat64.Vector) []float64 {
	return n.getOutput(currHidden).RawVector().Data
}

// getOutput just multiplies hiddenToHidden matrix by previous hidden layer
// (same as getting a weighted sum of inputs for each output neuron).
func (n *RNN) getOutput(currHidden *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(n.OutputDim, nil)
	out.MulVec(n.HiddenToOutput, currHidden)
	return common.GetVectorSoftmax(out)
}
