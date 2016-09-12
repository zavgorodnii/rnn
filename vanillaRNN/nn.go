package vanillaRNN

import (
	"fmt"
	"math"
	c "rnn/common"

	m "github.com/gonum/matrix/mat64"
)

// Args holds names parameters to the NewRNN() constructor.
type Args struct {
	Eta    float64
	NumInp int
	NumHid int
	NumOut int
	Depth  int
}

// RNN is a vanilla Recursive Neuron Network. All fields are exported for
// easier inspection.
type RNN struct {
	η      float64
	NumInp int
	NumHid int
	NumOut int
	Depth  int
	IH     *m.Dense
	HH     *m.Dense
	HO     *m.Dense
}

// NewRNN is a constructor for SimpleRNN. Initializes weight matrices.
func NewRNN(args *Args) *RNN {
	out := &RNN{
		η:      args.Eta,
		NumInp: args.NumInp,
		NumHid: args.NumHid,
		NumOut: args.NumOut,
		Depth:  args.Depth,
	}
	// Initialize matrix of weights from input to hidden layer
	out.IH = m.NewDense(out.NumHid, out.NumInp, nil)
	out.HH = m.NewDense(out.NumHid, out.NumHid, nil)
	out.HO = m.NewDense(out.NumOut, out.NumHid, nil)
	// Initialize at random as in (Glorot & Bengio, 2010):
	// Positive upper boundary for random init weights for inputToHidden
	inputToHidden := math.Sqrt(1. / float64(out.NumInp))
	// Positive upper boundary for random init weights for hiddenToAny
	hiddenToAny := math.Sqrt(1. / float64(out.NumHid))
	c.RandomDense(-inputToHidden, inputToHidden, out.IH)
	c.RandomDense(-hiddenToAny, hiddenToAny, out.HH)
	c.RandomDense(-hiddenToAny, hiddenToAny, out.HO)
	return out
}

// RunEpochs updates NN's weights for @input @numEpochs times.
func (n *RNN) RunEpochs(numEpochs int, input, expected *m.Dense) {
	// For each epoch
	for epoch := 0; epoch < numEpochs; epoch++ {
		// For each input
		n.BackProp(input, expected)
		_, acts := n.Forward(input)
		if epoch == numEpochs-1 {
			for _, prediction := range acts {
				for i := 0; i < prediction.Out.Len(); i++ {
					if prediction.Out.At(i, 0) >= 0.5 {
						prediction.Out.SetVec(i, 1.)
					} else {
						prediction.Out.SetVec(i, 0.)
					}
				}
				fmt.Println(prediction.Out)
			}
		}
	}
}

func (n *RNN) BackProp(input, expected *m.Dense) (
	ZErrZIH, ZErrZHH, ZErrZHO *m.Dense) {
	var (
		numSteps, _ = input.Dims()
		sums, acts  = n.Forward(input) // Outputs and hidden layers
	)
	// ZErrZIH = m.NewDense(n.NumHid, n.NumInp, nil)
	// ZErrZHH = m.NewDense(n.NumHid, n.NumHid, nil)
	// ZErrZHO = m.NewDense(n.NumOut, n.NumHid, nil)
	for t := 0; t < numSteps; t++ {
		// Calculate output layer error for sample t
		outError := n.GetOutError(acts[t].Out, sums[t].Out, expected.RowView(t))
		// Calculate derivatives for weights in HO
		ZErrZHO := c.GetOuterVec(outError, acts[t].Hid)
		// Calculate the changes for weights using the moment
		ηHO := c.GetDenseApply(ZErrZHO, func(val float64) float64 {
			return val * n.η
		})
		// Update HO weights
		n.HO.Sub(n.HO, ηHO)
		// Calculate initial hidden layer errors which will be used as
		// "previous" errors for the unfolding part
		prevHidErr := n.GetError(outError, sums[t].Hid, n.HO)
		for z := 0; z < n.Depth && t-z > 0; z++ {
			currHidErr := n.GetError(prevHidErr, sums[t-z-1].Hid, n.HH)
			// Calculate derivatives for weights in IH
			ZErrZIH := c.GetOuterVec(currHidErr, acts[t-z].Inp)
			ηIH := c.GetDenseApply(ZErrZIH, func(val float64) float64 {
				return val * n.η
			})
			// Update HO weights
			n.IH.Sub(n.IH, ηIH)
			// Calculate derivatives for weights in IH
			ZErrZHH := c.GetOuterVec(currHidErr, acts[t-z-1].Hid)
			ηHH := c.GetDenseApply(ZErrZHH, func(val float64) float64 {
				return val * n.η
			})
			// Update HO weights
			n.HH.Sub(n.HH, ηHH)
			// Update the "previous" hidden layer errors
			prevHidErr = currHidErr
		}
	}
	return
}

func (n *RNN) Forward(input *m.Dense) (sums []*Sums, acts []*Acts) {
	numSteps, _ := input.Dims()
	// Allocate space for all weighted sums that we get while propagating
	sums = make([]*Sums, numSteps)
	for i := 0; i < numSteps; i++ {
		sums[i] = &Sums{}
	}
	// Allocate space for all activations that we get while propagating
	acts = make([]*Acts, numSteps)
	for i := 0; i < numSteps; i++ {
		acts[i] = &Acts{}
	}
	// At first time step we have no previous hidden state, so we explicitly
	// create it with a zero magnitude t-1 hidden state and the first (initial)
	// training sample
	acts[0].Inp = input.RowView(0)
	sums[0].Hid, acts[0].Hid = n.getHidden(
		m.NewVector(n.NumHid, nil), acts[0].Inp,
	)
	sums[0].Out, acts[0].Out = n.getOutput(acts[0].Hid)
	// For each time step
	for t := 1; t < numSteps; t++ {
		currSample := input.RowView(t)
		prevHidden := acts[t-1].Hid
		acts[t].Inp = currSample
		sums[t].Hid, acts[t].Hid = n.getHidden(prevHidden, currSample)
		currHidden := acts[t].Hid
		sums[t].Out, acts[t].Out = n.getOutput(currHidden)
	}
	return
}

// GetOutError returns the output layer error as (output activations − expected
// activations) ⊙ sigmoidPrime(output sums).
func (n *RNN) GetOutError(outActs, outSums, expected *m.Vector) *m.Vector {
	outError := c.GetSubVec(outActs, expected)
	return c.GetMulElemVec(outError, c.GetVectorSigmoidPrime(outSums))
}

// GetError returns errors for each neuron in any single layer (L) using the
// errors in the layer just after it (L+1). The errors of (L+1) are propagated
// backwards to (L) using the same (L-to-L+1) weights that we used when passing
// (L)-activations to (L+1). Of course, we need to get a transposed version
// of (L-to-L+1) weights to make the matrix operations possible.
// After this backward-pass we multiply the (L)-errors by sigmoidPrime(L-sums),
// just as in GetOutError().
func (n *RNN) GetError(prevErrs, currSums *m.Vector, w *m.Dense) *m.Vector {
	wT := c.GetTransposed(w)
	propagated := c.GetMulVec(wT, prevErrs)
	return c.GetMulElemVec(propagated,
		c.GetVectorSigmoidPrime(currSums))
}

// getHidden calculates current hidden state as follows:
//	1. 	Multiplies inputToHidden matrix by the input sample (same as getting
//		a weighted sum of inputs for each hidden neuron);
// 	2.	Multiplies hiddenToHidden matrix by previous hidden layer (same as
//		getting a weighted sum of inputs for each hidden neuron);
//  3.  Sums the results from steps 1, 2 and applies activation function
//		(hyperbolic tanhent in this case).
func (n *RNN) getHidden(prevHidden, sample *m.Vector) (sums, acts *m.Vector) {
	fromInput := c.GetMulVec(n.IH, sample)
	fromHidden := c.GetMulVec(n.HH, prevHidden)
	sums = c.GetAddVec(fromInput, fromHidden)
	acts = c.GetVectorSigmoid(sums)
	return
}

// getOutput just multiplies hiddenToHidden matrix by previous hidden layer
// (same as getting a weighted sum of inputs for each output neuron).
func (n *RNN) getOutput(currHidden *m.Vector) (sums, acts *m.Vector) {
	sums = c.GetMulVec(n.HO, currHidden)
	acts = c.GetVectorSigmoid(sums)
	return
}

// Sums is used to keep weighted sums received by each neuron of output and
// hidden layers. So, Sums.Hid.At(2, 0) is the weighted sum received by the 3rd
// neuron in hidden layer.
type Sums struct {
	Out *m.Vector
	Hid *m.Vector
}

// Acts is used to keep activations of each neuron of output and hidden layers.
// So, Acts.Hid.At(2, 0) is the activation of the 3rd neuron in hidden layer.
type Acts struct {
	Out *m.Vector
	Hid *m.Vector
	Inp *m.Vector
}
