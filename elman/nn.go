package elman

import (
	"fmt"
	"math"
	c "rnn/common"

	m "github.com/gonum/matrix/mat64"
)

// Args holds names parameters to the NewElman() constructor.
type Args struct {
	Eta    float64
	NumInp int
	NumHid int
	NumOut int
	Depth  int
}

// Elman is a simple Recursive Neuron Network which has recurrent connections
// from hidden layer neurons at time step (t-1) to hidden layer neurons at time
// step (t). We use this simplified model (without the possibility to add
// arbitrary number of hidden layers) to reduce the number of obscure indices
// and to use only named entities. We also use no biases (again, for
// simplicity).
type Elman struct {
	NumInp int
	NumHid int
	NumOut int
	η      float64  // Learning rate
	Depth  int      // Number of steps down the unfolded network
	IH     *m.Dense // Weights from input to hidden layer
	HH     *m.Dense // Weights from hidden to hidden layer
	HO     *m.Dense // Weights from hidden to output layer
}

// NewElman is a constructor for SimpleElman. Initializes weight matrices.
func NewElman(args *Args) *Elman {
	out := &Elman{
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

// RunEpochs executes BPTT algorithm for @input @numEpochs times.
func (n *Elman) RunEpochs(numEpochs int, input, expected *m.Dense) {
	// For each epoch
	for epoch := 0; epoch < numEpochs; epoch++ {
		// For each input do the BPTT algorithm
		n.BPTT(input, expected)
		_, acts := n.Forward(input)
		// Print only the last epoch results
		if epoch == numEpochs-1 {
			for idx, prediction := range acts {
				for i := 0; i < prediction.Out.Len(); i++ {
					if prediction.Out.At(i, 0) >= 0.5 {
						prediction.Out.SetVec(i, 1.)
					} else {
						prediction.Out.SetVec(i, 0.)
					}
				}
				// Some pretty printing (I know, this looks awful)
				fmt.Printf("Input: ")
				c.PrintVectorStripSub(input.RowView(idx), 0., ".")
				fmt.Printf("\t")
				fmt.Printf("Expected: ")
				c.PrintVectorStripSub(expected.RowView(idx), 0., ".")
				fmt.Printf("\t")
				fmt.Printf("Predicted: ")
				c.PrintVectorStripSub(prediction.Out, 0., ".")
				fmt.Printf("\n")
			}
		}
	}
}

// BPTT executes the Backpropagation Through Time algorithm to learn the
// network's weight. As BPTT is a variation of standard Backpropagation, it
// might be useful to look at basicNN code and look for similarities.
// Note that we don't have a separate Update() method; all weights are updated
// "on the go".
func (n *Elman) BPTT(input, expected *m.Dense) (
	dErrdIH, dErrdHH, dErrdHO *m.Dense) {
	numSteps, _ := input.Dims()
	// Forward pass: get sums and activations for each layer for @numSteps
	// samples. See n.Forward() for details.
	sums, acts := n.Forward(input)
	// For each sample @t in the @input
	for t := 0; t < numSteps; t++ {
		// We start just as in basicNN. Calculate output layer error for @t
		outError := n.GetOutError(acts[t].Out, sums[t].Out, expected.RowView(t))
		// Calculate derivatives for weights in HO using output layer error
		dErrdHO := c.GetOuterVec(outError, acts[t].Hid)
		// Calculate changes for HO weights based on the derivatives from
		// previous step (this was done in a separate method in basicNN)
		ηHO := c.GetDenseApply(dErrdHO, func(val float64) float64 {
			return val * n.η
		})
		// Update HO weights
		n.HO.Sub(n.HO, ηHO)
		// Like in basicNN, we calculate hidden layer errors by backpropagating
		// output layer errors. These errors will be used as the starting point
		// for the recursive calculation of hidden layer errors in the
		// unfolding procedure.
		currHidErr := n.GetError(outError, sums[t].Hid, n.HO)
		// Start unfolding the network @n.Depth steps back. This is a "moving
		// backwards" procedure, and @z is the number of steps back through the
		// unfolded network
		for z := 0; z < n.Depth && t-z > 0; z++ {
			// Now we update the IH weights just as we did in basicNN. First we
			// calculate derivatives for weights in IH
			dErrdIH := c.GetOuterVec(currHidErr, acts[t-z].Inp)
			// Then we find the momentum-driven changes
			ηIH := c.GetDenseApply(dErrdIH, func(val float64) float64 {
				return val * n.η
			})
			// Finally we update IH weights
			n.IH.Sub(n.IH, ηIH)
			// Now the same for HH weights from (t-z-1) to (t-z)
			dErrdHH := c.GetOuterVec(currHidErr, acts[t-z-1].Hid)
			ηHH := c.GetDenseApply(dErrdHH, func(val float64) float64 {
				return val * n.η
			})
			n.HH.Sub(n.HH, ηHH)
			// In the next iteration we need hidden errors for layer (t-z-1).
			// We calculate them by propagating current (t-z) hidden errors
			// to (t-z-1) via HH weights.
			// When z is 0, @currHidErr is just the "normal" basicNN-style
			// hidden layer error propagated from the output layer (because we
			// need something to start with).
			currHidErr = n.GetError(currHidErr, sums[t-z-1].Hid, n.HH)
		}
	}
	return
}

// Forward accumulates sums and activations for each layer for each training
// sample.
func (n *Elman) Forward(input *m.Dense) (sums []*Sums, acts []*Acts) {
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
func (n *Elman) GetOutError(outActs, outSums, expected *m.Vector) *m.Vector {
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
func (n *Elman) GetError(prevErrs, currSums *m.Vector, w *m.Dense) *m.Vector {
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
func (n *Elman) getHidden(prevHidden, sample *m.Vector) (sums, acts *m.Vector) {
	fromInput := c.GetMulVec(n.IH, sample)
	fromHidden := c.GetMulVec(n.HH, prevHidden)
	sums = c.GetAddVec(fromInput, fromHidden)
	acts = c.GetVectorSigmoid(sums)
	return
}

// getOutput just multiplies hiddenToHidden matrix by previous hidden layer
// (same as getting a weighted sum of inputs for each output neuron).
func (n *Elman) getOutput(currHidden *m.Vector) (sums, acts *m.Vector) {
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
