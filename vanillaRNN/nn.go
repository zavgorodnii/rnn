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
		n.Update(input, expected)
		_, acts := n.Forward(input)
		if epoch == numEpochs-1 {
			for _, prediction := range acts {
				for i := 0; i < prediction.Out.Len(); i++ {
					if prediction.Out.At(i, 0) >= 0.5 {
						prediction.Out.SetVec(i, 0.)
					} else {
						prediction.Out.SetVec(i, 1.)
					}
				}
				fmt.Println(prediction.Out)
			}
		}
		// fmt.Println("=============================================")
	}
}

func (n *RNN) Update(input, expected *m.Dense) {
	numSteps, _ := input.Dims()
	ZErrZIH, ZErrZHH, ZErrZHO := n.BackProp(input, expected)
	for t := 0; t < numSteps; t++ {
		var (
			tZErrZHO = ZErrZHO[t]
			tZErrZHH = ZErrZHH[t]
			tZErrZIH = ZErrZIH[t]
		)
		ηIH := c.GetDenseApply(tZErrZIH, func(val float64) float64 {
			return val * n.η
		})
		ηHH := c.GetDenseApply(tZErrZHH, func(val float64) float64 {
			return val * n.η
		})
		ηHO := c.GetDenseApply(tZErrZHO, func(val float64) float64 {
			return val * n.η
		})
		n.IH.Sub(n.IH, ηIH)
		n.HH.Sub(n.HH, ηHH)
		n.HO.Sub(n.HO, ηHO)
	}
}

func (n *RNN) BackProp(input, expected *m.Dense) (
	ZErrZIH, ZErrZHH, ZErrZHO []*m.Dense) {
	var (
		numSteps, _ = input.Dims()
		sums, acts  = n.Forward(input) // Outputs and hidden layers
	)
	ZErrZIH = make([]*m.Dense, numSteps)
	ZErrZHH = make([]*m.Dense, numSteps)
	ZErrZHO = make([]*m.Dense, numSteps)
	// for t := (numSteps - 1); t >= 0; t-- {
	for t := 0; t < numSteps; t++ {
		// Deal with the output layer for current sample
		tExpected := expected.RowView(t)
		tOutErrs := n.GetOutError(sums[t].Out, acts[t].Out, tExpected)
		tZErrZHO := c.GetOuterVec(tOutErrs, acts[t].Hid)
		ZErrZHO[t] = tZErrZHO
		// Create accumulated derivatives for the IH and HH weights
		tZErrZIH := m.NewDense(n.NumHid, n.NumInp, nil)
		tZErrZHH := m.NewDense(n.NumHid, n.NumHid, nil)
		// Start accumulating the derivatives using @n.Depth-sized memory
		for memT := t; memT > c.MaxInt(0, t-n.Depth); memT-- {
			// Deal with IH weights. First get the expected output at tT
			memExpected := expected.RowView(t)
			// Calculate the output error at tT
			memOutErrs := n.GetOutError(sums[t].Out, acts[t].Out, memExpected)
			// Propagate the error to the hidden layer
			memHidErrs := n.GetError(memOutErrs, sums[memT].Hid, n.HO)
			// Calculate IH derivatives at tT and add them to the global
			// derivatives
			memZErrZIH := c.GetOuterVec(memHidErrs, acts[memT].Inp)
			tZErrZIH.Add(tZErrZIH, memZErrZIH)
			// Do the same for HH derivatives (yes, we're using memHidErrs here
			// too)
			memZErrZHH := c.GetOuterVec(memHidErrs, acts[memT].Hid)
			tZErrZHH.Add(tZErrZHH, memZErrZHH)
		}
		ZErrZHO[t] = tZErrZHO
		ZErrZHH[t] = tZErrZHH
		ZErrZIH[t] = tZErrZIH
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
