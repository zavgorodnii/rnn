package baseNN

import (
	"fmt"
	"math"
	c "rnn/common"

	m "github.com/gonum/matrix/mat64"
)

// Args is a holder for arguments to NewNN().
type Args struct {
	Eta    float64
	NumInp int
	NumHid int
	NumOut int
}

// NN is a simple feed-forward neural network that has an input, a hidden and
// an output layer. We use this simplified model (without the possibility to
// add an arbitrary number of hidden layers) to reduce the number of obscure
// indices and to use only named entities.
type NN struct {
	Eta float64
	IH  *m.Dense // Weights from input layer to hidden layer
	HO  *m.Dense // Weights from hidden layer to output layer
}

// NewNN is a constructor.
func NewNN(args *Args) *NN {
	out := &NN{
		Eta: args.Eta,
	}
	out.IH = m.NewDense(args.NumHid, args.NumInp, nil)
	out.HO = m.NewDense(args.NumOut, args.NumHid, nil)
	// Initialize at random as in (Glorot & Bengio, 2010):
	// Positive upper boundary for random init weights for inputToHidden
	maxIH := math.Sqrt(1. / float64(args.NumInp))
	// Positive upper boundary for random init weights for hiddenToAny
	maxHO := math.Sqrt(1. / float64(args.NumHid))
	c.RandomDense(-maxIH, maxIH, out.IH)
	c.RandomDense(-maxHO, maxHO, out.HO)
	return out
}

// Fit updates NN's weights for @input @numIterations times.
func (n *NN) Fit(input, expected *m.Dense, numIterations int) {
	numInputs, _ := input.Dims()
	for iteration := 0; iteration < numIterations; iteration++ {
		for i := 0; i < numInputs; i++ {
			currInp := input.RowView(i)
			currExp := expected.RowView(i)
			n.Update(currInp, currExp)
		}
		totalError := .0
		for i := 0; i < numInputs; i++ {
			currInp := input.RowView(i)
			currExp := expected.RowView(i)
			_, acts := n.ForwardSample(currInp)
			err := c.GetSubVec(acts.Out, currExp)
			totalError += c.GetVectorAbs(err)
		}
		fmt.Printf("Iteration %d; total error %f\n",
			iteration, totalError/float64(numInputs))
	}
}

// Update updates input-to-hidden and hidden-to-output weights using
// error gradients on those weights retrieved by backpropagation.
// Update are done by multiplying gradients by learning rate (Eta) and
// subtracting the result from the actual weights.
// A quick note on why we do it this way. The partial derivative of error with
// respect to any specific weight tells us how quickly the error grows when the
// weight grows. As we want the error to become smaller, we *subtract* the
// derivative times the learning rate from the actual weight.
func (n *NN) Update(input, expected *m.Vector) {
	dErrdIH, dErrdHO := n.BackPropSample(input, expected)
	etaIH := c.GetDenseApply(dErrdIH, func(val float64) float64 {
		return val * n.Eta
	})
	etaHO := c.GetDenseApply(dErrdHO, func(val float64) float64 {
		return val * n.Eta
	})
	n.IH.Sub(n.IH, etaIH)
	n.HO.Sub(n.HO, etaHO)
}

// BackPropSample performs a forward pass for the input vector, calculates the
// error and returns:
//	1. Error gradients on each of input-to-hidden weights (dErrdIH)
//	2. Error gradients on each of hidden-to-output weights (dErrdHO)
func (n *NN) BackPropSample(input, expected *m.Vector) (
	dErrdIH, dErrdHO *m.Dense) {
	// Get weighted sums and activations for all layers
	sums, acts := n.ForwardSample(input)
	// Calculate error for each neuron in the output layer
	outErrs := n.GetOutError(sums.Out, acts.Out, expected)
	// Calculate error for each neuron in the hidden layer using output layer's
	// errors
	hidErrs := n.GetError(outErrs, sums.Hid, n.HO)
	// Now the part where we find the derivative of error for each weight from
	// (k-th neuron in hidden layer) to (j-th neuron in output layer) as
	// (activation of k-th neuron in hidden layer) * (error of j-th neuron in
	// output layer). We could do this in a straightforward way like this:
	//
	// hoRows, hoCols := n.HO.Dims()
	// dErrdHO = m.NewDense(hoRows, hoCols, nil)
	// for j := 0; j < hoRows; j++ {
	// 	for k := 0; k < hoCols; k++ {
	// 		grad := acts.Hid.At(k, 0) * outErrs.At(j, 0)
	// 		dErrdHO.Set(j, k, grad)
	// 	}
	// }
	//
	// But it so happens that what we do in those loops can be described as the
	// *outer product* of any two vectors u and v, which is essentially the
	// same as matrix multiplication u * v.T(), provided that u is represented
	// as a m × 1 column vector and v as a n × 1 column vector (which makes
	// v.T() a row vector).
	//
	// If u.Len() is 2 ans v.Len() is 3 then Outer(u, v) is a 2 x 3 matrix.
	//
	// So we'll use this shortcut to calculate gradients of output error on
	// hidden-to-output weights by finding Outer(outErrs, acts.Hid) (which
	// has the same dims as m.HO):
	hoRows, hoCols := n.HO.Dims()
	dErrdHO = m.NewDense(hoRows, hoCols, nil)
	dErrdHO.Outer(1., outErrs, acts.Hid)
	// And then we'll do the same for weights from input to hidden layer:
	ihRows, ihCols := n.IH.Dims()
	dErrdIH = m.NewDense(ihRows, ihCols, nil)
	dErrdIH.Outer(1., hidErrs, acts.Inp)
	// Thus we get:
	//	1. Error gradients on each of input-to-hidden weights (dErrdIH)
	//	2. Error gradients on each of hidden-to-output weights (dErrdHO)
	return
}

// ForwardSample returns weighted sums (before applying the activation
// function) and activations (after applying the activation function) for each
// neuron in all layers. Note that we treat the @input vector as input layer
// activation (which is kind of natural) and don't save anything as
// input layer's sums (mostly because those values are not used anywhere).
// So, generally speaking it's just performing the forward pass and saving
// the intermediate results (which will be used for backpropagation).
func (n *NN) ForwardSample(input *m.Vector) (sums *Sums, acts *Acts) {
	// Same as getting the weighted sum of inputs for all hidden neurons
	hidSums := c.GetMulVec(n.IH, input)
	// Apply sigmoid function to each hidden neuron (getting the activation)
	hidActs := c.GetVectorSigmoid(hidSums)
	// Same as getting the weighted sum of inputs for all output neurons
	outSums := c.GetMulVec(n.HO, hidActs)
	// Apply sigmoid function to each output neuron (getting the activation)
	outActs := c.GetVectorSigmoid(outSums)
	sums = &Sums{
		Out: outSums,
		Hid: hidSums,
	}
	acts = &Acts{
		Out: outActs,
		Hid: hidActs,
		Inp: input,
	}
	return
}

// GetOutError returns the output layer error as (output activations − expected
// activations) ⊙ sigmoidPrime(output sums).
func (n *NN) GetOutError(outActs, outSums, expected *m.Vector) *m.Vector {
	outError := c.GetSubVec(outActs, expected)
	// fmt.Println(outSums)
	return c.GetMulElemVec(outError, c.GetVectorSigmoidPrime(outSums))
}

// GetError returns errors for each neuron in any single layer (L) using the
// errors in the layer just after it (L+1). The errors of (L+1) are propagated
// backwards to (L) using the same (L-to-L+1) weights that we used when passing
// (L)-activations to (L+1). Of course, we need to get a transposed version
// of (L-to-L+1) weights to make the matrix operations possible.
// After this backward-pass we multiply the (L)-errors by sigmoidPrime(L-sums),
// just as in GetOutError().
func (n *NN) GetError(prevErrs, currSums *m.Vector, w *m.Dense) *m.Vector {
	wT := c.GetTransposed(w)
	propagated := c.GetMulVec(wT, prevErrs)
	return c.GetMulElemVec(propagated,
		c.GetVectorSigmoidPrime(currSums))
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
