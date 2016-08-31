package baseNN

import (
	"rnn/common"

	m "github.com/gonum/matrix/mat64"
)

// NN is a simple feed-forward neural network that has an input, a hidden and
// an output layer. We use this simplified model (without the possibility to
// add an arbitrary number of hidden layers) to reduce the number of obscure
// indices and to use only named entities.
type NN struct {
	Eta    float64
	InpLen int
	HidLen int
	OutLen int
	IH     *m.Dense // Weights from input layer to hidden layer
	HO     *m.Dense // Weights from hidden layer to output layer
}

// ForwardSample returns weighted sums (before applying the activation
// function) and activations (after applying the activation function) for each
// neuron in all layers. Note that we treat the @input vector as activations of
// neurons in the input layer (which is natural) and don't save anything as
// input layer's sums (mostly because those values are not used anywhere).
// So, generally speaking it's just performing the forward pass and saving
// the intermediate results (which will be used for backpropagation).
func (n *NN) ForwardSample(input *m.Vector) (sums *Sums, acts *Acts) {
	// Same as getting the weighted sum of inputs for all hidden neurons
	hidSums := common.GetMulVec(n.IH, input)
	// Apply sigmoid function to each hidden neuron (getting the activation)
	hidActs := common.GetVectorSigmoid(hidSums)
	// Same as getting the weighted sum of inputs for all output neurons
	outSums := common.GetMulVec(n.HO, hidActs)
	// Apply sigmoid function to each output neuron (getting the activation)
	outActs := common.GetVectorSigmoid(outSums)
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

// BackProp performs a forward pass for the input vector, calculates the error
// and returns:
//	1. Gradients of Error on each of hidden-to-output weights (dErrdHO)
//	2. Gradients of Error on each of input-to-hidden weights (dErrdIH)
func (n *NN) BackProp(input, expected *m.Vector) (dErrdIH, dErrdHO *m.Dense) {
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
	dErrdHO.Outer(1., hidErrs, acts.Out)
	// Thus we get:
	//	1. Gradients of Error on each of hidden-to-output weights (dErrdHO)
	//	2. Gradients of Error on each of input-to-hidden weights (dErrdIH)
	return
}

// GetOutError returns the output layer error as (output activations − expected
// activations) ⊙ sigmoidPrime(output sums).
func (n *NN) GetOutError(outActs, outSums, expected *m.Vector) *m.Vector {
	outError := common.GetSubVec(outActs, expected)
	return common.GetMulElemVec(outError, common.GetVectorSigmoidPrime(outSums))
}

// GetError returns errors for each neuron in any single layer (L) using the
// errors in the layer just after it (L+1). The errors of (L+1) are propagated
// backwards to (L) using the same (L-to-L+1) weights that we used when passing
// (L)-activations to (L+1). Of course, we need to get a transposed version
// of (L-to-L+1) weights to make the matrix operations possible.
// After this backward-pass we multiply the (L)-errors by sigmoidPrime(L-sums),
// just as in GetOutError().
func (n *NN) GetError(prevErrs, currSums *m.Vector, w *m.Dense) *m.Vector {
	var (
		wT         = common.GetTransposed(w)
		propagated = common.GetMatrixByVector(wT, prevErrs)
		currErrs   = common.GetMulElemVec(
			propagated,
			common.GetVectorSigmoidPrime(currSums))
	)
	return currErrs
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
