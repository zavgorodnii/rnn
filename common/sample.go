package common

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// CrossEntropyError returns the total cross entropy error for a set
// of @predicted and @expected vectors. (Each row in those matrices is a
// vector-sample.)
func CrossEntropyError(predicted, expected *mat64.Dense) float64 {
	errorSum := .0
	numSamples, _ := predicted.Dims()
	for i := 0; i < numSamples; i++ {
		errorSum += SampleCrossEntropyError(
			expected.RowView(i), predicted.RowView(i),
		)
	}
	return errorSum / float64(numSamples)
}

// SampleCrossEntropyError calculates the error for a sample given a @predicted
// and @expected vector.
// If the prediction for a sample is [0.3, 0.3, 0.4] and the expected output
// is [.0, .0, 1.], then the cross entropy error is calculated as
// -( (ln(0.3)*0) + (ln(0.3)*0) + (ln(0.4)*1) ) = -ln(0.4).
func SampleCrossEntropyError(predicted, expected *mat64.Vector) float64 {
	out := mat64.NewVector(predicted.Len(), nil)
	out.CopyVec(predicted)
	for i := 0; i < predicted.Len(); i++ {
		out.SetVec(i, math.Log(out.At(i, 0)))
	}
	out.MulElemVec(expected, predicted)
	return -GetVectorSum(out)
}
