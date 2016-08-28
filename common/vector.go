package common

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

// SumVector returns the sum of all elements in a vector.
func SumVector(v *mat64.Vector) float64 {
	sum := .0
	for i := 0; i < v.Len(); i++ {
		sum += v.At(i, 0)
	}
	return sum
}

// TanhVector returns a vector where each @v's component is replaced with its
// hyperbolic tangent.
func TanhVector(v *mat64.Vector) *mat64.Vector {
	// Copy the input
	out := mat64.NewVector(v.Len(), v.RawVector().Data)
	for i := 0; i < out.Len(); i++ {
		out.SetVec(i, math.Tanh(v.At(i, 0)))
	}
	return out
}

// SoftmaxVector sets the softmax value for each component of @v.
func SoftmaxVector(v *mat64.Vector) *mat64.Vector {
	// Copy the input
	out := mat64.NewVector(v.Len(), v.RawVector().Data)
	for i := 0; i < out.Len(); i++ {
		sum, ith := .0, v.At(i, 0)
		for j := 0; j < v.Len(); j++ {
			sum += math.Pow(math.E, v.At(j, 0))
		}
		val := math.Pow(math.E, ith) / sum
		out.SetVec(i, val)
	}
	return out
}
