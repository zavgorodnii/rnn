package common

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

func GetVectorAbs(v *mat64.Vector) float64 {
	sum := .0
	for i := 0; i < v.Len(); i++ {
		sum += math.Pow(v.At(i, 0), 2)
	}
	return math.Sqrt(sum)
}

// GetVectorSigmoid applies sigmoid function to each element in @v.
func GetVectorSigmoid(v *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		sigm := 1. / (1. + math.Pow(math.E, -v.At(i, 0)))
		out.SetVec(i, sigm)
	}
	return out
}

// GetVectorSigmoidPrime returns derivative of VectorSigmoid.
func GetVectorSigmoidPrime(v *mat64.Vector) *mat64.Vector {
	sigm := GetVectorSigmoid(v)
	return GetMulElemVec(sigm, GetVectorApply(sigm, func(i float64) float64 {
		return 1.0 - i
	}))
}

// GetVectorApply applies @cg to each element in @v.
func GetVectorApply(v *mat64.Vector, cb func(float64) float64) *mat64.Vector {
	out := mat64.NewVector(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		out.SetVec(i, cb(v.At(i, 0)))
	}
	return out
}

// GetVectorSum returns the sum of all elements in @v.
func GetVectorSum(v *mat64.Vector) float64 {
	sum := .0
	for i := 0; i < v.Len(); i++ {
		sum += v.At(i, 0)
	}
	return sum
}

// GetVectorPow raises each element in @v to @power-th power.
func GetVectorPow(v *mat64.Vector, power float64) *mat64.Vector {
	out := mat64.NewVector(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		out.SetVec(i, math.Pow(v.At(i, 0), power))
	}
	return out
}

// GetVectorTanh applies hyperbolic tangent to each element in @v.
func GetVectorTanh(v *mat64.Vector) *mat64.Vector {
	// Copy the input
	out := mat64.NewVector(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		out.SetVec(i, math.Tanh(v.At(i, 0)))
	}
	return out
}

// GetVectorSoftmax sets the softmax value for each element in @v.
func GetVectorSoftmax(v *mat64.Vector) *mat64.Vector {
	// Copy the input
	out := mat64.NewVector(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		sum, ith := .0, v.At(i, 0)
		for j := 0; j < v.Len(); j++ {
			sum += math.Pow(math.E, v.At(j, 0))
		}
		val := math.Pow(math.E, ith) / sum
		out.SetVec(i, val)
	}
	return out
}
