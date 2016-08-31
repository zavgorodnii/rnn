package common

import "github.com/gonum/matrix/mat64"

// GetRandomDense generates @numSamples of random vectors of length @length.
func GetRandomDense(numSamples, length int) *mat64.Dense {
	out := mat64.NewDense(numSamples, length, nil)
	for i := 0; i < numSamples; i++ {
		curr := mat64.NewVector(length, nil)
		RandomVector(-10., 10., curr)
		out.SetRow(i, curr.RawVector().Data)
	}
	return out
}

// GetRandomVector generates a random vector of length @length.
func GetRandomVector(length int) *mat64.Vector {
	out := mat64.NewVector(length, nil)
	RandomVector(-10., 10., out)
	return out
}
