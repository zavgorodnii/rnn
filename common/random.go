package common

import (
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// RandomVector fills each component of a vector with random numbers in
// [min, max].
func RandomVector(min, max float64, v *mat64.Vector) {
	numComponents := v.Len()
	for idx := 0; idx < numComponents; idx++ {
		v.SetVec(idx, GetRandomInRange(min, max))
	}
}

// GetRandomVector generates a random vector of length @length.
func GetRandomVector(length int) *mat64.Vector {
	out := mat64.NewVector(length, nil)
	RandomVector(-10., 10., out)
	return out
}

// RandomDense fills each cell of a matrix with random numbers in [min, max].
func RandomDense(min, max float64, m *mat64.Dense) {
	numRows, numCols := m.Dims()
	for rowIdx := 0; rowIdx < numRows; rowIdx++ {
		for colIdx := 0; colIdx < numCols; colIdx++ {
			m.Set(rowIdx, colIdx, GetRandomInRange(min, max))
		}
	}
}

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

// GetRandomInRange returns a random float64 in [min, max].
func GetRandomInRange(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}
