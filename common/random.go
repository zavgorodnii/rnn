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

// RandomDense fills each cell of a matrix with random numbers in [min, max].
func RandomDense(min, max float64, m *mat64.Dense) {
	numRows, numCols := m.Dims()
	for rowIdx := 0; rowIdx < numRows; rowIdx++ {
		for colIdx := 0; colIdx < numCols; colIdx++ {
			m.Set(rowIdx, colIdx, GetRandomInRange(min, max))
		}
	}
}

// RandomRange returns a random float64 in [min, max].
func GetRandomInRange(min, max float64) float64 {
	return rand.Float64()*(max-min) + min
}
