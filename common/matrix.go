package common

import "github.com/gonum/matrix/mat64"

func GetTransposed(m *mat64.Dense) *mat64.Dense {
	return mat64.DenseCopyOf(m.T())
}
