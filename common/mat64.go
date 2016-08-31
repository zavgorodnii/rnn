package common

import "github.com/gonum/matrix/mat64"

func GetTransposed(m *mat64.Dense) *mat64.Dense {
	return mat64.DenseCopyOf(m.T())
}

func GetMatrixByVector(m *mat64.Dense, v *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v.Len(), nil)
	out.MulVec(m, v)
	return out
}

func GetMulElemVec(v1, v2 *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v1.Len(), nil)
	out.MulElemVec(v1, v2)
	return out
}

func GetMulVec(m *mat64.Dense, v *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v.Len(), nil)
	out.MulVec(m, v)
	return out
}

func GetSubVec(v1, v2 *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v1.Len(), nil)
	out.SubVec(v1, v2)
	return out
}
