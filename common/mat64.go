package common

import "github.com/gonum/matrix/mat64"

func GetTransposed(m *mat64.Dense) *mat64.Dense {
	return mat64.DenseCopyOf(m.T())
}

func GetMulElemVec(v1, v2 *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v1.Len(), nil)
	out.MulElemVec(v1, v2)
	return out
}

func GetMulVec(m *mat64.Dense, v *mat64.Vector) *mat64.Vector {
	mR, _ := m.Dims()
	out := mat64.NewVector(mR, nil)
	out.MulVec(m, v)
	return out
}

func GetAddVec(v1, v2 *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v1.Len(), nil)
	out.AddVec(v1, v2)
	return out
}

func GetSubVec(v1, v2 *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v1.Len(), nil)
	out.SubVec(v1, v2)
	return out
}

func GetDenseApply(m *mat64.Dense, cb func(float64) float64) *mat64.Dense {
	r, c := m.Dims()
	out := mat64.NewDense(r, c, nil)
	out.Apply(func(i, j int, val float64) float64 {
		return cb(val)
	}, m)
	return out
}
