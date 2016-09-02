package common

import "github.com/gonum/matrix/mat64"

//////////////////////////////////////////////////////////////////////////////
//
// This file contains non-inplace wrappers for mat64 vector/matrix operations.
//
//////////////////////////////////////////////////////////////////////////////

// GetTransposed returns a transposed Dense matrix.
func GetTransposed(m *mat64.Dense) *mat64.Dense {
	return mat64.DenseCopyOf(m.T())
}

// GetMulElemVec wraps mat64.MulElemVec and returns a new vector.
func GetMulElemVec(v1, v2 *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v1.Len(), nil)
	out.MulElemVec(v1, v2)
	return out
}

// GetMulVec wraps mat64.MulVec and returns a new vector.
func GetMulVec(m *mat64.Dense, v *mat64.Vector) *mat64.Vector {
	mR, _ := m.Dims()
	out := mat64.NewVector(mR, nil)
	out.MulVec(m, v)
	return out
}

// GetAddVec wraps mat64.AddVec and returns a new vector.
func GetAddVec(v1, v2 *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v1.Len(), nil)
	out.AddVec(v1, v2)
	return out
}

// GetSubVec wraps mat64.SubVec and returns a new vector.
func GetSubVec(v1, v2 *mat64.Vector) *mat64.Vector {
	out := mat64.NewVector(v1.Len(), nil)
	out.SubVec(v1, v2)
	return out
}

// GetDenseApply wraps mat64.Apply for dense matrices and returns a new vector.
// it slightly changes the API (see the callback signature).
func GetDenseApply(m *mat64.Dense, cb func(float64) float64) *mat64.Dense {
	r, c := m.Dims()
	out := mat64.NewDense(r, c, nil)
	out.Apply(func(i, j int, val float64) float64 {
		return cb(val)
	}, m)
	return out
}

// GetVectorApply applies @cg to each element in @v.
func GetVectorApply(v *mat64.Vector, cb func(float64) float64) *mat64.Vector {
	out := mat64.NewVector(v.Len(), nil)
	for i := 0; i < v.Len(); i++ {
		out.SetVec(i, cb(v.At(i, 0)))
	}
	return out
}
