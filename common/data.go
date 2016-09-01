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

func GetTwoClass(numSamples, length int) (input, expected *mat64.Dense) {
	input = mat64.NewDense(numSamples, length, nil)
	expected = mat64.NewDense(numSamples, 2, nil)
	for i := 0; i < numSamples; i++ {
		var (
			currInp = mat64.NewVector(length, nil)
			currExp *mat64.Vector
		)
		if (i % 2) == 0 {
			RandomVector(10., 15., currInp)
			currExp = mat64.NewVector(2, []float64{0., 1.})
		} else {
			RandomVector(100., 150., currInp)
			currExp = mat64.NewVector(2, []float64{1., 0.})
		}
		input.SetRow(i, currInp.RawVector().Data)
		expected.SetRow(i, currExp.RawVector().Data)
	}
	return
}
