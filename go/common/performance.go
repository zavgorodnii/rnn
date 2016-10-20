package common

import "github.com/gonum/matrix/mat64"

func GetClassAccuracy(outputs []*mat64.Vector, expected *mat64.Dense) (float64, int) {
	totalOk := 0.
	for idx, currOut := range outputs {
		currExp := expected.RowView(idx)
		if ClassPredictionCorrect(currOut, currExp) {
			totalOk++
		}
	}
	return totalOk / float64(len(outputs)), int(totalOk)
}

func ClassPredictionCorrect(out, exp *mat64.Vector) bool {
	requiredIdx := GetMaxIdx(exp)
	data := out.RawVector().Data
	for idx, value := range data {
		if idx == requiredIdx {
			continue
		}
		if data[requiredIdx] <= value {
			return false
		}
	}
	return true
}

func GetMaxIdx(v *mat64.Vector) int {
	maxIdx, max := 0, 0.
	for i := 0; i < v.Len(); i++ {
		if v.At(i, 0) > max {
			maxIdx, max = i, v.At(i, 0)
		}
	}
	return maxIdx
}
