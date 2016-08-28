package common

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

func PrintDense(m *mat64.Dense) {
	numRows, _ := m.Dims()
	for i := 0; i < numRows; i++ {
		PrintVectorNewline(m.RowView(i))
	}
	fmt.Printf("\n")
}

func PrintVectorNewline(v *mat64.Vector) {
	PrintVector(v)
	fmt.Printf("\n")
}

// PrintVector pretty prints @v.
func PrintVector(v *mat64.Vector) {
	for i := 0; i < v.Len(); i++ {
		fmt.Printf("%.3f\t", v.At(i, 0))
	}
}
