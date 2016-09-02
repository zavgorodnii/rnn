package common

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// PrintDense pretty prints a mat64.Dense matrix.
func PrintDense(m *mat64.Dense) {
	numRows, _ := m.Dims()
	for i := 0; i < numRows; i++ {
		printVector(m.RowView(i))
	}
	fmt.Printf("\n")
}

// PrintVector pretty prints @v and adds a newline.
func PrintVector(v *mat64.Vector) {
	printVector(v)
	fmt.Printf("\n")
}

// PrintVector pretty prints @v.
func printVector(v *mat64.Vector) {
	for i := 0; i < v.Len(); i++ {
		fmt.Printf("%.3f\t", v.At(i, 0))
	}
}
