package common

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// PrintDense pretty prints a mat64.Dense matrix.
func PrintDense(m *mat64.Dense) {
	numRows, _ := m.Dims()
	for i := 0; i < numRows; i++ {
		PrintVector(m.RowView(i))
	}
	fmt.Printf("\n")
}

// PrintVector pretty prints @v and adds a newline.
func PrintVector(v *mat64.Vector) {
	PrintVectorStrip(v)
	fmt.Printf("\n")
}

// PrintVectorStripSub is a stupid function that substitutes each component
// of @v that is equal to @val with @sub. This is used mostly for better
// visualizing time-series patterns.
func PrintVectorStripSub(v *mat64.Vector, val float64, sub string) {
	for i := 0; i < v.Len(); i++ {
		if v.At(i, 0) == val {
			fmt.Printf("%s\t", sub)
		} else {
			fmt.Printf("%.1f\t", v.At(i, 0))
		}
	}
}

// PrintVectorStrip pretty prints @v.
func PrintVectorStrip(v *mat64.Vector) {
	for i := 0; i < v.Len(); i++ {
		fmt.Printf("%.1f\t", v.At(i, 0))
	}
}
