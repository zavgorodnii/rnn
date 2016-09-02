package common

import (
	"bufio"
	"encoding/csv"
	"io"
	"os"
	"strconv"

	"github.com/gonum/matrix/mat64"
)

//////////////////////////////////////////////////////////////////////////////
//
// This file contains various functions that get you train/test data.
//
//////////////////////////////////////////////////////////////////////////////

// GetIris returns you X (samples) and Y (labels) for the Iris
// dataset. Labels are one-hot encoded.
func GetIris() (input, expected *mat64.Dense) {
	input = LoadFromCSV("data/iris_x.csv", 4)
	expected = LoadFromCSV("data/iris_y.csv", 3)
	return
}

// LoadFromCSV returns a matrix restored from a file (@path) which must be a
// valid csv with each line representing a float64-vector of length @vectorLen.
func LoadFromCSV(path string, vectorLen int) (out *mat64.Dense) {
	// Prepare storage for points
	points := [][]float64{}
	f, _ := os.Open(path)
	// Create a new reader.
	r := csv.NewReader(bufio.NewReader(f))
	i := 0
	for {
		record, err := r.Read()
		// Stop at EOF.
		if err == io.EOF {
			break
		}
		j := 0
		points = append(points, make([]float64, vectorLen))
		for value := range record {
			points[i][j], err = strconv.ParseFloat(record[value], 64)
			j++
		}
		i++
	}
	out = mat64.NewDense(len(points), vectorLen, nil)
	for idx, point := range points {
		out.SetRow(idx, point)
	}
	return out
}
