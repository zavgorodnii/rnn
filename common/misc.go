package common

import "github.com/gonum/matrix/mat64"

func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func GetLastVector(slice []*mat64.Vector) *mat64.Vector {
	return slice[len(slice)-1]
}
