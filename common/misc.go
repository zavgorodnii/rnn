package common

import "math/rand"

func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func Shuffle(a []int) {
	for i := range a {
		j := rand.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}

func GetRangeInt(lower, upper int) []int {
	out := []int{}
	for i := lower; i < upper; i++ {
		out = append(out, i)
	}
	return out
}
