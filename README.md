# rnn

Source code for a [series](http://chisquared.org) of posts about recurrent neural networks. (It's in Russian though, beware.)

You'll need to install the linear algebra library for Go:

```
$ go get github.com/gonum/matrix/mat64

```

You can execute any of the examples in the blog like this:

```
$ go run main.go [--basicNN | --Elman-- | --Jordan | --LSTM]

```

For example, you can train an Elman network:

```
$ go run main.go --Elman
====================================================
Testing basic Vanilla RNN on sample series dataset:
====================================================
Expected: .     1.0     .       .               Predicted: .    1.0     .       .
Expected: .     .       1.0     .               Predicted: .    .       1.0     .
Expected: .     .       .       1.0             Predicted: .    .       .       1.0
Expected: .     .       1.0     .               Predicted: .    .       1.0     .
Expected: .     1.0     .       .               Predicted: .    1.0     .       .
Expected: 1.0   .       .       .               Predicted: 1.0  .       .       .

```