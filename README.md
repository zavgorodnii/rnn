# rnn

Source code for a [series](http://chisquared.org/post/recurrent-neural-networks-part-1-basic-nn) of posts about recurrent neural networks. (It's in Russian though, beware.)

You'll need to install the linear algebra library for Go:

```
$ go get github.com/gonum/matrix/mat64

```

You can execute any of the examples in the blog like this:

```
$ go run main.go [--basicNN | --Elman | --Jordan | --LSTM]

```

For example, you can train an Elman network:

```
$ go run main.go --Elman
====================================================
Testing basic Vanilla RNN on sample series dataset:
====================================================
Epoch:  0
_________________________________________________________________
Input:  1  .  .  .  Expected:  .  1  .  .  Predicted:  1  .  .  1
Input:  .  1  .  .  Expected:  .  .  1  .  Predicted:  1  .  .  1
Input:  .  .  1  .  Expected:  .  .  .  1  Predicted:  1  .  .  1
Input:  .  .  .  1  Expected:  .  .  1  .  Predicted:  1  .  .  1
Input:  .  .  1  .  Expected:  .  1  .  .  Predicted:  1  .  .  1
Input:  .  1  .  .  Expected:  1  .  .  .  Predicted:  1  .  .  1

Epoch:  1000
_________________________________________________________________
Input:  1  .  .  .  Expected:  .  1  .  .  Predicted:  .  1  .  .
Input:  .  1  .  .  Expected:  .  .  1  .  Predicted:  .  .  .  .
Input:  .  .  1  .  Expected:  .  .  .  1  Predicted:  .  1  .  .
Input:  .  .  .  1  Expected:  .  .  1  .  Predicted:  .  .  1  .
Input:  .  .  1  .  Expected:  .  1  .  .  Predicted:  .  1  .  .
Input:  .  1  .  .  Expected:  1  .  .  .  Predicted:  .  .  1  .

Epoch:  2000
_________________________________________________________________
Input:  1  .  .  .  Expected:  .  1  .  .  Predicted:  .  1  .  .
Input:  .  1  .  .  Expected:  .  .  1  .  Predicted:  .  .  1  .
Input:  .  .  1  .  Expected:  .  .  .  1  Predicted:  .  1  .  .
Input:  .  .  .  1  Expected:  .  .  1  .  Predicted:  .  .  1  .
Input:  .  .  1  .  Expected:  .  1  .  .  Predicted:  .  1  .  .
Input:  .  1  .  .  Expected:  1  .  .  .  Predicted:  .  .  1  .

Epoch:  3000
_________________________________________________________________
Input:  1  .  .  .  Expected:  .  1  .  .  Predicted:  .  1  .  .
Input:  .  1  .  .  Expected:  .  .  1  .  Predicted:  .  .  1  .
Input:  .  .  1  .  Expected:  .  .  .  1  Predicted:  .  .  .  .
Input:  .  .  .  1  Expected:  .  .  1  .  Predicted:  .  .  1  .
Input:  .  .  1  .  Expected:  .  1  .  .  Predicted:  .  1  .  .
Input:  .  1  .  .  Expected:  1  .  .  .  Predicted:  .  .  1  .

Epoch:  4000
_________________________________________________________________
Input:  1  .  .  .  Expected:  .  1  .  .  Predicted:  .  1  .  .
Input:  .  1  .  .  Expected:  .  .  1  .  Predicted:  .  .  1  .
Input:  .  .  1  .  Expected:  .  .  .  1  Predicted:  .  .  .  1
Input:  .  .  .  1  Expected:  .  .  1  .  Predicted:  .  .  1  .
Input:  .  .  1  .  Expected:  .  1  .  .  Predicted:  .  1  .  .
Input:  .  1  .  .  Expected:  1  .  .  .  Predicted:  1  .  .  .
```

This repository also contains source code for RNNs implemented in Python using the awesome [Tensorflow](https://tensorflow.org) library. You'll need to install tensorflow to run those sources.