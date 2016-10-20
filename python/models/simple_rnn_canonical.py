# -*- coding: utf-8 -*-

import logging

import numpy as np
import tensorflow as tf


class SimpleRNNCanonical(object):

    num_classes = None
    unfolding_depth = None
    h_to_o_scope = "hidden_to_output"

    hidden = None
    init_hidden = None
    hidden_shape = None

    def __init__(self, num_hidden_neurons=4, eta=0.1):
        """Initializes local and tensorflow structures and dependencies.

        Args:
            num_hidden_neurons: Number of neurons in hodden layer.
            eta: learning rate.
        """
        self.eta = eta
        self.num_hidden_neurons = num_hidden_neurons

    def train(self, training_data):
        self.num_classes = training_data.num_classes
        self.unfolding_depth = training_data.truncate_at
        self.hidden_shape = [training_data.batch_size, self.num_hidden_neurons]
        batch_size = training_data.batch_size
        # A batch (unputs and labels).
        x = tf.placeholder(tf.int32, [batch_size, self.unfolding_depth])
        y = tf.placeholder(tf.int32, [batch_size, self.unfolding_depth])
        # Turn our x placeholder into a list of one-hot tensors. We just need
        # to tell `tf.one_hot` the size of our vocabulary (self.num_classes),
        # and it will generate appropriate one-hot vectors.
        # As `x` has shape (batch_size, truncate_at), `x_one_hot` will have
        # shape (batch_size, truncate_at, num_classes)
        x_one_hot = tf.one_hot(x, self.num_classes)
        # We can't use `x_one_hot` as is. After unpacking it along the
        # `num_classes` axis `rnn_inputs` is a list of length `truncate_at`
        # containing tensors with shape (batch_size, num_classes).
        #
        # O.K., we need something to be explained. `rnn_input[0]` will be a
        # "list" of first elements of all truncated sequences in a batch,
        # `rnn_input[1]` will be a "list" of second elements, etc.
        # This is because the easiest way to represent each type of duplicate
        # tensor (the rnn inputs, the rnn outputs (hidden state), the
        # predictions, and the loss) is as a LIST OF TENSORS.
        #
        # This is certainly not how we are used to imagine feeding sequences
        # to a network element by element, but conceptually it's the same
        # old story. Instead of collecting outputs and hidden states for a
        # single sequence, we can now do the same for the whole batch.
        # This is partly motivated by  computational efficiency: multiplying
        # larger matrices saves us CPU (GPU) time.
        rnn_inputs = tf.unpack(x_one_hot, axis=1)
        # As always with RNNs, before we get any input initial hidden state is
        # a zero-multitude vector. Note that we have a 2D hidden state due to
        # the data representation described in the comment section above.
        cell = tf.nn.rnn_cell.BasicRNNCell(self.num_hidden_neurons)
        self.init_hidden = cell.zero_state(batch_size, tf.float32)
        # After feeding another batch to network we have some final hidden
        # state that should be used as the "initial" (t_{-1}) state when
        # feeding the next batch. Note that `_final_hidden` here is a
        # tensorflow object that is not evaluated yet.
        self.hidden, _final_hidden = tf.nn.rnn(
            cell, rnn_inputs, initial_state=self.init_hidden)
        # Initialize weight matrices and biases as tensorflow variables.
        self._setup_model_params()
        train_step, _step_loss = self._setup_predictions(y)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch_num, epoch in enumerate(training_data):
                final_hidden = None
                step_loss = 0.
                for X, Y in epoch:
                    # `step_loss` is the mean loss for current step,
                    # `final_hidden` is the evaluated final hidden state for
                    # current batch. We tell tensorflow to use it as the
                    # "initial" (t_{-1}) state for the next batch by placing
                    # it in the `feed_dict` kwarg. Note that `self.init_hidden`
                    # is a tensorflow (unevaluated) tensorflow object that is
                    # used in `self._setup_input_processing`.
                    #
                    # Note that we pass `_step_loss` to `sess.run` just because
                    # we want to print something after each epoch. The only
                    # value that we really need to evaluate (and reuse) for
                    # training is `_final_hidden`.
                    feed_dict = {x: X, y: Y}
                    # First step already has valid (zero) initial hidden state.
                    # For further steps we need to use final hidden state for
                    # current batch as our new initial hidden state.
                    #
                    # `is not None` here is not bad style; `final_hidden` is
                    # going to become an array an will raise a `ValueError` if
                    # being used as a truth value.
                    if final_hidden is not None:
                        feed_dict[self.init_hidden] = final_hidden
                    step_loss, final_hidden, _ = sess.run(
                        [_step_loss, _final_hidden, train_step],
                        feed_dict=feed_dict)
                logging.info("Epoch %s, mean loss: %s",
                             epoch_num, np.mean(step_loss))

    def _setup_model_params(self):
        """Initializes weight matrices and biases as tensorflow variables.

        Our model has only one hidden layer. We need to initialize
        input-to-hidden and hidden-to-output weight matrices and bias vectors.

        Because we concatenate (t, current) input and (t-1, previous) hidden
        state into one vector, there's a single input-to-hidden weight matrix
        `W` and a single bias vector `b`. Its shape (num_classes + self.h_size,
        self.h_size) is due to the fact that input and output one-hot vectors
        have the same size (self.num_classes); obviously `W` treats inputs as
        row-vectors (not column-vectors).
        """
        num_classes, num_hidden = self.num_classes, self.num_hidden_neurons
        with tf.variable_scope(self.h_to_o_scope):
            tf.get_variable('W', [num_hidden, num_classes])
            tf.get_variable('b', [num_classes],
                            initializer=tf.constant_initializer(0.0))

    def _get_model_params(self, layer_scope):
        """Returns model parameters `W` and `b` from a certain layer.

        Args:
            layer_scope: name of the tensorflow scope containing a layer's
                params (stored as tensorflow variables).
        Returns:
            W: a 2D tensorflow variable representing a weight matrix.
            b: a 1D tensorflow variable representing a bias vector.
        Raises:
            An `Exception` if layer_scope is not self.i_to_h_scope or
            self.h_to_o_scope (we have a fixed model with one hidden layer).

        """
        if layer_scope not in [self.h_to_o_scope]:
            raise Exception("Unknown layer scope %s" % layer_scope)
        with tf.variable_scope(layer_scope, reuse=True):
            # Variables are reused, no shapes or initializers needed.
            W, b = tf.get_variable('W'), tf.get_variable('b')
        return W, b

    def _setup_predictions(self, Y):
        """Unfolds the network for a batch of truncated sequences ("in
        parallel") until the output layer using the precomputed hidden states.

        See `self._setup_input_processing` for previous unfolding step.

        Args:
            Y: A tensor with shape [batch_size, self.unfolding_depth].
                Represents all output labels organized in batches.
        """
        batch_size, truncate_at = Y.get_shape()
        W, b = self._get_model_params(self.h_to_o_scope)
        logits = [tf.matmul(hid, W) + b for hid in self.hidden]
        # Predictions (not used here directly).
        _ = [tf.nn.softmax(logit) for logit in logits]
        # Turn our y placeholder into a list of labels.
        y_as_list = [tf.squeeze(i, squeeze_dims=[1])
                     for i in tf.split(1, self.unfolding_depth, Y)]
        loss_weights = [tf.ones([batch_size]) for i in range(truncate_at)]
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            logits, y_as_list, loss_weights)
        train_step = tf.train.AdagradOptimizer(self.eta).minimize(losses)
        return train_step, losses


if __name__ == "__main__":
    pass
