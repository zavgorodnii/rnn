# -*- coding: utf-8 -*-

import logging

import numpy as np
import tensorflow as tf


class MultilayerGRU(object):

    num_classes = None
    unfolding_depth = None
    h_to_o_scope = "hidden_to_output"

    x = None
    y = None
    hidden = None
    hidden_shape = None
    _final_hidden = None
    _initial_hidden = None
    _logits = None
    _softmax = None

    def __init__(self,
                 eta=0.1,
                 num_hidden_neurons=4,
                 num_gru_layers=2,
                 dropout_prob=1.0):
        """Initializes local and tensorflow structures and dependencies.

        Args:
            eta: learning rate.
            num_hidden_neurons: Number of neurons in hodden layer.
            num_gru_layers: Number of stacked gru layers.
            dropout_prob: Float between 0 and 1, input keep probability; if it
                is 1, no input dropout will be added.
        """
        self.eta = eta
        self.dropout_prob = dropout_prob
        self.num_gru_layers = num_gru_layers
        self.num_hidden_neurons = num_hidden_neurons

    def fit(self, sess, data):
        self._setup_common(data)
        self._train(sess, data)

    def predict(self, sess, data):
        self._setup_common(data)
        return self._predict(sess, data)

    def _setup_common(self, data):
        self.num_classes = data.num_classes
        self.unfolding_depth = data.truncate_at
        self.hidden_shape = [data.batch_size, self.num_hidden_neurons]
        batch_size = data.batch_size
        # A batch (unputs and labels).
        self.x = tf.placeholder(tf.int32, [batch_size, self.unfolding_depth])
        # Turn our x placeholder into a list of one-hot tensors. We just need
        # to tell `tf.one_hot` the size of our vocabulary (self.num_classes),
        # and it will generate appropriate one-hot vectors.
        # As `x` has shape (batch_size, truncate_at), `x_one_hot` will have
        # shape (batch_size, truncate_at, num_classes)
        x_one_hot = tf.one_hot(self.x, self.num_classes)
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
        gru = tf.nn.rnn_cell.GRUCell(self.num_hidden_neurons)
        stacked_gru = tf.nn.rnn_cell.MultiRNNCell([gru] * self.num_gru_layers)
        # N.B.: LSTM's zero state is a little bit more complex that just a
        # single tensor.
        self._initial_hidden = stacked_gru.zero_state(batch_size, tf.float32)
        # After feeding another batch to network we have some final hidden
        # state that should be used as the "initial" (t_{-1}) state when
        # feeding the next batch. Note that `self._final_hidden` here is a
        # tensorflow object that is not evaluated yet.
        self.hidden, self._final_hidden = tf.nn.rnn(
            stacked_gru, rnn_inputs, initial_state=self._initial_hidden)
        num_classes, num_hidden = self.num_classes, self.num_hidden_neurons
        with tf.variable_scope(self.h_to_o_scope):
            W = tf.get_variable('W', [num_hidden, num_classes])
            b = tf.get_variable('b', [num_classes],
                                initializer=tf.constant_initializer(0.0))
        self._logits = [tf.matmul(hid, W) + b for hid in self.hidden]
        # Predictions (not used here directly).
        self._softmax = [tf.nn.softmax(logit) for logit in self._logits]

    def _setup_training(self, batch_size):
        """Unfolds the network for a batch of truncated sequences ("in
        parallel") until the output layer using the precomputed hidden states.

        See `self._setup_input_processing` for previous unfolding step.

        Args:
            Y: A tensor with shape [batch_size, self.unfolding_depth].
                Represents all output labels organized in batches.
        """
        # Turn our Y placeholder into a list of labels.
        depth = self.unfolding_depth
        self.y = tf.placeholder(tf.int32, [batch_size, depth])
        y_as_list = [tf.squeeze(i, squeeze_dims=[1])
                     for i in tf.split(1, depth, self.y)]
        loss_weights = [tf.ones([batch_size]) for i in range(depth)]
        losses = tf.nn.seq2seq.sequence_loss_by_example(
            self._logits, y_as_list, loss_weights)
        train_step = tf.train.AdagradOptimizer(self.eta).minimize(losses)
        return train_step, losses

    def _train(self, sess, train_data):
        _train_step, _step_loss = self._setup_training(train_data.batch_size)
        sess.run(tf.initialize_all_variables())
        for epoch_num, epoch in enumerate(train_data):
            step_loss = 0.
            # `final_hidden` is the evaluated final hidden state for
            # current batch. We tell tensorflow to use it as the
            # "initial" (t_{-1}) state for the next batch by placing
            # it in the `feed_dict` kwarg. Note that `self._initial_hidden`
            # is a tensorflow (unevaluated) tensorflow object that is
            # used in `self._setup_input_processing`.
            final_hidden = None
            for X, Y in epoch:
                # `step_loss` is the mean loss for current step. Note that
                # we pass `_step_loss` to `sess.run` just because we want
                # to print something after each epoch. The only value that
                # we really need to evaluate (and reuse) for training is
                # `_final_hidden`.
                feed_dict = {self.x: X, self.y: Y}
                # First step already has valid (zero) initial hidden state.
                # For further steps we need to use final hidden state for
                # current batch as our new initial hidden state.
                #
                # `is not None` here is not bad style; `final_hidden` is
                # going to become an array an will raise a `ValueError` if
                # being used as a truth value.
                if final_hidden is not None:
                    feed_dict[self._initial_hidden] = final_hidden
                step_loss, final_hidden, _ = sess.run(
                    [_step_loss, self._final_hidden, _train_step],
                    feed_dict=feed_dict)
            logging.info("Epoch %s, mean loss: %s",
                         epoch_num, np.mean(step_loss))
            if np.mean(step_loss) <= 0.42:
                logging.info("Model successfully trained")
                return

    def _predict(self, sess, data):
        final_hidden = None
        for X, Y in data:
            feed_dict = {self.x: X}
            # First step already has valid (zero) initial hidden state.
            # For further steps we need to use final hidden state for
            # current batch as our new initial hidden state.
            #
            # `is not None` here is not bad style; `final_hidden` is
            # going to become an array an will raise a `ValueError` if
            # being used as a truth value.
            if final_hidden is not None:
                feed_dict[self._initial_hidden] = final_hidden
            prediction, final_hidden = sess.run(
                [self._logits, self._final_hidden], feed_dict=feed_dict)
            logging.info(
                "Predicted: %s\tExpected:%s",
                np.argmax(prediction, axis=2).T[0], Y[0])


if __name__ == "__main__":
    pass
