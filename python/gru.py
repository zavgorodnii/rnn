# -*- coding: utf-8 -*-

import random

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader


class CharSequenceInput(object):

    def __init__(self,
                 raw_data,
                 num_epochs=5,
                 batch_size=200,
                 truncate_at=5):
        vocab = set(raw_data)
        self.vocab_size = len(vocab)
        self.idx_to_vocab = dict(enumerate(vocab))
        self.vocab_to_idx = dict(
            zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))
        translated_data = [self.vocab_to_idx[c] for c in raw_data]
        self.data = self._gen_epochs(
            translated_data, num_epochs, batch_size, truncate_at)

    def __iter__(self):
        return self

    def next(self):
        return self.data.next()

    def _gen_epochs(self, data, num_epochs, batch_size, truncate_at):
        for _ in range(num_epochs):
            yield reader.ptb_iterator(data, batch_size, truncate_at)


class GRUModel(object):

    sess = None

    def __init__(self,
                 state_size=128,
                 batch_size=64,
                 num_layers=2,
                 num_classes=None,
                 truncate_at=32,
                 learning_rate=1e-4):
        """Builds TF graph for the network.

        Also sets up a saver for the trained model.

        Args:
            state_size: Number of GRU cells in each of the hidden layers.
            batch_size: Number of sequences simultaneously propagated through
                the network.
            num_layers: Number of stacked hidden layers.
            num_classes: Number of output neurons.
            truncate_at: Length of each training truncated sequence, same as
                unfolded RNN depth, same as number of truncated backprop steps.
            learning_rate: Initial learning rate.
        """
        self._reset_graph()
        # Get placeholders for inputs and expected outputs.
        x = tf.placeholder(tf.int32, [batch_size, truncate_at])
        y = tf.placeholder(tf.int32, [batch_size, truncate_at])
        dropout = tf.constant(1.0)
        # Transform inputs to get suitable embeddings.
        embeddings = tf.get_variable(
            'embedding_matrix', [num_classes, state_size])
        rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
        # Set up stacked hidden layers with dropout.
        cell = tf.nn.rnn_cell.GRUCell(state_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
        init_state = cell.zero_state(batch_size, tf.float32)
        # Get hidden layer outputs.
        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell, rnn_inputs, initial_state=init_state)
        # Reshape hidden layer outputs and expected outputs.
        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
        y_reshaped = tf.reshape(y, [-1])
        # Pass hidden layer outputs to softmax output layer.
        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [state_size, num_classes])
            b = tf.get_variable(
                'b', [num_classes], initializer=tf.constant_initializer(0.0))
        logits = tf.matmul(rnn_outputs, W) + b
        predictions = tf.nn.softmax(logits)
        # Losses are minimized with Adam, but Adagrad/Adadelta/any other
        # optimization algorithm may be used.
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_reshaped))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        self._x = x
        self._y = y
        self._init_state = init_state
        self._final_state = final_state
        self._loss = loss
        self._train_step = train_step
        self._preds = predictions
        self._saver = tf.train.Saver()

    def train(self, data, checkpoint='saves/DefaultModel.tf', min_loss=0.5):
        """Trains the model and saves result to disk.

        Args:
            data: Batches of (X, Y) sequences organized in epochs.
            checkpoint: Path used to save the trained model.
        """
        tf.set_random_seed(2345)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for idx, epoch in enumerate(data):
                epoch_loss, steps = 0, 0
                # `training_state` is the activation of hidden layer(s) after
                # the last introduced batch. `None` if no samples were
                # introduced yet.
                training_state = None
                for X, Y in epoch:
                    steps += 1
                    feed_dict = {self._x: X, self._y: Y}
                    # Use the activation of hidden layer(s) after the last
                    # introduced batch as initial state for the next batch.
                    # `is not None` here is not bad style but necessity:
                    # tensors throw exceptions is used as booleans.
                    if training_state is not None:
                        feed_dict[self._init_state] = training_state
                    step_loss, training_state, _ = sess.run(
                        [self._loss, self._final_state, self._train_step],
                        feed_dict)
                    epoch_loss += step_loss
                print("Average training loss for Epoch",
                      idx, ":", epoch_loss / steps)
                if epoch_loss / steps < min_loss:
                    print("Reached minimal loss, training stopped")
            self._saver.save(sess, checkpoint)

    def generate_characters(self,
                            vocab,
                            num_chars=100,
                            top_chars=5,
                            checkpoint='saves/DefaultModel.tf'):
        """Uses a trained model to generate sequences of chars.

        Args:
            vocab: An object which has following attributes: `vocab_size`
                (number of items in vocabulary), `vocab_to_idx` (mapping from
                items to their integer codes), `idx_to_vocab` (mapping from
                integer codes to items).
            num_chars: Length of the generated sequence.
            checkpoint: Path used to save the trained model.


        """
        vocab_size = vocab.vocab_size
        vocab_to_idx = vocab.vocab_to_idx
        idx_to_vocab = vocab.idx_to_vocab
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self._saver.restore(sess, checkpoint)
            state = None
            current_char = random.choice(vocab_to_idx.values())
            chars = [current_char]
            for _ in range(num_chars):
                if state is not None:
                    feed_dict = {
                        self._x: [[current_char]], self._init_state: state}
                else:
                    feed_dict = {self._x: [[current_char]]}

                preds, state = sess.run(
                    [self._preds, self._final_state], feed_dict)
                p = np.squeeze(preds)
                p[np.argsort(p)[:-top_chars]] = 0
                p = p / np.sum(p)
                current_char = np.random.choice(vocab_size, 1, p=p)[0]
                chars.append(current_char)
        chars = [idx_to_vocab[x] for x in chars]
        print "".join(chars)
        return "".join(chars)

    def _reset_graph(self):
        if self.sess:
            self.sess.close()
        tf.reset_default_graph()


if __name__ == "__main__":
    pass
