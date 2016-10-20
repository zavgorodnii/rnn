# -*- coding: utf-8 -*-

import numpy as np


class SimpleSequenceInput(object):

    """
    (Iterable) Represents X (input) and Y (expected output) data for a RNN.

    Due to tensorflow specifics we must define our RNN in unfolded form
    with some fixed depth. This depth defines the length, :truncate_at, of
    (x, y) truncated sequences that will be fed to the network.

    So if both X, Y have length 99 and :truncate_at is 3, we must have 33
    (x, y) truncated sequences.

    For training purposes we further organize those truncated sequences into
    batches of size :batch_size (see self.gen_batch). In the example above, if
    :batch_size is 11, we will have 3 batches consisting of 11 truncated
    sequences.

    See self.get_X_Y for data characteristics.
    """

    num_classes = 2  # Number of types of sequence elements (0 and 1)

    def __init__(self,
                 num_epochs=1,
                 batch_size=200,
                 truncate_at=5,
                 sequence_len=1000000):
        """Initializes sequence data generator.

        Args:
            num_epochs: Number of training epochs.
            batch_size: Number of x,y-paired truncated sequences that go into a
                batch.
            truncate_at: Length of each training truncated sequence, same as
                unfolded RNN depth, same as number of truncated backprop steps.
            sequence_len: Number of elements in X and Y.
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.truncate_at = truncate_at
        self.data = self.gen_epochs(
            num_epochs, sequence_len, batch_size, truncate_at)

    def __iter__(self):
        return self

    def next(self):
        return self.data.next()

    def gen_epochs(self, num_epochs, sequence_len, batch_size, truncate_at):
        """Generates :num_epochs gen_batch() generators.

        Args:
            num_epochs: number of epochs
            truncate_at: Length of each training truncated sequence, same as
                unfolded RNN depth, same as number of truncated backprop steps.

        Yields:
            Another generator, which is gen_batch().
        """
        for _ in range(num_epochs):
            raw_X_Y = self.get_X_Y(sequence_len=sequence_len)
            yield self.gen_batch(raw_X_Y, batch_size, truncate_at)

    def gen_batch(self, raw_data, batch_size, truncate_at):
        """Generates (x, y) batches for training.

        Args:
            raw_data: Two sequences of floats: Input sequence (X) and Output
                sequence (Y). X and Y should have the same length. get_X_Y()
                generates a possible input.
            batch_size: Number of x,y-paired truncated sequences that go into a
                batch.
            truncate_at: Length of each training truncated sequence, same as
                unfolded RNN depth, same as number of truncated backprop steps.

        Yields:
            A batch of training data (a tuple of 2D numpy arrays).
            If X is [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            Y is [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
            batch_size=4 and truncate_at=2, the first yielded batch will be
            ([[ 1,  2],
              [ 4,  5],
              [ 7,  8],
              [10, 11]],

             [[ 10,  20],
              [ 40,  50],
              [ 70,  80],
              [100, 110]])
        """
        raw_x, raw_y = raw_data
        data_length = len(raw_x)
        # partition raw data into batches and stack them vertically in a data
        # matrix
        num_batches = data_length / batch_size
        data_x = np.zeros([batch_size, num_batches], dtype=np.int32)
        data_y = np.zeros([batch_size, num_batches], dtype=np.int32)
        # If raw_x is [0 1 2 3 4 5 6 7 8 9 10 11], data_x will be
        #   [[ 0  1  2]
        #    [ 3  4  5]
        #    [ 6  7  8]
        #    [ 9 10 11]]
        for i in range(batch_size):
            data_x[i] = raw_x[num_batches * i:num_batches * (i + 1)]
            data_y[i] = raw_y[num_batches * i:num_batches * (i + 1)]
        # Further divide batch partitions into truncate_at for truncated
        # backprop. Note that advanced numpy.ndarray indexing is used here.
        # If data_x is the  same as in comments above, data_x[0:, 0:2] will be
        #   [[ 0  1]
        #    [ 3  4]
        #    [ 6  7]
        #    [ 9 10]]
        epoch_size = num_batches / truncate_at
        for i in range(epoch_size):
            x = data_x[:, i * truncate_at:(i + 1) * truncate_at]
            y = data_y[:, i * truncate_at:(i + 1) * truncate_at]
            yield (x, y)

    def get_X_Y(self, sequence_len=1000000):
        """Returns X (input) and Y (expected output) for training the network.

        Args:
            sequence_len: Number of elements in X and Y.

        Returns:
            Input sequence (X): At time step t, X_t has a 50% chance of being 1
                (and a 50% chance of being 0). E.g., X might be
                [0, 0, 1, 0, ... ].
            Output sequence (Y): At time step t, Y_t has a base 50% chance of
                being 1 (and a 50% base chance to be 0). The chance of Y_t
                being 1 is increased by 50% (i.e., to 100%) if X_t−3 is 1, and
                decreased by 25% (i.e., to 25%) if X_t−8 is 1. If both X_t−3
                and X_t−8 are 1, the chance of Y_t being 1 is
                50% + 50% - 25% = 75%.
        """
        X = np.array(np.random.choice([0, 1], size=(sequence_len,)))
        Y = []
        for i in range(sequence_len):
            threshold = 0.5
            if X[i - 3] == 1:
                threshold += 0.5
            if X[i - 8] == 1:
                threshold -= 0.25
            if np.random.rand() > threshold:
                Y.append(0)
            else:
                Y.append(1)
        return X, np.array(Y)
