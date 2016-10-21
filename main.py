# -*- coding: utf-8 -*-

import sys
import logging

import tensorflow as tf

from src.inputs.sequence_input import TrainSequenceInput, TestSequenceInput
from src.models.multilayer_gru import MultilayerGRU

logging.basicConfig(stream=sys.stdout, format="", level=logging.INFO)

truncate_at = 10
batch_size = 50

num_hidden = 7
num_layers = 2


def main():
    train_data = TrainSequenceInput(
        sequence_len=10000000, num_epochs=50,
        batch_size=batch_size, truncate_at=truncate_at)
    test_data = TestSequenceInput(
        sequence_len=1000, truncate_at=truncate_at)
    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("model", reuse=None):
            model = MultilayerGRU(
                num_hidden_neurons=num_hidden, num_gru_layers=num_layers)
            model.fit(sess, train_data)
        with tf.variable_scope("model", reuse=True):
            eval_model = MultilayerGRU(
                num_hidden_neurons=num_hidden, num_gru_layers=num_layers)
            eval_model.predict(sess, test_data)


if __name__ == "__main__":
    main()
