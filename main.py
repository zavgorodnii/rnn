# -*- coding: utf-8 -*-

import sys
import logging

from python.inputs.simple_sequence_input import SimpleSequenceInput
from python.models.multilayer_gru import MultilayerGRU

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    training_data = SimpleSequenceInput(sequence_len=10000000)
    MultilayerGRU(num_hidden_neurons=10).train(training_data)


if __name__ == "__main__":
    main()
