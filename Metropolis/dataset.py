# data/dataset.py
import numpy as np

def create_teacher_forcing_inputs(y):

    decoder_input = np.zeros_like(y)
    decoder_input[:, 1:, :] = y[:, :-1, :]
    return decoder_input