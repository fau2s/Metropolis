import numpy as np


def noiseEstiomation(self, xoptbuy, Pid):
    return np.std(xoptbuy.dot(Pid))
