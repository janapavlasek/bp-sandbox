import numpy as np


def normalize_weights(weights, log=False):
    if not log:
        return weights / weights.sum()
    else:
        weights -= weights.max()
        return np.exp(weights) / np.exp(weights).sum()


def importance_sample(weights):
    N = len(weights)
    return np.random.choice(np.arange(N), N, p=weights)
