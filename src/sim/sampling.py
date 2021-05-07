import numpy as np


def normalize_weights(weights, log=False):
    if not log:
        return weights / weights.sum()
    else:
        weights -= weights.max()
        return np.exp(weights) / np.exp(weights).sum()


def importance_sample(particles, weights):
    return np.random.choice(particles, len(particles), p=weights)
