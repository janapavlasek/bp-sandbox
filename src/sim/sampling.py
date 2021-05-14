import numpy as np


def normalize_weights(weights, log=False):
    if not log:
        return weights / weights.sum()
    else:
        weights -= weights.max()
        return np.exp(weights) / np.exp(weights).sum()


def importance_sample(weights, keep_best=False):
    N = len(weights)
    pick = N if not keep_best else N - 1
    sample_idx = np.random.choice(np.arange(N), pick, p=weights)
    if keep_best:
        sample_idx = np.concatenate([[weights.argmax()], sample_idx])
    return sample_idx
