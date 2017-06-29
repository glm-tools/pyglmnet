""" A set of scoring functions. """

import numpy as np
from .pyglmnet import _logL


def deviance(y, yhat, ynull_, distr):
    """Deviance metrics."""
    if distr in ['softplus', 'poisson']:
        LS = _logL(distr, y, y)
    else:
        LS = 0

    L1 = _logL(distr, y, yhat)
    score = -2 * (L1 - LS)
    return score


def pseudo_R2(X, y, yhat, ynull_, distr):
    """Pseudo r2."""
    if distr in ['softplus', 'poisson']:
        LS = _logL(distr, y, y)
    else:
        LS = 0

    L0 = _logL(distr, y, ynull_)
    L1 = _logL(distr, y, yhat)

    if distr in ['softplus', 'poisson']:
        score = (1 - (LS - L1) / (LS - L0))
    else:
        score = (1 - L1 / L0)
    return score


def accuracy(y, yhat):
    """Accuracy."""
    return float(np.sum(y == yhat)) / yhat.shape[0]
