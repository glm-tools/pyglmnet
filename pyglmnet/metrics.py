""" A set of scoring functions. """
from . import utils
import numpy as np


def deviance(y, yhat, ynull_, distr):
    """Deviance metrics."""
    if distr in ['softplus', 'poisson']:
        LS = utils.log_likelihood(y, y, distr)
    else:
        LS = 0

    L1 = utils.log_likelihood(y, yhat, distr)
    score = -2 * (L1 - LS)
    return score


def pseudo_R2(X, y, yhat, ynull_, distr):
    """Pseudo r2."""
    if distr in ['softplus', 'poisson']:
        LS = utils.log_likelihood(y, y, distr)
    else:
        LS = 0

    L0 = utils.log_likelihood(y, ynull_, distr)
    L1 = utils.log_likelihood(y, yhat, distr)

    if distr in ['softplus', 'poisson']:
        score = (1 - (LS - L1) / (LS - L0))
    else:
        score = (1 - L1 / L0)
    return score


def accuracy(y, yhat):
    """Accuracy."""
    return float(np.sum(y == yhat)) / yhat.shape[0]
