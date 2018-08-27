""" A set of scoring functions. """

import numpy as np
from .pyglmnet import _logL


def deviance(y, yhat, distr):
    """Deviance metrics.

    Parameters
    ----------
    y : array
        Target labels of shape (n_samples, )

    yhat : array
        Predicted labels of shape (n_samples, )

    distr: str
        distribution

    Returns
    -------
    score : float
        Deviance of the predicted labels.
    """
    if distr in ['softplus', 'poisson']:
        LS = _logL(distr, y, y)
    else:
        LS = 0

    L1 = _logL(distr, y, yhat)
    score = -2 * (L1 - LS)
    return score


def pseudo_R2(X, y, yhat, ynull_, distr):
    """Pseudo-R2 metric.

    Parameters
    ----------
    y : array
        Target labels of shape (n_samples, )

    yhat : array
        Predicted labels of shape (n_samples, )

    ynull_ : float
        Mean of the target labels (null model prediction)

    distr: str
        distribution

    Returns
    -------
    score : float
        Pseudo-R2 score.
    """
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
    """Accuracy as ratio of correct predictions.

    Parameters
    ----------
    y : array
        Target labels of shape (n_samples, )

    yhat : array
        Predicted labels of shape (n_samples, )

    Returns
    -------
    accuracy : float
        Accuracy score.
    """
    return float(np.sum(y == yhat)) / yhat.shape[0]
