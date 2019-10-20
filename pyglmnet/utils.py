"""
A few miscellaneous helper functions for pyglmnet.py
"""

import numpy as np
from copy import copy
import logging


logger = logging.getLogger('pyglmnet')
logger.addHandler(logging.StreamHandler())


def softmax(w):
    """Softmax function of given array of number w.

    Parameters
    ----------
    w: array | list
        The array of numbers.

    Returns
    -------
    dist: array
        The resulting array with values ranging from 0 to 1.
    """
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist


def label_binarizer(y):
    """Mimics scikit learn's LabelBinarizer


    Parameters
    ----------
    y: ndarray, shape (n_samples, )
        one dimensional array of class labels

    Returns
    -------
    yb: array, shape (n_samples, n_classes)
        one-hot encoding of labels in y
    """
    if y.ndim != 1:
        raise ValueError('y has to be one-dimensional')
    y_flat = y.ravel()
    yb = np.zeros([len(y), y.max() + 1])
    yb[np.arange(len(y)), y_flat] = 1
    return yb


def tikhonov_from_prior(prior_cov, n_samples, threshold=0.0001):
    """Given a prior covariance matrix, returns a Tikhonov matrix

    Parameters
    ----------
    prior_cov: array \n
        prior covariance matrix of shape (n_features x n_features)
    n_samples: int \n
        number of samples
    threshold: float \n
        ratio of largest to smallest singular value to
        approximate matrix inversion using SVD

    Returns
    -------
    Tau: array \n
        Tikhonov matrix of shape (n_features x n_features)
    """

    [U, S, V] = np.linalg.svd(prior_cov, full_matrices=False)

    S_ratio = S / S.max()

    nonzero_indices = np.where(S_ratio > threshold)[0]
    zero_indices = np.where(S_ratio <= threshold)[0]

    S_inv = copy(np.sqrt(S))
    S_inv[zero_indices] = threshold
    S_inv[nonzero_indices] = 1. / S_inv[nonzero_indices]

    Tau = np.dot(np.diag(S_inv), V)
    n_features = Tau.shape[0]
    Tau = 1. / n_features * Tau
    return Tau


def _check_params(distr, max_iter, fit_intercept):
    from .pyglmnet import ALLOWED_DISTRS

    if distr not in ALLOWED_DISTRS:
        raise ValueError('distr must be one of %s, Got '
                         '%s' % (', '.join(ALLOWED_DISTRS), distr))

    if not isinstance(max_iter, int):
        raise ValueError('max_iter must be of type int')

    if not isinstance(fit_intercept, bool):
        raise ValueError('fit_intercept must be bool, got %s'
                         % type(fit_intercept))


def set_log_level(verbose):
    """Convenience function for setting the log level.

    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
    """
    if isinstance(verbose, bool):
        if verbose is True:
            verbose = 'INFO'
        else:
            verbose = 'WARNING'
    if isinstance(verbose, str):
        verbose = verbose.upper()
        logging_types = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                             WARNING=logging.WARNING, ERROR=logging.ERROR,
                             CRITICAL=logging.CRITICAL)
        if verbose not in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]
    logger.setLevel(verbose)
