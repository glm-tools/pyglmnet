"""
A few miscellaneous helper functions for pyglmnet.py
"""

import numpy as np
from copy import copy


def softmax(w):
    """Softmax function of given array of number w

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


def log_likelihood(y, yhat, distr):
    """Helper to compute the log likelihood."""
    eps = np.spacing(1)
    if distr in ['softplus', 'poisson']:
        return np.sum(y * np.log(eps + yhat) - yhat)
    elif distr == 'binomial':
        # Log likelihood of model under consideration
        return 2 * len(y) * \
            np.sum(y * np.log((yhat == 0) + yhat) / np.mean(yhat) +
                   (1 - y) * np.log((yhat == 1) +
                                    1 - yhat) / (1 - np.mean(yhat)))
    elif distr == 'gaussian':
        return np.sum((y - yhat)**2)
    elif distr == 'multinomial':
        y = label_binarizer(y)
        # yhat is the probability of each output
        if yhat.ndim != y.ndim:
            msg = 'yhat and ynull must be (n_samples, n_class) ndarrays'
            raise ValueError(msg)
        return np.sum(y * np.log(yhat))


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


def as_float_array(X, copy=True):
    """Converts an array-like to an array of floats

    The new dtype will be np.float32 or np.float64, depending on the original
    type. The function can create a copy or modify the argument depending
    on the argument copy.

    Parameters
    ----------
    X : {array-like, sparse matrix}

    copy : bool, optional
        If True, a copy of X will be created. If False, a copy may still be
        returned if X's dtype is not a floating point type.

    Returns
    -------
    XT : {array, sparse matrix}
        An array of type np.float
    """
    if isinstance(X, list):
        return np.array(X)
    if X.dtype in [np.float32, np.float64]:  # is numpy array
        return X.copy('F' if X.flags['F_CONTIGUOUS'] else 'C') if copy else X
    else:
        return X.astype(np.float32 if X.dtype == np.int32 else np.float64)


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)
