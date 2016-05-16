from copy import deepcopy
import numpy as np
from scipy.special import expit

def softmax_(w):
    """
    Softmax function of given array of number w
    """
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist


def label_binarizer_(y):
    """Mimics scikit learn's LabelBinarizer
    Parameters
    ---------
    y: ndarray (n_samples)
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

def qu_(distr, z, eta):
    """The non-linearity."""
    if distr == 'poisson':
        qu = np.log1p(np.exp(z))
    elif distr == 'poissonexp':
        qu = deepcopy(z)
        slope = np.exp(eta)
        intercept = (1 - eta) * slope
        qu[z > eta] = z[z > eta] * slope + intercept
        qu[z <= eta] = np.exp(z[z <= eta])
    elif distr == 'normal':
        qu = z
    elif distr == 'binomial':
        qu = expit(z)
    elif distr == 'multinomial':
        qu = softmax_(z)

    return qu

def lmb_(distr, beta0, beta, X, eta):
    """Conditional intensity function."""
    z = beta0 + np.dot(X, beta)
    l = qu_(distr, z, eta)
    return l

def logL_(distr, beta0, beta, X, y, eta):
    """The log likelihood."""
    l = lmb_(distr, beta0, beta, X, eta)
    if distr == 'poisson':
        logL = np.sum(y * np.log(l) - l)
    elif distr == 'poissonexp':
        logL = np.sum(y * l - l)
    elif distr == 'normal':
        logL = -0.5 * np.sum((y - l)**2)
    elif distr == 'binomial':
        z = beta0 + np.dot(X, beta)
        logL = np.sum(y * z - np.log(1 + np.exp(z)))
    elif distr == 'multinomial':
        logL = np.sum(y * np.log(l))
    return logL

def L2loss_(beta_concat, distr, alpha, reg_lambda, X, y, eta):
    """Quadratic loss."""
    beta0, beta = beta_concat[0], np.expand_dims(beta_concat[1:], axis=1)
    L = logL_(distr, beta0, beta, X, y, eta)
    P = 0.5 * (1 - alpha) * np.linalg.norm(beta, 2)
    J = -L + reg_lambda * P
    return J

def grad_L2loss_(beta_concat, distr, alpha, reg_lambda, X, y, eta):
    """The gradient."""
    beta0, beta = beta_concat[0], np.expand_dims(beta_concat[1:], axis=1)
    z = beta0 + np.dot(X, beta)
    s = expit(z)

    if distr == 'poisson':
        q = qu_(distr, z, eta)
        grad_beta0 = np.sum(s) - np.sum(y * s / q)
        grad_beta = np.transpose(np.dot(np.transpose(s), X) -
                                 np.dot(np.transpose(y * s / q), X)) + \
            reg_lambda * (1 - alpha) * beta
        # + reg_lambda*alpha*np.sign(beta)

    elif distr == 'poissonexp':
        q = qu_(distr, z, eta)

        grad_beta0 = np.sum(q[z <= eta] - y[z <= eta]) + \
            np.sum(1 - y[z > eta] / q[z > eta]) * eta

        grad_beta = np.zeros([X.shape[1], 1])
        selector = np.where(z.ravel() <= eta)[0]
        grad_beta += np.transpose(np.dot((q[selector] - y[selector]).T,
                                         X[selector, :]))
        selector = np.where(z.ravel() > eta)[0]
        grad_beta += eta * \
            np.transpose(np.dot((1 - y[selector] / q[selector]).T,
                                X[selector, :]))
        grad_beta += reg_lambda * (1 - alpha) * beta

    elif distr == 'normal':
        grad_beta0 = -np.sum(y - z)
        grad_beta = -np.transpose(np.dot(np.transpose(y - z), X)) \
            + reg_lambda * (1 - alpha) * beta

    elif distr == 'binomial':
        grad_beta0 = np.sum(s - y)
        grad_beta = np.transpose(np.dot(np.transpose(s - y), X)) \
            + reg_lambda * (1 - alpha) * beta

    elif distr == 'multinomial':
        # this assumes that y is already as a one-hot encoding
        pred = qu_(distr, z, eta)
        grad_beta0 = -np.sum(y - pred, axis=0)
        grad_beta = -np.transpose(np.dot(np.transpose(y - pred), X)) \
            + reg_lambda * (1 - alpha) * beta

    return np.concatenate([[grad_beta0], grad_beta.ravel()])

def grad_hess_(beta_concat, distr, alpha, reg_lambda, X, y, eta):
    """
    A callable function passed to optimize.newton_cg
    Returns gradient and a callable for hessian vector product
    """
    g = grad_L2loss_(beta_concat, distr, alpha, reg_lambda, X, y, eta)

    def hess_product(v):
        """The hessian matrix vector product"""
        g = grad_L2loss_(beta_concat, distr, alpha, reg_lambda, X, y, eta)
        r = 1e-3*np.min(g)
        gplus = grad_L2loss_(beta_concat + r * v, distr, alpha, reg_lambda, X, y, eta)
        gminus = grad_L2loss_(beta_concat - r * v, distr, alpha, reg_lambda, X, y, eta)
        Hv = (gplus - gminus)/(2.0 * r)
        return Hv

    return g, hess_product
