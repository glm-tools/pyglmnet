"""Python implementation of elastic-net regularized GLMs."""

import logging
from copy import deepcopy

import numpy as np
from scipy.special import expit
from . import utils

logger = logging.getLogger('pyglmnet')
logger.addHandler(logging.StreamHandler())


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


def _lmb(distr, beta0, beta, X, eta):
    """Conditional intensity function."""
    z = beta0 + np.dot(X, beta)
    return _qu(distr, z, eta)


def _qu(distr, z, eta):
    """The non-linearity."""
    if distr == 'softplus':
        qu = np.log1p(np.exp(z))
    elif distr == 'poisson':
        qu = deepcopy(z)
        slope = np.exp(eta)
        intercept = (1 - eta) * slope
        qu[z > eta] = z[z > eta] * slope + intercept
        qu[z <= eta] = np.exp(z[z <= eta])
    elif distr == 'gaussian':
        qu = z
    elif distr == 'binomial':
        qu = expit(z)
    elif distr == 'multinomial':
        qu = utils.softmax(z)
    return qu


def _logL(distr, beta0, beta, X, y, eta):
    """The log likelihood."""
    n_samples = np.float(X.shape[0])
    z = beta0 + np.dot(X, beta)
    l = _qu(distr, z, eta)
    if distr == 'softplus':
        logL = 1. / n_samples * np.sum(y * np.log(l) - l)
    elif distr == 'poisson':
        logL = 1. / n_samples * np.sum(y * np.log(l) - l)
    elif distr == 'gaussian':
        logL = -0.5 * 1. / n_samples * np.sum((y - l)**2)
    elif distr == 'binomial':
        # analytical formula
        # logL = np.sum(y*np.log(l) + (1-y)*np.log(1-l))

        # but this prevents underflow
        z = beta0 + np.dot(X, beta)
        logL = 1. / n_samples * np.sum(y * z - np.log(1 + np.exp(z)))
    elif distr == 'multinomial':
        logL = 1. / n_samples * np.sum(y * np.log(l))
    return logL


def _penalty(alpha, beta, Tau, group):
    """The penalty."""
    # Combine L1 and L2 penalty terms
    P = 0.5 * (1 - alpha) * _L2penalty(beta, Tau) + \
        alpha * _L1penalty(beta, group)
    return P


def _L2penalty(beta, Tau):
    """The L2 penalty"""
    # Compute the L2 penalty
    if Tau is None:
        # Ridge=like penalty
        L2penalty = np.linalg.norm(beta, 2) ** 2
    else:
        # Tikhonov penalty
        if (Tau.shape[0] != beta.shape[0] or
           Tau.shape[1] != beta.shape[0]):
            raise ValueError('Tau should be (n_features x n_features)')
        else:
            L2penalty = np.linalg.norm(np.dot(Tau, beta), 2) ** 2
    return L2penalty


def _L1penalty(beta, group=None):
    """The L1 penalty"""
    # Compute the L1 penalty
    if group is None:
        # Lasso-like penalty
        L1penalty = np.linalg.norm(beta, 1)
    else:
        # Group sparsity case: apply group sparsity operator
        group_ids = np.unique(group)
        L1penalty = 0.0
        for group_id in group_ids:
            if group_id != 0:
                L1penalty += \
                    np.linalg.norm(beta[group == group_id], 2)
        L1penalty += np.linalg.norm(beta[group == 0], 1)
    return L1penalty


def _loss(distr, alpha, beta0, beta, Tau, reg_lambda, X, y, eta, group):
    """Define the objective function for elastic net."""
    L = _logL(distr, beta0, beta, X, y, eta)
    P = _penalty(alpha, beta, Tau, group)
    J = -L + reg_lambda * P
    return J


def _grad_L2loss(distr, alpha, beta0, beta, reg_lambda, X, y, Tau, eta):
    """The gradient."""

    n_samples = np.float(X.shape[0])

    if Tau is None:
        Tau = np.eye(beta.shape[0])
    InvCov = np.dot(Tau.T, Tau)

    z = beta0 + np.dot(X, beta)
    s = expit(z)

    if distr == 'softplus':
        q = _qu(distr, z, eta)
        grad_beta0 = 1. / n_samples * (np.sum(s) - np.sum(y * s / q))
        grad_beta = 1. / n_samples * \
            (np.transpose(np.dot(np.transpose(s), X) -
                          np.dot(np.transpose(y * s / q), X))) + \
            reg_lambda * (1 - alpha) * \
            np.dot(InvCov, beta)

    elif distr == 'poisson':
        q = _qu(distr, z, eta)
        grad_beta0 = np.sum(q[z <= eta] - y[z <= eta]) + \
            np.sum(1 - y[z > eta] / q[z > eta]) * \
            np.exp(eta)
        grad_beta0 *= 1. / n_samples

        grad_beta = np.zeros([X.shape[1], 1])
        selector = np.where(z.ravel() <= eta)[0]
        grad_beta += np.transpose(np.dot((q[selector] - y[selector]).T,
                                         X[selector, :]))
        selector = np.where(z.ravel() > eta)[0]
        grad_beta += np.exp(eta) * \
            np.transpose(np.dot((1 - y[selector] / q[selector]).T,
                                X[selector, :]))
        grad_beta *= 1. / n_samples
        grad_beta += reg_lambda * (1 - alpha) * \
            np.dot(InvCov, beta)

    elif distr == 'gaussian':
        grad_beta0 = 1. / n_samples * np.sum(z - y)
        grad_beta = 1. / n_samples * \
            np.transpose(np.dot(np.transpose(z - y), X)) \
            + reg_lambda * (1 - alpha) * \
            np.dot(InvCov, beta)

    elif distr == 'binomial':
        grad_beta0 = 1. / n_samples * np.sum(s - y)
        grad_beta = 1. / n_samples * \
            np.transpose(np.dot(np.transpose(s - y), X)) \
            + reg_lambda * (1 - alpha) * \
            np.dot(InvCov, beta)

    elif distr == 'multinomial':
        # this assumes that y is already as a one-hot encoding
        pred = _qu(distr, z, eta)
        grad_beta0 = -1. / n_samples * np.sum(y - pred, axis=0)
        grad_beta = -1. / n_samples * \
            np.transpose(np.dot(np.transpose(y - pred), X)) \
            + reg_lambda * (1 - alpha) * \
            np.dot(InvCov, beta)

    return np.vstack((np.array([grad_beta0])[None, :], grad_beta))


class GLM(object):
    """Class for estimating regularized generalized linear models (GLM).
    The regularized GLM minimizes the penalized negative log likelihood:

    .. math::

        \min_{\\beta_0, \\beta} \\frac{1}{N}
        \sum_{i = 1}^N \mathcal{L} (y_i, \\beta_0 + \\beta^T x_i)
        + \lambda [ \\frac{1}{2}(1 - \\alpha) \mathcal{P}_2 +
                    \\alpha \mathcal{P}_1 ]

    where :math:`\mathcal{P}_2` and :math:`\mathcal{P}_1` are the generalized
    L2 (Tikhonov) and generalized L1 (Group Lasso) penalties, given by:

    .. math::

        \mathcal{P}_2 = \|\Gamma \\beta \|_2^2 \\
        \mathcal{P}_1 = \sum_g \|\\beta_{j,g}\|_2

    where :math:`\Gamma` is the Tikhonov matrix: a square factorization
    of the inverse covariance matrix and :math:`\\beta_{j,g}` is the
    :math:`j` th coefficient of group :math:`g`.

    The generalized L2 penalty defaults to the ridge penalty when
    :math:`\Gamma` is identity.

    The generalized L1 penalty defaults to the lasso penalty when each
    :math:`\\beta` belongs to its own group.

    Parameters
    ----------
    distr : str \n
        distribution family can be one of the following
        'gaussian' | 'binomial' | 'poisson' | 'softplus' | 'multinomial' \n
        default: 'poisson'.
    alpha : float \n
        the weighting between L1 penalty and L2 penalty term
        of the loss function. \n
        default: 0.5
    Tau : array | None \n
        the (n_features, n_features) Tikhonov matrix. \n
        default: None, in which case Tau is identity
        and the L2 penalty is ridge-like
    group : array | list | None \n
        the (n_features, )
        list or array of group identities for each parameter :math:`\\beta`. \n
        Each entry of the list/ array should contain an int from 1 to n_groups
        that specify group membership for each parameter
        (except :math:`\\beta_0`). \n
        If you do not want to specify a group for a specific parameter,
        set it to zero. \n
        default: None, in which case it defaults to L1 regularization
    reg_lambda : array | list | None \n
        array of regularized parameters :math:`\\lambda` of penalty term. \n
        default: None, a list of 10 floats spaced logarithmically (base e)
        between 0.5 and 0.01. \n
    solver : str \n
        optimization method, can be one of the following
        'batch-gradient' (vanilla batch gradient descent)
        'cdfast' (Newton coordinate gradient descent). \n
        default: 'batch-gradient'
    learning_rate : float \n
        learning rate for gradient descent. \n
        default: 2e-1
    max_iter : int \n
        maximum iterations for the model. \n
        default: 1000
    tol : float \n
        convergence threshold or stopping criteria.
        Optimization loop will stop below setting threshold. \n
        default: 1e-3
    eta : float \n
        a threshold parameter that linearizes the exp() function above eta. \n
        default: 2.0
    score_metric : str \n
        specifies the scoring metric.
        one of either 'deviance' or 'pseudo_R2'. \n
        default: 'deviance'
    random_state : int \n
        seed of the random number generator used to initialize the solution. \n
        default: 0
    verbose : boolean or int \n
        default: False

    Reference
    ---------
    Friedman, Hastie, Tibshirani (2010). Regularization Paths for Generalized
        Linear Models via Coordinate Descent, J Statistical Software.
        https://core.ac.uk/download/files/153/6287975.pdf

    Notes
    -----
    To select subset of fitted glm models, you can simply do:

    >>> glm = glm[1:3]
    >>> glm[2].predict(X_test)
    """

    def __init__(self, distr='poisson', alpha=0.5,
                 Tau=None, group=None,
                 reg_lambda=None,
                 solver='batch-gradient',
                 learning_rate=2e-1, max_iter=1000,
                 tol=1e-3, eta=2.0, score_metric='deviance',
                 random_state=0, verbose=False):

        if reg_lambda is None:
            reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10,
                                     base=np.exp(1))
        if not isinstance(reg_lambda, (list, np.ndarray)):
            reg_lambda = [reg_lambda]
        if not isinstance(max_iter, int):
            max_iter = int(max_iter)

        self.distr = distr
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.Tau = Tau
        self.group = group
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_ = None
        self.ynull_ = None
        self.tol = tol
        self.eta = eta
        self.score_metric = score_metric
        self.random_state = random_state
        self.verbose = verbose
        set_log_level(verbose)

    def get_params(self, deep=False):
        return dict(
            (
                ('distr', self.distr),
                ('alpha', self.alpha),
                ('Tau', self.Tau),
                ('group', self.group),
                ('reg_lambda', self.reg_lambda),
                ('learning_rate', self.learning_rate),
                ('max_iter', self.max_iter),
                ('tol', self.tol),
                ('eta', self.eta),
                ('score_metric', self.score_metric),
                ('random_state', self.random_state),
                ('verbose', self.verbose)
            )
        )

    def __repr__(self):
        """Description of the object."""
        reg_lambda = self.reg_lambda
        s = '<\nDistribution | %s' % self.distr
        s += '\nalpha | %0.2f' % self.alpha
        s += '\nmax_iter | %0.2f' % self.max_iter
        if len(reg_lambda) > 1:
            s += ('\nlambda: %0.2f to %0.2f\n>'
                  % (reg_lambda[0], reg_lambda[-1]))
        else:
            s += '\nlambda: %0.2f\n>' % reg_lambda[0]
        return s

    def __getitem__(self, key):
        """Return a GLM object with a subset of fitted lambdas."""
        glm = deepcopy(self)
        if self.fit_ is None:
            raise ValueError('Cannot slice object if the lambdas have'
                             ' not been fit.')
        if not isinstance(key, (slice, int)):
            raise IndexError('Invalid slice for GLM object')
        glm.fit_ = glm.fit_[key]
        glm.reg_lambda = glm.reg_lambda[key]
        return glm

    def copy(self):
        """Return a copy of the object.

        Parameters
        ----------
        none:

        Returns
        -------
        self: instance of GLM \n
            A copy of the GLM instance.
        """
        return deepcopy(self)

    def _prox(self, beta, thresh):
        """Proximal operator."""
        if self.group is None:
            # The default case: soft thresholding
            return np.sign(beta) * (np.abs(beta) - thresh) * \
                (np.abs(beta) > thresh)
        else:
            # Group sparsity case: apply group sparsity operator
            group_ids = np.unique(self.group)
            group_norms = np.abs(beta)

            for group_id in group_ids:
                if group_id != 0:
                    group_norms[self.group == group_id] = \
                        np.linalg.norm(beta[self.group == group_id], 2)

            nzero_norms = group_norms > 0.0
            over_thresh = group_norms > thresh
            idxs_to_update = nzero_norms & over_thresh

            result = beta
            result[idxs_to_update] = (beta[idxs_to_update] -
                                      thresh * beta[idxs_to_update] /
                                      group_norms[idxs_to_update])

            return result

    def _gradhess_logloss_1d(self, xk, y, z):
        """
        Computes gradient (1st derivative)
        and Hessian (2nd derivative)
        of log likelihood for a single coordinate

        Parameters
        ----------
        xk: float
            n_samples x 1
        y: float
            n_samples x n_classes
        z: float
            n_samples x n_classes

        Returns
        -------
        gk: float:
            n_classes
        hk: float:
            n_classes
        """
        n_samples = np.float(xk.shape[0])

        if self.distr == 'softplus':
            mu = _qu(self.distr, z, self.eta)
            s = expit(z)
            gk = np.sum(s * xk) - np.sum(y * s / mu * xk)

            grad_s = s * (1 - s)
            grad_s_by_mu = grad_s / mu - s / (mu ** 2)
            hk = np.sum(grad_s * xk ** 2) - np.sum(y * grad_s_by_mu * xk ** 2)

        elif self.distr == 'poisson':
            mu = _qu(self.distr, z, self.eta)
            s = expit(z)
            gk = np.sum((mu[z <= self.eta] - y[z <= self.eta]) *
                        xk[z <= self.eta]) + \
                np.exp(self.eta) * \
                np.sum((1 - y[z > self.eta] / mu[z > self.eta]) *
                       xk[z > self.eta])
            hk = np.sum(mu[z <= self.eta] * xk[z <= self.eta] ** 2) + \
                np.exp(self.eta) ** 2 * \
                np.sum(y[z > self.eta] / (mu[z > self.eta] ** 2) *
                       (xk[z > self.eta] ** 2))

        elif self.distr == 'gaussian':
            gk = np.sum((z - y) * xk)
            hk = np.sum(xk * xk)

        elif self.distr == 'binomial':
            mu = _qu(self.distr, z, self.eta)
            gk = np.sum((mu - y) * xk)
            hk = np.sum(mu * (1.0 - mu) * xk ** 2)

        elif self.distr == 'multinomial':
            mu = _qu(self.distr, z, self.eta)
            gk = np.ravel(np.dot(np.transpose(mu - y), xk))
            hk = np.ravel(np.dot(np.transpose(mu * (1.0 - mu)), (xk ** 2)))

        return 1. / n_samples * gk, 1. / n_samples * hk

    def _cdfast(self, X, y, z, ActiveSet, beta, rl):
        """
        Performs one cycle of Newton updates for all coordinates

        Parameters
        ----------
        X : array
            n_samples x n_features
            The input data
        y : array
            Labels to the data
            n_samples x 1
        z:  array
            n_samples x 1
            beta[0] + X * beta[1:]
        ActiveSet: array
            n_features + 1 x 1
            Active set storing which betas are non-zero
        beta: array
            n_features + 1 x 1
            Parameters to be updated
        rl: float
            Regularization lambda

        Returns
        -------
        beta: array
            (n_features + 1) x 1
            Updated parameters
        z: array
            beta[0] + X * beta[1:]
            (n_features + 1) x 1
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        reg_scale = rl * (1 - self.alpha)

        for k in range(0, np.int(n_features) + 1):
            # Only update parameters in active set
            if ActiveSet[k] != 0:
                if k > 0:
                    xk = np.expand_dims(X[:, k - 1], axis=1)
                else:
                    xk = np.ones((n_samples, 1))

                # Calculate grad and hess of log likelihood term
                gk, hk = self._gradhess_logloss_1d(xk, y, z)

                # Add grad and hess of regularization term
                if self.Tau is None:
                    gk_reg = beta[k]
                    hk_reg = 1.0
                else:
                    InvCov = np.dot(self.Tau.T, self.Tau)
                    gk_reg = np.sum(InvCov[k - 1, :] * beta[1:])
                    hk_reg = InvCov[k - 1, k - 1]
                gk += np.ravel([reg_scale * gk_reg if k > 0 else 0.0])
                hk += np.ravel([reg_scale * hk_reg if k > 0 else 0.0])

                # Update parameters, z
                update = 1. / hk * gk
                beta[k], z = beta[k] - update, z - update * xk
        return beta, z

    def fit(self, X, y):
        """The fit function.

        Parameters
        ----------
        X : array \n
            The input data of shape (n_samples, n_features)

        y : array \n
            The target data

        Returns
        -------
        self : instance of GLM \n
            The fitted model.
        """

        np.random.RandomState(self.random_state)

        # checks for group
        if self.group is not None:
            self.group = np.array(self.group)
            self.group.dtype = np.int64
            # shape check
            if self.group.shape[0] != X.shape[1]:
                raise ValueError('group should be (n_features,)')
            # int check
            if not np.all([isinstance(g, np.int64) for g in self.group]):
                raise ValueError('all entries of group should be integers')

        # type check for data matrix
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        n_features = np.float(X.shape[1])
        n_features = np.int64(n_features)

        if self.distr == 'multinomial':
            y = utils.label_binarizer(y)
        else:
            if y.ndim == 1:
                y = y[:, np.newaxis]

        n_classes = y.shape[1] if self.distr == 'multinomial' else 1
        n_classes = np.int64(n_classes)

        # Initialize parameters
        beta0_hat = 1 / (n_features + 1) * \
            np.random.normal(0.0, 1.0, n_classes)
        beta_hat = 1 / (n_features + 1) * \
            np.random.normal(0.0, 1.0, [n_features, n_classes])
        fit_params = list()

        logger.info('Looping through the regularization path')
        for l, rl in enumerate(self.reg_lambda):
            fit_params.append({'beta0': beta0_hat, 'beta': beta_hat})
            logger.info('Lambda: %6.4f' % rl)

            # Warm initialize parameters
            if l == 0:
                fit_params[-1]['beta0'] = beta0_hat
                fit_params[-1]['beta'] = beta_hat
            else:
                fit_params[-1]['beta0'] = fit_params[-2]['beta0']
                fit_params[-1]['beta'] = fit_params[-2]['beta']

            tol = self.tol
            alpha = self.alpha

            # Temporary parameters to update
            beta = np.zeros([n_features + 1, n_classes])
            beta[0] = fit_params[-1]['beta0']
            beta[1:] = fit_params[-1]['beta']

            if self.solver == 'batch-gradient':
                g = np.zeros([n_features + 1, n_classes])
            elif self.solver == 'cdfast':
                ActiveSet = np.ones(n_features + 1)     # init active set
                z = beta[0] + np.dot(X, beta[1:])       # cache z

            # Initialize loss accumulators
            L, DL = list(), list()
            for t in range(0, self.max_iter):
                if self.solver == 'batch-gradient':
                    grad = _grad_L2loss(self.distr,
                                        self.alpha,
                                        beta[0], beta[1:],
                                        rl, X, y, self.Tau, self.eta)

                    beta = beta - self.learning_rate * grad
                elif self.solver == 'cdfast':
                    beta, z = self._cdfast(X, y, z, ActiveSet, beta, rl)

                # Apply proximal operator
                beta[1:] = self._prox(beta[1:], rl * alpha)

                # Update active set
                if self.solver == 'cdfast':
                    ActiveSet[np.where(beta[1:] == 0)[0] + 1] = 0

                # Compute and save loss
                L.append(_loss(self.distr, self.alpha, beta[0], beta[1:], self.Tau, rl, X, y, self.eta, self.group))

                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if np.abs(DL[-1] / L[-1]) < tol:
                        msg = ('\tConverged. Loss function:'
                               ' {0:.2f}').format(L[-1])
                        logger.info(msg)
                        msg = ('\tdL/L: {0:.6f}\n'.format(DL[-1] / L[-1]))
                        logger.info(msg)
                        break

            fit_params[-1]['beta0'] = beta[0]
            fit_params[-1]['beta'] = beta[1:]

        # Update the estimated variables
        self.fit_ = fit_params
        self.ynull_ = np.mean(y) if self.distr != 'multinomial' \
            else np.mean(y, axis=0)

        # Return
        return self

    def predict(self, X):
        """Predict targets.

        Parameters
        ----------
        X : array \n
            Input data for prediction, of shape (n_samples, n_features)

        Returns
        -------
        yhat : array \n
            The predicted targets of shape ([n_lambda], n_samples) \n
            A 1D array if predicting on only
            one reg_lambda (compatible with scikit-learn API). \n
            Otherwise, returns a 2D array.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        if isinstance(self.fit_, list):
            yhat = list()
            for fit in self.fit_:
                yhat.append(_lmb(self.distr, fit['beta0'], fit['beta'], X, self.eta))
        else:
            yhat = _lmb(self.distr, self.fit_['beta0'], self.fit_['beta'], X, self.eta)
        yhat = np.asarray(yhat)
        yhat = yhat[..., 0] if self.distr != 'multinomial' else yhat

        # if multinomial get the argmax()
        if self.distr == 'multinomial':
            if isinstance(self.fit_, dict):
                yhat = yhat.argmax(axis=1)
            else:
                yhat = yhat.argmax(axis=2)

        return yhat

    def predict_proba(self, X):
        """Predict class probability for multinomial

        Parameters
        ----------
        X : array \n
            Input data for prediction, of shape (n_samples, n_features)

        Returns
        -------
        yhat : array \n
            The predicted targets of shape
            ([n_lambda], n_samples, n_classes). \n
            A 2D array if predicting on only
            one lambda (compatible with scikit-learn API). \n
            Otherwise, returns a 3D array.

        Raises
        ------
        Works only for the multinomial distribution.
        Raises error otherwise.

        """
        if self.distr != 'multinomial':
            raise ValueError('This is only applicable for \
                              the multinomial distribution.')

        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        if isinstance(self.fit_, list):
            yhat = list()
            for fit in self.fit_:
                yhat.append(_lmb(self.distr, fit['beta0'], fit['beta'], X, self.eta))
        else:
            yhat = _lmb(self.distr, self.fit_['beta0'], self.fit_['beta'], X, self.eta)
        yhat = np.asarray(yhat)

        return yhat

    def fit_predict(self, X, y):
        """Fit the model and predict on the same data.

        Parameters
        ----------
        X : array \n
            The input data to fit and predict,
            of shape (n_samples, n_features)


        Returns
        -------
        yhat : array \n
            The predicted targets of shape ([n_lambda], n_samples). \n
            A 1D array if predicting on only
            one lambda (compatible with scikit-learn API). \n
            Otherwise, returns a 2D array.
        """
        return self.fit(X, y).predict(X)

    def score(self, X, y):
        """Score the model.

        Parameters
        ----------
        X : array \n
            The input data whose prediction will be scored,
            of shape (n_samples, n_features).
        y : array \n
            The true targets against which to score the predicted targets,
            of shape (n_samples, [n_classes]).

        Returns
        -------
        score: array
            array when score is called by a list of estimators:
            :code:`glm.score()`\n
            singleton array when score is called by a sliced estimator:
            :code:`glm[0].score()`\n

            Note that if you want compatibility with sciki-learn's
            :code:`pipeline()`, :code:`cross_val_score()`,
            or :code:`GridSearchCV()` then you should
            only pass sliced estimators: \n

            .. code:: python

                from sklearn.grid_search import GridSearchCV
                from sklearn.cross_validation import cross_val_score
                grid = GridSearchCV(glm[0])
                grid = cross_val_score(glm[0], X, y, cv=10)
        """

        if self.score_metric not in ['deviance', 'pseudo_R2']:
            raise ValueError('score_metric has to be one of' +
                             ' deviance or pseudo_R2')

        # If the model has not been fit it cannot be scored
        if self.ynull_ is None:
            raise ValueError('Model must be fit before ' +
                             'prediction can be scored')

        y = y.ravel()

        yhat = self.predict(X) if self.distr != 'multinomial' \
            else self.predict_proba(X)

        score = list()
        # Check whether we have a list of estimators or a single estimator
        if isinstance(self.fit_, dict):
            yhat = yhat[np.newaxis, ...]

        if self.distr in ['softplus', 'poisson']:
            LS = utils.log_likelihood(y, y, self.distr)
        else:
            LS = 0
        if(self.score_metric == 'pseudo_R2'):
            if self.distr != 'multinomial':
                L0 = utils.log_likelihood(y, self.ynull_, self.distr)
            else:
                expand_ynull_ = np.tile(self.ynull_, (X.shape[0], 1))
                L0 = utils.log_likelihood(y, expand_ynull_, self.distr)

        # Compute array of scores for each model fit
        # (corresponding to a reg_lambda)
        for idx in range(yhat.shape[0]):
            if self.distr != 'multinomial':
                yhat_this = (yhat[idx, :]).ravel()
            else:
                yhat_this = yhat[idx, :, :]
            L1 = utils.log_likelihood(y, yhat_this, self.distr)

            if self.score_metric == 'deviance':
                score.append(-2 * (L1 - LS))
            elif self.score_metric == 'pseudo_R2':
                if self.distr in ['softplus', 'poisson']:
                    score.append(1 - (LS - L1) / (LS - L0))
                else:
                    score.append(1 - L1 / L0)

        return np.array(score)

    def simulate(self, beta0, beta, X):
        """Simulate target data under a generative model.

        Parameters
        ----------
        X: array
            design matrix of shape (n_samples, n_features)
        beta0: float
            intercept coefficient
        beta: array
            coefficients of shape (n_features, 1)

        Returns
        -------
        y: array
            simulated target data of shape (n_samples, 1)
        """
        np.random.RandomState(self.random_state)
        if self.distr == 'softplus' or self.distr == 'poisson':
            y = np.random.poisson(_lmb(self.distr, beta0, beta, X, self.eta))
        if self.distr == 'gaussian':
            y = np.random.normal(_lmb(self.distr, beta0, beta, X, self.eta))
        if self.distr == 'binomial':
            y = np.random.binomial(1, _lmb(self.distr, beta0, beta, X, self.eta))
        if self.distr == 'multinomial':
            y = np.array([np.random.multinomial(1, pvals)
                          for pvals in
                          _lmb(self.distr, beta0, beta, X, self.eta)]).argmax(0)
        return y
