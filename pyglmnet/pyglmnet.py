"""Python implementation of elastic-net regularized GLMs."""

import logging
from copy import deepcopy

import numpy as np
from scipy.special import expit

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


def softmax(w):
    """
    Softmax function of given array of number w
    """
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist


class GLM(object):
    """Generalized Linear Model (GLM)

    This class implements elastic-net regularized generalized linear models.
    The core algorithm is defined in the article.
        min_(beta0, beta) [-L + lamda * P]
    where
        L is log-likelihood term
        P is elastic-net penalty term

    Parameters
    ----------
    distr: str
        distribution family can be one of the following
        'poisson' or 'poissonexp' or 'normal' or 'binomial' or 'multinomial'
        default: 'poisson'
    alpha: float
        the weighting between L1 and L2 norm in the penalty term
        of the loss function i.e.
        P(beta) = 0.5 * (1-alpha) * |beta|_2^2 + alpha * |beta|_1
        default: 0.5
    reg_lambda: ndarray or list
        array of regularized parameters of penalty term i.e.
        min_(beta0, beta) -L + lambda * P
        where lambda is number in reg_lambda list
        default: None, a list of 10 floats spaced logarithmically (base e)
        between 0.5 and 0.01 is generated.
    learning_rate: float
        learning rate for gradient descent
        default: 1e-4
    max_iter: int
        maximum iterations for the model,
        default: 100
    tol: float
        convergence threshold or stopping criteria.
        Optimization loop will stop below setting threshold
        default: 1e-3
    eta: float
        a threshold parameter that linearizes the exp() function above eta
        default: 4.0
    random_state: int
        seed of the random number generator used to initialize the solution
    verbose: boolean or int
        if True it will print the output while iterating
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

    def __init__(self, distr='poisson', alpha=0.05,
                 reg_lambda=None,
                 learning_rate=1e-4, max_iter=100,
                 tol=1e-3, eta=4.0, random_state=0, verbose=False):

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
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_ = None
        self.tol = tol
        self.eta = eta
        self.random_state = random_state
        set_log_level(verbose)

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
        """Return a copy of the object."""
        return deepcopy(self)

    def qu(self, z):
        """The non-linearity."""
        if self.distr == 'poisson':
            qu = np.log1p(np.exp(z))
        elif self.distr == 'poissonexp':
            qu = deepcopy(z)
            slope = np.exp(self.eta)
            intercept = (1 - self.eta) * slope
            qu[z > self.eta] = z[z > self.eta] * slope + intercept
            qu[z <= self.eta] = np.exp(z[z <= self.eta])

            # qu = np.exp([zi if zi < 5.0 else 5.0 for zi in z.ravel()])
        elif self.distr == 'normal':
            qu = z
        elif self.distr == 'binomial':
            qu = expit(z)
        elif self.distr == 'multinomial':
            qu = softmax(z)

        return qu
        # qu = dict(poisson=np.log1p(np.exp(z)),
        #     poissonexp=np.exp([zi if zi < 3.0 else 3.0 for zi in z.ravel()]),
        #     normal=z, binomial=expit(z),
        #     multinomial=softmax(z))
        # return qu[self.distr]

    def lmb(self, beta0, beta, X):
        """Conditional intensity function."""
        z = beta0 + np.dot(X, beta)
        l = self.qu(z)
        return l

    def logL(self, beta0, beta, X, y):
        """The log likelihood."""
        l = self.lmb(beta0, beta, X)
        if self.distr == 'poisson':
            logL = np.sum(y * np.log(l) - l)
        elif self.distr == 'poissonexp':
            logL = np.sum(y * l - l)
        elif self.distr == 'normal':
            logL = -0.5 * np.sum((y - l)**2)
        elif self.distr == 'binomial':
            # analytical formula
            # logL = np.sum(y*np.log(l) + (1-y)*np.log(1-l))

            # but this prevents underflow
            z = beta0 + np.dot(X, beta)
            logL = np.sum(y * z - np.log(1 + np.exp(z)))
        elif self.distr == 'multinomial':
            logL = np.sum(y * np.log(l))
        return logL

    def penalty(self, beta):
        """The penalty."""
        alpha = self.alpha
        P = 0.5 * (1 - alpha) * np.linalg.norm(beta, 2) + \
            alpha * np.linalg.norm(beta, 1)
        return P

    def loss(self, beta0, beta, reg_lambda, X, y):
        """Define the objective function for elastic net."""
        L = self.logL(beta0, beta, X, y)
        P = self.penalty(beta)
        J = -L + reg_lambda * P
        return J

    def L2loss(self, beta0, beta, reg_lambda, X, y):
        """Quadratic loss."""
        alpha = self.alpha
        L = self.logL(beta0, beta, X, y)
        P = 0.5 * (1 - alpha) * np.linalg.norm(beta, 2)
        J = -L + reg_lambda * P
        return J

    def prox(self, X, l):
        """Proximal operator."""
        # sx = [0. if np.abs(y) <= l else np.sign(y)*np.abs(abs(y)-l)
        # for y in x]
        # return np.array(sx).reshape(x.shape)
        return np.sign(X) * (np.abs(X) - l) * (np.abs(X) > l)

    def grad_L2loss(self, beta0, beta, reg_lambda, X, y):
        """The gradient."""
        alpha = self.alpha
        z = beta0 + np.dot(X, beta)
        s = expit(z)

        if self.distr == 'poisson':
            q = self.qu(z)
            grad_beta0 = np.sum(s) - np.sum(y * s / q)
            grad_beta = np.transpose(np.dot(np.transpose(s), X) -
                                     np.dot(np.transpose(y * s / q), X)) + \
                reg_lambda * (1 - alpha) * beta
            # + reg_lambda*alpha*np.sign(beta)

        elif self.distr == 'poissonexp':
            q = self.qu(z)

            grad_beta0 = np.sum(q[z <= self.eta] - y[z <= self.eta]) + \
                np.sum(1 - y[z > self.eta] / q[z > self.eta]) * self.eta

            grad_beta = np.zeros([X.shape[1], 1])
            selector = np.where(z.ravel() <= self.eta)[0]
            grad_beta += np.transpose(np.dot((q[selector] - y[selector]).T,
                                             X[selector, :]))
            selector = np.where(z.ravel() > self.eta)[0]
            grad_beta += self.eta * \
                np.transpose(np.dot((1 - y[selector] / q[selector]).T,
                                    X[selector, :]))
            grad_beta += reg_lambda * (1 - alpha) * beta

        elif self.distr == 'normal':
            grad_beta0 = -np.sum(y - z)
            grad_beta = -np.transpose(np.dot(np.transpose(y - z), X)) \
                + reg_lambda * (1 - alpha) * beta
            # + reg_lambda*alpha*np.sign(beta)

        elif self.distr == 'binomial':
            grad_beta0 = np.sum(s - y)
            grad_beta = np.transpose(np.dot(np.transpose(s - y), X)) \
                + reg_lambda * (1 - alpha) * beta
            # + reg_lambda*alpha*np.sign(beta)

        elif self.distr == 'multinomial':
            # this assumes that y is already as a one-hot encoding
            pred = self.qu(z)
            grad_beta0 = -np.sum(y - pred, axis=0)
            grad_beta = -np.transpose(np.dot(np.transpose(y - pred), X)) \
                + reg_lambda * (1 - alpha) * beta

        return grad_beta0, grad_beta

    def fit(self, X, y):
        """The fit function.

        Parameters
        ----------
        X : array
            shape (n_samples, n_features)
            The input data
        y : array
            Labels to the data

        Returns
        -------
        self : instance of GLM
            The fitted model.
        """
        # Implements batch gradient descent (i.e. vanilla gradient descent by
        # computing gradient over entire training set)
        np.random.seed(self.random_state)

        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        # dataset shape
        n_features = X.shape[1]

        if self.distr == 'multinomial':
            # convert to 1-hot encoding in multinomial case
            y_bk = y.ravel()
            y = np.zeros([X.shape[0], y.max() + 1])
            y[np.arange(X.shape[0]), y_bk] = 1
        else:
            # else convert to column vector
            if y.ndim == 1:
                y = y[:, np.newaxis]

        # number of predictions
        n_classes = y.shape[1] if self.distr == 'multinomial' else 1

        # Initialize parameters
        beta0_hat = np.random.normal(0.0, 1.0, n_classes)
        beta_hat = np.random.normal(0.0, 1.0, [n_features, n_classes])
        # beta0_hat = np.zeros(n_classes)
        # beta_hat = np.zeros([n_features, n_classes])

        fit_params = list()

        # Outer loop with descending lambda
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

            # Iterate until convergence
            tol = self.tol
            alpha = self.alpha

            # Initialize parameters
            beta = np.zeros([n_features + 1, n_classes])
            beta[0] = fit_params[-1]['beta0']
            beta[1:] = fit_params[-1]['beta']

            g = np.zeros([n_features + 1, n_classes])

            # Initialize cost
            L, DL = list(), list()
            for t in range(0, self.max_iter):

                # Calculate gradient
                grad_beta0, grad_beta = self.grad_L2loss(
                    beta[0], beta[1:], rl, X, y)
                g[0] = grad_beta0
                g[1:] = grad_beta

                # Update parameters
                beta = beta - self.learning_rate * g

                # Apply proximal operator for L1-regularization
                beta[1:] = self.prox(beta[1:], rl * alpha)

                # Calculate loss and convergence criteria
                L.append(self.loss(beta[0], beta[1:], rl, X, y))

                # Delta loss and convergence criterion
                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if np.abs(DL[-1] / L[-1]) < tol:
                        msg = ('\tConverged. Loss function:'
                               ' {0:.2f}').format(L[-1])
                        logger.info(msg)
                        msg = ('\tdL/L: {0:.6f}\n'.format(DL[-1] / L[-1]))
                        logger.info(msg)
                        break

            # Store the parameters after convergence
            fit_params[-1]['beta0'] = beta[0]
            fit_params[-1]['beta'] = beta[1:]

        self.fit_ = fit_params
        return self

    def predict(self, X):
        """Predict labels.

        Parameters
        ----------
        X : array
            shape (n_samples, n_features)
            The data for prediction.

        Returns
        -------
        yhat : array
            shape ([n_lambda], n_samples)
            The predicted labels. A 1D array if predicting on only
            one lambda (compatible with scikit-learn API). Otherwise,
            returns a 2D array.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        if isinstance(self.fit_, list):
            yhat = list()
            for fit in self.fit_:
                yhat.append(self.lmb(fit['beta0'], fit['beta'], X))
        else:
            yhat = self.lmb(self.fit_['beta0'], self.fit_['beta'], X)
        yhat = np.asarray(yhat)
        yhat = yhat[..., 0] if self.distr != 'multinomial' else yhat
        return yhat

    def fit_predict(self, X, y):
        """Fit the model and predict on the same data.

        Parameters
        ----------
        X : array
            shape (n_samples, n_features)
            The data for fit and prediction.

        Returns
        -------
        yhat : array
            shape ([n_lambda], n_samples)
            The predicted labels. A 1D array if predicting on only
            one lambda (compatible with scikit-learn API). Otherwise,
            returns a 2D array.
        """
        return self.fit(X, y).predict(X)

    def pseudo_R2(self, y, yhat, ynull):
        """Define the pseudo-R2 function."""
        eps = np.spacing(1)
        y = y.ravel()
        yhat = yhat.ravel()

        if self.distr == 'poisson' or self.distr == 'poissonexp':
            # Log likelihood of model under consideration
            L1 = np.sum(y * np.log(eps + yhat) - yhat)

            # Log likelihood of homogeneous model
            L0 = np.sum(y * np.log(eps + ynull) - ynull)

            # Log likelihood of saturated model
            LS = np.sum(y * np.log(eps + y) - y)
            R2 = 1 - (LS - L1) / (LS - L0)

        elif self.distr == 'binomial':
            # Log likelihood of model under consideration
            L1 = 2 * len(y) * \
                np.sum(y * np.log((yhat == 0) + yhat) / np.mean(yhat) +
                       (1 - y) * np.log((yhat == 1) +
                                        1 - yhat) / (1 - np.mean(yhat)))

            # Log likelihood of homogeneous model
            L0 = 2 * len(y) * \
                np.sum(y * np.log((ynull == 0) + ynull) / np.mean(yhat) +
                       (1 - y) * np.log((ynull == 1) +
                                        1 - ynull) / (1 - np.mean(yhat)))
            R2 = 1 - L1 / L0

        elif self.distr == 'normal':
            R2 = 1 - np.sum((y - yhat)**2) / np.sum((y - ynull)**2)

        return R2

    def deviance(self, y, yhat):
        """The deviance function."""
        eps = np.spacing(1)
        y = y.ravel()
        yhat = yhat.ravel()
        # L1 = Log likelihood of model under consideration
        # LS = Log likelihood of saturated model
        if self.distr == 'poisson' or self.distr == 'poissonexp':
            L1 = np.sum(y * np.log(eps + yhat) - yhat)
            LS = np.sum(y * np.log(eps + y) - y)

        elif self.distr == 'binomial':
            L1 = 2 * len(y) * \
                np.sum(y * np.log((yhat == 0) + yhat) / np.mean(yhat) +
                       (1 - y) * np.log((yhat == 1) +
                                        1 - yhat) / (1 - np.mean(yhat)))
            LS = 0

        elif self.distr == 'normal':
            L1 = -np.sum((y - yhat) ** 2)
            LS = 0

        D = -2 * (L1 - LS)
        return D

    def simulate(self, beta0, beta, X):
        """Simulate data."""
        if self.distr == 'poisson' or self.distr == 'poissonexp':
            y = np.random.poisson(self.lmb(beta0, beta, X))
        if self.distr == 'normal':
            y = np.random.normal(self.lmb(beta0, beta, X))
        if self.distr == 'binomial':
            y = np.random.binomial(1, self.lmb(beta0, beta, X))
        return y
