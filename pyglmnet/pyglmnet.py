"""Python implementation of elastic-net regularized GLMs."""

import warnings
from copy import deepcopy

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from .utils import logger, set_log_level, _check_params
from .base import BaseEstimator, is_classifier, check_version


ALLOWED_DISTRS = ['gaussian', 'binomial', 'softplus', 'poisson',
                  'probit', 'gamma']


def _probit_g1(z, pdfz, cdfz, thresh=5):
    res = np.zeros_like(z)
    res[z < -thresh] = np.log(-pdfz[z < -thresh] / z[z < -thresh])
    res[np.abs(z) <= thresh] = np.log(cdfz[np.abs(z) <= thresh])
    res[z > thresh] = -pdfz[z > thresh] / z[z > thresh]
    return res


def _probit_g2(z, pdfz, cdfz, thresh=5):
    res = np.zeros_like(z)
    res[z < -thresh] = pdfz[z < -thresh] / z[z < -thresh]
    res[np.abs(z) <= thresh] = np.log(1 - cdfz[np.abs(z) <= thresh])
    res[z > thresh] = np.log(pdfz[z > thresh] / z[z > thresh])
    return res


def _probit_g3(z, pdfz, cdfz, thresh=5):
    res = np.zeros_like(z)
    res[z < -thresh] = -z[z < -thresh]
    res[np.abs(z) <= thresh] = \
        pdfz[np.abs(z) <= thresh] / cdfz[np.abs(z) <= thresh]
    res[z > thresh] = pdfz[z > thresh]
    return res


def _probit_g4(z, pdfz, cdfz, thresh=5):
    res = np.zeros_like(z)
    res[z < -thresh] = pdfz[z < -thresh]
    res[np.abs(z) <= thresh] = \
        pdfz[np.abs(z) <= thresh] / (1 - cdfz[np.abs(z) <= thresh])
    res[z > thresh] = z[z > thresh]
    return res


def _probit_g5(z, pdfz, cdfz, thresh=5):
    res = np.zeros_like(z)
    res[z < -thresh] = 0 * z[z < -thresh]
    res[np.abs(z) <= thresh] = \
        z[np.abs(z) <= thresh] * pdfz[np.abs(z) <= thresh] / \
        cdfz[np.abs(z) <= thresh] + (pdfz[np.abs(z) <= thresh] /
                                     cdfz[np.abs(z) <= thresh]) ** 2
    res[z > thresh] = z[z > thresh] * pdfz[z > thresh] + pdfz[z > thresh] ** 2
    return res


def _probit_g6(z, pdfz, cdfz, thresh=5):
    res = np.zeros_like(z)
    res[z < -thresh] = \
        pdfz[z < -thresh] ** 2 - z[z < -thresh] * pdfz[z < -thresh]
    res[np.abs(z) <= thresh] = \
        (pdfz[np.abs(z) <= thresh] / (1 - cdfz[np.abs(z) <= thresh])) ** 2 - \
        z[np.abs(z) <= thresh] * pdfz[np.abs(z) <= thresh] / \
        (1 - cdfz[np.abs(z) <= thresh])
    res[z > thresh] = 0 * z[z > thresh]
    return res


def _z(beta0, beta, X, fit_intercept):
    """Compute z to be passed through non-linearity"""
    if fit_intercept:
        z = beta0 + np.dot(X, beta)
    else:
        z = np.dot(X, np.r_[beta0, beta])
    return z


def _lmb(distr, beta0, beta, X, eta, fit_intercept=True):
    """Conditional intensity function."""
    z = _z(beta0, beta, X, fit_intercept)
    return _mu(distr, z, eta, fit_intercept)


def _mu(distr, z, eta, fit_intercept):
    """The non-linearity (inverse link)."""
    if distr in ['softplus', 'gamma']:
        mu = np.log1p(np.exp(z))
    elif distr == 'poisson':
        mu = z.copy()
        beta0 = (1 - eta) * np.exp(eta) if fit_intercept else 0.
        mu[z > eta] = z[z > eta] * np.exp(eta) + beta0
        mu[z <= eta] = np.exp(z[z <= eta])
    elif distr == 'gaussian':
        mu = z
    elif distr == 'binomial':
        mu = expit(z)
    elif distr == 'probit':
        mu = norm.cdf(z)
    return mu


def _grad_mu(distr, z, eta):
    """Derivative of the non-linearity."""
    if distr in ['softplus', 'gamma']:
        grad_mu = expit(z)
    elif distr == 'poisson':
        grad_mu = z.copy()
        grad_mu[z > eta] = np.ones_like(z)[z > eta] * np.exp(eta)
        grad_mu[z <= eta] = np.exp(z[z <= eta])
    elif distr == 'gaussian':
        grad_mu = np.ones_like(z)
    elif distr == 'binomial':
        grad_mu = expit(z) * (1 - expit(z))
    elif distr in 'probit':
        grad_mu = norm.pdf(z)
    return grad_mu


def _logL(distr, y, y_hat, z=None):
    """The log likelihood."""
    if distr in ['softplus', 'poisson']:
        eps = np.spacing(1)
        logL = np.sum(y * np.log(y_hat + eps) - y_hat)
    elif distr == 'gaussian':
        logL = -0.5 * np.sum((y - y_hat)**2)
    elif distr == 'binomial':

        # prevents underflow
        if z is not None:
            logL = np.sum(y * z - np.log(1 + np.exp(z)))
        # for scoring
        else:
            logL = np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    elif distr == 'probit':
        if z is not None:
            pdfz, cdfz = norm.pdf(z), norm.cdf(z)
            logL = np.sum(y * _probit_g1(z, pdfz, cdfz) +
                          (1 - y) * _probit_g2(z, pdfz, cdfz))
        else:
            logL = np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    elif distr == 'gamma':
        # see
        # https://www.statistics.ma.tum.de/fileadmin/w00bdb/www/czado/lec8.pdf
        nu = 1.  # shape parameter, exponential for now
        logL = np.sum(nu * (-y / y_hat - np.log(y_hat)))
    return logL


def _penalty(alpha, beta, Tau, group):
    """The penalty."""
    # Combine L1 and L2 penalty terms
    P = 0.5 * (1 - alpha) * _L2penalty(beta, Tau) + \
        alpha * _L1penalty(beta, group)
    return P


def _L2penalty(beta, Tau):
    """The L2 penalty."""
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
    """The L1 penalty."""
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


def _loss(distr, alpha, Tau, reg_lambda, X, y, eta, group, beta,
          fit_intercept=True):
    """Define the objective function for elastic net."""
    n_samples, n_features = X.shape
    z = _z(beta[0], beta[1:], X, fit_intercept)
    y_hat = _mu(distr, z, eta, fit_intercept)
    L = 1. / n_samples * _logL(distr, y, y_hat, z)
    if fit_intercept:
        P = _penalty(alpha, beta[1:], Tau, group)
    else:
        P = _penalty(alpha, beta, Tau, group)
    J = -L + reg_lambda * P
    return J


def _L2loss(distr, alpha, Tau, reg_lambda, X, y, eta, group, beta,
            fit_intercept=True):
    """Define the objective function for elastic net."""
    n_samples, n_features = X.shape
    z = _z(beta[0], beta[1:], X, fit_intercept)
    y_hat = _mu(distr, z, eta, fit_intercept)
    L = 1. / n_samples * _logL(distr, y, y_hat, z)
    if fit_intercept:
        P = 0.5 * (1 - alpha) * _L2penalty(beta[1:], Tau)
    else:
        P = 0.5 * (1 - alpha) * _L2penalty(beta, Tau)
    J = -L + reg_lambda * P
    return J


def _grad_L2loss(distr, alpha, Tau, reg_lambda, X, y, eta, beta,
                 fit_intercept=True):
    """The gradient."""
    n_samples, n_features = X.shape
    n_samples = np.float(n_samples)

    if Tau is None:
        if fit_intercept:
            Tau = np.eye(beta[1:].shape[0])
        else:
            Tau = np.eye(beta.shape[0])
    InvCov = np.dot(Tau.T, Tau)

    z = _z(beta[0], beta[1:], X, fit_intercept)
    mu = _mu(distr, z, eta, fit_intercept)
    grad_mu = _grad_mu(distr, z, eta)

    grad_beta0 = 0.
    if distr in ['poisson', 'softplus']:
        if fit_intercept:
            grad_beta0 = np.sum(grad_mu) - np.sum(y * grad_mu / mu)
        grad_beta = ((np.dot(grad_mu.T, X) -
                      np.dot((y * grad_mu / mu).T, X)).T)

    elif distr == 'gaussian':
        if fit_intercept:
            grad_beta0 = np.sum((mu - y) * grad_mu)
        grad_beta = np.dot((mu - y).T, X * grad_mu[:, None]).T

    elif distr == 'binomial':
        if fit_intercept:
            grad_beta0 = np.sum(mu - y)
        grad_beta = np.dot((mu - y).T, X).T

    elif distr == 'probit':
        grad_logl = (y * _probit_g3(z, grad_mu, mu) -
                     (1 - y) * _probit_g4(z, grad_mu, mu))
        if fit_intercept:
            grad_beta0 = -np.sum(grad_logl)
        grad_beta = -np.dot(grad_logl.T, X).T

    elif distr == 'gamma':
        nu = 1.
        grad_logl = (y / mu ** 2 - 1 / mu) * grad_mu
        if fit_intercept:
            grad_beta0 = -nu * np.sum(grad_logl)
        grad_beta = -nu * np.dot(grad_logl.T, X).T

    grad_beta0 *= 1. / n_samples
    grad_beta *= 1. / n_samples
    if fit_intercept:
        grad_beta += reg_lambda * (1 - alpha) * np.dot(InvCov, beta[1:])
        g = np.zeros((n_features + 1, ))
        g[0] = grad_beta0
        g[1:] = grad_beta
    else:
        grad_beta += reg_lambda * (1 - alpha) * np.dot(InvCov, beta)
        g = grad_beta

    return g


def _gradhess_logloss_1d(distr, xk, y, z, eta, fit_intercept=True):
    """
    Compute gradient (1st derivative)
    and Hessian (2nd derivative)
    of log likelihood for a single coordinate.

    Parameters
    ----------
    xk: float
        (n_samples,)
    y: float
        (n_samples,)
    z: float
        (n_samples,)

    Returns
    -------
    gk: gradient, float:
        (n_features + 1,)
    hk: float:
        (n_features + 1,)
    """
    n_samples = xk.shape[0]

    if distr == 'softplus':
        mu = _mu(distr, z, eta, fit_intercept)
        s = expit(z)
        gk = np.sum(s * xk) - np.sum(y * s / mu * xk)

        grad_s = s * (1 - s)
        grad_s_by_mu = grad_s / mu - s / (mu ** 2)
        hk = np.sum(grad_s * xk ** 2) - np.sum(y * grad_s_by_mu * xk ** 2)

    elif distr == 'poisson':
        mu = _mu(distr, z, eta, fit_intercept)
        s = expit(z)
        gk = np.sum((mu[z <= eta] - y[z <= eta]) *
                    xk[z <= eta]) + \
            np.exp(eta) * \
            np.sum((1 - y[z > eta] / mu[z > eta]) *
                   xk[z > eta])
        hk = np.sum(mu[z <= eta] * xk[z <= eta] ** 2) + \
            np.exp(eta) ** 2 * \
            np.sum(y[z > eta] / (mu[z > eta] ** 2) *
                   (xk[z > eta] ** 2))

    elif distr == 'gaussian':
        gk = np.sum((z - y) * xk)
        hk = np.sum(xk * xk)

    elif distr == 'binomial':
        mu = _mu(distr, z, eta, fit_intercept)
        gk = np.sum((mu - y) * xk)
        hk = np.sum(mu * (1.0 - mu) * xk * xk)

    elif distr == 'probit':
        pdfz = norm.pdf(z)
        cdfz = norm.cdf(z)
        gk = -np.sum((y * _probit_g3(z, pdfz, cdfz) -
                      (1 - y) * _probit_g4(z, pdfz, cdfz)) * xk)
        hk = np.sum((y * _probit_g5(z, pdfz, cdfz) +
                     (1 - y) * _probit_g6(z, pdfz, cdfz)) * (xk * xk))

    elif distr == 'gamma':
        raise NotImplementedError('cdfast is not implemented for Gamma '
                                  'distribution')

    return 1. / n_samples * gk, 1. / n_samples * hk


def simulate_glm(distr, beta0, beta, X, eta=2.0, random_state=None,
                 sample=False):
    """Simulate target data under a generative model.

    Parameters
    ----------
    distr: str
        distribution
    beta0: float
        intercept coefficient
    beta: array
        coefficients of shape (n_features,)
    X: array
        design matrix of shape (n_samples, n_features)
    eta: float
        parameter for poisson non-linearity
    random_state: float
        random state
    sample: bool
        If True, sample from distribution. Otherwise, return
        conditional intensity function

    Returns
    -------
    y: array
        simulated target data of shape (n_samples,)
    """
    if distr not in ALLOWED_DISTRS:
        raise ValueError("'distr' must be in %s, got %s"
                         % (repr(ALLOWED_DISTRS), distr))

    if not isinstance(beta0, float):
        raise ValueError("'beta0' must be float, got %s" % type(beta0))

    if beta.ndim != 1:
        raise ValueError("'beta' must be 1D, got %dD" % beta.ndim)

    if not sample:
        return _lmb(distr, beta0, beta, X, eta)

    _random_state = np.random.RandomState(random_state)
    if distr == 'softplus' or distr == 'poisson':
        y = _random_state.poisson(_lmb(distr, beta0, beta, X, eta))
    if distr == 'gaussian':
        y = _random_state.normal(_lmb(distr, beta0, beta, X, eta))
    if distr == 'binomial' or distr == 'probit':
        y = _random_state.binomial(1, _lmb(distr, beta0, beta, X, eta))
    if distr == 'gamma':
        mu = _lmb(distr, beta0, beta, X, eta)
        y = np.exp(mu)
    return y


class GLM(BaseEstimator):
    """Class for estimating regularized generalized linear models (GLM).
    The regularized GLM minimizes the penalized negative log likelihood:

    .. math::

        \\min_{\\beta_0, \\beta} \\frac{1}{N}
        \\sum_{i = 1}^N \\mathcal{L} (y_i, \\beta_0 + \\beta^T x_i)
        + \\lambda [ \\frac{1}{2}(1 - \\alpha) \\mathcal{P}_2 +
                    \\alpha \\mathcal{P}_1 ]

    where :math:`\\mathcal{P}_2` and :math:`\\mathcal{P}_1` are the generalized
    L2 (Tikhonov) and generalized L1 (Group Lasso) penalties, given by:

    .. math::

        \\mathcal{P}_2 = \\|\\Gamma \\beta \\|_2^2 \\
        \\mathcal{P}_1 = \\sum_g \\|\\beta_{j,g}\\|_2

    where :math:`\\Gamma` is the Tikhonov matrix: a square factorization
    of the inverse covariance matrix and :math:`\\beta_{j,g}` is the
    :math:`j` th coefficient of group :math:`g`.

    The generalized L2 penalty defaults to the ridge penalty when
    :math:`\\Gamma` is identity.

    The generalized L1 penalty defaults to the lasso penalty when each
    :math:`\\beta` belongs to its own group.

    Parameters
    ----------
    distr: str
        distribution family can be one of the following
        'gaussian' | 'binomial' | 'poisson' | 'softplus' | 'probit' | 'gamma'
        default: 'poisson'.
    alpha: float
        the weighting between L1 penalty and L2 penalty term
        of the loss function.
        default: 0.5
    Tau: array | None
        the (n_features, n_features) Tikhonov matrix.
        default: None, in which case Tau is identity
        and the L2 penalty is ridge-like
    group: array | list | None
        the (n_features, )
        list or array of group identities for each parameter :math:`\\beta`.
        Each entry of the list/ array should contain an int from 1 to n_groups
        that specify group membership for each parameter
        (except :math:`\\beta_0`).
        If you do not want to specify a group for a specific parameter,
        set it to zero.
        default: None, in which case it defaults to L1 regularization
    reg_lambda: float
        regularization parameter :math:`\\lambda` of penalty term.
        default: 0.1
    solver: str
        optimization method, can be one of the following
        'batch-gradient' (vanilla batch gradient descent)
        'cdfast' (Newton coordinate gradient descent).
        default: 'batch-gradient'
    learning_rate: float
        learning rate for gradient descent.
        default: 2e-1
    max_iter: int
        maximum iterations for the model.
        default: 1000
    tol: float
        convergence threshold or stopping criteria.
        Optimization loop will stop when norm(gradient) is below the threshold.
        default: 1e-6
    eta: float
        a threshold parameter that linearizes the exp() function above eta.
        default: 2.0
    score_metric: str
        specifies the scoring metric.
        one of either 'deviance' or 'pseudo_R2'.
        default: 'deviance'
    fit_intercept: boolean
        specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        default: True
    random_state : int
        seed of the random number generator used to initialize the solution.
        default: 0
    verbose: boolean or int
        default: False

    Attributes
    ----------
    beta0_: int
        The intercept
    beta_: array, (n_features)
        The learned betas
    n_iter_: int
        The number of iterations

    Examples
    --------
    >>> import numpy as np
    >>> random_state = 1
    >>> n_samples, n_features = 100, 4
    >>> rng = np.random.RandomState(random_state)
    >>> X = rng.normal(0, 1, (n_samples, n_features))
    >>> y = 2.2 * X[:, 0] -1.0 * X[:, 1] + 0.3 * X[:, 3] + 1.0
    >>> glm = GLM(distr='gaussian', verbose=False, random_state=random_state)
    >>> glm = glm.fit(X, y)
    >>> glm.beta0_ # The intercept
    1.005380485553247
    >>> glm.beta_ # The coefficients
    array([ 1.90216711, -0.78782533, -0.        ,  0.03227455])
    >>> y_pred = glm.predict(X)


    Reference
    ---------
    Friedman, Hastie, Tibshirani (2010). Regularization Paths for Generalized
        Linear Models via Coordinate Descent, J Statistical Software.
        https://core.ac.uk/download/files/153/6287975.pdf
    """

    def __init__(self, distr='poisson', alpha=0.5,
                 Tau=None, group=None,
                 reg_lambda=0.1,
                 solver='batch-gradient',
                 learning_rate=2e-1, max_iter=1000,
                 tol=1e-6, eta=2.0, score_metric='deviance',
                 fit_intercept=True,
                 random_state=0, callback=None, verbose=False):

        _check_params(distr=distr, max_iter=max_iter,
                      fit_intercept=fit_intercept)

        self.distr = distr
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.Tau = Tau
        self.group = group
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta0_ = None
        self.beta_ = None
        self.ynull_ = None
        self.n_iter_ = 0
        self.tol = tol
        self.eta = eta
        self.score_metric = score_metric
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state)
        self.callback = callback
        self.verbose = verbose
        set_log_level(verbose)

    def _set_cv(cv, estimator=None, X=None, y=None):
        """Set the default CV depending on whether clf
           is classifier/regressor."""
        # Detect whether classification or regression
        if estimator in ['classifier', 'regressor']:
            est_is_classifier = estimator == 'classifier'
        else:
            est_is_classifier = is_classifier(estimator)
        # Setup CV
        if check_version('sklearn', '0.18'):
            from sklearn import model_selection as models
            from sklearn.model_selection import (check_cv,
                                                 StratifiedKFold, KFold)
            if isinstance(cv, (int, np.int)):
                XFold = StratifiedKFold if est_is_classifier else KFold
                cv = XFold(n_splits=cv)
            elif isinstance(cv, str):
                if not hasattr(models, cv):
                    raise ValueError('Unknown cross-validation')
                cv = getattr(models, cv)
                cv = cv()
            cv = check_cv(cv=cv, y=y, classifier=est_is_classifier)
        else:
            from sklearn import cross_validation as models
            from sklearn.cross_validation import (check_cv,
                                                  StratifiedKFold, KFold)
            if isinstance(cv, (int, np.int)):
                if est_is_classifier:
                    cv = StratifiedKFold(y=y, n_folds=cv)
                else:
                    cv = KFold(n=len(y), n_folds=cv)
            elif isinstance(cv, str):
                if not hasattr(models, cv):
                    raise ValueError('Unknown cross-validation')
                cv = getattr(models, cv)
                if cv.__name__ not in ['KFold', 'LeaveOneOut']:
                    raise NotImplementedError('CV cannot be defined with str'
                                              ' for sklearn < .017.')
                cv = cv(len(y))
            cv = check_cv(cv=cv, X=X, y=y, classifier=est_is_classifier)

        # Extract train and test set to retrieve them at predict time
        if hasattr(cv, 'split'):
            cv_splits = [(train, test) for train, test in
                         cv.split(X=np.zeros_like(y), y=y)]
        else:
            # XXX support sklearn.cross_validation cv
            cv_splits = [(train, test) for train, test in cv]

        if not np.all([len(train) for train, _ in cv_splits]):
            raise ValueError('Some folds do not have any train epochs.')

        return cv, cv_splits

    def __repr__(self):
        """Description of the object."""
        reg_lambda = self.reg_lambda
        s = '<\nDistribution | %s' % self.distr
        s += '\nalpha | %0.2f' % self.alpha
        s += '\nmax_iter | %0.2f' % self.max_iter
        s += '\nlambda: %0.2f\n>' % reg_lambda
        return s

    def copy(self):
        """Return a copy of the object.

        Parameters
        ----------
        none

        Returns
        -------
        self: instance of GLM
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
            result[~idxs_to_update] = 0.0

            return result

    def _cdfast(self, X, y, ActiveSet, beta, rl, fit_intercept=True):
        """
        Perform one cycle of Newton updates for all coordinates.

        Parameters
        ----------
        X: array
            n_samples x n_features
            The input data
        y: array
            Labels to the data
            n_samples x 1
        ActiveSet: array
            n_features + 1 x 1, or n_features
            Active set storing which betas are non-zero
        beta: array
            n_features + 1 x 1, or n_features
            Parameters to be updated
        rl: float
            Regularization lambda

        Returns
        -------
        beta: array
            (n_features + 1) x 1, or (n_features)
            Updated parameters
        """
        n_samples, n_features = X.shape
        reg_scale = rl * (1 - self.alpha)
        z = _z(beta[0], beta[1:], X, fit_intercept)

        for k in range(0, n_features + int(fit_intercept)):
            # Only update parameters in active set
            if ActiveSet[k] != 0:
                if fit_intercept:
                    if k == 0:
                        xk = np.ones((n_samples, ))
                    else:
                        xk = X[:, k - 1]
                else:
                    xk = X[:, k]

                # Calculate grad and hess of log likelihood term
                gk, hk = _gradhess_logloss_1d(self.distr, xk, y, z, self.eta,
                                              fit_intercept)

                # Add grad and hess of regularization term
                if self.Tau is None:
                    if k == 0 and fit_intercept:
                        gk_reg, hk_reg = 0.0, 0.0
                    else:
                        gk_reg, hk_reg = beta[k], 1.0
                else:
                    InvCov = np.dot(self.Tau.T, self.Tau)
                    if fit_intercept:
                        gk_reg = np.sum(InvCov[k - 1, :] * beta[1:])
                        hk_reg = InvCov[k - 1, k - 1]
                    else:
                        gk_reg = np.sum(InvCov[k, :] * beta)
                        hk_reg = InvCov[k, k]
                gk += reg_scale * gk_reg
                hk += reg_scale * hk_reg

                # Update parameters, z
                update = 1. / hk * gk
                beta[k], z = beta[k] - update, z - update * xk
        return beta

    def fit(self, X, y):
        """The fit function.

        Parameters
        ----------
        X: array
            The 2D input data of shape (n_samples, n_features)

        y: array
            The 1D target data of shape (n_samples,)

        Returns
        -------
        self: instance of GLM
            The fitted model.
        """
        # checks for group
        if self.group is not None:
            self.group = np.array(self.group)
            self.group = self.group.astype(np.int64)
            # shape check
            if self.group.shape[0] != X.shape[1]:
                raise ValueError('group should be (n_features,)')
            # int check
            if not np.all([isinstance(g, np.int64) for g in self.group]):
                raise ValueError('all entries of group should be integers')

        # type check for data
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            msg = ("Input must be ndarray. Got {} and {}"
                   .format(type(X), type(y)))
            raise ValueError(msg)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array, got %sD" % X.ndim)

        if y.ndim != 1:
            raise ValueError("y must be 1D, got %sD" % y.ndim)

        if hasattr(self, '_allow_refit') and self._allow_refit is False:
            raise ValueError("This glm object has already been fit before."
                             "A refit is not allowed")

        n_observations, n_features = X.shape

        if n_observations != len(y):
            raise ValueError('Shape mismatch.' +
                             'X has {} observations, y has {}.'
                             .format(n_observations, len(y)))

        # Initialize parameters
        beta = np.zeros((n_features + int(self.fit_intercept),))
        if self.fit_intercept:
            if self.beta0_ is None and self.beta_ is None:
                beta[0] = 1 / (n_features + 1) * \
                    self.rng.normal(0.0, 1.0, 1)
                beta[1:] = 1 / (n_features + 1) * \
                    self.rng.normal(0.0, 1.0, (n_features, ))
            else:
                beta[0] = self.beta0_
                beta[1:] = self.beta_
        else:
            if self.beta0_ is None and self.beta_ is None:
                beta = 1 / (n_features + 1) * \
                    self.rng.normal(0.0, 1.0, (n_features, ))
            else:
                beta = self.beta_

        logger.info('Lambda: %6.4f' % self.reg_lambda)

        tol = self.tol
        alpha = self.alpha
        reg_lambda = self.reg_lambda

        if self.solver == 'cdfast':
            # init active set
            ActiveSet = np.ones_like(beta)

        # Iterative updates
        for t in range(0, self.max_iter):
            self.n_iter_ += 1
            beta_old = beta.copy()
            if self.solver == 'batch-gradient':
                grad = _grad_L2loss(self.distr,
                                    alpha, self.Tau,
                                    reg_lambda, X, y, self.eta,
                                    beta, self.fit_intercept)
                # Update
                beta = beta - self.learning_rate * grad

            elif self.solver == 'cdfast':
                beta = \
                    self._cdfast(X, y, ActiveSet, beta, reg_lambda,
                                 self.fit_intercept)

            else:
                raise ValueError("solver must be one of "
                                 "'('batch-gradient', 'cdfast'), got %s."
                                 % (self.solver))

            # Apply proximal operator
            if self.fit_intercept:
                beta[1:] = self._prox(beta[1:], reg_lambda * alpha)
            else:
                beta = self._prox(beta, reg_lambda * alpha)

            # Update active set
            if self.solver == 'cdfast':
                ActiveSet[beta == 0] = 0
                if self.fit_intercept:
                    ActiveSet[0] = 1.

            # Convergence by relative parameter change tolerance
            norm_update = np.linalg.norm(beta - beta_old)
            if t > 1 and (norm_update / np.linalg.norm(beta)) < tol:
                msg = ('\tParameter update tolerance. ' +
                       'Converged in {0:d} iterations'.format(t))
                logger.info(msg)
                break

            # Compute and save loss if callbacks are requested
            if callable(self.callback):
                self.callback(beta)

        if self.n_iter_ == self.max_iter:
            warnings.warn(
                "Reached max number of iterations without convergence.")

        # Update the estimated variables
        if self.fit_intercept:
            self.beta0_ = beta[0]
            self.beta_ = beta[1:]
        else:
            self.beta0_ = 0
            self.beta_ = beta
        self.ynull_ = np.mean(y)
        self._allow_refit = False
        return self

    def predict(self, X):
        """Predict targets.

        Parameters
        ----------
        X: array
            Input data for prediction, of shape (n_samples, n_features)

        Returns
        -------
        yhat: array
            The predicted targets of shape (n_samples,)
        """
        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        yhat = _lmb(self.distr, self.beta0_, self.beta_, X, self.eta,
                    fit_intercept=True)

        if self.distr == 'binomial':
            yhat = (yhat > 0.5).astype(int)
        yhat = np.asarray(yhat)
        return yhat

    def predict_proba(self, X):
        """Predict class probability for binomial.

        Parameters
        ----------
        X: array
            Input data for prediction, of shape (n_samples, n_features)

        Returns
        -------
        yhat: array
            The predicted targets of shape (n_samples,).

        Raises
        ------
        Works only for the binomial distribution.
        Raises error otherwise.

        """
        if self.distr not in ['binomial', 'probit']:
            raise ValueError('This is only applicable for \
                              the binomial distribution.')

        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        yhat = _lmb(self.distr, self.beta0_, self.beta_, X, self.eta,
                    fit_intercept=True)
        yhat = np.asarray(yhat)
        return yhat

    def fit_predict(self, X, y):
        """Fit the model and predict on the same data.

        Parameters
        ----------
        X: array
            The input data to fit and predict,
            of shape (n_samples, n_features)


        Returns
        -------
        yhat: array
            The predicted targets of shape (n_samples,).
        """
        return self.fit(X, y).predict(X)

    def score(self, X, y):
        """Score the model.

        Parameters
        ----------
        X: array
            The input data whose prediction will be scored,
            of shape (n_samples, n_features).
        y: array
            The true targets against which to score the predicted targets,
            of shape (n_samples,).

        Returns
        -------
        score: float
            The score metric
        """
        from . import metrics
        valid_metrics = ['deviance', 'pseudo_R2', 'accuracy']
        if self.score_metric not in valid_metrics:
            raise ValueError("score_metric has to be one of: "
                             ",".join(valid_metrics))

        # If the model has not been fit it cannot be scored
        if self.ynull_ is None:
            raise ValueError('Model must be fit before ' +
                             'prediction can be scored')

        # For f1 as well
        if self.score_metric in ['accuracy']:
            if self.distr not in ['binomial', 'multinomial']:
                raise ValueError(self.score_metric +
                                 ' is only defined for binomial ' +
                                 'or multinomial distributions')

        y = np.asarray(y).ravel()

        if self.distr in ['binomial', 'probit'] and \
           self.score_metric != 'accuracy':
            yhat = self.predict_proba(X)
        else:
            yhat = self.predict(X)

        # Check whether we have a list of estimators or a single estimator
        if self.score_metric == 'deviance':
            return metrics.deviance(y, yhat, self.distr)
        elif self.score_metric == 'pseudo_R2':
            return metrics.pseudo_R2(X, y, yhat, self.ynull_, self.distr)
        if self.score_metric == 'accuracy':
            return metrics.accuracy(y, yhat)


class GLMCV(object):
    """Class for estimating regularized generalized linear models (GLM)
    along a regularization path with warm restarts.

    The regularized GLM minimizes the penalized negative log likelihood:

    .. math::

        \\min_{\\beta_0, \\beta} \\frac{1}{N}
        \\sum_{i = 1}^N \\mathcal{L} (y_i, \\beta_0 + \\beta^T x_i)
        + \\lambda [ \\frac{1}{2}(1 - \\alpha) \\mathcal{P}_2 +
                    \\alpha \\mathcal{P}_1 ]

    where :math:`\\mathcal{P}_2` and :math:`\\mathcal{P}_1` are the generalized
    L2 (Tikhonov) and generalized L1 (Group Lasso) penalties, given by:

    .. math::

        \\mathcal{P}_2 = \\|\\Gamma \\beta \\|_2^2 \\
        \\mathcal{P}_1 = \\sum_g \\|\\beta_{j,g}\\|_2
    where :math:`\\Gamma` is the Tikhonov matrix: a square factorization
    of the inverse covariance matrix and :math:`\\beta_{j,g}` is the
    :math:`j` th coefficient of group :math:`g`.

    The generalized L2 penalty defaults to the ridge penalty when
    :math:`\\Gamma` is identity.

    The generalized L1 penalty defaults to the lasso penalty when each
    :math:`\\beta` belongs to its own group.

    Parameters
    ----------
    distr: str
        distribution family can be one of the following
        'gaussian' | 'binomial' | 'poisson' | 'softplus' | 'probit' | 'gamma'
        default: 'poisson'.
    alpha: float
        the weighting between L1 penalty and L2 penalty term
        of the loss function.
        default: 0.5
    Tau: array | None
        the (n_features, n_features) Tikhonov matrix.
        default: None, in which case Tau is identity
        and the L2 penalty is ridge-like
    group: array | list | None
        the (n_features, )
        list or array of group identities for each parameter :math:`\\beta`.
        Each entry of the list/ array should contain an int from 1 to n_groups
        that specify group membership for each parameter
        (except :math:`\\beta_0`).
        If you do not want to specify a group for a specific parameter,
        set it to zero.
        default: None, in which case it defaults to L1 regularization
    reg_lambda: array | list | None
        array of regularized parameters :math:`\\lambda` of penalty term.
        default: None, a list of 10 floats spaced logarithmically (base e)
        between 0.5 and 0.01.
    cv: cross validation object (default 10)
        Iterator for doing cross validation
    solver: str
        optimization method, can be one of the following
        'batch-gradient' (vanilla batch gradient descent)
        'cdfast' (Newton coordinate gradient descent).
        default: 'batch-gradient'
    learning_rate: float
        learning rate for gradient descent.
        default: 2e-1
    max_iter: int
        maximum iterations for the model.
        default: 1000
    tol: float
        convergence threshold or stopping criteria.
        Optimization loop will stop when norm(gradient) is below the threshold.
        default: 1e-6
    eta: float
        a threshold parameter that linearizes the exp() function above eta.
        default: 2.0
    score_metric: str
        specifies the scoring metric.
        one of either 'deviance' or 'pseudo_R2'.
        default: 'deviance'
    fit_intercept: boolean
        specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
        default: True
    random_state : int
        seed of the random number generator used to initialize the solution.
        default: 0
    verbose: boolean or int
        default: False

    Attributes
    ----------
    beta0_: int
        The intercept
    beta_: array, (n_features)
        The learned betas
    glm_: instance of GLM
        The GLM object with best score
    reg_lambda_opt_: float
        The reg_lambda parameter for best GLM object
    n_iter_: int
        The number of iterations

    Reference
    ---------
    Friedman, Hastie, Tibshirani (2010). Regularization Paths for Generalized
        Linear Models via Coordinate Descent, J Statistical Software.
        https://core.ac.uk/download/files/153/6287975.pdf

    Notes
    -----
    To select subset of fitted glm models, you can simply do:

    glm = glm[1:3]
    glm[2].predict(X_test)
    """

    def __init__(self, distr='poisson', alpha=0.5,
                 Tau=None, group=None,
                 reg_lambda=None, cv=10,
                 solver='batch-gradient',
                 learning_rate=2e-1, max_iter=1000,
                 tol=1e-6, eta=2.0, score_metric='deviance',
                 fit_intercept=True,
                 random_state=0, verbose=False):

        if reg_lambda is None:
            reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10,
                                     base=np.exp(1))
        if not isinstance(reg_lambda, (list, np.ndarray)):
            reg_lambda = [reg_lambda]

        _check_params(distr=distr, max_iter=max_iter,
                      fit_intercept=fit_intercept)

        self.distr = distr
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.cv = cv
        self.Tau = Tau
        self.group = group
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta0_ = None
        self.beta_ = None
        self.reg_lambda_opt_ = None
        self.glm_ = None
        self.scores_ = None
        self.ynull_ = None
        self.tol = tol
        self.eta = eta
        self.score_metric = score_metric
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.verbose = verbose
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

    def copy(self):
        """Return a copy of the object.

        Parameters
        ----------
        none

        Returns
        -------
        self: instance of GLM
            A copy of the GLM instance.
        """
        return deepcopy(self)

    def fit(self, X, y):
        """The fit function.

        Parameters
        ----------
        X: array
            The input data of shape (n_samples, n_features)

        y: array
            The target data

        Returns
        -------
        self: instance of GLM
            The fitted model.
        """
        logger.info('Looping through the regularization path')
        glms, scores = list(), list()
        self.ynull_ = np.mean(y)

        if not type(int):
            raise ValueError('cv must be int. We do not support scikit-learn '
                             'cv objects at the moment')

        idxs = np.arange(y.shape[0])
        np.random.shuffle(idxs)
        cv_splits = np.array_split(idxs, self.cv)

        for idx, rl in enumerate(self.reg_lambda):
            glm = GLM(distr=self.distr,
                      alpha=self.alpha,
                      Tau=self.Tau,
                      group=self.group,
                      reg_lambda=0.1,
                      solver=self.solver,
                      learning_rate=self.learning_rate,
                      max_iter=self.max_iter,
                      tol=self.tol,
                      eta=self.eta,
                      score_metric=self.score_metric,
                      fit_intercept=self.fit_intercept,
                      random_state=self.random_state,
                      verbose=self.verbose)
            logger.info('Lambda: %6.4f' % rl)
            glm.reg_lambda = rl

            scores_fold = list()
            for fold in range(self.cv):
                val = cv_splits[fold]
                train = np.setdiff1d(idxs, val)
                if idx == 0:
                    glm.beta0_, glm.beta_ = self.beta0_, self.beta_
                else:
                    glm.beta0_, glm.beta_ = glms[-1].beta0_, glms[-1].beta_

                glm.n_iter_ = 0
                glm.fit(X[train], y[train])
                glm._allow_refit = True
                scores_fold.append(glm.score(X[val], y[val]))
            scores.append(np.mean(scores_fold))

            if idx == 0:
                glm.beta0_, glm.beta_ = self.beta0_, self.beta_
            else:
                glm.beta0_, glm.beta_ = glms[-1].beta0_, glms[-1].beta_

            glm.n_iter_ = 0
            glm.fit(X, y)
            glms.append(glm)

        for glm in glms:
            glm._allow_refit = False

        # Update the estimated variables
        if self.score_metric == 'deviance':
            opt = np.array(scores).argmin()
        elif self.score_metric in ['pseudo_R2', 'accuracy']:
            opt = np.array(scores).argmax()
        else:
            raise ValueError("Unknown score_metric: %s" % (self.score_metric))

        self.beta0_, self.beta_ = glms[opt].beta0_, glms[opt].beta_
        self.reg_lambda_opt_ = self.reg_lambda[opt]
        self.glm_ = glms[opt]
        self.scores_ = scores
        return self

    def predict(self, X):
        """Predict targets.

        Parameters
        ----------
        X: array
            Input data for prediction, of shape (n_samples, n_features)

        Returns
        -------
        yhat: array
            The predicted targets of shape based on the model with optimal
            reg_lambda (n_samples,)
        """
        return self.glm_.predict(X)

    def predict_proba(self, X):
        """Predict class probability for binomial.

        Parameters
        ----------
        X: array
            Input data for prediction, of shape (n_samples, n_features)

        Returns
        -------
        yhat: array
            The predicted targets of shape (n_samples, ).

        Raises
        ------
        Works only for the binomial distribution.
        Raises error otherwise.

        """
        return self.glm_.predict_proba(X)

    def fit_predict(self, X, y):
        """Fit the model and predict on the same data.

        Parameters
        ----------
        X: array
            The input data to fit and predict,
            of shape (n_samples, n_features)

        Returns
        -------
        yhat: array
            The predicted targets of shape based on the model with optimal
            reg_lambda (n_samples,)
        """
        self.fit(X, y)
        return self.glm_.predict(X)

    def score(self, X, y):
        """Score the model.

        Parameters
        ----------
        X: array
            The input data whose prediction will be scored,
            of shape (n_samples, n_features).
        y: array
            The true targets against which to score the predicted targets,
            of shape (n_samples,).

        Returns
        -------
        score: float
            The score metric for the optimal reg_lambda
        """
        return self.glm_.score(X, y)
