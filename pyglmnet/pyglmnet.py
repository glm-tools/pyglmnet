"""Python implementation of elastic-net regularized GLMs."""

import warnings
from copy import deepcopy

import numpy as np

from .utils import logger, set_log_level, _check_params, \
    _verbose_iterable, _tqdm_log
from .base import BaseEstimator, is_classifier, check_version

from .externals.sklearn.utils import check_random_state, check_array, check_X_y
from .externals.sklearn.utils.validation import check_is_fitted

from .distributions import BaseDistribution, Gaussian, Poisson, \
    PoissonSoftplus, NegBinomialSoftplus, Binomial, Probit, GammaSoftplus

ALLOWED_DISTRS = ['gaussian', 'binomial', 'softplus', 'poisson',
                  'probit', 'gamma', 'neg-binomial']


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


def _loss(distr, alpha, Tau, reg_lambda, X, y, eta, theta, group, beta,
          fit_intercept=True):
    """Define the objective function for elastic net."""
    n_samples, n_features = X.shape
    if fit_intercept:
        z = distr._z(beta[0], beta[1:], X)
    else:
        z = distr._z(0., beta, X)
    y_hat = distr.mu(z)
    if isinstance(distr, (Binomial, Probit)):
        L = 1. / n_samples * distr.log_likelihood(y, y_hat, z)
    else:
        L = 1. / n_samples * distr.log_likelihood(y, y_hat)
    if fit_intercept:
        P = _penalty(alpha, beta[1:], Tau, group)
    else:
        P = _penalty(alpha, beta, Tau, group)
    J = -L + reg_lambda * P
    return J


def _L2loss(distr, alpha, Tau, reg_lambda, X, y, eta, theta, group, beta,
            fit_intercept=True):
    """Define the objective function for elastic net."""
    n_samples, n_features = X.shape
    if fit_intercept:
        z = distr._z(beta[0], beta[1:], X)
    else:
        z = distr._z(0., beta, X)
    y_hat = distr.mu(z)
    if isinstance(distr, (Binomial, Probit)):
        L = 1. / n_samples * distr.log_likelihood(y, y_hat, z)
    else:
        L = 1. / n_samples * distr.log_likelihood(y, y_hat)
    if fit_intercept:
        P = 0.5 * (1 - alpha) * _L2penalty(beta[1:], Tau)
    else:
        P = 0.5 * (1 - alpha) * _L2penalty(beta, Tau)
    J = -L + reg_lambda * P
    return J


def _grad_L2loss(distr, alpha, Tau, reg_lambda, X, y, eta, theta, beta,
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

    if fit_intercept:
        beta0_, beta_ = beta[0], beta[1:]
    else:
        beta0_, beta_ = 0., beta
    grad_beta0, grad_beta = distr.grad_log_likelihood(X, y, beta0_, beta_)
    grad_beta0 = grad_beta0 if fit_intercept else 0.

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


def simulate_glm(distr, beta0, beta, X, eta=2.0, random_state=None,
                 sample=False, theta=1.0, fit_intercept=True):
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
    err_msg = ('distr must be one of %s or a subclass of BaseDistribution. '
               'Got %s' % (', '.join(ALLOWED_DISTRS), distr))
    if isinstance(distr, str) and distr not in ALLOWED_DISTRS:
        raise ValueError(err_msg)
    if not isinstance(distr, str) and not isinstance(distr, BaseDistribution):
        raise TypeError(err_msg)

    glm = GLM(distr=distr)
    glm._set_distr()

    if not isinstance(beta0, float):
        raise ValueError("'beta0' must be float, got %s" % type(beta0))

    if beta.ndim != 1:
        raise ValueError("'beta' must be 1D, got %dD" % beta.ndim)

    if not fit_intercept:
        beta0 = 0.

    y = glm.distr_.simulate(beta0, beta, X, random_state, sample)
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
    distr: str or object subclassed from BaseDistribution
        if str, distribution family can be one of the following
        'gaussian' | 'binomial' | 'poisson' | 'softplus'
        | 'probit' | 'gamma' | 'neg-binomial'
        default: 'poisson'.
    alpha: float
        the weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.)
        term of the loss function.
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
        maximum number of iterations for the solver.
        default: 1000
    tol: float
        convergence threshold or stopping criteria.
        Optimization loop will stop when relative change
        in parameter norm is below the threshold.
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
    theta: float
        shape parameter of the negative binomial distribution (number of
        successes before the first failure). It is used only if distr is
        equal to neg-binomial, otherwise it is ignored.
        default: 1.0
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
    is_fitted_: bool
        if True, the model is previously fitted

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
                 random_state=0, theta=1.0, callback=None, verbose=False):

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
        self.tol = tol
        self.eta = eta
        self.score_metric = score_metric
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.callback = callback
        self.verbose = verbose
        self.theta = theta
        set_log_level(verbose)

    def _set_distr(self):
        distr_lookup = {
            'gaussian': Gaussian(),
            'poisson': Poisson(),
            'softplus': PoissonSoftplus(),
            'neg-binomial': NegBinomialSoftplus(),
            'binomial': Binomial(),
            'probit': Probit(),
            'gamma': GammaSoftplus()
        }
        self.distr_ = distr_lookup[self.distr]
        if isinstance(self.distr_, Poisson):
            self.distr_.eta = self.eta
        if isinstance(self.distr_, NegBinomialSoftplus):
            self.distr_.theta = self.theta

    def _set_cv(cv, estimator=None, X=None, y=None):
        """Set the default CV.

        Depends on whether clf is classifier/regressor.
        """
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
        if fit_intercept:
            z = self.distr_._z(beta[0], beta[1:], X)
        else:
            z = self.distr_._z(0., beta, X)
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
                # gk, hk = _gradhess_logloss_1d(self.distr, xk, y, z, self.eta,
                #                               self.theta, fit_intercept)
                gk, hk = self.distr_.gradhess_log_likelihood_1d(xk, y, z)
                gk = 1. / n_samples * gk
                hk = 1. / n_samples * hk

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

                # Ensure that update does not blow up if Hessian is small
                update = 1. / hk * gk if hk > 1. else self.learning_rate * gk

                # Update parameters, z
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
        X, y = check_X_y(X, y, accept_sparse=False)
        self._set_distr()
        self.beta0_ = None
        self.beta_ = None
        self.ynull_ = None
        self.n_iter_ = 0
        self.random_state_ = check_random_state(self.random_state)

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
                    self.random_state_.normal(0.0, 1.0, 1)
                beta[1:] = 1 / (n_features + 1) * \
                    self.random_state_.normal(0.0, 1.0, (n_features, ))
            else:
                beta[0] = self.beta0_
                beta[1:] = self.beta_
        else:
            if self.beta0_ is None and self.beta_ is None:
                beta = 1 / (n_features + 1) * \
                    self.random_state_.normal(0.0, 1.0, (n_features, ))
            else:
                beta = self.beta_

        _tqdm_log('Lambda: %6.4f' % self.reg_lambda)

        tol = self.tol
        alpha = self.alpha
        reg_lambda = self.reg_lambda

        if self.solver == 'cdfast':
            # init active set
            ActiveSet = np.ones_like(beta)

        self._convergence = list()
        training_iterations = _verbose_iterable(range(self.max_iter))

        # Iterative updates
        for t in training_iterations:
            self.n_iter_ += 1
            beta_old = beta.copy()
            if self.solver == 'batch-gradient':
                grad = _grad_L2loss(self.distr_,
                                    alpha, self.Tau,
                                    reg_lambda, X, y, self.eta,
                                    self.theta, beta, self.fit_intercept)
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
                beta[1:] = self._prox(beta[1:],
                                      self.learning_rate * reg_lambda * alpha)
            else:
                beta = self._prox(beta,
                                  self.learning_rate * reg_lambda * alpha)

            # Update active set
            if self.solver == 'cdfast':
                ActiveSet[beta == 0] = 0
                if self.fit_intercept:
                    ActiveSet[0] = 1.

            # Convergence by relative parameter change tolerance
            norm_update = np.linalg.norm(beta - beta_old)
            norm_update /= np.linalg.norm(beta)
            self._convergence.append(norm_update)
            if t > 1 and self._convergence[-1] < tol:
                msg = ('\tParameter update tolerance. ' +
                       'Converged in {0:d} iterations'.format(t))
                _tqdm_log(msg)
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
        self.is_fitted_ = True
        return self

    def plot_convergence(self, ax=None, show=True):
        """Plot convergence.

        Parameters
        ----------
        ax : matplotlib.pyplot.axes object
            If not None, plot in this axis.
        show : bool
            If True, call plt.show()

        Returns
        -------
        fig : matplotlib.Figure
            The matplotlib figure handle
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.semilogy(self._convergence)
        ax.set_xlim((-20, self.max_iter + 20))
        ax.axhline(self.tol, linestyle='--', color='r', label='tol')
        ax.set_ylabel(r'$\Vert\beta_{t} - \beta_{t-1}\Vert/\Vert\beta_t\Vert$')
        ax.legend()

        if show:
            plt.show()

        return ax.get_figure()

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
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        z = self.distr_._z(self.beta0_, self.beta_, X)
        yhat = self.distr_.mu(z)

        if isinstance(self.distr_, Binomial):
            yhat = (yhat > 0.5).astype(int)
        yhat = np.asarray(yhat)
        return yhat

    def _predict_proba(self, X):
        """Predict class probability for a binomial or probit distribution.

        Parameters
        ----------
        X: array
            Input data for prediction, of shape (n_samples, n_features)

        Returns
        -------
        yhat: array
            The predicted targets of shape (n_samples,).

        """
        if not isinstance(self.distr_, (Binomial, Probit)):
            raise ValueError('This is only applicable for \
                              the binomial distribution.')

        if not isinstance(X, np.ndarray):
            raise ValueError('Input data should be of type ndarray (got %s).'
                             % type(X))

        z = self.distr_._z(self.beta0_, self.beta_, X)
        yhat = self.distr_.mu(z)
        yhat = np.asarray(yhat)
        return yhat

    def predict_proba(self, X):
        """Predict class probability for a binomial or probit distribution.

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
        Warn the output otherwise.

        """
        X = check_array(X, accept_sparse=False)
        check_is_fitted(self, 'is_fitted_')

        if isinstance(self.distr_, (Binomial, Probit)):
            return self._predict_proba(X)
        else:
            warnings.warn('This is only applicable for \
                           the binomial distribution. \
                           We returns predict as an output here.')
            return self.predict(X)

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
        check_is_fitted(self, 'is_fitted_')
        from . import metrics
        valid_metrics = ['deviance', 'pseudo_R2', 'accuracy']
        if self.score_metric not in valid_metrics:
            raise ValueError("score_metric has to be one of: "
                             ",".join(valid_metrics))

        # If the model has not been fit it cannot be scored
        if not hasattr(self, 'ynull_'):
            raise ValueError('Model must be fit before ' +
                             'prediction can be scored')

        # For f1 as well
        if self.score_metric in ['accuracy']:
            if not isinstance(self.distr_, (Binomial, Probit)):
                raise ValueError(self.score_metric +
                                 ' is only defined for binomial ' +
                                 'or multinomial distributions')

        y = np.asarray(y).ravel()

        if isinstance(self.distr_, (Binomial, Probit)) and \
           self.score_metric != 'accuracy':
            yhat = self.predict_proba(X)
        else:
            yhat = self.predict(X)

        # Check whether we have a list of estimators or a single estimator
        if self.score_metric == 'deviance':
            return metrics.deviance(y, yhat, self.distr_, self.theta)
        elif self.score_metric == 'pseudo_R2':
            return metrics.pseudo_R2(X, y, yhat, self.ynull_,
                                     self.distr_, self.theta)
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
        the weighting between L1 penalty (alpha=1.) and L2 penalty (alpha=0.)
        term of the loss function.
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
    cv: int
        number of cross validation repeats
        default: 10
    solver: str
        optimization method, can be one of the following
        'batch-gradient' (vanilla batch gradient descent)
        'cdfast' (Newton coordinate gradient descent).
        default: 'batch-gradient'
    learning_rate: float
        learning rate for gradient descent.
        default: 2e-1
    max_iter: int
        maximum number of iterations for the solver.
        default: 1000
    tol: float
        convergence threshold or stopping criteria.
        Optimization loop will stop when relative change
        in parameter norm is below the threshold.
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
    theta: float
        shape parameter of the negative binomial distribution (number of
        successes before the first failure). It is used only if distr is
        equal to neg-binomial, otherwise it is ignored.
        default: 1.0
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
                 random_state=0, theta=1.0, verbose=False):

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
        self.theta = theta
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

        cv_training_iterations = _verbose_iterable(self.reg_lambda)

        for idx, rl in enumerate(cv_training_iterations):
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
                      theta=self.theta,
                      score_metric=self.score_metric,
                      fit_intercept=self.fit_intercept,
                      random_state=self.random_state,
                      verbose=self.verbose)
            _tqdm_log('Lambda: %6.4f' % rl)
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
                scores_fold.append(glm.score(X[val], y[val]))
            scores.append(np.mean(scores_fold))

            if idx == 0:
                glm.beta0_, glm.beta_ = self.beta0_, self.beta_
            else:
                glm.beta0_, glm.beta_ = glms[-1].beta0_, glms[-1].beta_

            glm.n_iter_ = 0
            glm.fit(X, y)
            glms.append(glm)

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
