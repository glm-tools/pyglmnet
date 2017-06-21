from functools import partial

import numpy as np
from numpy.testing import assert_allclose

import scipy.sparse as sps
from scipy.optimize import approx_fprime

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

from nose.tools import assert_true, assert_equal, assert_raises

from pyglmnet import GLM, GLMCV, _grad_L2loss, _L2loss, simulate_glm


def test_gradients():
    """Test gradient accuracy."""
    # data
    scaler = StandardScaler()
    n_samples, n_features = 1000, 100
    X = np.random.normal(0.0, 1.0, [n_samples, n_features])
    X = scaler.fit_transform(X)

    density = 0.1
    beta_ = np.zeros(n_features + 1)
    beta_[0] = np.random.rand()
    beta_[1:] = sps.rand(n_features, 1, density=density).toarray()[:, 0]

    reg_lambda = 0.1
    distrs = ['gaussian', 'binomial', 'softplus', 'poisson', 'probit', 'gamma']
    for distr in distrs:
        glm = GLM(distr=distr, reg_lambda=reg_lambda)
        y = simulate_glm(glm.distr, beta_[0], beta_[1:], X)

        func = partial(_L2loss, distr, glm.alpha,
                       glm.Tau, reg_lambda, X, y, glm.eta, glm.group)
        grad = partial(_grad_L2loss, distr, glm.alpha, glm.Tau,
                       reg_lambda, X, y,
                       glm.eta)
        approx_grad = approx_fprime(beta_, func, 1.5e-8)
        analytical_grad = grad(beta_)
        assert_allclose(approx_grad, analytical_grad, rtol=1e-5, atol=1e-3)


def test_tikhonov():
    """Tikhonov regularization test."""
    n_samples, n_features = 100, 10

    # design covariance matrix of parameters
    Gam = 15.
    PriorCov = np.zeros([n_features, n_features])
    for i in np.arange(0, n_features):
        for j in np.arange(i, n_features):
            PriorCov[i, j] = np.exp(-Gam * 1. / (np.float(n_features) ** 2) *
                                    (np.float(i) - np.float(j)) ** 2)
            PriorCov[j, i] = PriorCov[i, j]
            if i == j:
                PriorCov[i, j] += 0.01
    PriorCov = 1. / np.max(PriorCov) * PriorCov

    # sample parameters as multivariate normal
    beta0 = np.random.randn()
    beta = np.random.multivariate_normal(np.zeros(n_features), PriorCov)

    # sample train and test data
    glm_sim = GLM(distr='softplus', score_metric='pseudo_R2')
    X = np.random.randn(n_samples, n_features)
    y = simulate_glm(glm_sim.distr, beta0, beta, X)

    from sklearn.cross_validation import train_test_split
    Xtrain, Xtest, ytrain, ytest = \
        train_test_split(X, y, test_size=0.5, random_state=42)

    # design tikhonov matrix
    [U, S, V] = np.linalg.svd(PriorCov, full_matrices=False)
    Tau = np.dot(np.diag(1. / np.sqrt(S)), U)
    Tau = 1. / np.sqrt(np.float(n_samples)) * Tau / Tau.max()

    # fit model with batch gradient
    glm_tikhonov = GLM(distr='softplus',
                       alpha=0.0,
                       Tau=Tau,
                       solver='batch-gradient',
                       tol=1e-5,
                       score_metric='pseudo_R2')
    glm_tikhonov.fit(Xtrain, ytrain)

    R2_train, R2_test = dict(), dict()
    R2_train['tikhonov'] = glm_tikhonov.score(Xtrain, ytrain)
    R2_test['tikhonov'] = glm_tikhonov.score(Xtest, ytest)

    # fit model with cdfast
    glm_tikhonov = GLM(distr='softplus',
                       alpha=0.0,
                       Tau=Tau,
                       solver='cdfast',
                       tol=1e-5,
                       score_metric='pseudo_R2')
    glm_tikhonov.fit(Xtrain, ytrain)

    R2_train, R2_test = dict(), dict()
    R2_train['tikhonov'] = glm_tikhonov.score(Xtrain, ytrain)
    R2_test['tikhonov'] = glm_tikhonov.score(Xtest, ytest)


def test_group_lasso():
    """Group Lasso test."""
    n_samples, n_features = 100, 90

    # assign group ids
    groups = np.zeros(90)
    groups[0:29] = 1
    groups[30:59] = 2
    groups[60:] = 3

    # sample random coefficients
    beta0 = np.random.normal(0.0, 1.0, 1)
    beta = np.random.normal(0.0, 1.0, n_features)
    beta[groups == 2] = 0.

    # create an instance of the GLM class
    glm_group = GLM(distr='softplus', alpha=1.)

    # simulate training data
    Xr = np.random.normal(0.0, 1.0, [n_samples, n_features])
    yr = simulate_glm(glm_group.distr, beta0, beta, Xr)

    # scale and fit
    scaler = StandardScaler().fit(Xr)
    glm_group.fit(scaler.transform(Xr), yr)


def test_glmnet():
    """Test glmnet."""
    scaler = StandardScaler()
    n_samples, n_features = 100, 10

    # coefficients
    beta0 = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0)
    beta = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0, (n_features,))

    distrs = ['softplus', 'gaussian', 'poisson', 'binomial', 'probit']
    solvers = ['batch-gradient', 'cdfast']
    score_metric = 'pseudo_R2'
    learning_rate = 2e-1

    for solver in solvers:
        for distr in distrs:

            glm = GLM(distr, learning_rate=learning_rate,
                      solver=solver, score_metric=score_metric)

            assert_true(repr(glm))

            np.random.seed(glm.random_state)
            X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
            y_train = simulate_glm(glm.distr, beta0, beta, X_train)

            X_train = scaler.fit_transform(X_train)
            glm.fit(X_train, y_train)

            beta_ = glm.beta_
            assert_allclose(beta, beta_, atol=0.5)  # check fit

            y_pred = glm.predict(scaler.transform(X_train))
            assert_equal(y_pred.shape[0], X_train.shape[0])

    # test fit_predict
    glm_poisson = GLM(distr='softplus')
    glm_poisson.fit_predict(X_train, y_train)
    assert_raises(ValueError, glm_poisson.fit_predict,
                  X_train[None, ...], y_train)


def test_glmcv():
    """Test GLMCV class."""
    scaler = StandardScaler()
    n_samples, n_features = 100, 10

    # coefficients
    beta0 = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0)
    beta = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0, (n_features,))

    distrs = ['softplus', 'gaussian', 'poisson', 'binomial', 'probit']
    solvers = ['batch-gradient', 'cdfast']
    score_metric = 'pseudo_R2'
    learning_rate = 2e-1

    for solver in solvers:
        for distr in distrs:

            glm = GLMCV(distr, learning_rate=learning_rate,
                        solver=solver, score_metric=score_metric)

            assert_true(repr(glm))

            np.random.seed(glm.random_state)
            X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
            y_train = simulate_glm(glm.distr, beta0, beta, X_train)

            X_train = scaler.fit_transform(X_train)
            glm.fit(X_train, y_train)

            beta_ = glm.beta_
            assert_allclose(beta, beta_, atol=0.5)  # check fit

            y_pred = glm.predict(scaler.transform(X_train))
            assert_equal(y_pred.shape[0], X_train.shape[0])

def simple_cv_scorer(obj, X, y):
    """Simple scorer takes average pseudo-R2 from regularization path."""
    yhats = obj.predict(X)
    return np.mean([obj.score(X, y) for yhat in yhats])


def test_cv():
    """Simple CV check."""
    # XXX: don't use scikit-learn for tests.
    X, y = make_regression()

    glm_normal = GLM(distr='gaussian', alpha=0.01,
                     reg_lambda=0.1)
    glm_normal.fit(X, y)

    cv = KFold(X.shape[0], 5)

    # check that it returns 5 scores
    assert_equal(len(cross_val_score(glm_normal, X, y, cv=cv,
                 scoring=simple_cv_scorer)), 5)


def test_cdfast():
    """Test all functionality related to fast coordinate descent"""
    scaler = StandardScaler()
    n_samples = 1000
    n_features = 100
    n_classes = 5
    density = 0.1

    distrs = ['softplus', 'gaussian', 'binomial', 'poisson', 'probit']
    for distr in distrs:
        glm = GLM(distr, solver='cdfast')

        np.random.seed(glm.random_state)

        # coefficients
        beta0 = np.random.rand()
        beta = sps.rand(n_features, 1, density=density).toarray()[:, 0]
        # data
        X = np.random.normal(0.0, 1.0, [n_samples, n_features])
        X = scaler.fit_transform(X)
        y = simulate_glm(glm.distr, beta0, beta, X)

        # compute grad and hess
        beta_ = np.zeros((n_features + 1,))
        beta_[0] = beta0
        beta_[1:] = beta
        z = beta_[0] + np.dot(X, beta_[1:])
        k = 1
        xk = X[:, k - 1]
        gk, hk = glm._gradhess_logloss_1d(xk, y, z)

        # test grad and hess
        if distr != 'multinomial':
            assert_equal(np.size(gk), 1)
            assert_equal(np.size(hk), 1)
            assert_true(isinstance(gk, float))
            assert_true(isinstance(hk, float))
        else:
            assert_equal(gk.shape[0], n_classes)
            assert_equal(hk.shape[0], n_classes)
            assert_true(isinstance(gk, np.ndarray))
            assert_true(isinstance(hk, np.ndarray))
            assert_equal(gk.ndim, 1)
            assert_equal(hk.ndim, 1)

        # test cdfast
        ActiveSet = np.ones(n_features + 1)
        beta_ret, z_ret = glm._cdfast(X, y, z,
                                      ActiveSet, beta_, glm.reg_lambda)
        assert_equal(beta_ret.shape, beta_.shape)
        assert_equal(z_ret.shape, z.shape)
