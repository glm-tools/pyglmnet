from functools import partial

import numpy as np
from numpy.testing import assert_allclose

import scipy.sparse as sps
from scipy.optimize import approx_fprime

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score

from nose.tools import assert_true, assert_equal, assert_raises

from pyglmnet import (GLM, GLMCV, _grad_L2loss, _L2loss, simulate_glm,
                      _gradhess_logloss_1d, _loss, datasets)


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

    lams = [0.5, 0.3237394, 0.2096144, 0.13572088, 0.08787639,
            0.0568981, 0.03684031, 0.02385332, 0.01544452, 0.01]
    alpha = 1.0

    for lam in lams:
        # create an instance of the GLM class
        glm_group = GLM(distr='softplus', alpha=alpha, reg_lambda=lam, group=groups)

        # simulate training data
        np.random.seed(0)
        Xr = np.random.normal(0.0, 1.0, [n_samples, n_features])
        yr = simulate_glm(glm_group.distr, beta0, beta, Xr)

        # scale and fit
        scaler = StandardScaler().fit(Xr)
        glm_group.fit(scaler.transform(Xr), yr)

        # count number of nonzero coefs for each group.
        # in each group, coef must be [all nonzero] or [all zero].
        unique_group_idxs = np.unique(groups)
        beta = glm_group.beta_
        group_norms = np.abs(beta)
        for target_group_idx in unique_group_idxs:
            if target_group_idx == 0:
                continue

            target_beta = beta[groups == target_group_idx]
            n_nonzero = (target_beta != 0.0).sum()
            assert n_nonzero in (len(target_beta), 0)
            group_norms[groups == target_group_idx] = np.linalg.norm(beta[groups == target_group_idx], 2)

        # beta where those absolute values are smaller than the threshold must be 0.
        thresh = lam * alpha
        assert (beta[group_norms <= thresh] == 0.0).all()


def test_glmnet():
    """Test glmnet."""
    assert_raises(ValueError, GLM, distr='blah')
    assert_raises(ValueError, GLM, distr='gaussian', max_iter=1.8)

    n_samples, n_features = 100, 10

    # coefficients
    beta0 = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0)
    beta = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0, (n_features,))

    distrs = ['softplus', 'gaussian', 'poisson', 'binomial']  # , 'probit']
    solvers = ['batch-gradient', 'cdfast']
    score_metric = 'pseudo_R2'
    learning_rate = 2e-1

    for distr in distrs:
        betas_ = list()
        for solver in solvers:

            glm = GLM(distr, learning_rate=learning_rate,
                      reg_lambda=0., tol=1e-7, max_iter=5000,
                      alpha=0., solver=solver, score_metric=score_metric)
            assert_true(repr(glm))

            np.random.seed(glm.random_state)
            X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
            y_train = simulate_glm(glm.distr, beta0, beta, X_train,
                                   sample=False)

            glm.fit(X_train, y_train)
            assert_true(np.all(np.diff(glm._loss) <= 1e-7))  # loss decreases

            # verify loss at convergence = loss when beta=beta_
            l_true = _loss(distr, 0., np.eye(beta.shape[0]), 0.,
                           X_train, y_train, 2.0, None,
                           np.concatenate(([beta0], beta)))
            assert_allclose(glm._loss[-1], l_true, rtol=1e-4, atol=1e-7)
            # beta=beta_ when reg_lambda = 0.
            assert_allclose(beta, glm.beta_, rtol=0.05, atol=1e-2)
            betas_.append(glm.beta_)

            y_pred = glm.predict(X_train)
            assert_equal(y_pred.shape[0], X_train.shape[0])

        # compare all solvers pairwise to make sure they're close
        for i, first_beta in enumerate(betas_[:-1]):
            for second_beta in betas_[i + 1:]:
                assert_allclose(first_beta, second_beta, rtol=0.05, atol=1e-2)

    # test fit_predict
    glm_poisson = GLM(distr='softplus')
    glm_poisson.fit_predict(X_train, y_train)
    assert_raises(ValueError, glm_poisson.fit_predict,
                  X_train[None, ...], y_train)


def test_glmcv():
    """Test GLMCV class."""
    assert_raises(ValueError, GLM, distr='blah')
    assert_raises(ValueError, GLM, distr='gaussian', max_iter=1.8)

    scaler = StandardScaler()
    n_samples, n_features = 100, 10

    # coefficients
    beta0 = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0)
    beta = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0, (n_features,))

    distrs = ['softplus', 'gaussian', 'poisson', 'binomial', 'gamma']
    # XXX: 'probit'
    solvers = ['batch-gradient', 'cdfast']
    score_metric = 'pseudo_R2'
    learning_rate = 2e-1

    for solver in solvers:
        for distr in distrs:

            if distr == 'gamma' and solver == 'cdfast':
                continue

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


def test_cv():
    """Simple CV check."""
    # XXX: don't use scikit-learn for tests.
    X, y = make_regression()
    cv = KFold(X.shape[0], 5)

    glm_normal = GLM(distr='gaussian', alpha=0.01, reg_lambda=0.1)
    # check that it returns 5 scores
    scores = cross_val_score(glm_normal, X, y, cv=cv)
    assert_equal(len(scores), 5)

    param_grid = [{'alpha': np.linspace(0.01, 0.99, 2)},
                  {'reg_lambda': np.logspace(np.log(0.5), np.log(0.01),
                                             10, base=np.exp(1))}]
    glmcv = GridSearchCV(glm_normal, param_grid, cv=cv)
    glmcv.fit(X, y)


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
        gk, hk = _gradhess_logloss_1d(glm.distr, xk, y, z, glm.eta)

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


def test_fetch_datasets():
    """Test fetching datasets."""
    datasets.fetch_community_crime_data('/tmp/glm-tools')
