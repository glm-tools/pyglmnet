import subprocess
import os.path as op

from functools import partial
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pytest import raises

import scipy.sparse as sps
from scipy.optimize import approx_fprime

from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import ElasticNet

from pyglmnet import (GLM, GLMCV, _grad_L2loss, _L2loss, simulate_glm,
                      _gradhess_logloss_1d, _loss, datasets, ALLOWED_DISTRS)


@pytest.mark.parametrize("distr", ALLOWED_DISTRS)
def test_gradients(distr):
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

    from sklearn.model_selection import train_test_split
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
                       tol=1e-3,
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
                       tol=1e-3,
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
    beta0 = np.random.rand()
    beta = np.random.normal(0.0, 1.0, n_features)
    beta[groups == 2] = 0.

    # create an instance of the GLM class
    glm_group = GLM(distr='softplus', alpha=1., reg_lambda=0.2, group=groups)

    # simulate training data
    np.random.seed(glm_group.random_state)
    Xr = np.random.normal(0.0, 1.0, [n_samples, n_features])
    yr = simulate_glm(glm_group.distr, beta0, beta, Xr)

    # scale and fit
    scaler = StandardScaler().fit(Xr)
    glm_group.fit(scaler.transform(Xr), yr)

    # count number of nonzero coefs for each group.
    # in each group, coef must be [all nonzero] or [all zero].
    beta = glm_group.beta_
    group_ids = np.unique(groups)
    for group_id in group_ids:
        if group_id == 0:
            continue

        target_beta = beta[groups == group_id]
        n_nonzero = (target_beta != 0.0).sum()
        assert n_nonzero in (len(target_beta), 0)

    # one of the groups must be [all zero]
    assert np.any([beta[groups == group_id].sum() == 0
                   for group_id in group_ids if group_id != 0])


@pytest.mark.parametrize("distr", ALLOWED_DISTRS)
@pytest.mark.parametrize("reg_lambda", [0.0, 0.1])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_glmnet(distr, reg_lambda, fit_intercept):
    """Test glmnet."""
    raises(ValueError, GLM, distr='blah')
    raises(ValueError, GLM, distr='gaussian', max_iter=1.8)

    n_samples, n_features = 100, 10

    # coefficients
    beta0 = 0.
    if fit_intercept:
        beta0 = 1. / (np.float(n_features) + 1.) * \
            np.random.normal(0.0, 1.0)
    beta = 1. / (np.float(n_features) + int(fit_intercept)) * \
        np.random.normal(0.0, 1.0, (n_features,))

    solvers = ['batch-gradient', 'cdfast']

    score_metric = 'pseudo_R2'
    learning_rate = 2e-1
    random_state = 0

    betas_ = list()
    for solver in solvers:

        if distr == 'gamma' and solver == 'cdfast':
            continue

        np.random.seed(random_state)

        X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
        y_train = simulate_glm(distr, beta0, beta, X_train,
                               sample=False)

        alpha = 0.
        loss_trace = list()
        eta = 2.0
        group = None
        Tau = None

        def callback(beta):
            Tau = None
            loss_trace.append(
                _loss(distr, alpha, Tau, reg_lambda,
                      X_train, y_train, eta, group, beta,
                      fit_intercept=fit_intercept))

        glm = GLM(distr, learning_rate=learning_rate,
                  reg_lambda=reg_lambda, tol=1e-5, max_iter=5000,
                  alpha=alpha, solver=solver, score_metric=score_metric,
                  random_state=random_state, callback=callback,
                  fit_intercept=fit_intercept)
        assert(repr(glm))

        glm.fit(X_train, y_train)

        # verify loss decreases
        assert(np.all(np.diff(loss_trace) <= 1e-7))

        # true loss and beta should be recovered when reg_lambda == 0
        if reg_lambda == 0.:
            # verify loss at convergence = loss when beta=beta_
            l_true = _loss(distr, alpha, Tau, reg_lambda,
                           X_train, y_train, eta, group,
                           np.concatenate(([beta0], beta)))
            assert_allclose(loss_trace[-1], l_true, rtol=1e-4, atol=1e-5)
            # beta=beta_ when reg_lambda = 0.
            assert_allclose(beta, glm.beta_, rtol=0.05, atol=1e-2)
        betas_.append(glm.beta_)

        y_pred = glm.predict(X_train)
        assert(y_pred.shape[0] == X_train.shape[0])

    # compare all solvers pairwise to make sure they're close
    for i, first_beta in enumerate(betas_[:-1]):
        for second_beta in betas_[i + 1:]:
            assert_allclose(first_beta, second_beta, rtol=0.05, atol=1e-2)

    # test fit_predict
    glm_poisson = GLM(distr='softplus')
    glm_poisson.fit_predict(X_train, y_train)
    raises(ValueError, glm_poisson.fit_predict, X_train[None, ...], y_train)


@pytest.mark.parametrize("distr", ALLOWED_DISTRS)
def test_glmcv(distr):
    """Test GLMCV class."""
    raises(ValueError, GLM, distr='blah')
    raises(ValueError, GLM, distr='gaussian', max_iter=1.8)

    scaler = StandardScaler()
    n_samples, n_features = 100, 10

    # coefficients
    beta0 = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0)
    beta = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0, (n_features,))

    solvers = ['batch-gradient', 'cdfast']
    score_metric = 'pseudo_R2'
    learning_rate = 2e-1

    for solver in solvers:

        if distr == 'gamma' and solver == 'cdfast':
            continue

        glm = GLMCV(distr, learning_rate=learning_rate,
                    solver=solver, score_metric=score_metric, cv=2)

        assert(repr(glm))

        np.random.seed(glm.random_state)
        X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
        y_train = simulate_glm(glm.distr, beta0, beta, X_train)

        X_train = scaler.fit_transform(X_train)
        glm.fit(X_train, y_train)

        beta_ = glm.beta_
        assert_allclose(beta, beta_, atol=0.5)  # check fit

        y_pred = glm.predict(scaler.transform(X_train))
        assert(y_pred.shape[0] == X_train.shape[0])

    # test picky score_metric check within fit().
    glm.score_metric = 'bad_score_metric'  # reuse last glm
    raises(ValueError, glm.fit, X_train, y_train)


def test_cv():
    """Simple CV check."""
    # XXX: don't use scikit-learn for tests.
    X, y = make_regression()
    cv = KFold(n_splits=5)

    glm_normal = GLM(distr='gaussian', alpha=0.01, reg_lambda=0.1)
    # check that it returns 5 scores
    scores = cross_val_score(glm_normal, X, y, cv=cv)
    assert(len(scores) == 5)

    param_grid = [{'alpha': np.linspace(0.01, 0.99, 2)},
                  {'reg_lambda': np.logspace(np.log(0.5), np.log(0.01),
                                             10, base=np.exp(1))}]
    glmcv = GridSearchCV(glm_normal, param_grid, cv=cv)
    glmcv.fit(X, y)


@pytest.mark.parametrize("solver", ['batch-gradient', 'cdfast'])
def test_compare_sklearn(solver):
    """Test results against sklearn."""
    def rmse(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    X, Y, coef_ = make_regression(
        n_samples=1000, n_features=1000,
        noise=0.1, n_informative=10, coef=True,
        random_state=42)

    alpha = 0.1
    l1_ratio = 0.5

    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-5)
    clf.fit(X, Y)
    glm = GLM(distr='gaussian', alpha=l1_ratio, reg_lambda=alpha,
              solver=solver, tol=1e-5, max_iter=70)
    glm.fit(X, Y)

    y_sk = clf.predict(X)
    y_pg = glm.predict(X)
    assert abs(rmse(Y, y_sk) - rmse(Y, y_pg)) < 1.0

    glm = GLM(distr='gaussian', alpha=l1_ratio, reg_lambda=alpha,
              solver=solver, tol=1e-5, max_iter=5, fit_intercept=False)
    glm.fit(X, Y)
    assert glm.beta0_ == 0.

    glm.predict(X)


@pytest.mark.parametrize("distr", ALLOWED_DISTRS)
def test_cdfast(distr):
    """Test all functionality related to fast coordinate descent."""
    scaler = StandardScaler()
    n_samples = 1000
    n_features = 100
    n_classes = 5
    density = 0.1

    # Batch gradient not available for gamma
    if distr == 'gamma':
        return

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
        assert(np.size(gk) == 1)
        assert(np.size(hk) == 1)
        assert(isinstance(gk, float))
        assert(isinstance(hk, float))
    else:
        assert(gk.shape[0] == n_classes)
        assert(hk.shape[0] == n_classes)
        assert(isinstance(gk, np.ndarray))
        assert(isinstance(hk, np.ndarray))
        assert(gk.ndim == 1)
        assert(hk.ndim == 1)

    # test cdfast
    ActiveSet = np.ones(n_features + 1)
    beta_ret = glm._cdfast(X, y, ActiveSet, beta_, glm.reg_lambda)
    assert(beta_ret.shape == beta_.shape)
    assert(True not in np.isnan(beta_ret))


def test_fetch_datasets():
    """Test fetching datasets."""
    datasets.fetch_community_crime_data()


def test_random_state_consistency():
    """Test model's random_state."""
    # Generate the dataset
    n_samples, n_features = 1000, 10

    beta0 = 1. / (np.float(n_features) + 1.) * np.random.normal(0.0, 1.0)
    beta = 1. / (np.float(n_features) + 1.) * \
        np.random.normal(0.0, 1.0, (n_features,))
    Xtrain = np.random.normal(0.0, 1.0, [n_samples, n_features])

    ytrain = simulate_glm("gaussian", beta0, beta, Xtrain,
                          sample=False, random_state=42)

    # Test simple glm
    glm_a = GLM(distr="gaussian", random_state=1)
    ypred_a = glm_a.fit_predict(Xtrain, ytrain)
    glm_b = GLM(distr="gaussian", random_state=1)
    ypred_b = glm_b.fit_predict(Xtrain, ytrain)
    match = "This glm object has already been fit"
    with pytest.raises(ValueError, match=match):
        ypred_c = glm_b.fit_predict(Xtrain, ytrain)

    # Consistency between two different models
    assert_array_equal(ypred_a, ypred_b)

    # Test also cross-validation
    glm_cv_a = GLMCV(distr="gaussian", cv=3, random_state=1)
    ypred_a = glm_cv_a.fit_predict(Xtrain, ytrain)
    glm_cv_b = GLMCV(distr="gaussian", cv=3, random_state=1)
    ypred_b = glm_cv_b.fit_predict(Xtrain, ytrain)
    ypred_c = glm_cv_b.fit_predict(Xtrain, ytrain)

    assert_array_equal(ypred_a, ypred_b)
    assert_array_equal(ypred_b, ypred_c)


@pytest.mark.parametrize("distr", ALLOWED_DISTRS)
def test_simulate_glm(distr):
    """Test that every generative model can be simulated from."""

    random_state = 1
    state = np.random.RandomState(random_state)
    n_samples, n_features = 10, 3

    # sample random coefficients
    beta0 = state.rand()
    beta = state.normal(0.0, 1.0, n_features)

    X = state.normal(0.0, 1.0, [n_samples, n_features])
    simulate_glm(distr, beta0, beta, X, random_state=random_state)

    with pytest.raises(ValueError, match="'beta0' must be float"):
        simulate_glm(distr, np.array([1.0]), beta, X, random_state)

    with pytest.raises(ValueError, match="'beta' must be 1D"):
        simulate_glm(distr, 1.0, np.atleast_2d(beta), X, random_state)

    # If the distribution name is garbage it will fail
    distr = 'multivariate_gaussian_poisson'
    with pytest.raises(ValueError, match="'distr' must be in"):
        simulate_glm(distr, 1.0, 1.0, np.array([[1.0]]))


def test_api_input():
    """Test that the input value of y can be of different types."""

    random_state = 1
    state = np.random.RandomState(random_state)
    n_samples, n_features = 100, 5

    X = state.normal(0, 1, (n_samples, n_features))
    y = state.normal(0, 1, (n_samples, ))

    glm = GLM(distr='gaussian')

    # Test that a list will not work - the types have to be ndarray
    with pytest.raises(ValueError):
        glm.fit(X, list(y))

    # Test that ValueError is raised when the shapes mismatch
    with pytest.raises(ValueError):
        GLM().fit(X, y[3:])

    # This would work without errors
    glm.fit(X, y)
    glm.predict(X)
    glm.score(X, y)
    glm = GLM(distr='gaussian', solver='test')

    with pytest.raises(ValueError, match="solver must be one of"):
        glm.fit(X, y)

    with pytest.raises(ValueError, match="fit_intercept must be"):
        glm = GLM(distr='gaussian', fit_intercept='blah')

    glm = GLM(distr='gaussian', max_iter=2)
    with pytest.warns(UserWarning, match='Reached max number of iterat'):
        glm.fit(X, y)


def test_intro_example():
    """Test that the intro example works."""
    base, _ = op.split(op.realpath(__file__))
    fname = op.join(base, '..', 'README.rst')

    start_idx = 0  # where does Python code start?
    code_lines = []
    for idx, line in enumerate(open(fname, "r")):
        if '.. code:: python' in line:
            start_idx = idx
        if start_idx > 0 and idx >= start_idx + 2:
            if line.startswith('`More'):
                break
            code_lines.append(line.strip())
    subprocess.run(['python', '-c', '\n'.join(code_lines)], check=True)
