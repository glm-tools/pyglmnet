"""Tests for distributions."""

from functools import partial

import pytest

import numpy as np
from numpy.testing import assert_allclose

import scipy.sparse as sps
from scipy.optimize import approx_fprime
from sklearn.preprocessing import StandardScaler

from pyglmnet import ALLOWED_DISTRS, simulate_glm, GLM, _grad_L2loss, _L2loss
from pyglmnet.distributions import BaseDistribution


def test_base_distribution():
    """Test the base distribution."""
    class TestDistr(BaseDistribution):
        def mu():
            pass

        def grad_mu():
            pass

    class TestDistr2(TestDistr):
        def log_likelihood():
            pass

    with pytest.raises(TypeError, match='abstract methods log_likelihood'):
        distr = TestDistr()

    msg = 'Gradients of log likelihood are not specified'
    with pytest.raises(NotImplementedError, match=msg):
        distr = TestDistr2()
        distr.grad_log_likelihood()

    msg = 'distr must be one of'
    with pytest.raises(ValueError, match=msg):
        GLM(distr='blah')


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
    glm._set_distr()
    y = simulate_glm(glm.distr, beta_[0], beta_[1:], X)

    func = partial(_L2loss, distr, glm.alpha,
                   glm.Tau, reg_lambda, X, y, glm.eta, glm.theta, glm.group)
    grad = partial(_grad_L2loss, glm.distr_, glm.alpha, glm.Tau,
                   reg_lambda, X, y,
                   glm.eta, glm.theta)
    approx_grad = approx_fprime(beta_, func, 1.5e-8)
    analytical_grad = grad(beta_)
    assert_allclose(approx_grad, analytical_grad, rtol=1e-5, atol=1e-3)
