"""Tests for metrics."""

import numpy as np
from pyglmnet import GLM, simulate_glm
from pyglmnet.metrics import deviance, pseudo_R2


def test_deviance():
    """Test deviance."""
    n_samples, n_features = 1000, 100

    beta0 = np.random.rand()
    beta = np.random.normal(0.0, 1.0, n_features)

    # sample train and test data
    glm_sim = GLM(score_metric='deviance')
    X = np.random.randn(n_samples, n_features)
    y = simulate_glm(glm_sim.distr, beta0, beta, X)

    glm_sim.fit(X, y)
    score = glm_sim.score(X, y)

    assert(isinstance(score, float))


def test_pseudoR2():
    """Test pseudo r2."""
    n_samples, n_features = 1000, 100

    beta0 = np.random.rand()
    beta = np.random.normal(0.0, 1.0, n_features)

    # sample train and test data
    glm_sim = GLM(score_metric='pseudo_R2')
    X = np.random.randn(n_samples, n_features)
    y = simulate_glm(glm_sim.distr, beta0, beta, X)

    glm_sim.fit(X, y)
    score = glm_sim.score(X, y)

    assert(isinstance(score, float))


def test_accuracy():
    """Testing accuracy."""
    n_samples, n_features, n_classes = 1000, 100, 2

    beta0 = np.random.rand()
    betas = np.random.normal(0.0, 1.0, (n_features, n_classes))

    # sample train and test data
    glm_sim = GLM(distr='binomial', score_metric='accuracy')
    X = np.random.randn(n_samples, n_features)
    y = np.zeros((n_samples, 2))
    for idx, beta in enumerate(betas.T):
        y[:, idx] = simulate_glm(glm_sim.distr, beta0, beta, X)
    y = np.argmax(y, axis=1)
    glm_sim.fit(X, y)
    score = glm_sim.score(X, y)

    assert(isinstance(score, float))


def test_bernoulli_loglik_endpoints():
    """Bernoulli log likelihood is finite at probabilities 0 and 1."""
    from pyglmnet.pyglmnet import _logL
    y = np.array([0., 1., 0., 1.])
    for distr in ['binomial', 'probit']:
        # perfect predictions at the endpoints: log likelihood is 0
        assert _logL(distr, y, y.copy()) == 0.
        assert deviance(y, y.copy(), distr, theta=1.0) == 0.
        # integer probabilities are not wrapped by the negation
        assert _logL(distr, np.array([0.]),
                     np.array([1], dtype=np.uint8)) == -np.inf
    # large logits must not overflow
    z = np.array([-1000., 1000., -1000., 1000.])
    assert _logL('binomial', y, None, z=z) == 0.


def test_gamma_deviance_saturated():
    """Gamma deviance is 0 for perfect predictions and matches the formula."""
    y = np.array([0.1, 0.2, 0.4])
    assert np.isclose(deviance(y, y.copy(), 'gamma', theta=1.0), 0.)
    mu = np.array([0.15, 0.25, 0.3])
    expected = 2 * np.sum((y - mu) / mu - np.log(y / mu))
    assert np.isclose(deviance(y, mu, 'gamma', theta=1.0), expected)
    # unchanged when y and mu are rescaled together
    assert np.isclose(deviance(10 * y, 10 * mu, 'gamma', theta=1.0), expected)
    # pseudo-R2 is 1 - D / D_null: 1 for a perfect fit, 0 for the null model
    X = np.zeros((3, 1))
    ynull = np.mean(y)
    d_null = deviance(y, np.full(3, ynull), 'gamma', theta=1.0)
    assert np.isclose(pseudo_R2(X, y, y.copy(), ynull, 'gamma', 1.0), 1.)
    assert np.isclose(pseudo_R2(X, y, np.full(3, ynull), ynull, 'gamma', 1.0),
                      0.)
    r2 = pseudo_R2(X, y, mu, ynull, 'gamma', 1.0)
    assert np.isclose(r2, 1 - expected / d_null)
    assert np.isclose(pseudo_R2(X, 10 * y, 10 * mu, 10 * ynull, 'gamma', 1.0),
                      r2)
