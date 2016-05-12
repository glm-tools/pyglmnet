import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_allclose

from pyglmnet import GLM


def test_glmnet():
    """Test glmnet."""
    scaler = StandardScaler()
    n_samples, n_features = 10000, 100
    density = 0.1

    # coefficients
    beta0 = np.random.rand()
    beta = sps.rand(n_features, 1, density=density).toarray()

    distrs = ['poisson', 'poissonexp', 'normal', 'binomial']
    for distr in distrs:

        glm = GLM(distr)
        n_lambda = 10
        # FIXME: why do we need such lambda & learning rate?
        if distr == 'poissonexp':
            learning_rate, reg_lambda = 1e-5, [0.05, 0.01]
            n_lambda = 2
            glm = GLM(distr=distr, learning_rate=learning_rate,
                      reg_lambda=reg_lambda)

        assert_true(repr(glm))

        X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
        y_train = glm.simulate(beta0, beta, X_train)

        X_train = scaler.fit_transform(X_train)
        glm.fit(X_train, y_train)

        beta_ = glm.fit_[-2]['beta'][:]
        assert_allclose(beta[:], beta_, atol=0.1)  # check fit
        density_ = np.sum(beta_ > 0.1) / float(n_features)
        assert_allclose(density_, density, atol=0.05)  # check density

        y_pred = glm.predict(scaler.transform(X_train))
        assert_equal(y_pred.shape, (n_lambda, X_train.shape[0]))

    # checks for slicing.
    glm = glm[:3]
    glm_copy = glm.copy()
    assert_true(glm_copy is not glm)
    assert_equal(len(glm.reg_lambda), 3)
    y_pred = glm[:2].predict(scaler.transform(X_train))
    assert_equal(y_pred.shape, (2, X_train.shape[0]))
    y_pred = glm[2].predict(scaler.transform(X_train))
    assert_equal(y_pred.shape, (X_train.shape[0], ))
    assert_raises(IndexError, glm.__getitem__, [2])
    glm.deviance(y_train, y_pred)

    # don't allow slicing if model has not been fit yet.
    glm = GLM(distr='poisson')
    assert_raises(ValueError, glm.__getitem__, 2)

    # test fit_predict
    glm.fit_predict(X_train, y_train)
    assert_raises(ValueError, glm.fit_predict, X_train[None, ...], y_train)

def test_multinomial_gradient():
    """Gradient of intercept params is different"""
    glm = GLM(distr='multinomial')
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([0, 1])
    beta = np.zeros([4, 2])
    grad_beta0, grad_beta = glm.grad_L2loss(beta[0], beta[1:], 0, X, y)
    glm.fit(X, y)
    y_pred = glm.predict(X)
    assert_equal(y_pred.shape, (10, 2, 2))  # n_classes x n_samples x n_classes
    assert grad_beta0[0] != grad_beta0[1]
