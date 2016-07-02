import numpy as np
import scipy.sparse as sps
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from nose.tools import assert_true, assert_equal, assert_raises
from numpy.testing import assert_allclose

from pyglmnet import GLM


def test_glmnet():
    """Test glmnet."""
    scaler = StandardScaler()
    n_samples, n_features = 10000, 100
    density = 0.1
    n_lambda = 10

    # coefficients
    beta0 = np.random.rand()
    beta = sps.rand(n_features, 1, density=density).toarray()

    distrs = ['poisson', 'poissonexp', 'normal', 'binomial']
    learning_rates = {'poisson': 2e-1, 'poissonexp': 1e-1, 'normal': 1e-2, 'binomial': 2e-1}
    for distr in distrs:

        # FIXME: why do we need such different learning rates?
        learning_rate = learning_rates[distr]
        glm = GLM(distr, learning_rate=learning_rate)

        assert_true(repr(glm))

        np.random.seed(glm.random_state)
        X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
        y_train = glm.simulate(beta0, beta, X_train)

        X_train = scaler.fit_transform(X_train)
        glm.fit(X_train, y_train)

        beta_ = glm.fit_[-2]['beta'][:]
        assert_allclose(beta[:], beta_, atol=0.5)  # check fit
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
    glm.general_score(y_train, y_pred)

    # don't allow slicing if model has not been fit yet.
    glm_poisson = GLM(distr='poisson')
    assert_raises(ValueError, glm_poisson.__getitem__, 2)

    # test fit_predict
    glm_poisson.fit_predict(X_train, y_train)
    assert_raises(ValueError, glm_poisson.fit_predict, X_train[None, ...], y_train)


def simple_cv_scorer(estimator, X, y):
    """Simple scorer takes average pseudo-R2 from regularization path"""
    scores = estimator.score(X, y)
    return np.mean(scores)


def test_multinomial():
    """Test all multinomial functionality"""
    glm_mn = GLM(distr='multinomial', reg_lambda=np.array([0.0, 0.1, 0.2]),
                 learning_rate = 2e-1, tol=1e-10)
    X = np.array([[-1, -2, -3], [4, 5, 6]])
    y = np.array([1, 0])

    # test gradient
    beta = np.zeros([4, 2])
    grad_beta0, grad_beta = glm_mn._grad_L2loss(beta[0], beta[1:], 0, X, y)
    assert_true(grad_beta0[0] != grad_beta0[1])
    glm_mn.fit(X, y)
    y_pred = glm_mn.predict(X)
    assert_equal(y_pred.shape, (3, X.shape[0], 2))  # n_lambdas x n_samples x n_classes

    # pick one as yhat
    yhat = y_pred[0]
    # uniform prediction
    ynull = np.ones(yhat.shape) / yhat.shape[1]
    # pseudo_R2 should be greater than 0
    assert_true(glm_mn.general_score(y, yhat, ynull, method='pseudo_R2') > 0.)
    glm_mn.general_score(y, yhat)
    assert_equal(len(glm_mn.simulate(glm_mn.fit_[0]['beta0'],
                                  glm_mn.fit_[0]['beta'],
                                  X)),
                 X.shape[0])
    # these should raise an exception
    assert_raises(ValueError, glm_mn.general_score, y, y, y, 'pseudo_R2')
    assert_raises(ValueError, glm_mn.general_score, y, y, None, 'deviance')


def test_sklearn_compatibility():
    X, y = make_regression(n_samples=1000, n_features=10)

    # Test whether cross validation works
    glm_normal = GLM(distr='normal', alpha=0.01,
                     reg_lambda=[0.0, 0.1, 0.2])
    glm_normal.fit(X, y)
    cv = KFold(X.shape[0], 5)
    # check that it returns 5 scores
    assert_equal(len(cross_val_score(glm_normal, X, y, cv=cv,
                                     scoring=simple_cv_scorer)), 5)

    # Test sklearn pipelines
    scaler = StandardScaler()
    glm = GLM(distr='normal', alpha=0.01,
              reg_lambda=[0.0, 0.1, 0.2])
    clf = Pipeline([("scaler", scaler), ("glm", glm)])
    assert_equal(len(cross_val_score(clf, X, y, cv=cv,
                                     scoring=simple_cv_scorer)), 5)
