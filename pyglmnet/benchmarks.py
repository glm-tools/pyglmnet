"""
A class that assists performance comparison across different software tools:
At present, we compare scikit-learn, R, statsmodels and pyglmnet
"""
import time
import numpy as np

from pyglmnet import GLM
import statsmodels.api as sm
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.metrics import r2_score, accuracy_score

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()


class BenchmarkGLM(object):
    """
    envs: list
        each entry is a string:
        one of 'sklearn', 'pyglmnet', 'R', or 'statsmodels'

    distr: str
        one of 'gaussian', 'binomial', 'poisson'

    alpha: float
        the ratio between L1 and L2 regularization in elastic net

    reg_lambda: float
        regularization parameter
    """

    def __init__(self,
                 envs=['pyglmnet', 'sklearn', 'statsmodels'],
                 distr='gaussian',
                 alpha=0.1,
                 reg_lambda=0.1,
                 n_repeats=100):
        self.envs = envs
        self.distr = distr
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.n_repeats = n_repeats

    def get_benchmarks(self, X_train, y_train, X_test, y_test):
        """
        """
        n_repeats = self.n_repeats
        distr = self.distr

        res = dict()
        for env in self.envs:
            res[env] = dict()
            if env == 'pyglmnet':
                # initialize model
                model = GLM(distr=distr,
                            reg_lambda=[self.reg_lambda],
                            alpha=self.alpha,
                            solver='batch-gradient',
                            score_metric='pseudo_R2')

                # fit-predict-score
                model.fit(X_train, y_train)
                y_test_hat = model[-1].predict(X_test)
                y_test_hat = np.squeeze(y_test_hat)

                if distr in ['gaussian', 'poisson']:
                    res[env]['score'] = \
                        r2_score(y_test, y_test_hat)
                elif distr == 'binomial':
                    res[env]['score'] = \
                        accuracy_score(y_test,
                                       (y_test_hat > 0.5).astype(int))

                # time
                tmp = list()
                for r in range(n_repeats):
                    start = time.time()
                    model.fit(X_train, y_train)
                    stop = time.time()
                    tmp.append(stop - start)
                res[env]['time'] = np.min(tmp) * 1e3

            if env == 'sklearn':
                if distr in ['gaussian', 'binomial']:
                    # initialize model
                    if distr == 'gaussian':
                        model = ElasticNet(alpha=self.reg_lambda,
                                           l1_ratio=self.alpha)
                    elif distr == 'binomial':

                        model = SGDClassifier(loss='log',
                                              penalty='elasticnet',
                                              alpha=self.reg_lambda,
                                              l1_ratio=self.alpha)

                    # fit-predict-score
                    model.fit(X_train, y_train)
                    y_test_hat = model.predict(X_test)
                    res[env]['score'] = model.score(X_test, y_test)

                    # time
                    tmp = list()
                    for r in range(n_repeats):
                        start = time.time()
                        model.fit(X_train, y_train)
                        stop = time.time()
                        tmp.append(stop - start)
                    res[env]['time'] = np.min(tmp) * 1e3
                else:
                    res[env]['score'] = -999.
                    res[env]['time'] = -999.

            if env == 'statsmodels':
                # initialize model
                if distr == 'gaussian':
                    model = sm.GLM(y_train,
                                   sm.add_constant(X_train),
                                   family=sm.families.Gaussian())
                elif distr == 'binomial':
                    model = sm.GLM(y_train,
                                   sm.add_constant(X_train),
                                   family=sm.families.Binomial())
                elif distr == 'poisson':
                    model = sm.GLM(y_train,
                                   sm.add_constant(X_train),
                                   family=sm.families.Poisson())

                # fit-predict-score
                statsmodels_res = model.fit()
                y_test_hat = model.predict(statsmodels_res.params,
                                           exog=sm.add_constant(X_test))
                y_test_hat = np.array(y_test_hat)

                if distr in ['gaussian', 'poisson']:
                    res[env]['score'] = \
                        r2_score(y_test, y_test_hat)
                elif distr == 'binomial':
                    res[env]['score'] = \
                        accuracy_score(y_test,
                                       (y_test_hat > 0.5).astype(int))

                # time
                tmp = list()
                for r in range(n_repeats):
                    start = time.time()
                    statsmodels_res = model.fit()
                    stop = time.time()
                    tmp.append(stop - start)
                res[env]['time'] = np.min(tmp) * 1e3

            if env == 'R':
                # initialize model
                glmnet = importr('glmnet')
                predict = robjects.r('predict')

                # fit-predict-score
                try:
                    fit = glmnet.glmnet(X_train,
                                        y_train,
                                        family=distr,
                                        alpha=self.alpha,
                                        nlambda=1)
                    tmp = predict(fit, newx=X_test, s=0)

                    y_test_hat = np.zeros(y_test.shape[0])
                    for i in range(y_test.shape[0]):
                        y_test_hat[i] = tmp[i]

                    if distr in ['gaussian', 'poisson']:
                        res[env]['score'] = \
                            r2_score(y_test, y_test_hat)
                    elif distr == 'binomial':
                        res[env]['score'] = \
                            accuracy_score(y_test,
                                           (y_test_hat > 0.5).astype(int))

                    # time
                    tmp = list()
                    for r in range(n_repeats):
                        start = time.time()
                        fit = glmnet.glmnet(X_train,
                                            y_train,
                                            family=distr,
                                            alpha=self.alpha,
                                            nlambda=1)
                        stop = time.time()
                        tmp.append(stop - start)
                    res[env]['time'] = np.min(tmp) * 1e3
                except Exception:
                    res[env]['score'] = -999.
                    res[env]['time'] = -999.

        return res
