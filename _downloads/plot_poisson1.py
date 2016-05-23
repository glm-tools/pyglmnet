# -*- coding: utf-8 -*-
"""
=================================
Poisson (Basic) Distribution
=================================

This is an example demonstrating how pyglmnet
works with a basic Poisson distribution.

"""

########################################################
# First, we can import useful libraries that we will use it later on.

########################################################

# Author: Pavan Ramkumar <pavan.ramkumar@gmail.com>
# License: MIT

import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

########################################################
# Here are inputs that you can provide when you instantiate the `GLM` class.
# If not provided, it will be set to the respective defaults
#
# - `distr`: str (`'poisson'` or `'normal'` or `'binomial'` or `'multinomial'`)
#     default: `'poisson'`
# - `alpha`: float (the weighting between L1 and L2 norm)
#     default: 0.5
# - `reg_lambda`: array (array of regularized parameters)
#     default: `np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1))`
# - `learning_rate`: float (learning rate for gradient descent)
#     default: 1e-4
# - `max_iter`: int (maximum iteration for the model)
#     default: 100

########################################################

########################################################
# Import ``GLM`` class from ``pyglmnet``

########################################################

# import GLM model
from pyglmnet import GLM

# create regularize parameters for model
reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1))
glm_poisson = GLM(distr='poisson', verbose=False, alpha=0.05,
            max_iter=1000, learning_rate=1e-4,
            reg_lambda=reg_lambda)

##########################################################
# Simulate a dataset
# ------------------
# The ``GLM`` class has a very useful method called ``simulate()``.
#
# Since a canonical link function is already specified by the distribution
# parameters, or provided by the user, ``simulate()`` requires
# only the independent variables ``X`` and the coefficients ``beta0``
# and ``beta``

##########################################################

n_samples, n_features = 10000, 100

# coefficients
beta0 = np.random.normal(0.0, 1.0, 1)
beta = sps.rand(n_features, 1, 0.1)
beta = np.array(beta.todense())

# training data
Xr = np.random.normal(0.0, 1.0, [n_samples, n_features])
yr = glm_poisson.simulate(beta0, beta, Xr)

# testing data
Xt = np.random.normal(0.0, 1.0, [n_samples, n_features])
yt = glm_poisson.simulate(beta0, beta, Xt)

##########################################################
# Fit the model
# ^^^^^^^^^^^^^
# Fitting the model is accomplished by a single GLM method called `fit()`.
# You can provide data and output pair `(X, y)` i.e.

##########################################################

scaler = StandardScaler().fit(Xr)
glm_poisson.fit(scaler.transform(Xr), yr)

##########################################################
# Visualize the fit coefficients
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The estimated coefficients are stored in an instance variable called ``.fit_``
# which is a list of dictionaries. Each dictionary corresponds to a
# particular ``reg_lambda``

##########################################################

fit_param = glm_poisson[0].fit_
plt.plot(beta[:], 'bo', label ='bo')
plt.plot(fit_param['beta'][:], 'ro', label='ro')
plt.xlabel('samples')
plt.ylabel('outputs')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=2, borderaxespad=0.)
plt.show()

##########################################################
# Slicing the model object
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Although the model is fit to all values of reg_lambda specified by a regularization
# path, often we are only interested in further analysis for a particular value of
# ``reg_lambda``. We can easily do this by slicing the object.
#
# For instance model[0] returns an object identical to model but with ``.fit_``
# as a dictionary corresponding to the estimated coefficients for ``reg_lambda[0]``.

##########################################################
# Make predictions based on fit model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The ``predict()`` method takes two parameters: a numpy 2d array of independent
# variables and a dictionary of fit parameters. It returns a vector of
# predicted targets.

##########################################################

# Predict targets from test set
yrhat = glm_poisson[0].predict(scaler.transform(Xr))
ythat = glm_poisson[0].predict(scaler.transform(Xt))

plt.plot(yt[:100], label='tr')
plt.plot(ythat[:100], 'r', label='pr')
plt.xlabel('samples')
plt.ylabel('true and predicted outputs')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=2, borderaxespad=0.)
plt.show()

##########################################################
# Goodness of fit
# ^^^^^^^^^^^^^^^
# The GLM class provides two methods for evaluating goodness of fit: ``deviance()``
# and ``pseudo_R2()``. Both of them require the true targets and the predicted targets
# as inputs. ``pseudo_R2()`` additionally requires a null model, which is typically
# the mean of the target variables in the training set.

##########################################################

# Compute model deviance
Dr = glm_poisson[0].score(yr, yrhat)
Dt = glm_poisson[0].score(yt, ythat)
print('Dr = %f' % Dr, 'Dt = %f' % Dt)

# Compute pseudo-R2s
R2r = glm_poisson[0].score(yr, yrhat, np.mean(yr), method='pseudo_R2')
R2t = glm_poisson[0].score(yt, ythat, np.mean(yr), method='pseudo_R2')
print('  R2r =  %f' % R2r, ' R2r = %f' % R2t)

##########################################################
# Multinomial example
# ^^^^^^^^^^^^^^^^^^^
# We can also use ``pyglmnet`` with multinomial case
# where you can provide array of class.

##########################################################

from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, n_classes=5,
                           n_informative=100, n_features=100, n_redundant=0)

glm_mn = GLM(distr='multinomial', alpha=0.01,
               reg_lambda=np.array([0.02, 0.01]), verbose=False)
glm_mn.threshold = 1e-5
glm_mn.fit(X, y)
y_pred = glm_mn[-1].predict(X).argmax(axis=1)
print('Output performance = %f percent.' % (y_pred == y).mean())
