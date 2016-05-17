# -*- coding: utf-8 -*-
"""
=============================================
pyglmnet for Poisson exponential distribution
=============================================

This is an example demonstrating how pyglmnet with
poisson exponential distribution works.

"""

########################################################
# first, we can import useful libraries that we will use it later on

########################################################

import numpy as np
import scipy.sparse as sps
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

########################################################
# import ``GLM`` class from ``pyglmnet``

########################################################

# import GLM model
from pyglmnet import GLM

# create regularization parameters for model
reg_lambda = np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1))
model = GLM(distr='poissonexp', verbose=False, alpha=0.05,
            max_iter=1000, learning_rate=1e-5,
            reg_lambda=reg_lambda, eta=4.0)

########################################################
#
# .. math::
#
#     J = \sum_i \lambda_i - y_i \log \lambda_i
#
# where
#
# .. math::
#
#     \lambda_i =
#     \begin{cases}
#         \exp(z_i), & \text{if}\ z_i \leq \eta \\
#         \\
#          \exp(\eta)z_i + (1-\eta)\exp(\eta), & \text{if}\ z_i \gt \eta
#     \end{cases}
#
# and
#
# .. math::
#
#     z_i = \beta_0 + \sum_j \beta_j x_{ij}
#
# Taking gradients,
#
# .. math::
#
#     \frac{\partial J}{\partial \beta_j} = \sum_i \frac{\partial J}{\partial \lambda_i} \frac{\partial \lambda_i}{\partial z_i} \frac{\partial z_i}{\partial \beta_j}
#
#
# .. math::
#
#     \frac{\partial J}{\partial \beta_0} =
#     \begin{cases}
#         \sum_i \Big(\lambda_i - y_i\Big), & \text{if}\ z_i \leq \eta \\
#         \\
#         \exp(\eta) \sum_i \Big(1 - \frac{\lambda_i}{y_i}\Big), & \text{if}\ z_i \gt \eta
#     \end{cases}
#
# .. math::
#     \frac{\partial J}{\partial \beta_j} =
#     \begin{cases}
#         \sum_i \Big(\lambda_i - y_i\Big)x_{ij}, & \text{if}\ z_i \leq \eta \\
#         \\
#         \exp(\eta) \sum_i \Big(1 - \frac{\lambda_i}{y_i}\Big)x_{ij}, & \text{if}\ z_i \gt \eta
#     \end{cases}

########################################################

z = np.linspace(0., 10., 100)
qu = model.qu(z)
plt.plot(z, qu)
plt.hold
plt.plot(z, np.exp(z))
plt.ylim([0, 1000])
plt.show()

########################################################
#

########################################################

# Dataset size
n_samples, n_features = 10000, 100

# baseline term
beta0 = np.random.normal(0.0, 1.0, 1)
# sparse model terms
beta = sps.rand(n_features, 1, 0.1)
beta = np.array(beta.todense())

# training data
Xr = np.random.normal(0.0, 1.0, [n_samples, n_features])
yr = model.simulate(beta0, beta, Xr)

# testing data
Xt = np.random.normal(0.0, 1.0, [n_samples, n_features])
yt = model.simulate(beta0, beta, Xt)

########################################################
# fit model to training data

########################################################

scaler = StandardScaler().fit(Xr)
model.fit(scaler.transform(Xr),yr);

########################################################
# gradient of loss fucntion

########################################################

grad_beta0, grad_beta = model.grad_L2loss(model.fit_[-1]['beta0'], model.fit_[-1]['beta'], 0.01, Xr, yr)
print(grad_beta[:5])

########################################################
# use one model to predict

########################################################

m = model[-1]
this_model_param = m.fit_
yrhat = m.predict(scaler.transform(Xr))
ythat = m.predict(scaler.transform(Xt))

########################################################
# visualize predicted output

########################################################

plt.plot(yt[:100])
plt.hold(True)
plt.plot(ythat[:100], 'r')
plt.show()

########################################################
# compute pseudo R-square

########################################################

print(m.score(yt, ythat, np.mean(yr), method='pseudo_R2'))
