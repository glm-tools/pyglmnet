"""
======================
Rolling out custom GLM
======================

This is an example demonstrating rolling out your custom
GLM class using Pyglmnet.
"""
########################################################

# Author: Pavan Ramkumar <pavan.ramkumar@gmail.com>
# License: MIT

from sklearn.model_selection import train_test_split
from pyglmnet import GLMCV, datasets

########################################################
# Download and preprocess data files

X, y = datasets.fetch_community_crime_data()
n_samples, n_features = X.shape

########################################################
# Split the data into training and test sets

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.33, random_state=0)

########################################################
# Now we define our own distribution class. This must
# inherit from BaseDistribution. The BaseDistribution
# class requires defining the following methods:
#   - mu: inverse link function
#   - grad_mu: gradient of the inverse link function
#   - log_likelihood: the log likelihood function
#   - grad_log_likelihood: the gradient of the log 
#     likelihood.
# All distributions in pyglmnet inherit from BaseDistribution
#
# This is really powerful. For instance, we can start from
# the existing Binomial distribution and override mu and grad_mu
# if we want to use the cloglog link function.

import numpy as np
from pyglmnet.distributions import Binomial


class CustomBinomial(Binomial):
    """Custom binomial distribution."""

    def mu(self, z):
        """clogclog inverse link"""
        mu = 1 - np.exp(-np.exp(z))
        return mu

    def grad_mu(self, z):
        """Gradient of inverse link."""
        grad_mu = np.exp(1 - np.exp(z))
        return grad_mu


distr = CustomBinomial()

########################################################
# Now we pass it to the GLMCV class just as before.

# use the default value for reg_lambda
glm = GLMCV(distr=distr, alpha=0.05, score_metric='pseudo_R2', cv=3,
            tol=1e-4)

# fit model
glm.fit(X_train, y_train)

# score the test set prediction
y_test_hat = glm.predict_proba(X_test)
print("test set pseudo $R^2$ = %f" % glm.score(X_test, y_test))
