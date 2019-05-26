"""
This example matches the one in the README, and is run
on travis to ensure that it works on every build.
"""
import numpy as np
import scipy.sparse as sps
from pyglmnet import GLM, simulate_glm

# create an instance of the GLM class
glm = GLM(distr="poisson")

# sample random coefficients
n_samples, n_features = 1000, 100
beta0 = np.random.normal(0.0, 1.0, 1)
beta = sps.rand(n_features, 1, 0.1)
beta = np.array(beta.todense())

# simulate training data
X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
y_train = simulate_glm("poisson", beta0, beta, X_train)

# simulate testing data
X_test = np.random.normal(0.0, 1.0, [n_samples, n_features])
y_test = simulate_glm("poisson", beta0, beta, X_test)

# fit the model on the training data
glm.fit(X_train, y_train.squeeze())

# predict using fitted model on the test data
yhat_test = glm.predict(X_test)

# score the model
deviance = glm.score(X_test, y_test)
