"""
=========================
Multinomial Distribution
=========================

This is an example demonstrating Pyglmnet with
multinomial distributed targets, typical in classification problems.

This canonical link can be used by specifying ``distr`` = ``'multinomial'``.

"""
##########################################################
# Multinomial example
# ^^^^^^^^^^^^^^^^^^^
# In the multinomial case
# you can provide an array of class labels as targets.

##########################################################

########################################################

# Author: Pavan Ramkumar <pavan.ramkumar@gmail.com>
# License: MIT

import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

########################################################

########################################################
# Setup the model using scikit learn convenience functions

########################################################

########################################################
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, n_classes=5,
                           n_informative=100, n_features=100, n_redundant=0)

########################################################

########################################################
# Fit the model

########################################################

########################################################
from pyglmnet import GLM
glm_mn = GLM(distr='multinomial', alpha=0.01,
               reg_lambda=np.array([0.02, 0.01]), verbose=False)
glm_mn.threshold = 1e-5
glm_mn.fit(X, y)

########################################################

########################################################
# Predict and score the output

########################################################

y_pred = glm_mn[-1].predict(X)
print('Percentage correct = %f percent.' % (y_pred == y).mean())

########################################################
