# -*- coding: utf-8 -*-
"""
======================
Community and Crime
======================

This is a real dataset of per capita violent crime, with demographic
data comprising 128 attributes from 1994 counties in the US.

The original dataset can be found here:
http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

The target variables (per capita violent crime) are normalized to lie in
a [0, 1] range. We preprocessed this dataset to exclude attributes with
missing values.
"""

########################################################

# Author: Vinicius Marques <vini.type@gmail.com>
# License: MIT

########################################################
# Imports

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from pyglmnet import GLM, datasets

########################################################
# Download and preprocess data files

X, y = datasets.fetch_community_crime_data('/tmp/glm-tools')
n_samples, n_features = X.shape

########################################################
# Split the data into training and test sets

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.33, random_state=0)

########################################################
# Fit a gaussian distributed GLM with elastic net regularization

# use the default value for reg_lambda
glm = GLM(distr='gaussian', alpha=0.05, score_metric='pseudo_R2')

# fit model
glm.fit(X_train, y_train)

# score the test set prediction
y_test_hat = glm[-1].predict(X_test)
print ("test set pseudo $R^2$ = %f" % glm[-1].score(X_test, y_test))

########################################################
# Plot the true and predicted test set target values

plt.plot(y_test[:50], 'ko-')
plt.plot(y_test_hat[:50], 'ro-')
plt.legend(['true', 'pred'], frameon=False)
plt.xlabel('Counties')
plt.ylabel('Per capita violent crime')

plt.tick_params(axis='y', right='off')
plt.tick_params(axis='x', top='off')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
