# -*- coding: utf-8 -*-
"""
=======================================
GLM with Negative Binomial Distribution
=======================================

This is an example of GLM with negative binomial distribution.
We wrote this example taking inspiration from the R community
below
https://stats.idre.ucla.edu/r/dae/negative-binomial-regression/

Here, we would like to predict the days absence of high school
juniors at two schools from there type of program they are enrolled,
and their math score.
"""

########################################################

# Author: Titipat Achakulvisut <my.titipat@gmail.com>
#         Giovanni De Toni <giovanni.det@gmail.com>
# License: MIT

########################################################


########################################################
# Import relevance libraries

import pandas as pd
from pyglmnet import GLM

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

########################################################
# Read and preprocess data
df = pd.read_stata("https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta")

# Histogram of type of program they are enrolled
df.hist(column='daysabs', by=['prog'])
plt.show()

# Print mean and standard deviation for each program enrolled
df.groupby('prog').agg({'daysabs': ['mean', 'std']})

########################################################
# Feature
X = df.drop('daysabs', axis=1)
y = df['daysabs'].values

# design matrix
program_df = pd.get_dummies(df.prog)
Xdsgn = pd.concat((df['math'], program_df.drop(3.0, axis=1)), axis=1).values

# Split the dataset into training and test
Xtrain, Xtest, ytrain, ytest = train_test_split(Xdsgn, y, test_size=0.2)

########################################################
# Fit the model using the GLM
glm_neg_bino = GLM(distr='neg-binomial',
                   alpha=0.0,
                   reg_lambda=0.0,
                   max_iter=10000,
                   score_metric='pseudo_R2')
glm_neg_bino.fit(Xtrain, ytrain)

########################################################
# Predict
y_hat = glm_neg_bino.predict(Xtest)

########################################################
# Return the learned betas and the score
print(glm_neg_bino.beta0_, glm_neg_bino.beta_)
print(glm_neg_bino.score(Xtest, y_hat))