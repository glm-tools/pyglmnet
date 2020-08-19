# -*- coding: utf-8 -*-
"""
=======================================
GLM with Negative Binomial Distribution
=======================================

This is an example of GLM with negative binomial distribution.
We wrote this example taking inspiration from the R community
below
https://stats.idre.ucla.edu/r/dae/negative-binomial-regression/

Here, we would like to predict the number of days absence of high school
juniors at two schools from there type of program they are enrolled,
and their math score.

The nature of the empirical data suggests that we need to model
count data (the number of days absent). In such scenarios, a common model
we could use is the Poisson regression.

However, if we inspect the dataset more closely, we will notice that
the dataset is over-dispersed since the conditional mean exceeds the
conditional variance. We would need to apply another model which is the
Negative Binomial regression.

The Negative Binomial regression can be seen as a mixture of Poisson
regression in which the mean of the Poisson distribution can be seen
as a random variable drawn from a Gamma distribution.

This gives us an extra parameter which can be used to account for the over
dispersion.

In this example, we will apply both Negative Binomial regression and
Poisson regression on the dataset.
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

import matplotlib.pyplot as plt

########################################################
# Read and preprocess data
df = pd.read_stata("https://stats.idre.ucla.edu/stat/stata/dae/nb_data.dta")

########################################################
# Change the program type to string (we don't need it)
df['prog'].replace({1:"General", 2:"Academic", 3:"Vocational"}, inplace=True)

########################################################
# Histogram of type of program they are enrolled
df.hist(column='daysabs', by=['prog'])
plt.show()

# Print mean and standard deviation for each program enrolled.
# We can see from here that the variance is higher that then mean for all
# the levels, therefore hinting for over-dispersion.
prog_mean = df.groupby('prog').agg({'daysabs': ['mean', 'std']})
print(prog_mean)

########################################################
# Feature
X = df.drop('daysabs', axis=1)
y = df['daysabs'].values

########################################################
# design matrix
program_df = pd.get_dummies(df.prog)
program_df_cleaned = program_df.drop('Vocational', axis=1)[["General", "Academic"]]
Xdsgn = pd.concat((df['math'], program_df_cleaned), axis=1).values

########################################################
# Fit the model using the GLM
glm_nb = GLM(distr='neg-binomial',
             alpha=0.0,
             reg_lambda=0.0,
             score_metric='pseudo_R2',
             verbose=True,
             learning_rate=1e-6,
             theta=1.032713156)
glm_nb.fit(Xdsgn, y)
print(glm_nb.beta0_, glm_nb.beta_)

########################################################
# Fit the model using the Poisson regression instead
glm_poisson = GLM(distr='poisson',
             alpha=0.0,
             reg_lambda=0.0,
             score_metric='pseudo_R2',
             verbose=True,
             learning_rate=1e-6)
glm_poisson.fit(Xdsgn, y)
print(glm_poisson.beta0_, glm_poisson.beta_)


########################################################
# Plot convergence information for both negative binomial and poisson
glm_nb.plot_convergence()
glm_poisson.plot_convergence()
plt.show()