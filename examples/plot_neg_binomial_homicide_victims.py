# -*- coding: utf-8 -*-
"""
=======================================
Number of Homicide victims
=======================================

This is an example of GLM with negative binomial distribution.
We wrote this example taking inspiration from the R community
below
https://data.library.virginia.edu/getting-started-with-negative-binomial-regression-modeling/

The data used are taken from a survey which asked people
how many homicide victims they knew. The variables are "resp" and "race".
The former indicates how many victims the respondent knew, the latter
the ethnic group of the respondent (black or white).

The nature of the empirical data suggests that we need to model
count data (the number of homicide victims). In such scenarios,
a common model we could use is the Poisson regression.

However, if we inspect the dataset more closely, we will notice that
the dataset is over-dispersed since the conditional mean exceeds the
conditional variance. Basically, for each race (black or white,
the variance is double the mean.

We would need to apply another model which is the Negative Binomial regression.

In this example, we will see how the Negative Binomial model will produce better
results thanks to his dispersion parameter.

"""

########################################################
# Author: Giovanni De Toni <giovanni.det@gmail.com>
# License: MIT
########################################################

########################################################
# Import libraries

import pandas as pd
import numpy as np
from pyglmnet import GLM

import matplotlib.pyplot as plt

########################################################
# Read and preprocess data
df = pd.read_csv("./homicide-dataset.csv")[['resp', 'race']]

########################################################
# Histogram of type of program they are enrolled
df.hist(column='resp', by=['race'])
plt.show()

# Print mean and standard deviation for each program enrolled.
# We can see from here that the variance is higher that then mean for all
# the levels, therefore hinting for over-dispersion.
prog_mean = df.groupby('race').agg({'resp': ['mean', 'std']})
print(prog_mean)

########################################################
# Feature
# Model the race as a binary categorical feature
df.race = pd.Categorical(df.race)
df["race_code"] = df.race.cat.codes

Xdsgn = df.drop(['race', 'resp'], axis=1)
y = df['resp'].values

########################################################
# Fit the model using the Negative Binomial
glm_nb = GLM(distr='neg-binomial',
             alpha=0.0,
             reg_lambda=0.0,
             score_metric='pseudo_R2',
             verbose=True,
             learning_rate=1e-1,
             max_iter=5000,
             theta=0.20)
glm_nb.fit(Xdsgn, y)
print(glm_nb.beta0_, glm_nb.beta_)

########################################################
# Fit the model using the Poisson regression instead
glm_poisson = GLM(distr='poisson',
             alpha=0.0,
             reg_lambda=0.0,
             score_metric='pseudo_R2',
             verbose=True,
             max_iter=5000,
             learning_rate=1e-1)
glm_poisson.fit(Xdsgn, y)
print(glm_poisson.beta0_, glm_poisson.beta_)

########################################################
# Plot convergence information for both negative binomial and poisson
glm_nb.plot_convergence()
glm_poisson.plot_convergence()
plt.show()

########################################################
# Simulate the prediction given new data
#
# The Poisson model outputs the predicted mean (and therefore variance) of the distribution.
# However, we can see from the exploratory analysis that the observed standard deviations are much larger.
# The Negative Binomal generates the same mean, but we can use the dispersion parameter to
# compute a more accurate estimate of the standard deviations (for both white and black classes).
#
pred_nb = np.array(glm_nb.predict([[0], [1]]))
pred_poisson =  np.array(glm_poisson.predict([[0], [1]]))
print("")
print("NB Predicted means+std (black/white): {}, {}".format(pred_nb, np.sqrt(pred_nb+(pred_nb**2)*1/0.20)))
print("Poisson Predicted means+std (black/white): {}, {}".format(pred_poisson, np.sqrt(pred_poisson)))