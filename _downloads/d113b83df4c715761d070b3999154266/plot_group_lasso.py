"""
===========================
Group Lasso Regularization
===========================

This is an example demonstrating Pyglmnet with group lasso regularization,
typical in regression problems where it is reasonable to impose penalties
to model parameters in a group-wise fashion based on domain knowledge.

"""
########################################################

# Author: Matthew Antalek <matthew.antalek@northwestern.edu>
# License: MIT

########################################################

from pyglmnet import GLMCV
from pyglmnet.datasets import fetch_group_lasso_datasets
import matplotlib.pyplot as plt

##########################################################
#
# Group Lasso Example
# applied to the same dataset found in:
# ftp://ftp.stat.math.ethz.ch/Manuscripts/buhlmann/lukas-sara-peter.pdf
#
# The task here is to determine which base pairs and positions within a 7-mer
# sequence are predictive of whether the sequence contains a splice
# site or not.
#

##########################################################
# Read and preprocess data

df, group_idxs = fetch_group_lasso_datasets()
print(df.head())

##########################################################
# Set up the training and testing sets

from sklearn.model_selection import train_test_split # noqa

X = df[df.columns.difference(["Label"])].values
y = df.loc[:, "Label"].values

Xtrain, Xtest, ytrain, ytest = \
    train_test_split(X, y, test_size=0.2, random_state=42)

##########################################################
# Setup the models

# set up the group lasso GLM model
gl_glm = GLMCV(distr="binomial", tol=1e-3,
               group=group_idxs, score_metric="pseudo_R2",
               alpha=1.0, learning_rate=3, max_iter=100, cv=3, verbose=True)


# set up the lasso model
glm = GLMCV(distr="binomial", tol=1e-3,
            score_metric="pseudo_R2",
            alpha=1.0, learning_rate=3, max_iter=100, cv=3, verbose=True)

print("gl_glm: ", gl_glm)
print("glm: ", glm)

##########################################################
# Fit models

gl_glm.fit(Xtrain, ytrain)
glm.fit(Xtrain, ytrain)

##########################################################
# Visualize model scores on test set

plt.figure()
plt.semilogx(gl_glm.reg_lambda, gl_glm.scores_, 'go-')
plt.semilogx(glm.reg_lambda, glm.scores_, 'ro--')
plt.legend(['Group Lasso', 'Lasso'], frameon=False,
           loc='best')
plt.xlabel('$\lambda$')
plt.ylabel('pseudo-$R^2$')

plt.tick_params(axis='y', right='off')
plt.tick_params(axis='x', top='off')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
