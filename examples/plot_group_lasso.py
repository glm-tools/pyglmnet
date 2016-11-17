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

from pyglmnet import GLM
from pyglmnet.datasets import fetch_group_lasso_datasets
import numpy as np
import matplotlib.pyplot as plt

##########################################################
#
#Group Lasso Example
#applied to the same dataset found in:
#ftp://ftp.stat.math.ethz.ch/Manuscripts/buhlmann/lukas-sara-peter.pdf
#
#The task here is to determine which base pairs and positions within a 7-mer
#sequence are predictive of whether the sequence contains a splice
#site or not.
#

##########################################################
# Read and preprocess data

df , group_idxs= fetch_group_lasso_datasets()
print(df.head())

##########################################################
# Set up the training and testing sets

X = df[df.columns.difference(["Label"])].values
y = df.loc[:, "Label"].values

from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = \
    train_test_split(X, y, test_size=0.2, random_state=42)

##########################################################
# Setup the models

#set up the group lasso GLM model
gl_glm = GLM(distr="binomial",
             tol=1e-2,
             group=group_idxs,
             score_metric="pseudo_R2",
             alpha=1.0,
             reg_lambda=np.logspace(np.log(100), np.log(0.01), 5, base=np.exp(1)))


#set up the lasso model
glm = GLM(distr="binomial",
          tol=1e-2,
          score_metric="pseudo_R2",
          alpha=1.0,
          reg_lambda=np.logspace(np.log(100), np.log(0.01), 5, base=np.exp(1)))

print("gl_glm: ", gl_glm)
print("glm: ", glm)

##########################################################
# Fit models

gl_glm.fit(Xtrain, ytrain)
glm.fit(Xtrain, ytrain)

##########################################################
# Visualize model scores on test set

plt.figure()
plt.semilogx(gl_glm.reg_lambda, gl_glm.score(Xtest, ytest), 'go-')
plt.semilogx(gl_glm.reg_lambda, gl_glm.score(Xtrain, ytrain), 'go--')
plt.semilogx(glm.reg_lambda, glm.score(Xtest, ytest), 'ro-')
plt.semilogx(glm.reg_lambda, glm.score(Xtrain, ytrain), 'ro--')
plt.legend(['Group Lasso: test',
            'Group Lasso: train',
            'Lasso: test',
            'Lasso: train'], frameon=False, loc='best')
plt.xlabel('$\lambda$')
plt.ylabel('pseudo-$R^2$')
plt.ylim([-0.1, 0.7])

plt.tick_params(axis='y', right='off')
plt.tick_params(axis='x', top='off')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
