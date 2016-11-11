"""
=========================
Group Lasso Example
=========================

This is an example demonstrating Pyglmnet with
multinomial the group lasso regularization, typical in regression
problems where it is reasonable to impose penalties to model parameters
in a group-wise fashion based on domain knowledge.


"""

from pyglmnet import GLM
from pyglmnet.datasets import fetch_group_lasso_datasets
import numpy as np

##########################################################
#
#Group Lasso Example
#similar to method found in:
#ftp://ftp.stat.math.ethz.ch/Manuscripts/buhlmann/lukas-sara-peter.pdf
#
#The task here is to determine which base pairs and positions within a 7-mer
#sequence are most important to predicting if the sequence contains a splice
#site or not.
#
##########################################################

print("Retrieving data...")

df , group_idxs= fetch_group_lasso_datasets()


print("Data retrieved")
print("Dataframe: ")
print df.head()

#set up the group lasso GLM model

gl_glm = GLM(distr="binomial",
             group=group_idxs,
             max_iter=10000,
             tol=1e-3,
             score_metric="deviance",
             alpha=1.0,
             reg_lambda=np.logspace(np.log(100), np.log(0.01), 10, base=np.exp(1)))


#set up the non group GLM model

glm = GLM(distr="binomial",
          max_iter=10000,
          tol=1e-3,
          score_metric="deviance",
          alpha=1.0,
          reg_lambda=np.logspace(np.log(100), np.log(0.01), 10, base=np.exp(1)))

print("gl_glm: \n", gl_glm)
print("glm: \n", glm)

X = df[df.columns.difference(["Label"]).values]
y = df.loc[:, "Label"]

print("Fitting models")
gl_glm.fit(X.values, y.values)
glm.fit(X.values, y.values)
print("Model fitting complete.")
print("\n\n")


print("Group lasso post fitting score: ", gl_glm.score(X.values, y.values))
print("Non-group lasso post fitting score: ", glm.score(X.values, y.values))
