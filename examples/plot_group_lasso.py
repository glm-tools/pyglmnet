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
import random

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
print(df.head())

#set up the group lasso GLM model

gl_glm = GLM(distr="binomial",
             group=group_idxs,
             tol=1e-2,
             score_metric="pseudo_R2",
             alpha=1.0,
             reg_lambda=np.logspace(np.log(100), np.log(0.01), 5, base=np.exp(1)))


#set up the non group GLM model

glm = GLM(distr="binomial",
          tol=1e-2,
          score_metric="pseudo_R2",
          alpha=1.0,
          reg_lambda=np.logspace(np.log(100), np.log(0.01), 5, base=np.exp(1)))

print("gl_glm: ", gl_glm)
print("glm: ", glm)

# Set up the training and testing sets.
X = df[df.columns.difference(["Label"]).values]

test_idxs = random.sample(list(range(X.shape[0])), 1000)
train_idxs = list( set(list(range(X.shape[0]))).difference(set(test_idxs)) )

X_train = X.iloc[train_idxs, :]
X_test = X.iloc[test_idxs, :]

y = df.loc[:, "Label"]
y_train = y.iloc[train_idxs]
y_test = y.iloc[test_idxs]


print("Fitting models")
gl_glm.fit(X_train.values, y_train.values)
glm.fit(X_train.values, y_train.values)
print("Model fitting complete.")
print("\n\n")


print("Group lasso post fitting score: ", gl_glm.score(X_test.values, y_test.values))
print("Non-group lasso post fitting score: ", glm.score(X_test.values, y_test.values))
