"""
==============================
Communities and Crime - sklearn and pyglmnet
==============================

This is a real application example comparing pyglmnet against scikit using R_2 measure and mean squared error.
The used dataset was preprocessed fom the original dataset (it can be found at: http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime)

A quick view to the dataset information:
"Many variables are included so that algorithms that select or learn weights for attributes could be tested.
However, clearly unrelated attributes were not included; attributes were picked if there was any plausible connection
to crime (N=122), plus the attribute to be predicted (Per Capita Violent Crimes). The variables included in the dataset
involve the community, such as the percent of the population considered urban, and the median family income, and
involving law enforcement, such as per capita number of police officers, and percent of officers assigned to drug units."

We've tried to use the same values as possible for similiar parameters used by the two Classes.

"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Split data in train set and test set
from sklearn.cross_validation import train_test_split

ds = pd.read_csv('community_crime.csv',header=0)
X = ds.values # it returns a numpy array
n_samples, n_features = X.shape

# att128 is the labeled attribute and it's defined as:
#   ViolentCrimesPerPop: total number of violent crimes per 100K popuation (numeric - decimal) GOAL attribute (to be predicted)
X, y = np.array(ds.drop(['att128'],axis=1)), np.array(ds['att128'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)

########################################################
# pyglmnet

########################################################

from pyglmnet import GLM

# use the default value for reg_lambda
glm = GLM(distr='poissonexp', alpha=0.2, learning_rate=1e-3, verbose=False, max_iter=100)

glm.fit(X_train, y_train)

y_pred_glm = glm[-1].predict(X_test)

r2_score_glm = glm[-1].score(y_test, y_pred_glm, np.mean(y_train), method='pseudo_R2')
print("r^2 on test data using pyglmnet : %f" % r2_score_glm)
mean_square_score_glm = mean_squared_error(y_test, y_pred_glm)
print("mean square error on test data using pyglmnet : %f" % mean_square_score_glm)

########################################################
# sklearn

########################################################
from sklearn.linear_model import ElasticNet

alpha = glm[-1].reg_lambda

# l1_ratio is similar to alpha in GLM class
# alpha is similar to reg_lambda in GLM class
enet = ElasticNet(alpha=alpha, l1_ratio=0.2, max_iter=100)
y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print("r^2 on test data using sklearn : %f" % r2_score_enet)
mean_square_score_enet = mean_squared_error(y_test, y_pred_enet)
print("mean square error on test data using sklearn : %f" % mean_square_score_enet)

########################################################
# Plot the values of the test dataset, the predicted values computed by scikit and pyglmnet

########################################################

#plotting the predictions
plt.plot(y_test, label='real testing values')
plt.plot(y_pred_enet, 'r', label='scikit prediction')
plt.plot(y_pred_glm, 'g', label='pyglmnet prediction')
plt.xlabel('sample')
plt.ylabel('predicted value')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1,
           ncol=2, borderaxespad=0.)
plt.show()