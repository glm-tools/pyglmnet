from . import utils
import numpy as np

def deviance(y, yhat, ynull_, distr):

	if distr in ['softplus', 'poisson']:
		LS = utils.log_likelihood(y, y, distr)
	else:
		LS = 0

	scores = list()
	
	for idx in range(yhat.shape[0]):
		if distr != 'multinomial':
			yhat_this = (yhat[idx, :]).ravel()
		else :
			yhat_this = yhat[idx, :, :]
		
		L1 = utils.log_likelihood(y, yhat_this, distr)
		scores.append(-2 * (L1 - LS))
	

	return np.array(scores)

def pseudo_R2(X, y, yhat, ynull_, distr):

	if distr in ['softplus', 'poisson']:
		LS = utils.log_likelihood(y, y, distr)
	else:
		LS = 0
    
	if distr != 'multinomial':
		L0 = utils.log_likelihood(y, ynull_, distr)
	else:
		expand_ynull_ = np.tile(ynull_, (X.shape[0], 1))
		L0 = utils.log_likelihood(y, expand_ynull_, distr)
	
	scores = list()
	for idx in range(yhat.shape[0]):
		
		if distr != 'multinomial':
			yhat_this = (yhat[idx, :]).ravel()
		else:
			yhat_this = yhat[idx, :, :]

		L1 = utils.log_likelihood(y, yhat_this, distr)
		        
		if distr in ['softplus', 'poisson']:
			scores.append(1 - (LS - L1) / (LS - L0))
		else:
			scores.append(1 - L1 / L0)
	
	return np.array(scores)

def accuracy(y, yhat):

	scores = list()
	for idx in range(yhat.shape[0]):
		accuracy = float(np.sum(y == yhat[idx])) / len(yhat[idx])
		scores.append(accuracy)
	return np.array(scores)

