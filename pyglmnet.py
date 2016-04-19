import numpy as np
from scipy.special import expit

# Define a class for a glm solver
class glm:

    # Instance object
    distr='poisson'

    # Initialize with distr as poisson by default
    def __init__(self, distr='poisson'):
        self.distr = distr

    #--------------------------
    # Define the nonlinearity
    #--------------------------
    def qu(self, z):
        if(self.distr=='poisson'):
            eps = 0.1
            q = np.log(1+eps+np.exp(z))
        elif(self.distr=='normal'):
            q = z
        elif(self.distr=='binomial'):
            q = expit(z)
        return q

    #-------------------------------------------
    # Define the conditional intensity function
    #-------------------------------------------
    def lmb(self, beta0, beta, x):
        z = beta0 + np.dot(x,beta)
        l = self.qu(z)
        return l

    #---------------------------
    # Define the log-likelihood
    #---------------------------
    def logL(self, beta0, beta, x, y):
        l = self.lmb(beta0, beta, x)
        if(self.distr=='poisson'):
            logL = np.sum(y*np.log(l) - l)
        elif(self.distr=='normal'):
            logL = np.sum((y-l)**2)
        elif(self.distr=='binomial'):
            logL = np.sum(y*np.log(l) - (n-y)*np.log(1-l))
        return logL

    #--------------------
    # Define the penalty
    #--------------------
    def penalty(self, alpha, beta):
        P = 0.5*(1-alpha)*np.linalg.norm(beta,2) + alpha*np.linalg.norm(beta,1)
        return P

    #-----------------------------------------------
    # Define the objective function for elastic net
    #-----------------------------------------------
    def loss(self, beta0, beta, alpha, reg_lambda, x, y):
        L = self.logL(beta0, beta, x, y)
        P = self.penalty(alpha, beta)
        J = -L + reg_lambda*P
        return J

    #------------------------------------------------------
    # Define only the differentiable part of the objective
    #------------------------------------------------------
    def L2loss(self, beta0, beta, alpha, reg_lambda, x, y):
        L = self.logL(beta0, beta, x, y)
        P = 0.5*(1-alpha)*np.linalg.norm(beta,2)
        J = -L + reg_lambda*P
        return J

    #------------------------------
    # Define the proximal operator
    #------------------------------
    def prox(self,x,l):
        sx = [0. if np.abs(y) <= l else np.sign(y)*np.abs(abs(y)-l) for y in x]
        return np.array(sx).reshape(x.shape)

    #---------------------
    # Define the gradient
    #---------------------
    def grad_L2loss(self, beta0, beta, alpha, reg_lambda, x, y):
        z = beta0 + np.dot(x, beta)
        q = self.qu(z)
        s = expit(z)

        if(self.distr=='poisson'):
            grad_beta0 = np.sum(s) - np.sum(y*s/q)

            # This is a matrix implementation
            grad_beta = np.transpose(np.dot(np.transpose(s), x) - np.dot(np.transpose(y*s/q), x)) \
                + reg_lambda*(1-alpha)*beta# + reg_lambda*alpha*np.sign(beta)

        return grad_beta0, grad_beta

    #-------------------------
    # Define the fit function
    #-------------------------
    def fit(self, x, y, reg_params, opt_params):
    # Implements batch gradient descent (i.e. vanilla gradient descent by computing gradient over entire training set)

        # Dataset shape
        n = x.shape[0]
        p = x.shape[1]

        # Regularization parameters
        reg_lambda = reg_params['reg_lambda']
        alpha = reg_params['alpha']

        # Optimization parameters
        max_iter = opt_params['max_iter']
        e = opt_params['learning_rate']

        # Initialize parameters
        beta0_hat = np.random.normal(0.0,1.0,1)
        beta_hat = np.random.normal(0.0,1.0,[p,1])
        fit = []

        # Outer loop with descending lambda
        for l,rl in enumerate(reg_lambda):
            fit.append({'beta0': 0., 'beta': np.zeros([p,1])})
            print('Lambda: {}\n').format(rl)

            # Warm initialize parameters
            if(l == 0):
                fit[-1]['beta0'] = beta0_hat
                fit[-1]['beta'] = beta_hat
            else:
                fit[-1]['beta0'] = fit[-2]['beta0']
                fit[-1]['beta'] = fit[-2]['beta']

            #---------------------------
            # Iterate until convergence
            #---------------------------
            no_convergence = 1
            convergence_threshold = 1e-3
            t = 0

            # Initialize parameters
            beta = np.zeros([p+1,1])
            beta[0] = beta0_hat[:]
            beta[1:] = beta_hat[:]

            # Initialize moment parameters
            g = np.zeros([p+1,1])

            # Initialize cost
            L = []
            DL = []

            while(no_convergence and t < max_iter):

                #Calculate gradient
                grad_beta0, grad_beta = self.grad_L2loss(beta[0], beta[1:], alpha, rl, x, y)
                g[0] = grad_beta0
                g[1:] = grad_beta

                # Update time step
                t = t+1

                # Update parameters
                beta = beta -e*g

                # Apply proximal operator for L1-regularization
                beta[1:] = self.prox(beta[1:], rl*alpha)

                # Calculate loss and convergence criteria
                L.append(self.loss(beta[0], beta[1:], alpha, rl, x, y))

                # Delta loss and convergence criterion
                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if(np.abs(DL[-1]/L[-1]) < convergence_threshold):
                        no_convergence = 0
                        print('Converged')
                        print('    Loss function: {}').format(L[-1])
                        print('    dL/L: {}\n').format(DL[-1]/L[-1])

            #Store the parameters after convergence
            fit[-1]['beta0'] = beta[0]
            fit[-1]['beta'] = beta[1:]

        return fit

    #-----------------------------
    # Define the predict function
    #-----------------------------
    def predict(self, x, fitparams):
        yhat = self.lmb(fitparams['beta0'], fitparams['beta'], zscore(x))
        return yhat

    #-------------------------------
    # Define the pseudo-R2 function
    #-------------------------------
    def pseudo_R2(self, y, yhat, ynull):
        eps = 0.1
        if(self.distr=='poisson'):
            # Log likelihood of model under consideration
            L1 = np.sum(y*np.log(eps+yhat) - yhat)

            # Log likelihood of homogeneous model
            L0 = np.sum(y*np.log(eps+ynull) - ynull)

            # Log likelihood of saturated model
            LS = np.sum(y*np.log(eps+y) - y)
            R2 = 1-(LS-L1)/(LS-L0)

        elif(self.distr=='binomial'):
            # Log likelihood of model under consideration
            L1 = 2*len(y)*np.sum(y*np.log((yhat==0)+yhat)/np.mean(yhat) + \
                                (1-y)*np.log((yhat==1)+1-yhat)/(1-np.mean(yhat)))

            # Log likelihood of homogeneous model
            L0 = 2*len(y)*np.sum(y*np.log((ynull==0)+ynull)/np.mean(yhat) + \
                                (1-y)*np.log((ynull==1)+1-ynull)/(1-np.mean(yhat)))
            R2 = 1 - L1/L0

        elif(self.distr=='normal'):
            R2 = 1 - np.sum((y - yhat)**2)/np.sum((y - ynull)**2)

        return R2
