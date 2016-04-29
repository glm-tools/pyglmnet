import numpy as np
from scipy.special import expit
from scipy.stats import zscore


def softmax(w):
    """
    Softmax function of given array of number w
    """
    w = np.array(w)
    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist


class GLM:
    """Generalized Linear Model (GLM)

    This is class implements  elastic-net regularized generalized linear models.
    The core algorithm is defined in the ariticle

    Parameters
    ----------
    family: str, 'poisson' or 'normal' or 'binomial' or 'multinomial'
        default: 'poisson'
    alpha: float, the weighting between L1 and L2 norm in penalty term
        loss function i.e.
            P(beta) = 0.5*(1-alpha)*|beta|_2^2 + alpha*|beta|_1
        default: 0.5
    reg_lambda: array or list, array of regularized parameters of penalty term i.e.
            (1/2*N) sum(y - beta*X) + lambda*P
        where lambda is number in reg_lambda list
        default: np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1))
    learning_rate: float, learning rate for gradient descent,
        default: 1e-4
    max_iter: int, maximum iteration for the model, default: 100
    threshold: float, threshold for convergence. Optimization loop will stop
        below setting threshold, default: 1e-3
    verbose: boolean, if True it will print output while iterating

    Reference
    ---------
    Friedman, Hastie, Tibshirani (2010). Regularization Paths for Generalized Linear
        Models via Coordinate Descent, J Statistical Software.
        https://core.ac.uk/download/files/153/6287975.pdf

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pyglmnet import GLM
    >>>
    >>> n_samples, n_features = 1000, 20
    >>> beta0 = np.random.normal(0.0, 1.0, 1)
    >>> beta = np.array(sps.rand(n_features,1,0.1).todense())
    >>> X = np.random.normal(0.0, 1.0, [n_samples, n_features])
    >>> y = model.simulate(beta0, beta, X)
    >>> model = GLM(family='poisson', alpha=0.05, reg_lambda=[0.1])
    >>> model.fit(X, y)
    >>> fit_param = model.fit_params[-1]
    >>> yhat = model.predict(X, fit_param)
    >>> plt.plot(beta, '.b')
    >>> plt.plot(model.fit_params[-1]['beta'], '.r')
    """

    def __init__(self, family='poisson', alpha=0.5,
                 reg_lambda=np.logspace(np.log(0.5), np.log(0.01), 10, base=np.exp(1)),
                 learning_rate=1e-4, max_iter=100, verbose=False):
        self.family = family
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_params = None
        self.verbose = False
        self.threshold = 1e-3


    def qu(self, z):
        """
        Define the nonlinearity given input value z
        """
        if(self.family=='poisson'):
            eps = np.spacing(1)
            q = np.log(1+eps+np.exp(z))
        elif(self.family=='normal'):
            q = z
        elif(self.family=='binomial'):
            q = expit(z)
        elif(self.family=='multinomial'):
            q = softmax(z)
        return q


    def lmb(self, beta0, beta, X):
        """
        Define the conditional intensity function
        """
        z = beta0 + np.dot(X,beta)
        l = self.qu(z)
        return l


    def logL(self, beta0, beta, X, y):
        """
        Log-likelihood of given parameter of GLM, data X and
        output y
        """
        l = self.lmb(beta0, beta, X)
        if(self.family=='poisson'):
            logL = np.sum(y*np.log(l) - l)
        elif(self.family=='normal'):
            logL = -0.5*np.sum((y-l)**2)
        elif(self.family=='binomial'):
            # analytical formula
            # logL = np.sum(y*np.log(l) + (1-y)*np.log(1-l))

            # this prevents underflow
            z = beta0 + np.dot(X,beta)
            logL = np.sum(y*z - np.log(1+np.exp(z)))
        elif(self.family=='multinomial'):
            logL = -np.sum(y*np.log(l))
        return logL


    def penalty(self, alpha, beta):
        """
        l1 norm and l2 norm penalty of GLM
        """
        P = 0.5*(1-alpha)*np.linalg.norm(beta,2) + alpha*np.linalg.norm(beta,1)
        return P


    def loss(self, beta0, beta, alpha, reg_lambda, X, y):
        """
        Objective function for elastic net
        """
        L = self.logL(beta0, beta, X, y)
        P = self.penalty(alpha, beta)
        J = -L + reg_lambda*P
        return J


    def L2loss(self, beta0, beta, alpha, reg_lambda, X, y):
        """
        Differentiable part of the objective (l2-norm loss)
        """
        L = self.logL(beta0, beta, X, y)
        P = 0.5*(1-alpha)*np.linalg.norm(beta,2)
        J = -L + reg_lambda*P
        return J


    def prox(self, X, l):
        """
        Proximal operator
        """
        #sx = [0. if np.abs(y) <= l else np.sign(y)*np.abs(abs(y)-l) for y in x]
        #return np.array(sx).reshape(x.shape)
        return np.sign(X) * (np.abs(X) - l) * (np.abs(X) > l)


    def grad_L2loss(self, beta0, beta, alpha, reg_lambda, X, y):
        """
        Gradient of loss function
        """
        z = beta0 + np.dot(X, beta)

        if(self.family=='poisson'):
            q = self.qu(z)
            s = expit(z)
            grad_beta0 = np.sum(s) - np.sum(y*s/q)
            grad_beta = np.transpose(np.dot(np.transpose(s), X) - np.dot(np.transpose(y*s/q), X)) \
                        + reg_lambda*(1-alpha)*beta# + reg_lambda*alpha*np.sign(beta)

        elif(self.family=='normal'):
            grad_beta0 = -np.sum(y-z)
            grad_beta = -np.transpose(np.dot(np.transpose(y-z), X)) \
                        + reg_lambda*(1-alpha)*beta# + reg_lambda*alpha*np.sign(beta)

        elif(self.family=='binomial'):
            s = expit(z)
            grad_beta0 =  np.sum(s-y)
            grad_beta = np.transpose(np.dot(np.transpose(s-y), X)) \
                        + reg_lambda*(1-alpha)*beta# + reg_lambda*alpha*np.sign(beta)
        elif(self.family=='multinomial'):
            # this assumes that y is already as a one-hot encoding
            pred = self.qu(z)
            grad_beta0 = -np.sum(y - pred)
            grad_beta = -np.transpose(np.dot(np.transpose(y - pred), X)) \
                        + reg_lambda*(1-alpha)*beta


        return grad_beta0, grad_beta


    def fit(self, X, y):
        """
        Fit function. Implements batch gradient descent (i.e. vanilla gradient
        descent by computing gradient over entire training set)
        """

        n, p = X.shape # input dataset shape

        if(len(y.shape) == 1):
            # convert to 1-hot encoding
            y_bk = y
            y = np.zeros([X.shape[0], y.max()+1])
            for i in range(X.shape[0]):
                y[i, y_bk[i]] = 1.

        # number of predictions
        if(self.family=='multinomial'):
            k = y.shape[1]
        else:
            k = 1

        # Regularization parameters
        reg_lambda = self.reg_lambda
        alpha = self.alpha

        # Optimization parameters
        max_iter = self.max_iter
        learning_rate = self.learning_rate

        # Initialize parameters
        beta0_hat = np.random.normal(0.0, 1.0, k)
        beta_hat = np.random.normal(0.0, 1.0, [p,k])
        fit_params = []

        # Outer loop with descending lambda
        if(self.verbose==True):
            print('----------------------------------------')
            print('Looping through the regularization path')
            print('----------------------------------------')

        for (l, rl) in enumerate(reg_lambda):
            fit_params.append({'beta0': beta0_hat, 'beta':beta_hat})
            if(self.verbose==True):
                print('Lambda: %6.4f') % rl

            # Warm initialize parameters
            if(l == 0):
                fit_params[-1]['beta0'] = beta0_hat
                fit_params[-1]['beta'] = beta_hat
            else:
                fit_params[-1]['beta0'] = fit_params[-2]['beta0']
                fit_params[-1]['beta'] = fit_params[-2]['beta']

            #---------------------------
            # Iterate until convergence
            #---------------------------
            no_convergence = 1
            t = 0

            # Initialize parameters
            beta = np.zeros([p+1,k])
            beta[0] = fit_params[-1]['beta0']
            beta[1:] = fit_params[-1]['beta']

            g = np.zeros([p+1, k])
            # Initialize cost
            L = []
            DL = []

            while(no_convergence and t < max_iter):

                # Calculate gradient
                grad_beta0, grad_beta = self.grad_L2loss(beta[0], beta[1:], alpha, rl, X, y)
                g[0] = grad_beta0
                g[1:] = grad_beta

                t = t+1 # Update time step

                beta = beta - learning_rate*g # Update parameters

                # Apply proximal operator for L1-regularization
                beta[1:] = self.prox(beta[1:], rl*alpha)

                # Calculate loss and convergence criteria
                L.append(self.loss(beta[0], beta[1:], alpha, rl, X, y))

                # Delta loss and convergence criterion
                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if(np.abs(DL[-1]/L[-1]) < self.threshold):
                        no_convergence = 0
                        if(self.verbose==True):
                            print('    Converged. Loss function: {0:.2f}').format(L[-1])
                            print('    dL/L: {0:.6f}\n').format(DL[-1]/L[-1])

            # Store the parameters after convergence
            fit_params[-1]['beta0'] = beta[0]
            fit_params[-1]['beta'] = beta[1:]

        self.fit_params = fit_params
        return self


    def predict(self, X, fit_param):
        """
        Predict output given data X and dictionary of fit_param

        Parameters
        ----------
        X: array, numpy array of shape (N, p) where
            N is data length and
            p is dimension or number of features
        fit_param: dict, dictionary of parameter including 2 main keys
            ['beta0', 'beta'] where fit_param['beta0'] is intercept and
            fit_param['beta'] is other parameters
        Return
        ------
        yhat: array, numpy array of predicted output of size (N, 1)
        """
        yhat = self.lmb(fit_param['beta0'], fit_param['beta'], zscore(X))
        return yhat


    def pseudo_R2(self, y, yhat, ynull):
        """
        pseudo-R2 function
        """

        eps = np.spacing(1)
        if(self.family=='poisson'):
            # Log likelihood of model under consideration
            L1 = np.sum(y*np.log(eps+yhat) - yhat)

            # Log likelihood of homogeneous model
            L0 = np.sum(y*np.log(eps+ynull) - ynull)

            # Log likelihood of saturated model
            LS = np.sum(y*np.log(eps+y) - y)
            R2 = 1-(LS-L1)/(LS-L0)

        elif(self.family=='binomial'):
            # Log likelihood of model under consideration
            L1 = 2*len(y)*np.sum(y*np.log((yhat==0)+yhat)/np.mean(yhat) + \
                                (1-y)*np.log((yhat==1)+1-yhat)/(1-np.mean(yhat)))

            # Log likelihood of homogeneous model
            L0 = 2*len(y)*np.sum(y*np.log((ynull==0)+ynull)/np.mean(yhat) + \
                                (1-y)*np.log((ynull==1)+1-ynull)/(1-np.mean(yhat)))
            R2 = 1 - L1/L0

        elif(self.family=='normal'):
            R2 = 1 - np.sum((y - yhat)**2)/np.sum((y - ynull)**2)

        return R2


    def deviance(self, y, yhat):
        """
        deviance function of y and predicted y (yhat)

        Parameters
        ----------
        y: array of output value
        yhat: array of predicted output value
        """

        eps = np.spacing(1)
        # L1 = Log likelihood of model under consideration
        # LS = Log likelihood of saturated model
        if(self.family=='poisson'):
            L1 = np.sum(y*np.log(eps+yhat) - yhat)
            LS = np.sum(y*np.log(eps+y) - y)

        elif(self.family=='binomial'):
            L1 = 2*len(y)*np.sum(y*np.log((yhat==0)+yhat)/np.mean(yhat) + \
                                (1-y)*np.log((yhat==1)+1-yhat)/(1-np.mean(yhat)))
            LS = 0

        elif(self.family=='normal'):
            L1 = -np.sum((y - yhat)**2)
            LS = 0

        D = -2*(L1-LS)
        return D


    def simulate(self, beta0, beta, X):
        """
        function to simulate data with given
        data X and parameter beta0, beta
        """
        if(self.family=='poisson'):
            y = np.random.poisson(self.lmb(beta0, beta, zscore(X)))
        if(self.family=='normal'):
            y = np.random.normal(self.lmb(beta0, beta, zscore(X)))
        if(self.family=='binomial'):
            y = np.random.binomial(1, self.lmb(beta0, beta, zscore(X)))
        return y
