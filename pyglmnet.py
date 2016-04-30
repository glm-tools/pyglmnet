import numpy as np
from scipy.special import expit
from scipy.stats import zscore


def softmax(w):
    w = np.array(w)

    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1, keepdims=True)
    return dist

# Define a class for a glm solver


class glm:

    # Instance object
    distr = 'poisson'

    # Initialize with distr as poisson by default
    def __init__(self, distr='poisson'):
        self.distr = distr

    # Define the nonlinearity
    def qu(self, z):
        if self.distr == 'poisson':
            eps = np.spacing(1)
            q = np.log(1 + eps + np.exp(z))
        elif self.distr == 'normal':
            q = z
        elif self.distr == 'binomial':
            q = expit(z)
        elif self.distr == 'multinomial':
            q = softmax(z)
        return q

    # Define the conditional intensity function
    def lmb(self, beta0, beta, x):
        z = beta0 + np.dot(x, beta)
        l = self.qu(z)
        return l

    def logL(self, beta0, beta, x, y):
        """The log likelihood."""
        l = self.lmb(beta0, beta, x)
        if(self.distr == 'poisson'):
            logL = np.sum(y * np.log(l) - l)
        elif(self.distr == 'normal'):
            logL = -0.5 * np.sum((y - l)**2)
        elif(self.distr == 'binomial'):
            # analytical formula
            #logL = np.sum(y*np.log(l) + (1-y)*np.log(1-l))

            # this prevents underflow
            z = beta0 + np.dot(x, beta)
            logL = np.sum(y * z - np.log(1 + np.exp(z)))
        elif(self.distr == 'multinomial'):
            logL = -np.sum(y * np.log(l))
        return logL

    def penalty(self, alpha, beta):
        """The penalty."""
        P = 0.5 * (1 - alpha) * np.linalg.norm(beta, 2) + \
            alpha * np.linalg.norm(beta, 1)
        return P

    def loss(self, beta0, beta, alpha, reg_lambda, x, y):
        """Define the objective function for elastic net."""
        L = self.logL(beta0, beta, x, y)
        P = self.penalty(alpha, beta)
        J = -L + reg_lambda * P
        return J

    def L2loss(self, beta0, beta, alpha, reg_lambda, x, y):
        """Quadratic loss."""
        L = self.logL(beta0, beta, x, y)
        P = 0.5 * (1 - alpha) * np.linalg.norm(beta, 2)
        J = -L + reg_lambda * P
        return J

    def prox(self, x, l):
        """Proximal operator."""
        # sx = [0. if np.abs(y) <= l else np.sign(y)*np.abs(abs(y)-l) for y in x]
        # return np.array(sx).reshape(x.shape)
        return np.sign(x) * (np.abs(x) - l) * (np.abs(x) > l)

    # Define the gradient
    def grad_L2loss(self, beta0, beta, alpha, reg_lambda, x, y):
        z = beta0 + np.dot(x, beta)

        if(self.distr == 'poisson'):
            q = self.qu(z)
            s = expit(z)
            grad_beta0 = np.sum(s) - np.sum(y * s / q)
            grad_beta = np.transpose(np.dot(np.transpose(s), x) -
                                     np.dot(np.transpose(y * s / q), x)) + \
                reg_lambda * (1 - alpha) * beta
            # + reg_lambda*alpha*np.sign(beta)

        elif(self.distr == 'normal'):
            grad_beta0 = -np.sum(y - z)
            grad_beta = -np.transpose(np.dot(np.transpose(y - z), x)) \
                + reg_lambda * (1 - alpha) * beta
            # + reg_lambda*alpha*np.sign(beta)

        elif(self.distr == 'binomial'):
            s = expit(z)
            grad_beta0 = np.sum(s - y)
            grad_beta = np.transpose(np.dot(np.transpose(s - y), x)) \
                + reg_lambda * (1 - alpha) * beta
            # + reg_lambda*alpha*np.sign(beta)
        elif(self.distr == 'multinomial'):
            # this assumes that y is already as a one-hot encoding
            pred = self.qu(z)
            grad_beta0 = -np.sum(y - pred)
            grad_beta = -np.transpose(np.dot(np.transpose(y - pred), x)) \
                + reg_lambda * (1 - alpha) * beta

        return grad_beta0, grad_beta

    def fit(self, x, y, reg_params, opt_params, verbose):
        """The fit function."""
        # Implements batch gradient descent (i.e. vanilla gradient descent by
        # computing gradient over entire training set)

        # Dataset shape
        p = x.shape[1]

        if len(y.shape) == 1:
            # convert to 1-hot encoding
            y_bk = y
            y = np.zeros([x.shape[0], y.max() + 1])
            for i in range(x.shape[0]):
                y[i, y_bk[i]] = 1.

        # number of predictions
        if self.distr == 'multinomial':
            k = y.shape[1]
        else:
            k = 1

        # Regularization parameters
        reg_lambda = reg_params['reg_lambda']
        alpha = reg_params['alpha']

        # Optimization parameters
        max_iter = opt_params['max_iter']
        e = opt_params['learning_rate']

        # Initialize parameters
        beta0_hat = np.random.normal(0.0, 1.0, k)
        beta_hat = np.random.normal(0.0, 1.0, [p, k])
        fit = []

        # Outer loop with descending lambda
        if verbose is True:
            print('----------------------------------------')
            print('Looping through the regularization path')
            print('----------------------------------------')
        for l, rl in enumerate(reg_lambda):
            fit.append({'beta0': beta0_hat, 'beta': beta_hat})
            if verbose is True:
                print('Lambda: %6.4f') % rl

            # Warm initialize parameters
            if l == 0:
                fit[-1]['beta0'] = beta0_hat
                fit[-1]['beta'] = beta_hat
            else:
                fit[-1]['beta0'] = fit[-2]['beta0']
                fit[-1]['beta'] = fit[-2]['beta']

            # Iterate until convergence
            no_convergence = 1
            convergence_threshold = 1e-3
            t = 0

            # Initialize parameters
            beta = np.zeros([p + 1, k])
            beta[0] = fit[-1]['beta0']
            beta[1:] = fit[-1]['beta']

            g = np.zeros([p + 1, k])
            # Initialize cost
            L = []
            DL = []

            while(no_convergence and t < max_iter):

                # Calculate gradient
                grad_beta0, grad_beta = self.grad_L2loss(
                    beta[0], beta[1:], alpha, rl, x, y)
                g[0] = grad_beta0
                g[1:] = grad_beta

                # Update time step
                t = t + 1

                # Update parameters
                beta = beta - e * g

                # Apply proximal operator for L1-regularization
                beta[1:] = self.prox(beta[1:], rl * alpha)

                # Calculate loss and convergence criteria
                L.append(self.loss(beta[0], beta[1:], alpha, rl, x, y))

                # Delta loss and convergence criterion
                if t > 1:
                    DL.append(L[-1] - L[-2])
                    if np.abs(DL[-1] / L[-1]) < convergence_threshold:
                        no_convergence = 0
                        if verbose is True:
                            print('    Converged. Loss function: {0:.2f}').format(
                                L[-1])
                            print('    dL/L: {0:.6f}\n').format(DL[-1] / L[-1])

            # Store the parameters after convergence
            fit[-1]['beta0'] = beta[0]
            fit[-1]['beta'] = beta[1:]

        return fit

    def predict(self, x, fitparams):
        """Define the predict function."""
        yhat = self.lmb(fitparams['beta0'], fitparams['beta'], zscore(x))
        return yhat

    def pseudo_R2(self, y, yhat, ynull):
        """Define the pseudo-R2 function."""
        eps = np.spacing(1)
        if self.distr == 'poisson':
            # Log likelihood of model under consideration
            L1 = np.sum(y * np.log(eps + yhat) - yhat)

            # Log likelihood of homogeneous model
            L0 = np.sum(y * np.log(eps + ynull) - ynull)

            # Log likelihood of saturated model
            LS = np.sum(y * np.log(eps + y) - y)
            R2 = 1 - (LS - L1) / (LS - L0)

        elif self.distr == 'binomial':
            # Log likelihood of model under consideration
            L1 = 2 * len(y) * np.sum(y * np.log((yhat == 0) + yhat) / np.mean(yhat) +
                                     (1 - y) * np.log((yhat == 1) + 1 - yhat) / (1 - np.mean(yhat)))

            # Log likelihood of homogeneous model
            L0 = 2 * len(y) * np.sum(y * np.log((ynull == 0) + ynull) / np.mean(yhat) +
                                     (1 - y) * np.log((ynull == 1) + 1 - ynull) / (1 - np.mean(yhat)))
            R2 = 1 - L1 / L0

        elif self.distr == 'normal':
            R2 = 1 - np.sum((y - yhat)**2) / np.sum((y - ynull)**2)

        return R2

    def deviance(self, y, yhat):
        """The deviance function."""
        eps = np.spacing(1)
        # L1 = Log likelihood of model under consideration
        # LS = Log likelihood of saturated model
        if self.distr == 'poisson':
            L1 = np.sum(y * np.log(eps + yhat) - yhat)
            LS = np.sum(y * np.log(eps + y) - y)

        elif self.distr == 'binomial':
            L1 = 2 * len(y) * np.sum(y * np.log((yhat == 0) + yhat) / np.mean(yhat) +
                                     (1 - y) * np.log((yhat == 1) + 1 - yhat) / (1 - np.mean(yhat)))
            LS = 0

        elif self.distr == 'normal':
            L1 = -np.sum((y - yhat)**2)
            LS = 0

        D = -2 * (L1 - LS)
        return D

    def simulate(self, beta0, beta, x):
        """Simulate data."""
        if self.distr == 'poisson':
            y = np.random.poisson(self.lmb(beta0, beta, zscore(x)))
        if self.distr == 'normal':
            y = np.random.normal(self.lmb(beta0, beta, zscore(x)))
        if self.distr == 'binomial':
            y = np.random.binomial(1, self.lmb(beta0, beta, zscore(x)))
        return y
