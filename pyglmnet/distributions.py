"""Exponential family distributions for GLM estimators."""

from abc import ABC, abstractmethod
import numpy as np


class BaseDistribution(ABC):
    """Base class for distributions."""

    def __init__(self):
        """init."""
        pass

    @abstractmethod
    def mu(self, z):
        """Inverse link function."""
        pass

    @abstractmethod
    def grad_mu(self, z):
        """Gradient of inverse link."""
        pass

    @abstractmethod
    def log_likelihood(self, y, y_hat):
        """Log L2-penalized likelihood."""
        pass

    def grad_log_likelihood(self):
        """Gradient of L2-penalized log likelihood."""
        msg = (f"""Gradients of log likelihood are not specified for {self.__class__} distribution""") # noqa
        raise NotImplementedError(msg)

    def gradhess_log_likelihood_1d(self):
        """One-dimensional Gradient and Hessian of log likelihood."""
        msg = (f"""Gradients and Hessians of 1-d log likelihood are not specified for {self.__class__} distribution""") # noqa
        raise NotImplementedError(msg)

    def _z(self, beta0, beta, X):
        """Compute z to be passed through non-linearity."""
        return beta0 + np.dot(X, beta)


class Gaussian(BaseDistribution):
    """Class for Gaussian distribution."""

    def __init__(self):
        """init."""
        pass

    def mu(self, z):
        """Inverse link function."""
        mu = z
        return mu

    def grad_mu(self, z):
        """Gradient of inverse link."""
        grad_mu = np.ones_like(z)
        return grad_mu

    def log_likelihood(self, y, y_hat):
        """Log L2-penalized likelihood."""
        log_likelihood = -0.5 * np.sum((y - y_hat) ** 2)
        return log_likelihood

    def grad_log_likelihood(self, X, y, beta0, beta):
        """Gradient of L2-penalized log likelihood."""
        z = self._z(beta0, beta, X)
        mu = self.mu(z)
        grad_mu = self.grad_mu(z)
        grad_beta0 = np.sum((mu - y) * grad_mu)
        grad_beta = np.dot((mu - y).T, X * grad_mu[:, None]).T
        return grad_beta0, grad_beta

    def gradhess_log_likelihood_1d(self, xk, y, beta0, beta):
        """One-dimensional Gradient and Hessian of log likelihood."""
        z = self._z(beta0, beta, X)
        gk = np.sum((z - y) * xk)
        hk = np.sum(xk * xk)
        return gk, hk


class Poisson(BaseDistribution):
    """Class for Poisson distribution."""

    def __init__(self):
        """init."""
        self.eta = None

    def mu(self, z):
        """Inverse link function."""
        mu = z.copy()
        mu0 = (1 - self.eta) * np.exp(self.eta)
        mu[z > self.eta] = z[z > self.eta] * np.exp(self.eta) + mu0
        mu[z <= self.eta] = np.exp(z[z <= self.eta])
        return mu

    def grad_mu(self, z):
        """Gradient of inverse link."""
        grad_mu = z.copy()
        grad_mu[z > self.eta] = \
            np.ones_like(z)[z > self.eta] * np.exp(self.eta)
        grad_mu[z <= self.eta] = np.exp(z[z <= self.eta])
        return grad_mu

    def log_likelihood(self, y, y_hat):
        """Log L2-penalized likelihood."""
        eps = np.spacing(1)
        log_likelihood = np.sum(y * np.log(y_hat + eps) - y_hat)
        return log_likelihood

    def grad_log_likelihood(self, X, y, beta0, beta):
        """Gradient of L2-penalized log likelihood."""
        z = self._z(beta0, beta, X)
        mu = self.mu(z)
        grad_mu = self.grad_mu(z)
        grad_beta0 = np.sum(grad_mu) - np.sum(y * grad_mu / mu)
        grad_beta = ((np.dot(grad_mu.T, X) -
                      np.dot((y * grad_mu / mu).T, X)).T)
        return grad_beta0, grad_beta

    def gradhess_log_likelihood_1d(self, xk, y, beta0, beta):
        """One-dimensional Gradient and Hessian of log likelihood."""
        z = self._z(beta0, beta, X)
        mu = self.mu(z)
        s = expit(z)
        gk = np.sum((mu[z <= self.eta] - y[z <= self.eta]) *
                    xk[z <= self.eta]) + \
            np.exp(self.eta) * \
            np.sum((1 - y[z > self.eta] / mu[z > self.eta]) *
                   xk[z > self.eta])
        hk = np.sum(mu[z <= self.eta] * xk[z <= self.eta] ** 2) + \
            np.exp(self.eta) ** 2 * \
            np.sum(y[z > self.eta] / (mu[z > self.eta] ** 2) *
                   (xk[z > self.eta] ** 2))
        return gk, hk
