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
    """Base class for distributions."""

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
