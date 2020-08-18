"""Exponential family distributions for GLM estimators."""

from abc import ABC, abstractmethod
import numpy as np
from scipy.special import expit, log1p, loggamma
from scipy.stats import norm


def softplus(z):
    """Numerically stable version of log(1 + exp(z))."""
    # see stabilizing softplus: http://sachinashanbhag.blogspot.com/2014/05/numerically-approximation-of-log-1-expy.html # noqa
    mu = z.copy()
    mu[z > 35] = z[z > 35]
    mu[z < -10] = np.exp(z[z < -10])
    mu[(z >= -10) & (z <= 35)] = log1p(np.exp(z[(z >= -10) & (z <= 35)]))
    return mu


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


class PoissonSoftplus(BaseDistribution):
    """Class for Poisson distribution with softplus inverse link."""

    def __init__(self):
        """init."""
        pass

    def mu(self, z):
        """Inverse link function."""
        mu = softplus(z)
        return mu

    def grad_mu(self, z):
        """Gradient of inverse link."""
        grad_mu = expit(z)
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
        gk = np.sum(s * xk) - np.sum(y * s / mu * xk)

        grad_s = s * (1 - s)
        grad_s_by_mu = grad_s / mu - s / (mu ** 2)
        hk = np.sum(grad_s * xk ** 2) - np.sum(y * grad_s_by_mu * xk ** 2)
        return gk, hk


class NegBinomialSoftplus(BaseDistribution):
    """Class for Negative binomial distribution with softplus inverse link."""

    def __init__(self):
        """init."""
        self.theta = None

    def mu(self, z):
        """Inverse link function."""
        mu = softplus(z)
        return mu

    def grad_mu(self, z):
        """Gradient of inverse link."""
        grad_mu = expit(z)
        return grad_mu

    def log_likelihood(self, y, y_hat):
        """Log L2-penalized likelihood."""
        theta = self.theta
        log_likelihood = \
            np.sum(loggamma(y + theta) - loggamma(theta) - loggamma(y + 1) +
                   theta * np.log(theta) + y * np.log(y_hat) - (theta + y) *
                   np.log(y_hat + theta))
        return log_likelihood

    def grad_log_likelihood(self, X, y, beta0, beta):
        """Gradient of L2-penalized log likelihood."""
        z = self._z(beta0, beta, X)
        mu = self.mu(z)
        grad_mu = self.grad_mu(z)
        theta = self.theta
        partial_beta_0 = grad_mu * ((theta + y) / (mu + theta) - y / mu)
        grad_beta0 = np.sum(partial_beta_0)
        grad_beta = np.dot(partial_beta_0.T, X)
        return grad_beta0, grad_beta

    def gradhess_log_likelihood_1d(self, xk, y, beta0, beta):
        """One-dimensional Gradient and Hessian of log likelihood."""
        z = self._z(beta0, beta, X)
        mu = self.mu(z)
        grad_mu = expit(z)
        hess_mu = np.exp(-z) * expit(z) ** 2

        gradient_beta_j = -grad_mu * (y / mu - (y + theta) / (mu + theta))
        partial_beta_0_1 = hess_mu * (y / mu - (y + theta) / (mu + theta))
        partial_beta_0_2 = grad_mu ** 2 * \
            ((y + theta) / (mu + theta) ** 2 - y / mu ** 2)
        partial_beta_0 = -(partial_beta_0_1 + partial_beta_0_2)
        gk = np.dot(gradient_beta_j.T, xk)
        hk = np.dot(partial_beta_0.T, xk ** 2)
        return gk, hk


class Binomial(BaseDistribution):
    """Class for binomial distribution."""

    def __init__(self):
        """init."""
        pass

    def mu(self, z):
        """Inverse link function."""
        mu = expit(z)
        return mu

    def grad_mu(self, z):
        """Gradient of inverse link."""
        grad_mu = expit(z) * (1 - expit(z))
        return grad_mu

    def log_likelihood(self, y, y_hat, z=None):
        """Log L2-penalized likelihood."""
        # prevents underflow
        if z is not None:
            log_likelihood = np.sum(y * z - np.log(1 + np.exp(z)))
        # for scoring
        else:
            log_likelihood = \
                np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return log_likelihood

    def grad_log_likelihood(self, X, y, beta0, beta):
        """Gradient of L2-penalized log likelihood."""
        z = self._z(beta0, beta, X)
        mu = self.mu(z)
        grad_beta0 = np.sum(mu - y)
        grad_beta = np.dot((mu - y).T, X).T
        return grad_beta0, grad_beta

    def gradhess_log_likelihood_1d(self, xk, y, beta0, beta):
        """One-dimensional Gradient and Hessian of log likelihood."""
        z = self._z(beta0, beta, X)
        mu = self.mu(z)
        gk = np.sum((mu - y) * xk)
        hk = np.sum(mu * (1.0 - mu) * xk * xk)
        return gk, hk


class Probit(BaseDistribution):
    """Class for probit distribution."""

    def __init__(self):
        """init."""
        pass

    def _probit_g1(self, z, pdfz, cdfz, thresh=5):
        res = np.zeros_like(z)
        res[z < -thresh] = np.log(-pdfz[z < -thresh] / z[z < -thresh])
        res[np.abs(z) <= thresh] = np.log(cdfz[np.abs(z) <= thresh])
        res[z > thresh] = -pdfz[z > thresh] / z[z > thresh]
        return res

    def _probit_g2(self, z, pdfz, cdfz, thresh=5):
        res = np.zeros_like(z)
        res[z < -thresh] = pdfz[z < -thresh] / z[z < -thresh]
        res[np.abs(z) <= thresh] = np.log(1 - cdfz[np.abs(z) <= thresh])
        res[z > thresh] = np.log(pdfz[z > thresh] / z[z > thresh])
        return res

    def _probit_g3(self, z, pdfz, cdfz, thresh=5):
        res = np.zeros_like(z)
        res[z < -thresh] = -z[z < -thresh]
        res[np.abs(z) <= thresh] = \
            pdfz[np.abs(z) <= thresh] / cdfz[np.abs(z) <= thresh]
        res[z > thresh] = pdfz[z > thresh]
        return res

    def _probit_g4(self, z, pdfz, cdfz, thresh=5):
        res = np.zeros_like(z)
        res[z < -thresh] = pdfz[z < -thresh]
        res[np.abs(z) <= thresh] = \
            pdfz[np.abs(z) <= thresh] / (1 - cdfz[np.abs(z) <= thresh])
        res[z > thresh] = z[z > thresh]
        return res

    def _probit_g5(self, z, pdfz, cdfz, thresh=5):
        res = np.zeros_like(z)
        res[z < -thresh] = 0 * z[z < -thresh]
        res[np.abs(z) <= thresh] = \
            z[np.abs(z) <= thresh] * pdfz[np.abs(z) <= thresh] / \
            cdfz[np.abs(z) <= thresh] + (pdfz[np.abs(z) <= thresh] /
                                         cdfz[np.abs(z) <= thresh]) ** 2
        res[z > thresh] = z[z > thresh] * pdfz[z > thresh] + pdfz[z > thresh] ** 2
        return res

    def _probit_g6(self, z, pdfz, cdfz, thresh=5):
        res = np.zeros_like(z)
        res[z < -thresh] = \
            pdfz[z < -thresh] ** 2 - z[z < -thresh] * pdfz[z < -thresh]
        res[np.abs(z) <= thresh] = \
            (pdfz[np.abs(z) <= thresh] / (1 - cdfz[np.abs(z) <= thresh])) ** 2 - \
            z[np.abs(z) <= thresh] * pdfz[np.abs(z) <= thresh] / \
            (1 - cdfz[np.abs(z) <= thresh])
        res[z > thresh] = 0 * z[z > thresh]
        return res

    def mu(self, z):
        """Inverse link function."""
        mu = norm.cdf(z)
        return mu

    def grad_mu(self, z):
        """Gradient of inverse link."""
        grad_mu = norm.pdf(z)
        return grad_mu

    def log_likelihood(self, y, y_hat, z=None):
        """Log L2-penalized likelihood."""
        if z is not None:
            pdfz, cdfz = norm.pdf(z), norm.cdf(z)
            log_likelihood = \
                np.sum(y * self._probit_g1(z, pdfz, cdfz) +
                       (1 - y) * self._probit_g2(z, pdfz, cdfz))
        else:
            log_likelihood = \
                np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return log_likelihood

    def grad_log_likelihood(self, X, y, beta0, beta):
        """Gradient of L2-penalized log likelihood."""
        z = self._z(beta0, beta, X)
        mu = self.mu(z)
        grad_beta0 = np.sum(mu - y)
        grad_beta = np.dot((mu - y).T, X).T
        return grad_beta0, grad_beta

    def gradhess_log_likelihood_1d(self, xk, y, beta0, beta):
        """One-dimensional Gradient and Hessian of log likelihood."""
        z = self._z(beta0, beta, X)
        pdfz = norm.pdf(z)
        cdfz = norm.cdf(z)
        gk = -np.sum((y * self._probit_g3(z, pdfz, cdfz) -
                      (1 - y) * self._probit_g4(z, pdfz, cdfz)) * xk)
        hk = np.sum((y * self._probit_g5(z, pdfz, cdfz) +
                     (1 - y) * self._probit_g6(z, pdfz, cdfz)) * (xk * xk))
        return gk, hk
