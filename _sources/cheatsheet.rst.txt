==========
Cheatsheet
==========

This is a simple cheatsheet with the gradients and Hessians
of the penalized log likelihood loss to use as updates in the
Newton coordinate descent algorithm for GLMs.

Poisson: `softplus`
-------------------

**Mean Function**

.. math::

    z_i = \beta_0 + \sum_j \beta_j x_{ij} \\
    \mu_i = \log( 1 + \exp(z_i) )

**Log-likelihood function**

.. math::

    \mathcal{L} = \sum_i y_i \log(\mu_i) - \sum_i \mu_i

**L2-penalized loss function**

.. math::

    J = \frac{1}{n}\sum_i \left\{ \log( 1 + \exp( \beta_0 + \sum_j \beta_j x_{ij} ) ) \right\} \\
    - \frac{1}{n}\sum_i \left\{ y_i \log( \log( 1 + \exp(\beta_0 + \sum_j \beta_j x_{ij} ) ) ) \right\} \\
    + \lambda (1-\alpha) \frac{1}{2} \sum_j \beta_j^2

**Gradient**

.. math::

    \mu(z_i) &= \log(1 + \exp(z_i)) \\
    \sigma(z_i) &= \frac{1}{1 + \exp(-z_i)} \\
    \frac{\partial J}{\partial \beta_0} &= \frac{1}{n}\sum_i \sigma(z_i) - \frac{1}{n}\sum_i y_i \frac{\sigma(z_i)}{\mu(z_i)} \\
    \frac{\partial J}{\partial \beta_j} &= \frac{1}{n}\sum_i \sigma(z_i) x_{ij} - \frac{1}{n}\sum_i \sigma(z_i) y_i \frac{\sigma(z_i)}{\mu(z_i)}x_{ij} + \lambda (1 - \alpha) \beta_j

**Hessian**

.. math::

    \mu(z_i) &= \log(1 + \exp(z_i)) \\
    \sigma(z_i) &= \frac{1}{1 + \exp(-z_i)} \\
    \frac{\partial^2 J}{\partial \beta_0^2} &= \frac{1}{n}\sum_i \sigma(z_i) (1 - \sigma(z_i))
    - \frac{1}{n}\sum_i y_i \left\{ \frac{\sigma(z_i) (1 - \sigma(z_i))}{\mu(z_i)} - \frac{\sigma(z_i)}{\mu(z_i)^2} \right\} \\
    \frac{\partial^2 J}{\partial \beta_j^2} &=  \frac{1}{n}\sum_i \sigma(z_i) (1 - \sigma(z_i)) x_{ij}^2
    - \frac{1}{n}\sum_i y_i \left\{ \frac{\sigma(z_i) (1 - \sigma(z_i))}{\mu(z_i)} - \frac{\sigma(z_i)}{\mu(z_i)^2} \right\} x_{ij}^2
    + \lambda (1 - \alpha)

Poisson (linearized): `poisson`
-------------------------------

**Mean Function**

.. math::

    z_i &= \beta_0 + \sum_j \beta_j x_{ij} \\
    \mu_i &=
    \begin{cases}
    \exp(z_i), & z_i \leq \eta \\
    \\
    \exp(\eta)z_i + (1-\eta)\exp(\eta), & z_i > \eta
    \end{cases}

**Log-likelihood function**

.. math::

  \mathcal{L} = \sum_i y_i \log(\mu_i) - \sum_i \mu_i

**L2-penalized loss function**

.. math::

    J = -\frac{1}{n} \mathcal{L} + \lambda (1 - \alpha) \frac{1}{2} \sum_j \beta_j^2

**Gradient**

.. math::

    \mu_i &=
    \begin{cases}
    \exp(z_i),  & z_i \leq \eta \\
    \\
    \exp(\eta)z_i + (1-\eta)\exp(\eta),  & z_i > \eta
    \end{cases}
    \\
    \frac{\partial J}{\partial \beta_0} &= \frac{1}{n}\sum_{i; z_i \leq \eta} (\mu_i - y_i)
    + \frac{1}{n}\sum_{i; z_i > \eta} \exp(\eta) (1 - y_i/\mu_i) \\
    \frac{\partial J}{\partial \beta_j} &= \frac{1}{n}\sum_{i; z_i \leq \eta} (\mu_i - y_i) x_{ij}
    + \frac{1}{n}\sum_{i; z_i > \eta} \exp(\eta) (1 - y_i/\mu_i) x_{ij}

**Hessian**

.. math::

    \mu_i &=
    \begin{cases}
    \exp(z_i),  & z_i \leq \eta \\
    \\
    \exp(\eta)z_i + (1-\eta)\exp(\eta),  & z_i > \eta
    \end{cases}
    \\
    \frac{\partial^2 J}{\partial \beta_0^2} &= \frac{1}{n}\sum_{i; z_i \leq \eta} \mu_i
    + \frac{1}{n}\sum_{i; z_i > \eta} \exp(\eta)^2 \frac{y_i}{\mu_i^2}  \\
    \frac{\partial^2 J}{\partial \beta_j^2} &=  \frac{1}{n}\sum_{i; z_i \leq \eta} \mu_i x_{ij}^2
    + \frac{1}{n}\sum_{i; z_i > \eta} \exp(\eta)^2 \frac{y_i}{\mu_i^2} x_{ij}^2
    + \lambda (1 - \alpha)

Gaussian: `gaussian`
--------------------

**Mean Function**

.. math::

    z_i &= \beta_0 + \sum_j \beta_j x_{ij} \\
    \mu_i &= z_i

**Log-likelihood function**

.. math::

    \mathcal{L} = -\frac{1}{2} \sum_i (y_i - \mu_i)^2 \\

**L2-penalized loss function**

.. math::

    J = \frac{1}{2n}\sum_i (y_i - (\beta_0 + \sum_j \beta_j x_{ij}))^2 +
    \lambda (1 - \alpha) \frac{1}{2}\sum_j \beta_j^2\\

**Gradient**

.. math::

    \mu(z_i) &= z_i \\
    \frac{\partial J}{\partial \beta_0} &= -\frac{1}{n}\sum_i (y_i - \mu_i) \\
    \frac{\partial J}{\partial \beta_j} &= -\frac{1}{n}\sum_i (y_i - \mu_i) x_{ij}
    + \lambda (1 - \alpha) \beta_j

**Hessian**

.. math::

    \frac{\partial^2 J}{\partial \beta_0^2} &= 1 \\
    \frac{\partial^2 J}{\partial \beta_j^2} &=  \frac{1}{n}\sum_i x_{ij}^2
    + \lambda (1 - \alpha)

Logistic: `binomial`
--------------------

**Mean Function**

.. math::

    z_i &= \beta_0 + \sum_j \beta_j x_{ij} \\
    \mu_i &= \frac{1}{1+\exp(-z_i)}

**Log-likelihood function**

.. math::

    \mathcal{L} = \sum_i \left\{ y_i \log(\mu_i) + (1-y_i) \log(1 - \mu_i) \right\} \\

**L2-penalized loss function**

.. math::

    J = -\frac{1}{n}\sum_i \left\{ y_i \log(\mu_i) +
    (1-y_i) \log(1 - \mu_i) \right\}
    + \lambda (1 - \alpha) \frac{1}{2}\sum_j \beta_j^2\\


**Gradient**

.. math::

    \mu(z_i) &= \frac{1}{1 + \exp(-z_i)} \\
    \frac{\partial J}{\partial \beta_0} &= -\frac{1}{n}\sum_i (y_i - \mu_i) \\
    \frac{\partial J}{\partial \beta_j} &= -\frac{1}{n}\sum_i (y_i - \mu_i) x_{ij}
    + \lambda (1 - \alpha) \beta_j

**Hessian**

.. math::

    \frac{\partial^2 J}{\partial \beta_0^2} &= \frac{1}{n}\sum_i \mu_i (1 - \mu_i) \\
    \frac{\partial^2 J}{\partial \beta_j^2} &=  \frac{1}{n}\sum_i \mu_i (1 - \mu_i) x_{ij}^2
    + \lambda (1 - \alpha)

Logistic: `probit`
------------------

**Mean Function**

.. math::

    z_i &= \beta_0 + \sum_j \beta_j x_{ij} \\
    \mu_i &= \Phi(z_i)

where :math:`\Phi(z_i)` is the standard normal cumulative distribution function.

**Log-likelihood function**

.. math::

    \mathcal{L} = \sum_i \left\{ y_i \log(\mu_i) + (1-y_i) \log(1 - \mu_i) \right\} \\

**L2-penalized loss function**

.. math::

    J = -\frac{1}{n}\sum_i \left\{ y_i \log(\mu_i) +
    (1-y_i) \log(1 - \mu_i) \right\}
    + \lambda (1 - \alpha) \frac{1}{2}\sum_j \beta_j^2\\


**Gradient**

.. math::

    \mu(z_i) &= \Phi(z_i) \\
    \mu'(z_i) &= \phi(z_i)


where :math:`\Phi(z_i)` and :math:`\phi(z_i)` are the standard normal cdf and pdf.

.. math::

    \frac{\partial J}{\partial \beta_0} &=
      -\frac{1}{n}\sum_i \Bigg\{y_i \frac{\mu'(z_i)}{\mu(z_i)} - (1 - y_i)\frac{\mu'(z_i)}{1 - \mu(z_i)}\Bigg\} \\
      \frac{\partial J}{\partial \beta_j} &=
        -\frac{1}{n}\sum_i \Bigg\{y_i \frac{\mu'(z_i)}{\mu(z_i)} - (1 - y_i)\frac{\mu'(z_i)}{1 - \mu(z_i)}\Bigg\} x_{ij}
    + \lambda (1 - \alpha) \beta_j


**Hessian**

.. math::

    \frac{\partial^2 J}{\partial \beta_0^2} &=
      \frac{1}{n}\sum_i \mu'(z_i) \Bigg\{y_i \frac{z_i\mu(z_i) + \mu'(z_i)}{\mu^2(z_i)} +
      (1 - y_i)\frac{-z_i(1 - \mu(z_i)) + \mu'(z_i)}{(1 - \mu(z_i))^2} \Bigg\} \\
      \frac{\partial^2 J}{\partial \beta_j^2} &=
        \frac{1}{n}\sum_i \mu'(z_i) \Bigg\{y_i \frac{z_i\mu(z_i) + \mu'(z_i)}{\mu^2(z_i)} +
        (1 - y_i)\frac{-z_i(1 - \mu(z_i)) + \mu'(z_i)}{(1 - \mu(z_i))^2} \Bigg\} x_{ij}^2
    + \lambda (1 - \alpha)

In practice, the probit gradients suffer from instability primarily due to precision of evaluating the normal cdf.
Thus, in pyglmnet we use approximate formulas for computing the loss, gradients, and hessians from `Demidenko et al. (2001)
<https://pdfs.semanticscholar.org/0c03/0537919f09575b9f2c0a98c62f6571bdceee.pdf>`_.
For more details, see Eqns. 17-20 in the paper.

Gamma
-----

**Mean function**

.. math::

    z_i = \beta_0 + \sum_j \beta_j x_{ij} \\
    \mu_i = \log(1 + \exp(z_i))

**Log-likelihood function**

.. math::

    \mathcal{L} = \sum_{i} \nu\Bigg\{\frac{-y_i}{\mu_i} - \log(\mu_i)\Bigg\}

where :math:`\nu` is the shape parameter. It is exponential for :math:`\nu = 1`
and normal for :math:`\nu = \infty`.

**L2-penalized loss function**

.. math::

    J = -\frac{1}{n}\sum_{i} \nu\Bigg\{\frac{-y_i}{\mu_i} - \log(\mu_i)\Bigg\}
    + \lambda (1 - \alpha) \frac{1}{2}\sum_j \beta_j^2\\

**Gradient**

.. math::

    \frac{\partial J}{\partial \beta_0} &= \frac{1}{n} \sum_{i} \nu\Bigg\{\frac{y_i}{\mu_i^2}
    - \frac{1}{\mu_i}\Bigg\}{\mu_i'} \\
    \frac{\partial J}{\partial \beta_j} &= \frac{1}{n} \sum_{i} \nu\Bigg\{\frac{y_i}{\mu_i^2}
    - \frac{1}{\mu_i}\Bigg\}{\mu_i'}x_{ij} + \lambda (1 - \alpha) \beta_j

where :math:`\mu_i' = \frac{1}{1 + \exp(-z_i)}`.
