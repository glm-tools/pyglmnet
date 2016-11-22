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

    J = -\frac{1}{n}\sum_i \left\{ y_i \log(\beta_0 + \sum_j \beta_j x_{ij}) +
    (1-y_i) \log(1 - (\beta_0 + \sum_j \beta_j x_{ij})) \right\}
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
