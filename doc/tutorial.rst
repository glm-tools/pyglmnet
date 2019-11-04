========
Tutorial
========

This is a tutorial on elastic net regularized generalized linear models.
We will go through the math to setup the penalized negative log-likelihood
loss function and the coordinate descent algorithm for optimization.

Here are some other resources from a
`PyData 2016 talk <https:github.com/pavanramkumar/pydata-chicago-2016>`_.

At present this tutorial does not cover Tikhonov regularization or group lasso,
but we look forward to adding more material shortly.

**Reference**
Jerome Friedman, Trevor Hastie and Rob Tibshirani. (2010).
Regularization Paths for Generalized Linear Models via Coordinate Descent.
Journal of Statistical Software, Vol. 33(1), 1-22 `[pdf]
<https://core.ac.uk/download/files/153/6287975.pdf>`_.


.. code-block:: python

   # Author: Pavan Ramkumar
   # License: MIT

   import numpy as np
   from scipy.special import expit



GLM with elastic net penalty
----------------------------

In the elastic net regularized generalized Linear Model (GLM), we
want to solve the following convex optimization problem.

.. math::

    \min_{\beta_0, \beta} \frac{1}{N} \sum_{i = 1}^N \mathcal{L} (y_i, \beta_0 + \beta^T x_i)
    + \lambda [\frac{1}{2}(1 - \alpha)\| \beta \|_2^2 + \alpha \| \beta \|_1]

where :math:`\mathcal{L} (y_i, \beta_0 + \beta^T x_i)` is the negative log-likelihood of
observation :math:`i`. We will go through the softplus link function case
and show how we optimize the cost function.

Poisson-like GLM
----------------
The `pyglmnet` implementation comes with `gaussian`, `binomial`, `probit`
`gamma`, `poisson` and `softplus` distributions, but for illustration,
we will walk you through `softplus`: a particular adaptation of the canonical
Poisson generalized linear model (GLM).

For the Poisson GLM, :math:`\lambda_i` is the rate parameter of an
inhomogeneous linear-nonlinear Poisson (LNP) process with instantaneous
mean given by:

.. math::   \lambda_i = \exp(\beta_0 + \beta^T x_i)

where :math:`x_i \in \mathcal{R}^{p \times 1}, i = \{1, 2, \dots, n\}` are
the observed independent variables (predictors),
:math:`\beta_0 \in \mathcal{R}^{1 \times 1}`,
:math:`\beta \in \mathcal{R}^{p \times 1}`
are linear coefficients. The rate parameter :math:`\lambda_i` is also known as
the conditional intensity function, conditioned on :math:`(\beta_0, \beta)` and
:math:`q(z) = \exp(z)` is the nonlinearity.

For numerical reasons, let's adopt a stabilizing non-linearity, known as the
`softplus` or the smooth rectifier (see `Dugas et al., 2001
<http://papers.nips.cc/paper/1920-incorporating-second-order-functional-knowledge-for-better-option-pricing.pdf>`_),
that has been adopted by Jonathan Pillow's and Liam Paninski's groups for neural data
analysis.
See for instance `Park et al., 2014
<http://www.nature.com/neuro/journal/v17/n10/abs/nn.3800.html>`_.

.. math::    q(z) = \log(1+\exp(z))

The `softplus` prevents :math:`\lambda` in the canonical inverse link function
from exploding when the argument to the exponent is large. In this
modification, the formulation is no longer an exact LNP, nor an exact GLM,
but :math:`\mathcal{L}(\beta_0, \beta)` is still concave (convex) and we
can use gradient ascent (descent) to optimize it.

.. math::    \lambda_i = q(\beta_0 + \beta^T x_i) = \log(1 + \exp(\beta_0 +
                           \beta^T x_i))

.. code-block:: python

   def qu(z):
      """The non-linearity."""
      qu = np.log1p(np.exp(z))

   def lmb(self, beta0, beta, X):
      """Conditional intensity function."""
      z = beta0 + np.dot(X, beta)
      l = qu(z)
      return l


Poisson Log-likelihood
----------------------
The likelihood of observing the spike count :math:`y_i` under the Poisson
likelihood function with inhomogeneous rate :math:`\lambda_i` is given by:

.. math::    \prod_i P(y = y_i | X) = \prod_i \frac{e^{-\lambda_i} \lambda_i^{y_i}}{y_i!}

The log-likelihood is given by:

.. math::    \mathcal{L} = \sum_i \bigg\{y_i \log(\lambda_i) - \lambda_i
                           - \log(y_i!)\bigg\}

However, we are interested in maximizing the log-likelihood with respect to
:math:`\beta_0` and :math:`\beta`. Thus, we can drop the factorial term:

.. math::    \mathcal{L}(\beta_0, \beta) = \sum_i \bigg\{y_i \log(\lambda_i)
                                          - \lambda_i\bigg\}

Elastic net penalty
-------------------
For large models we need to penalize the log likelihood term in order to
prevent overfitting. The elastic net penalty is given by:

.. math::    \mathcal{P}_\alpha(\beta) = (1-\alpha)\frac{1}{2} \|\beta\|^2_{\mathcal{l}_2} + \alpha\|\beta\|_{\mathcal{l}_1}

The elastic net interpolates between two extremes.
When :math:`\alpha = 0` the penalized model is known as ridge regression and
when :math:`\alpha = 1` it is known as LASSO. Note that we do not penalize the
baseline term :math:`\beta_0`.


.. code-block:: python

   def penalty(alpha, beta):
      """the penalty term"""
      P = 0.5 * (1 - alpha) * np.linalg.norm(beta, 2) ** 2 + \
            alpha * np.linalg.norm(beta, 1)
      return P

Objective function
------------------

We minimize the objective function:

.. math::

     J(\beta_0, \beta) = -\mathcal{L}(\beta_0, \beta) + \lambda \mathcal{P}_\alpha(\beta)

where :math:`\mathcal{L}(\beta_0, \beta)` is the Poisson log-likelihood and
:math:`\mathcal{P}_\alpha(\beta)` is the elastic net penalty term and
:math:`\lambda` and :math:`\alpha` are regularization parameters.

.. code-block:: python

   def loss(beta0, beta, reg_lambda, X, y):
      """Define the objective function for elastic net."""
      L = logL(beta0, beta, X, y)
      P = penalty(beta)
      J = -L + reg_lambda * P
      return J


Gradient descent
----------------

To calculate the gradients of the cost function with respect to :math:`\beta_0` and
:math:`\beta`, let's plug in the definitions for the log likelihood and penalty terms from above.

.. math::

     \begin{eqnarray}
         J(\beta_0, \beta) &= \sum_i \bigg\{ \log(1 + \exp(\beta_0 + \beta^T x_i))\\
           & - y_i \log(\log(1 + \exp(\beta_0 + \beta^T x_i)))\bigg\}\\
           & + \lambda(1 - \alpha)\frac{1}{2} \|\beta\|^2_{\mathcal{l_2}}
           + \lambda\alpha\|\beta\|_{\mathcal{l_1}}
     \end{eqnarray}


Since we will apply coordinate descent, let's rewrite this cost in terms of each
scalar parameter :math:`\beta_j`

.. math::

     \begin{eqnarray}
         J(\beta_0, \beta) &= \sum_i \bigg\{ \log(1 + \exp(\beta_0 + \sum_j \beta_j x_{ij}))
         & - y_i \log(\log(1 + \exp(\beta_0 + \sum_j \beta_j x_{ij})))\bigg\}\\
         & + \lambda(1-\alpha)\frac{1}{2} \sum_j \beta_j^2 + \lambda\alpha\sum_j \mid\beta_j\mid
     \end{eqnarray}

Let's take the derivatives of some big expressions using chain rule.
Define :math:`z_i = \beta_0 + \sum_j \beta_j x_{ij}`.

For the nonlinearity in the first term :math:`y = q(z) = \log(1+e^{z(\theta)})`,

.. math::

     \begin{eqnarray}
     \frac{\partial y}{\partial \theta} &= \frac{\partial q}{\partial z}\frac{\partial z}{\partial \theta}\\
     & = \frac{e^z}{1+e^z}\frac{\partial z}{\partial \theta}\\
     & = \sigma(z)\frac{\partial z}{\partial \theta}
     \end{eqnarray}

For the nonlinearity in the second term :math:`y = \log(q(z)) = \log(\log(1+e^{z(\theta)}))`,

.. math::

     \begin{eqnarray}
     \frac{\partial y}{\partial \theta} & = \frac{1}{q(z)}\frac{\partial q}{\partial z}\frac{\partial z}{\partial \theta}\\
     & = \frac{\sigma(z)}{q(z)}\frac{\partial z}{\partial \theta}
     \end{eqnarray}

where :math:`\dot q(z)` happens to be be the sigmoid function

.. math::

     \sigma(z) = \frac{e^z}{1+e^z}.

Putting it all together we have:

.. math::

     \frac{\partial J}{\partial \beta_0} = \sum_i \sigma(z_i) - \sum_i y_i\frac{\sigma(z_i)}{q(z_i)}

.. math::

     \frac{\partial J}{\partial \beta_j} = \sum_i \sigma(z_i) x_{ij} - \sum_i y_i \frac{\sigma(z_i)}{q(z_i)}x_{ij}
     + \lambda(1-\alpha)\beta_j + \lambda\alpha \text{sgn}(\beta_j)

Let's define these gradients:

.. code-block:: python

   def grad_L2loss(beta0, beta, reg_lambda, X, y):
      z = beta0 + np.dot(X, beta)
      s = expit(z)
      q = qu(z)
      grad_beta0 = np.sum(s) - np.sum(y * s / q)
        grad_beta = np.transpose(np.dot(np.transpose(s), X) -
                    np.dot(np.transpose(y * s / q), X)) + \
        reg_lambda * (1 - alpha) * beta
        return grad_beta0, grad_beta


Note that this is all we need for a classic batch gradient descent implementation,
implemented in the ``'batch-gradient'`` solver.

However, let's also derive the Hessian terms that will be useful for second-order
optimization methods implemented in the ``'cdfast'`` solver.

Hessian terms
-------------

Second-order derivatives can accelerate convergence to local minima by providing
optimal step sizes. However, they are expensive to compute.

This is where coordinate descent shines. Since we update only one parameter
:math:`\beta_j` per step, we can simply use the :math:`j^{th}` diagonal term in
the Hessian matrix to perform an approximate Newton update as:

.. math::

     \beta_j^{t+1} = \beta_j^{t} - \bigg\{\frac{\partial^2 J}{\partial \beta_j^2}\bigg\}^{-1} \frac{\partial J}{\partial \beta_j}

Let's use calculus again to compute these diagonal terms. Recall that:

.. math::

     \begin{eqnarray}
     \dot q(z) & = \sigma(z)\\
     \dot\sigma(z) & = \sigma(z)(1-\sigma(z))
     \end{eqnarray}

Using these, and applying the product rule

.. math::

    \frac{\partial}{\partial z}\bigg\{ \frac{\sigma(z)}{q(z)} \bigg\} = \frac{\sigma(z)(1-\sigma(z))}{q(z)} - \frac{\sigma(z)}{q(z)^2}

Plugging all these in, we get

.. math::
     \frac{\partial^2 J}{\partial \beta_0^2} = \sum_i \sigma(z_i)(1 - \sigma(z_i)) - \sum_i y_i \bigg\{ \frac{\sigma(z_i) (1 - \sigma(z_i))}{q(z_i)} - \frac{\sigma(z_i)}{q(z_i)^2} \bigg\}

.. math::

     \begin{eqnarray}
     \frac{\partial^2 J}{\partial \beta_j^2} & = \sum_i \sigma(z_i)(1 - \sigma(z_i)) x_{ij}^2 \\
     & - \sum_i y_i \bigg\{ \frac{\sigma(z_i) (1 - \sigma(z_i))}{q(z_i)} \\
     & - \frac{\sigma(z_i)}{q(z_i)^2} \bigg\}x_{ij}^2 + \lambda(1-\alpha)
     \end{eqnarray}


.. code-block:: python

    def hessian_loss(beta0, beta, alpha, reg_lambda, X, y):
        z = beta0 + np.dot(X, beta)
        q = qu(z)
        s = expit(z)
        grad_s = s * (1-s)
        grad_s_by_q = grad_s/q - s/(q * q)
        hess_beta0 = np.sum(grad_s) - np.sum(y * grad_s_by_q)
        hess_beta = np.transpose(np.dot(np.transpose(grad_s), X * X)
                    - np.dot(np.transpose(y * grad_s_by_q), X * X))\
                    + reg_lambda * (1-alpha)
        return hess_beta0, hess_beta


Also see the `cheatsheet <http://glm-tools.github.io/pyglmnet/cheatsheet.html>`_ for the calculus required
to derive gradients and Hessians for other distributions.


Cyclical coordinate descent
---------------------------

**Parameter update step**

In cylical coordinate descent with elastic net, we store an active set
:math:`\mathcal{K}` of parameter indices that we update. Since the :math:`\mathcal{l}_1`
terms :math:`|\beta_j|` are not differentiable at zero, we use the gradient without
the :math:`\lambda\alpha \text{sgn}(\beta_j)` term to update :math:`\beta_j`.
Let's call these gradient terms :math:`\tilde{g}_k`.

We start by initializing :math:`\mathcal{K}` to contain all parameter indices.
Let's say only the :math:`k^{th}` parameter is updated at time step :math:`t`.

.. math::

     \begin{eqnarray}
         \beta_k^{t} & = \beta_k^{t-1} - (h_k^{t-1})^{-1} \tilde{g}_k^{t-1} \\
         \beta_j^{t} & = \beta_j^{t-1}, \forall j \neq k
     \end{eqnarray}

In practice, while implementing the update step, we check to see
if :math:`h_k^{t-1}` is above a tolerance so that its inverse does not explode.

Next we apply a soft thresholding step for :math:`k \neq 0` after every update iteration, as follows.
:math:`\beta_k^{t} = \mathcal{S}_{\lambda\alpha}(\beta_k^{t})`

where

.. math::

     S_\lambda(x) =
     \begin{cases}
     0 & \text{if} & |x| \leq \lambda\\
     \text{sgn}(x)||x|-\lambda| & \text{if} & |x| > \lambda
     \end{cases}

If :math:`\beta_k^{t}` has been zero-ed out, we remove :math:`k` from the active set.

.. math::

     \mathcal{K} = \mathcal{K} \setminus \left\{k\right\}


.. code-block:: python

    def prox(X, l):
        """Proximal operator."""
        return np.sign(X) * (np.abs(X) - l) * (np.abs(X) > l)


**Caching the z update step**

Next we want to update :math:`\beta_{k+1}` at the next time step :math:`t+1`.
For this we need the gradient and Hessian terms, :math:`\tilde{g}_{k+1}` and
:math:`h_{k+1}`. If we update them instead of recalculating them, we can save on
a lot of multiplications and additions. This is possible because we only update
one parameter at a time. Let's calculate how to make these updates.

.. math::

    z_i^{t} = z_i^{t-1} - \beta_k^{t-1}x_{ik} + \beta_k^{t}x_{ik}

.. math::

    z_i^{t} = z_i^{t-1} - (h_k^{t-1})^{-1} \tilde{g}_k^{t-1}x_{ik}


**Gradient update**

If :math:`k = 0`,

.. math::

     \tilde{g}_{k+1}^t = \sum_i \sigma(z_i^t) - \sum_i y_i \frac{\sigma(z_i^t)}{q(z_i^t)}

If :math:`k > 0`,

.. math::

     \begin{eqnarray}
         \tilde{g}_{k+1}^t & = \sum_i \sigma(z_i^t) x_{i,k+1} - \sum_i y_i \frac{\sigma(z_i^t)}{q(z_i^t)} x_{i,k+1}
           & + \lambda(1-\alpha)\beta_{k+1}^t
     \end{eqnarray}

.. code-block:: python

    def grad_loss_k(z, beta_k, alpha, rl, xk, y, k):
        """Gradient update for a single coordinate
        """
        q = qu(z)
        s = expit(z)
        if(k == 0):
            gk = np.sum(s) - np.sum(y*s/q)
        else:
            gk = np.sum(s*xk) - np.sum(y*s/q*xk) + rl*(1-alpha)*beta_k
        return gk


**Hessian update**

If :math:`k = 0`,

.. math::

    h_{k+1}^t & = \sum_i \sigma(z_i^t)(1 - \sigma(z_i^t)) \\
    & - \sum_i y_i \bigg\{ \frac{\sigma(z_i^t) (1 - \sigma(z_i^t))}{q(z_i^t)} - \frac{\sigma(z_i^t)}{q(z_i^t)^2} \bigg\}


If :math:`k > 0`,

.. math::

    \begin{eqnarray}
    h_{k+1}^t & = \sum_i \sigma(z_i^t)(1 - \sigma(z_i^t)) x_{i,k+1}^2 \\
    & - \sum_i y_i \bigg\{ \frac{\sigma(z_i^t) (1 - \sigma(z_i^t))}{q(z_i^t)}
    & - \frac{\sigma(z_i^t)}{q(z_i^t)^2} \bigg\}x_{i,k+1}^2 + \lambda(1-\alpha)
    \end{eqnarray}

.. code-block:: python

    def hess_loss_k(z, alpha, rl, xk, y, k):
        """Hessian update for a single coordinate
        """
        q = qu(z)
        s = expit(z)
        grad_s = s*(1-s)
        grad_s_by_q = grad_s/q - s/(q*q)
        if(k == 0):
            hk = np.sum(grad_s) - np.sum(y*grad_s_by_q)
        else:
            hk = np.sum(grad_s*xk*xk) - np.sum(y*grad_s_by_q*xk*xk) + rl*(1-alpha)
        return hk


Regularization paths and warm restarts
--------------------------------------

We often find the optimal regularization parameter :math:`\lambda` through cross-validation.
In practice we therefore often fit the model several times over a range of :math:`\lambda`'s
:math:`\{ \lambda_{max} \geq \dots \geq \lambda_0\}`.

Instead of re-fitting the model each time, we can solve the problem for the
most-regularized model (:math:`\lambda_{max}`) and then initialize the subsequent
model with this solution. The path that each parameter takes through the range of
regularization parameters is known as the regularization path, and the trick of
initializing each model with the previous model's solution is known as a warm restart.

In practice, this significantly speeds up convergence.
