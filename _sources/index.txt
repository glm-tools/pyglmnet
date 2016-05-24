.. pyglmnet documentation master file, created by
   sphinx-quickstart on Mon May  23 01:07:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================================================================
Python implementation of elastic-net regularized generalized linear models
==========================================================================

``pyglmnet 0.1.dev`` is a Python library implementing elastic-net
regularized generalized linear models (GLM).

At present ``scikit-learn`` only offers
`elastic net regularization <http://scikit-learn.org/stable/modules/linear_model.html#elastic-net>`__
for linear regression.

Here, we offer a pure Python implementation of linear, logistic, multinomial, and Poisson regression
with elastic net regularization.

We follow the same notations and approach as in
`Friedman, J., Hastie, T., & Tibshirani, R. (2010) <https://core.ac.uk/download/files/153/6287975.pdf>`_.
The implementation is aimed to suit users of the popular GLM
`R package <https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html>`_.
To the extent possible we have designed the package to conform to
`scikit-learn <https://scikit-learn.org>`_'s API standards.

[`Repository <https://github.com/pavanramkumar/pyglmnet>`_]
[`Documentation <http://pavanramkumar.github.io/pyglmnet>`_]


A brief introduction to GLMs
============================

Linear models are estimated as

.. math::
    y = \beta_0 + X\beta + \epsilon

The parameters :math:`\beta_0, \beta` are estimated using ordinary least squares, under the
implicit assumption the :math:`y` is normally distributed.

Generalized linear models allow us to generalize this approach to point-wise
nonlinearities :math:`q(.)` and a family of exponential distributions for :math:`y`.

.. math::
    y = q(\beta_0 + X\beta) + \epsilon

Regularized GLMs are estimated by minimizing a loss function specified by
the penalized negative log-likelihood. The elastic net penalty interpolates
between L2 and L1 norm. Thus, we solve the following optimization problem:

.. math::
    \min_{\beta_0, \beta} \frac{1}{N} \sum_{i = 1}^N \mathcal{L} (y_i, \beta_0 + \beta^T x_i)
    + \lambda [ \frac{1}{2}(1 - \alpha)\| \beta \|_2^2 + \alpha \| \beta \|_1 ]

Contents
========

.. toctree::
   :maxdepth: 1

   install
   start
   auto_examples/index
   tutorials/index
   api
   developers
   resources

Questions / Errors / Bugs
=========================

If you have a question about the code or find errors or bugs,
please `report it here <https://github.com/pavanramkumar/pyglmnet/issues>`__.
For more specific question, feel free to email us directly.
