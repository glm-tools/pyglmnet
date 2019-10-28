.. pyglmnet documentation master file, created by
   sphinx-quickstart on Mon May  23 01:07:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==============================================================
Python implementation of regularized generalized linear models
==============================================================

Pyglmnet is a Python 3.5+ library implementing generalized linear models (GLMs)
with advanced regularization options. It provides a wide range of noise models
(with paired canonical link functions) including gaussian, binomial, probit,
gamma, poisson, and softplus. It supports a wide range of regularizers: ridge, lasso,
elastic net, group lasso, and Tikhonov regularization.

[`Repository <https://github.com/glm-tools/pyglmnet>`_]
[`Documentation (stable release) <http://glm-tools.github.io/pyglmnet>`_]
[`Documentation (development version) <https://circleci.com/api/v1.1/project/github/glm-tools/pyglmnet/latest/artifacts/0/html/index.html?branch=master>`_]

A brief introduction to GLMs
============================

For linear models specified as

.. math::
    y = \beta_0 + X\beta + \epsilon.

The parameters :math:`\beta_0, \beta` are estimated using ordinary least squares, under the
implicit assumption that :math:`y` is normally distributed.

Generalized linear models allow us to generalize this approach to point-wise
nonlinearities :math:`q(\cdot)` and corresponding exponential family noise distributions for :math:`\epsilon`.

.. math::
    y = q(\beta_0 + X\beta) + \epsilon

Regularized GLMs are estimated by minimizing a loss function specified by
the penalized negative log-likelihood. The elastic net penalty interpolates
between the L2 and L1 norm. We solve the following optimization problem:

.. math::
    \min_{\beta_0, \beta} \frac{1}{N} \sum_{i = 1}^N \mathcal{L} (y_i, \beta_0 + \beta^T x_i)
    + \lambda [ \frac{1}{2}(1 - \alpha) \mathcal{P}_2 + \alpha \mathcal{P}_1 ]

where :math:`\mathcal{P}_2` and :math:`\mathcal{P}_1` are the generalized
L2 (Tikhonov) and generalized L1 (Group Lasso) penalties, given by:

.. math::
    \mathcal{P}_2 & = & \|\Gamma \beta \|_2^2 \\
    \mathcal{P}_1 & = & \sum_g \|\beta_{j,g}\|_2

where :math:`\Gamma` is the Tikhonov matrix: a square factorization of the inverse
covariance matrix and :math:`\beta_{j,g}` is the :math:`j` th coefficient of
group :math:`g`.

Contents
========

.. toctree::
   :maxdepth: 1

   install
   start
   tutorial
   cheatsheet
   auto_examples/index
   api
   contributing
   resources
   whats_new

Questions / Errors / Bugs
=========================

If you have questions about the code or find errors or bugs,
please `report it here <https://github.com/glm-tools/pyglmnet/issues>`__.
For more specific questions, feel free to email us directly.
