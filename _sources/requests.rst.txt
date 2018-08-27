==========================
Requests for pull requests
==========================

We always welcome new feature requests and suggestions, but if you are looking
to contribute, here are some features that we plan to add soon.

Exponential family distributions
--------------------------------

One prominent next step is to increase the number of distributions
available. In rough order of importance, we are looking at:

* Probit regression
  (see `issue #159 <https://github.com/glm-tools/pyglmnet/issues/159>`_).

* Negative binomial regression
  (see `issue #163 <https://github.com/glm-tools/pyglmnet/issues/163>`_).

* Other count distributions (e.g. Quasi Poisson, hurdle)
  (see `issue #163 <https://github.com/glm-tools/pyglmnet/issues/163>`_).

* Cox regression

For each distribution, the typical workflow would involve

* computing the gradients and Hessian and updating the
  `cheatsheet <glm-tools.github.io/pyglmnet/cheatsheet.html>`_.

* implementing it by updating ``_qu()``, ``_logL()``, ``_grad_L2loss()``,
  ``_gradhess_logloss_1d()``, ``score()``, and ``simulate()``.

* adding an `example <glm-tools.github.io/pyglmnet/auto_examples/index.html>`_
  to illustrate the distribution, preferably with real data.

Cython implementation of coordinate descent
-------------------------------------------

Coordinate descent requires nested for loops that can be sped up with cython.
This would involve translating ``_gradhess_logloss_1d()`` and ``_cdfast()``
into cython code
(see `issue #104 <https://github.com/glm-tools/pyglmnet/issues/104>`_).

Provide `GLMCV` class for warm restarts
---------------------------------------

Currently, the `GLM` class returns a list of estimators corresponding to
each value of :math:`\lambda` using a warm restart approach. This not readily
compatible with some of ``scikit-learn``'s  cross-validation and grid search
features.

This PR would require an overhaul of the existing ``GLM`` class in addition to
writing a new class
(see `issue #158 <https://github.com/glm-tools/pyglmnet/issues/158>`_).
