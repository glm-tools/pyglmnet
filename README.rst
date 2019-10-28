pyglmnet
========

A python implementation of elastic-net regularized generalized linear models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|License| |Travis| |Codecov| |Circle| |Gitter| |DOI|

`[Documentation (stable version)]`_ `[Documentation (development version)]`_

`Generalized linear
models <https://en.wikipedia.org/wiki/Generalized_linear_model>`__ are
well-established tools for regression and classification and are widely
applied across the sciences, economics, business, and finance. They are
uniquely identifiable due to their convex loss and easy to interpret due
to their point-wise non-linearities and well-defined noise models.

In the era of exploratory data analyses with a large number of predictor
variables, it is important to regularize. Regularization prevents
overfitting by penalizing the negative log likelihood and can be used to
articulate prior knowledge about the parameters in a structured form.

Despite the attractiveness of regularized GLMs, the available tools in
the Python data science eco-system are highly fragmented. More
specifically,

-  `statsmodels <http://statsmodels.sourceforge.net/devel/glm.html>`__
   provides a wide range of link functions but no regularization.
-  `scikit-learn <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`__
   provides elastic net regularization but only for linear models.
-  `lightning <https://github.com/scikit-learn-contrib/lightning>`__
   provides elastic net and group lasso regularization, but only for
   linear and logistic regression.

**Pyglmnet** is a response to this fragmentation. It runs on Python 3.5+,
and here are some of the highlights.

-  Pyglmnet provides a wide range of noise models (and paired canonical
   link functions): ``'gaussian'``, ``'binomial'``, ``'probit'``,
   ``'gamma'``, '``poisson``', and ``'softplus'``.

-  It supports a wide range of regularizers: ridge, lasso, elastic net,
   `group
   lasso <https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning#Group_lasso>`__,
   and `Tikhonov
   regularization <https://en.wikipedia.org/wiki/Tikhonov_regularization>`__.

-  Pyglmnet's API is designed to be compatible with scikit-learn, so you
   can deploy ``Pipeline`` tools such as ``GridSearchCV()`` and
   ``cross_val_score()``.

-  We follow the same approach and notations as in `Friedman, J.,
   Hastie, T., & Tibshirani, R.
   (2010) <https://core.ac.uk/download/files/153/6287975.pdf>`__ and the
   accompanying widely popular `R
   package <https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html>`__.

-  We have implemented a cyclical coordinate descent optimizer with
   Newton update, active sets, update caching, and warm restarts. This
   optimization approach is identical to the one used in R package.

-  A number of Python wrappers exist for the R glmnet package (e.g.
   `here <https://github.com/civisanalytics/python-glmnet>`__ and
   `here <https://github.com/dwf/glmnet-python>`__) but in contrast to
   these, Pyglmnet is a pure python implementation. Therefore, it is
   easy to modify and introduce additional noise models and regularizers
   in the future.

Installation
~~~~~~~~~~~~

Install the stable PyPI version with ``pip``

.. code:: bash

    $ pip install pyglmnet

For the bleeding edge development version:

Clone the repository.

.. code:: bash

    $ pip install https://api.github.com/repos/glm-tools/pyglmnet/zipball/master

Getting Started
~~~~~~~~~~~~~~~


Here is an example on how to use the ``GLM`` estimator.

.. code:: python

   import numpy as np
   import scipy.sparse as sps
   from pyglmnet import GLM, simulate_glm

   n_samples, n_features = 1000, 100
   distr = 'poisson'

   # sample a sparse model
   beta0 = np.random.rand()
   beta = np.random.random(n_features)
   beta[beta < 0.9] = 0

   # simulate data
   Xtrain = np.random.normal(0.0, 1.0, [n_samples, n_features])
   ytrain = simulate_glm('poisson', beta0, beta, Xtrain)
   Xtest = np.random.normal(0.0, 1.0, [n_samples, n_features])
   ytest = simulate_glm('poisson', beta0, beta, Xtest)

   # create an instance of the GLM class
   glm = GLM(distr='poisson', score_metric='deviance')

   # fit the model on the training data
   glm.fit(Xtrain, ytrain)

   # predict using fitted model on the test data
   yhat = glm.predict(Xtest)

   # score the model on test data
   deviance = glm.score(Xtest, ytest)

`More pyglmnet examples and use
cases <http://glm-tools.github.io/pyglmnet/auto_examples/index.html>`__.

Tutorial
~~~~~~~~

Here is an `extensive
tutorial <http://glm-tools.github.io/pyglmnet/tutorial.html>`__ on GLMs,
optimization and pseudo-code.

Here are
`slides <https://pavanramkumar.github.io/pydata-chicago-2016>`__ from a
talk at `PyData Chicago
2016 <http://pydata.org/chicago2016/schedule/presentation/15/>`__,
corresponding `tutorial
notebooks <http://github.com/pavanramkumar/pydata-chicago-2016>`__ and a
`video <https://www.youtube.com/watch?v=zXec96KD1uA>`__.

How to contribute?
~~~~~~~~~~~~~~~~~~

We welcome pull requests. Please see our `developer documentation
page <http://glm-tools.github.io/pyglmnet/developers.html>`__ for more
details.

Acknowledgments
~~~~~~~~~~~~~~~

-  `Konrad Kording <http://kordinglab.com>`__ for funding and support
-  `Sara
   Solla <http://www.physics.northwestern.edu/people/joint-faculty/sara-solla.html>`__
   for masterful GLM lectures

License
~~~~~~~

MIT License Copyright (c) 2016-2019 Pavan Ramkumar

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: https://github.com/glm-tools/pyglmnet/blob/master/LICENSE
.. |Travis| image:: https://api.travis-ci.org/glm-tools/pyglmnet.svg?branch=master
   :target: https://travis-ci.org/glm-tools/pyglmnet
.. |Codecov| image:: https://codecov.io/github/glm-tools/pyglmnet/coverage.svg?precision=0
   :target: https://codecov.io/gh/glm-tools/pyglmnet
.. |Circle| image:: https://circleci.com/gh/glm-tools/pyglmnet.svg?style=svg
   :target: https://circleci.com/gh/glm-tools/pyglmnet
.. |Gitter| image:: https://badges.gitter.im/glm-tools/pyglmnet.svg
   :target: https://gitter.im/pavanramkumar/pyglmnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
.. |DOI| image:: https://zenodo.org/badge/55302570.svg
   :target: https://zenodo.org/badge/latestdoi/55302570
.. _[Documentation (stable version)]: http://glm-tools.github.io/pyglmnet
.. _[Documentation (development version)]: https://circleci.com/api/v1.1/project/github/glm-tools/pyglmnet/latest/artifacts/0/html/index.html?branch=master
