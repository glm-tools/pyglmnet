pyglmnet
========

A python implementation of elastic-net regularized generalized linear models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|License| |Travis| |Codecov| |Gitter|

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

**Pyglmnet** is a response to this fragmentation. Here are some
highlights.

-  Pyglmnet provides a wide range of noise models (and paired canonical
   link functions): ``'gaussian'``, ``'binomial'``, ``'multinomial'``,
   '``poisson``', and ``'softplus'``.

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

Benchmarks
~~~~~~~~~~

Here is table comparing ``pyglmnet`` against ``scikit-learn``'s
``linear_model``, ``statsmodels``, and ``R``.

The numbers below are run time (in milliseconds) to fit a :math:`1000`
samples x :math:`100` predictors sparse matrix (density :math:`0.05`).
This was done on a c. 2011 Macbook Pro, so your numbers may vary.

+------------+------------+----------------+---------------+---------+
| distr      | pyglmnet   | scikit-learn   | statsmodels   | R       |
+============+============+================+===============+=========+
| gaussian   | 6.8        | 1.2            | 29.8          | 10.3    |
+------------+------------+----------------+---------------+---------+
| binomial   | 16.3       | 4.5            | 89.3          | --      |
+------------+------------+----------------+---------------+---------+
| poisson    | 5.8        | --             | 117.2         | 156.1   |
+------------+------------+----------------+---------------+---------+

We provide a function called ``BenchMarkGLM()`` in ``pyglmnet.datasets``
if you would like to run these benchmarks yourself, but you need to take
care of the dependencies: ``scikit-learn``, ``Rpy2``, and
``statsmodels`` yourself.

Installation
~~~~~~~~~~~~

Now ``pip`` installable!

.. code:: bash

    $ pip install pyglmnet

Manual installation instructions below:

Clone the repository.

.. code:: bash

    $ git clone http://github.com/glm-tools/pyglmnet

Install ``pyglmnet`` using ``setup.py`` as follows

.. code:: bash

    $ python setup.py develop install

Getting Started
~~~~~~~~~~~~~~~

Here is an example on how to use the ``GLM`` estimator.

.. code:: python

    import numpy as np
    import scipy.sparse as sps
    from sklearn.preprocessing import StandardScaler
    from pyglmnet import GLM

    # create an instance of the GLM class
    glm = GLM(distr='poisson')

    n_samples, n_features = 10000, 100

    # sample random coefficients
    beta0 = np.random.normal(0.0, 1.0, 1)
    beta = sps.rand(n_features, 1, 0.1)
    beta = np.array(beta.todense())

    # simulate training data
    X_train = np.random.normal(0.0, 1.0, [n_samples, n_features])
    y_train = glm.simulate(beta0, beta, X_train)

    # simulate testing data
    X_test = np.random.normal(0.0, 1.0, [n_samples, n_features])
    y_test = glm.simulate(beta0, beta, X_test)

    # fit the model on the training data
    scaler = StandardScaler().fit(X_train)
    glm.fit(scaler.transform(X_train), y_train)

    # predict using fitted model on the test data
    yhat_test = glm.predict(scaler.transform(X_test))

    # score the model
    deviance = glm.score(X_test, y_test)

`More pyglmnet examples and use
cases <http://glm-tools.github.io/pyglmnet/auto_examples/index.html>`__.

Tutorial
~~~~~~~~

Here is an `extensive
tutorial <http://glm-tools.github.io/pyglmnet/tutorial.html>`__ on GLMs,
optimization and pseudo-code.

Here are
`slides <https://pavanramkumar.github.io/pydata-chicago-2016>`__ from a
recent talk at `PyData Chicago
2016 <http://pydata.org/chicago2016/schedule/presentation/15/>`__,
corresponding `tutorial
notebooks <http://github.com/pavanramkumar/pydata-chicago-2016>`__ and a
`video <https://www.youtube.com/watch?v=zXec96KD1uA>`__.

How to contribute?
~~~~~~~~~~~~~~~~~~

We welcome pull requests. Please see our `developer documentation
page <http://glm-tools.github.io/pyglmnet/developers.html>`__ for more
details.

Author
~~~~~~

-  `Pavan Ramkumar <http:/github.com/pavanramkumar>`__

Contributors
~~~~~~~~~~~~

-  `Mainak Jas <http:/github.com/jasmainak>`__
-  `Titipat Achakulvisut <http:/github.com/titipata>`__
-  `Aid Idrizović <http:/github.com/the872>`__
-  `Vinicius Marques <http:/github.com/marquesVF>`__
-  `Daniel Acuna <http:/github.com/daniel-acuna>`__
-  `Hugo Fernandes <http:/github.com/hugoguh>`__
-  `Eva Dyer <http:/github.com/evadyer>`__
-  `Matt Antalek <https://github.com/themantalope>`__

Acknowledgments
~~~~~~~~~~~~~~~

-  `Konrad Kording <http://kordinglab.com>`__ for funding and support
-  `Sara
   Solla <http://www.physics.northwestern.edu/people/joint-faculty/sara-solla.html>`__
   for masterful GLM lectures

License
~~~~~~~

MIT License Copyright (c) 2016 Pavan Ramkumar

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
   :target: https://github.com/glm-tools/pyglmnet/blob/master/LICENSE
.. |Travis| image:: https://api.travis-ci.org/glm-tools/pyglmnet.svg?branch=master
   :target: https://travis-ci.org/glm-tools/pyglmnet
.. |Codecov| image:: https://codecov.io/github/glm-tools/pyglmnet/coverage.svg?precision=0
   :target: https://codecov.io/gh/glm-tools/pyglmnet
.. |Gitter| image:: https://badges.gitter.im/glm-tools/pyglmnet.svg
   :target: https://gitter.im/pavanramkumar/pyglmnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
