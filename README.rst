pyglmnet
========

A python implementation of elastic-net regularized generalized linear models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|License| |Travis| |Codecov| |Circle| |Gitter| |DOI| |JOSS|

`[Documentation (stable version)]`_ `[Documentation (development version)]`_

.. image:: https://user-images.githubusercontent.com/15852194/67919367-70482600-fb76-11e9-9b86-891969bd2bee.jpg

-  Pyglmnet provides a wide range of noise models (and paired canonical
   link functions): ``'gaussian'``, ``'binomial'``, ``'probit'``,
   ``'gamma'``, '``poisson``', and ``'softplus'``.

-  It supports a wide range of regularizers: ridge, lasso, elastic net,
   `group
   lasso <https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning#Group_lasso>`__,
   and `Tikhonov
   regularization <https://en.wikipedia.org/wiki/Tikhonov_regularization>`__.

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

    import matplotlib.pyplot as plt
    from pyglmnet import GLM, simulate_glm

    n_samples, n_features = 1000, 100
    distr = 'poisson'

    # sample a sparse model
    np.random.seed(42)
    beta0 = np.random.rand()
    beta = sps.random(1, n_features, density=0.2).toarray()[0]

    # simulate data
    Xtrain = np.random.normal(0.0, 1.0, [n_samples, n_features])
    ytrain = simulate_glm('poisson', beta0, beta, Xtrain)
    Xtest = np.random.normal(0.0, 1.0, [n_samples, n_features])
    ytest = simulate_glm('poisson', beta0, beta, Xtest)

    # create an instance of the GLM class
    glm = GLM(distr='poisson', score_metric='pseudo_R2', reg_lambda=0.01)

    # fit the model on the training data
    glm.fit(Xtrain, ytrain)

    # predict using fitted model on the test data
    yhat = glm.predict(Xtest)

    # score the model on test data
    pseudo_R2 = glm.score(Xtest, ytest)
    print('Pseudo R^2 is %.3f' % pseudo_R2)

    # plot the true coefficients and the estimated ones
    plt.stem(beta, markerfmt='r.', label='True coefficients')
    plt.stem(glm.beta_, markerfmt='b.', label='Estimated coefficients')
    plt.ylabel(r'$\beta$')
    plt.legend(loc='upper right')

    # plot the true vs predicted label
    plt.figure()
    plt.plot(ytest, yhat, '.')
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.plot([0, ytest.max()], [0, ytest.max()], 'r--')
    plt.show()

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
page <https://glm-tools.github.io/pyglmnet/contributing.html>`__ for more
details.

Citation
~~~~~~~~

If you use ``pyglmnet`` package in your publication, please cite us from
our `JOSS publication <https://doi.org/10.21105/joss.01959>`__ using the following BibTex

.. code::

   @article{Jas2020,
   doi = {10.21105/joss.01959},
   url = {https://doi.org/10.21105/joss.01959},
   year = {2020},
   publisher = {The Open Journal},
   volume = {5},
   number = {47},
   pages = {1959},
   author = {Mainak Jas and Titipat Achakulvisut and Aid Idrizović
             and Daniel Acuna and Matthew Antalek and Vinicius Marques
             and Tommy Odland and Ravi Garg and Mayank Agrawal
             and Yu Umegaki and Peter Foley and Hugo Fernandes
             and Drew Harris and Beibin Li and Olivier Pieters
             and Scott Otterson and Giovanni De Toni and Chris Rodgers
             and Eva Dyer and Matti Hamalainen and Konrad Kording and Pavan Ramkumar},
   title = {{P}yglmnet: {P}ython implementation of elastic-net regularized generalized linear models},
   journal = {Journal of Open Source Software}
   }

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
.. |JOSS| image:: https://joss.theoj.org/papers/10.21105/joss.01959/status.svg
   :target: https://doi.org/10.21105/joss.01959
.. _[Documentation (stable version)]: http://glm-tools.github.io/pyglmnet
.. _[Documentation (development version)]: https://circleci.com/api/v1.1/project/github/glm-tools/pyglmnet/latest/artifacts/0/html/index.html?branch=master
