===============
Getting Started
===============

Here is a brief example of how to use the ``GLM()`` class.

.. code:: python

    import numpy as np
    import scipy.sparse as sps
    from pyglmnet import GLM, simulate_glm

.. code:: python

    n_samples, n_features = 1000, 100
    distr = 'poisson'

    # sample a sparse model
    beta0 = np.random.normal(0.0, 1.0, 1)
    beta = sps.rand(n_features, 1, 0.1)
    beta = np.array(beta.todense())

    # simulate data
    Xtrain = np.random.normal(0.0, 1.0, [n_samples, n_features])
    ytrain = simulate_glm('poisson', beta0, beta, Xtrain)
    Xtest = np.random.normal(0.0, 1.0, [n_samples, n_features])
    ytest = simulate_glm('poisson', beta0, beta, Xtest)

.. code:: python

    # create an instance of the GLM class
    glm = GLM(distr='poisson', score_metric='deviance')

    # fit the model on the training data
    glm.fit(Xtrain, ytrain)

    # predict using fitted model on the test data
    yhat = glm.predict(Xtest)

    # score the model on test data
    deviance = glm.score(Xtest, ytest)
