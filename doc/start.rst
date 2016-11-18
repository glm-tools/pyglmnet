===============
Getting Started
===============

Here is a brief example of how to use the ``GLM()`` class.

.. code:: python

    import numpy as np
    import scipy.sparse as sps
    from sklearn.preprocessing import StandardScaler
    from pyglmnet import GLM

.. code:: python

    # create an instance of the GLM class
    glm = GLM(distr='poisson')

.. code:: python

    n_samples, n_features = 10000, 100

.. code:: python

    # sample random coefficients
    beta0 = np.random.normal(0.0, 1.0, 1)
    beta = sps.rand(n_features, 1, 0.1)
    beta = np.array(beta.todense())

.. code:: python

    # simulate training data
    Xtrain = np.random.normal(0.0, 1.0, [n_samples, n_features])
    ytrain = glm.simulate(beta0, beta, Xtrain)

.. code:: python

    # simulate testing data
    Xtest = np.random.normal(0.0, 1.0, [n_samples, n_features])
    ytest = glm.simulate(beta0, beta, Xtest)

.. code:: python

    # fit the model on the training data
    scaler = StandardScaler().fit(Xtrain)
    glm.fit(scaler.transform(Xtrain), ytrain)

.. code:: python

    # predict using fitted model on the test data
    yhat = glm.predict(scaler.transform(Xtest))

.. code:: python

    # score the model on test data
    deviance = glm.score(scaler.transform(Xtest), ytest)
