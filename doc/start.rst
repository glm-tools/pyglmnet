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
    glm = GLM(distr='poisson', verbose=True, alpha=0.05)

.. code:: python

    n_samples, n_features = 10000, 100

.. code:: python

    # sample random coefficients
    beta0 = np.random.normal(0.0, 1.0, 1)
    beta = sps.rand(n_features, 1, 0.1)
    beta = np.array(beta.todense())

.. code:: python

    # simulate training data
    Xr = np.random.normal(0.0, 1.0, [n_samples, n_features])
    yr = glm.simulate(beta0, beta, Xr)

.. code:: python

    # simulate testing data
    Xt = np.random.normal(0.0, 1.0, [n_samples, n_features])
    yt = glm.simulate(beta0, beta, Xt)

.. code:: python

    # fit the model on the training data
    scaler = StandardScaler().fit(Xr)
    glm.fit(scaler.transform(Xr), yr)

.. code:: python

    # predict using fitted model on the test data
    yhat = glm.predict(scaler.transform(Xt))
