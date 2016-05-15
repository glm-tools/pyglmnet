=================
API Documentation
=================

:class:`GLM` Objects
--------------------

.. py:class:: GLM

   Initialize a Generalized Linear Model (GLM) object

   :param str distr: distribution family can be one of the
       following ``poisson`` or ``poissonexp`` or ``normal`` or ``binomial`` or
       ``multinomial`` default: ``poisson``

   :param float alpha: the weighting between L1 and L2 norm in the penalty term
        of the loss function i.e. P(beta) = 0.5 * (1-alpha) * |beta|_2^2 + alpha * |beta|_1
        default: 0.05

   :param ndarray reg_lambda: array of regularized parameters of penalty term i.e.
        min_(beta0, beta) -L + lambda * P
        where lambda is number in reg_lambda list
        default: None, a list of 10 floats spaced logarithmically (base e)
        between 0.5 and 0.01 is generated.

   :param float learning_rate: learning rate for gradient descent
        default: 1e-4

   :param int max_iter: maximum iterations for the model,
        default: 100

   :param float tol: convergence threshold or stopping criteria.
        Optimization loop will stop below setting threshold
        default: 1e-3

   :param float eta: a threshold parameter that linearizes the `exp()` function
        above eta default: 4.0

   :param int random_state: seed of the random number generator used to
        initialize the solution

   :param boolean/int verbose: if True it will print the output while iterating
        default: False

   Example to create ``GLM`` object

   .. code:: python

      from pyglmnet import GLM
      distr = 'poisson'
      learning_rate = 1e-5 if distr == 'poissonexp' else 1e-4
      glm = GLM(distr, learning_rate=learning_rate)


   .. py:method:: fit(X, y)

      The fit function

      :param ndarray X: array input of data shape `(n_samples, n_features)`

      :param ndarray y: array input of data shape `(n_samples)` or `(n_samples, 1)`

   .. py:method:: predict(X)

      Predict labels

      :param ndarray X: The data for prediction shape `(n_samples, n_features)`

   .. py:method:: fit_predict(X, y)

      :param ndarray X:  The data for fit and prediction shape `(n_samples, n_features)`

      :param ndarray yhat: The predicted labels shape `([n_lambda], n_samples)`
          The predicted labels. A 1D array if predicting on only
          one lambda (compatible with scikit-learn API). Otherwise,
          returns a 2D array.

   .. py:method:: pseudo_R2(y, yhat, ynull)

   .. py:method:: deviance(y, yhat)

   .. py:method:: simulate(beta0, beta, X)
