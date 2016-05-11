|Travis|_

.. |Travis| image:: https://api.travis-ci.org/pavanramkumar/pyglmnet.png?branch=master
.. _Travis: https://travis-ci.org/pavanramkumar/pyglmnet

# pyglmnet

Python implementation of elastic-net regularized generalized linear models.

I follow the same approach and notations as in
[Friedman, J., Hastie, T., & Tibshirani, R. (2010)](https://core.ac.uk/download/files/153/6287975.pdf)
and the accompanying widely popular [R package](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html).

The key difference is that we use ordinary batch gradient descent instead of
co-ordinate descent, which is very fast for `N x p` of up to `10000 x 1000`.

You can find some resources [here](doc/resources.md).


### Simulating data and fitting a GLM in 5 minutes

Clone the repository.

```bash
$ git clone http://github.com/pavanramkumar/pyglmnet
```

Install `pyglmnet` using `setup.py` as following

```bash
$ python setup.py develop install
```


### Getting Started

Here is an example on how to use `GLM` class.

```python
import numpy as np
import scipy.sparse as sps
from sklearn.preprocessing import StandardScaler
from pyglmnet import GLM

# create an instance of the GLM class
model = GLM(distr='poisson', verbose=True, alpha=0.05)

n_samples, n_features = 10000, 100

# coefficients
beta0 = np.random.normal(0.0, 1.0, 1)
beta = sps.rand(n_features, 1, 0.1)
beta = np.array(beta.todense())

# training data
Xr = np.random.normal(0.0, 1.0, [n_samples, n_features])
yr = model.simulate(beta0, beta, Xr)

# testing data
Xt = np.random.normal(0.0, 1.0, [n_samples, n_features])
yt = model.simulate(beta0, beta, Xt)

# fit Generalized Linear Model
scaler = StandardScaler().fit(Xr)
model.fit(scaler.transform(Xr), yr)

# we'll get .fit_params after .fit(), here we get one set of fit parameters
fit_param = model[-1].fit_

# we can use fitted parameters to predict
yhat = model.predict(scaler.transform(Xt))
```

You can also work through given Jupyter notebook demo
[`pyglmnet_example.ipynb`](http://nbviewer.jupyter.org/github/pavanramkumar/pyglmnet/blob/master/notebooks/pyglmnet_example.ipynb)


### Tutorial

A more extensive tutorial on posing and fitting the GLM is in
[`glmnet_tutorial.ipynb`](http://nbviewer.jupyter.org/github/pavanramkumar/pyglmnet/blob/master/notebooks/glmnet_tutorial.ipynb)

### Note

We don't use the canonical link function ```exp()``` for ```'poisson'``` targets.
Instead, we use the softplus function: ```log(1+exp())``` for numerical stability.

### To contribute

We welcome any pull requests. You can run
`nosetests tests` before for making pull requests
to ensure that the changes work.

### Author

* [Pavan Ramkumar](http:/github.com/pavanramkumar)

### Contributors

* [Daniel Acuna](http:/github.com/daniel-acuna)
* [Titipat Achakulvisut](http:/github.com/titipata)
* [Hugo Fernandes](http:/github.com/hugoguh)
* [Mainak Jas](http:/github.com/jasmainak)

### License

MIT License Copyright (c) 2016 Pavan Ramkumar
