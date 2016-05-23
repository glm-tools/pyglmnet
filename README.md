# Python implementation of elastic-net regularized generalized linear models

[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/pavanramkumar/pyglmnet/blob/master/LICENSE) [![Travis](https://api.travis-ci.org/pavanramkumar/pyglmnet.png?branch=master "Travis")](https://travis-ci.org/pavanramkumar/pyglmnet)
[![Coverage Status](https://coveralls.io/repos/github/pavanramkumar/pyglmnet/badge.svg?branch=master)](https://coveralls.io/github/pavanramkumar/pyglmnet?branch=master)
[![Gitter](https://badges.gitter.im/pavanramkumar/pyglmnet.svg)](https://gitter.im/pavanramkumar/pyglmnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

We follow the same approach and notations as in
[Friedman, J., Hastie, T., & Tibshirani, R. (2010)](https://core.ac.uk/download/files/153/6287975.pdf)
and the accompanying widely popular [R package](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html).

### Installation

Clone the repository.

```bash
$ git clone http://github.com/pavanramkumar/pyglmnet
```

Install `pyglmnet` using `setup.py` as follows

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
glm = GLM(distr='poisson', verbose=True, alpha=0.05)

n_samples, n_features = 10000, 100

# sample random coefficients
beta0 = np.random.normal(0.0, 1.0, 1)
beta = sps.rand(n_features, 1, 0.1)
beta = np.array(beta.todense())

# simulate training data
Xr = np.random.normal(0.0, 1.0, [n_samples, n_features])
yr = glm.simulate(beta0, beta, Xr)

# simulate testing data
Xt = np.random.normal(0.0, 1.0, [n_samples, n_features])
yt = glm.simulate(beta0, beta, Xt)

# fit the model on the training data
scaler = StandardScaler().fit(Xr)
glm.fit(scaler.transform(Xr), yr)

# predict using fitted model on the test data
yhat = glm.predict(scaler.transform(Xt))
```

[More `pyglmnet` examples and use cases](http://pavanramkumar.github.io/pyglmnet/auto_examples/plot_poisson.html)


### Tutorial

Here is an [extensive tutorial](http://pavanramkumar.github.io/pyglmnet/auto_examples/index.html) on GLMs with optimization and pseudo-code.

### How to contribute?

We welcome pull requests. Please see our [developer documentation page](http://pavanramkumar.github.io/pyglmnet/developers.html) for more details.

### Author

* [Pavan Ramkumar](http:/github.com/pavanramkumar)

### Contributors

* [Daniel Acuna](http:/github.com/daniel-acuna)
* [Titipat Achakulvisut](http:/github.com/titipata)
* [Hugo Fernandes](http:/github.com/hugoguh)
* [Mainak Jas](http:/github.com/jasmainak)

### License

MIT License Copyright (c) 2016 Pavan Ramkumar
