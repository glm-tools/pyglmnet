# pyglmnet

Python implementation of elastic-net regularized generalized linear models.

I follow the same approach and notations as in
[Friedman, J., Hastie, T., & Tibshirani, R. (2010)](https://core.ac.uk/download/files/153/6287975.pdf)
and the accompanying widely popular [R package](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html).

The key difference is that I use ordinary batch gradient descent instead of co-ordinate descent, which is very fast for `N x p` of up to `10000 x 1000`.

You can find some resources [here](resources.md).


### Simulating data and fitting a GLM in 5 minutes

Clone the repository.

```bash
$ git clone http://github.com/pavanramkumar/pyglmnet
```


### Documentation

Here is an example on how to use `GLM` class.

```python
import numpy as np
import scipy.sparse as sps
from pyglmnet import GLM

# create Generalized Linear model
model = GLM(distr='poisson', verbose=True, alpha=0.05)

N, p = 10000, 100

# coefficients
beta0 = np.random.normal(0.0, 1.0, 1)
beta = sps.rand(p,1,0.1)
beta = np.array(beta.todense())

# training data
Xr = np.random.normal(0.0, 1.0, [N,p])
yr = model.simulate(beta0, beta, Xr)

# Test data
Xt = np.random.normal(0.0, 1.0, [N,p])
yt = model.simulate(beta0, beta, Xt)

# fit Generalized Linear Model
model.fit(zscore(Xr),yr)

# get one set of fit parameters
fit_param = model.fit_params[-2]
```

You can also work through given Jupyter notebook demo
[`pyglmnet_example.ipynb`](pyglmnet_example.ipynb)


### Tutorial

A more extensive tutorial on posing and fitting the GLM is in
[`glmnet_tutorial.ipynb`](glmnet_tutorial.ipynb)


### Author

* [Pavan Ramkumar](http:/github.com/pavanramkumar)


### Contributors

* [Daniel Acuna](http:/github.com/daniel-acuna)
* [Titipat Achakulvisut](http:/github.com/titipata)
* [Hugo Fernandes](http:/github.com/hugoguh)

### License
MIT License Copyright (c) 2016 Pavan Ramkumar
