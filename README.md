## pyglmnet
Python implementation of elastic-net regularized generalized linear models.

I follow the same approach and notations as in:

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), 1.

and the accompanying widely popular [R package](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html).

I use ordinary batch gradient descent instead of co-ordinate descent, which is very fast for N x p of up to 10000 x 1000.

#### Simulating data and fitting a GLM in 5 minutes
Clone the repository.
```
home> $ git clone http://github.com/pavanramkumar/pyglmnet
```
Work through the demo.
```
pyglmnet_example.ipynb
```

A more extensive tutorial is coming soon.
