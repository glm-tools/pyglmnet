## pyglmnet
Python implementation of elastic-net regularized generalized linear models.

Same approach and notations as [Friedman, Hastie, Tibshirani, 2009].

I use ordinary batch gradient descent instead of co-ordinate descent, which is very fast for $N x p$ of $10000 x 1000$.

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
