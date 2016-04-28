# pyglmnet

Python implementation of elastic-net regularized generalized linear models.

I follow the same approach and notations as in:

[Friedman, J., Hastie, T., & Tibshirani, R. (2010). _Regularization paths for generalized linear models via coordinate descent_. Journal of statistical software, 33(1), 1.](https://core.ac.uk/download/files/153/6287975.pdf)

and the accompanying widely popular [R package](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html).
I use ordinary batch gradient descent instead of co-ordinate descent, which is very fast for `N x p` of up to `10000 x 1000`.
You can see more resources [here](resources.md).

### Simulating data and fitting a GLM in 5 minutes

Clone the repository.

```bash
$ git clone http://github.com/pavanramkumar/pyglmnet
```

Work through the demo.

```
pyglmnet_example.ipynb
```

### Tutorial
A more extensive tutorial on posing and fitting the GLM is here:

```
glmnet_tutorial.ipynb
```

### Author
* [Pavan Ramkumar](http:/github.com/pavanramkumar)

### Contributors
* [Daniel Acuna](http:/github.com/daniel-acuna)
* [Titipat Achakulvisut](http:/github.com/titipata)

### License
MIT License Copyright (c) 2016 Pavan Ramkumar