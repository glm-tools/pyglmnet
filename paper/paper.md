---
title: 'Pyglmnet: Python implementation of elastic-net regularized generalized linear models'
tags:
  - Python
  - glm
  - machine-learning
  - lasso
  - elastic-net
authors:
  - name: Mainak Jas
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2"
  - name: Pavan Ramkumar
    affiliation: 3
  - name: Hugo Fernandes
    affiliation: 10
  - name: Daniel Acuna
    affiliation: 6
  - name: Ravi Garg
    affiliation: 5
  - name: Aid Idrizović
    affiliation: 4
  - name: Vinicius Marques
    affiliation: 4
  - name: Tommy Odland
    affiliation: 13
  - name: Giovanni De Toni
    affiliation: 14
  - name: Peter Foley
    affiliation: 11
  - name: Chris Rodgers
    affiliation: 8
  - name: Beibin Li
    affiliation: 12
  - name: Eva Dyer
    affiliation: 9
  - name: Matti Hamalainen
    affiliation: "1, 2"
  - name: Konrad Kording
    affiliation: 5
affiliations:
 - name: Massachusetts General Hospital
   index: 1
 - name: Harvard Medical School
   index: 2
 - name: Balbix
   index: 3
 - name: Unknown
   index: 4
 - name: University of Pennsylvania
   index: 5
 - name: University of Syracuse
   index: 6
 - name: Columbia University
   index: 8
 - name: Georgia Tech
   index: 9
 - name: Rockets of Awesome
   index: 10
 - name: 605
   index: 11
 - name: University of Washington
   index: 12
 - name: Sonat Consulting
   index: 13
 - name: University of Trento
   index: 14

date: 6 September 2019
bibliography: paper.bib
---

# Summary

[Generalized linear models](GLMs) are
well-established tools for regression and classification and are widely
applied across the sciences, economics, business, and finance. They are
uniquely identifiable due to their convex loss and easy to interpret due
to their point-wise non-linearities and well-defined noise models. Mathematically,
we want to solve problems of the form:

$$\min_{\beta_0, \beta} \frac{1}{N} \sum_{i = 1}^N \ell (y_i, \beta_0 + \beta^T x_i)
+ \lambda \mathcal{P}(\beta)$$

where $\ell (y_i, \beta_0 + \beta^T x_i)$ is the negative log-likelihood of an 
observation $i$. and $\mathcal{P}(\cdot)$ is the penalty that regularizes the solution.

In the era of exploratory data analyses with a large number of predictor
variables, it is important to regularize. Regularization prevents
overfitting by penalizing the negative log likelihood and can be used to
articulate prior knowledge about the parameters in a structured form. In Pyglmnet, we offer
users the ability to combine different types of regularization with different noise
distributions in the GLMs.

Despite the attractiveness of regularized GLMs, the available tools in
the Python data science eco-system are highly fragmented. More
specifically,

-  [statsmodels] provides a wide range of link functions but no regularization.
-  [scikit-learn] provides elastic net regularization but only for linear models.
-  [lightning] provides elastic net and group lasso regularization, but only for
   linear and logistic regression.

[Pyglmnet] is a response to this fragmentation. Here are is a comparison with existing toolboxes.

|                    | [pyglmnet] | [scikit-learn] | [statsmodels] |   [lightning]   |   [py-glm]    | [Matlab]|   [glmnet] in R |
|--------------------|:----------:|:--------------:|:-------------:|:---------------:|:-------------:|:-------:|:---------------:|
| **distributions**  |            |                |               |                 |               |         |                 |
| gaussian           |    x       |      x         |      x        |       x         |      x        |    x    |  x              |
| binomial           |    x       |      x         |      x        |       x         |      x        |    x    |  x              |
| poisson            |    x       |                |      x        |                 |      x        |    x    |  x              |
| softplus           |    x       |                |               |                 |               |         |                 |
| probit             |    x       |                |               |                 |               |         |                 |
| gamma              |    x       |                |               |                 |               |    x    |                 |
| **regularization** |            |                |               |                 |               |         |                 |
| l2                 |    x       |      x         |               |       x         |               |         |                 |
| lasso              |    x       |      x         |               |       x         |               |         |  x              |
| group lasso        |    x       |                |               |       x         |               |         |  x              |
| tikhonov           |    x       |                |               |                 |               |         |                 |

It runs on Python 3.5+. The implementation is compatible with the existing data science ecosystem.
Pyglmnet's API is designed to be compatible with scikit-learn, thus it is possible to do::


```py
           glm.fit(X, y)
           glm.predict(y)
```

As a result of this compatibility, we do not reinvent the wheel and scikit-learn tools
for building pipelines, cross-validation and grid search can be reused.

Pyglmnet has already been used in a number of published research investigations
`[bertran2018active; rybakken2019decoding; hofling2019probing; benjamin2017modern]`

It is unit tested and includes documentation in the form of tutorials, docstrings and
examples that are run through continuous integration.

# Acknowledgements

...

[Generalized linear models]: https://en.wikipedia.org/wiki/Generalized_linear_model>`__
[statsmodel]: https://www.statsmodels.org/
[py-glm]: https://github.com/madrury/py-glm/
[scikit-learn]: https://scikit-learn.org/stable/
[statsmodels]:  http://statsmodels.sourceforge.net/devel/glm.html
[lightning]: https://github.com/scikit-learn-contrib/lightning
[Matlab]: https://www.mathworks.com/help/stats/glmfit.html
[pyglmnet]: http://github.com/glm-tools/pyglmnet/
[glmnet]: https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
