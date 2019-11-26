---
title: 'Pyglmnet: Python implementation of elastic-net regularized generalized linear models'
tags:
  - Python
  - glm
  - machine-learning
  - lasso
  - elastic-net
  - group-lasso
authors:
  - name: Mainak Jas
    orcid: 0000-0002-3199-9027
    affiliation: "1, 2"
  - name: Titipat Achakulvisut
    affiliation: 3
  - name: Aid Idrizović
    affiliation: 4
  - name: Daniel Acuna
    affiliation: 5
  - name: Matthew Antalek
    affiliation: 6
  - name: Vinicius Marques
    affiliation: 4
  - name: Tommy Odland
    affiliation: 7
  - name: Ravi Prakash Garg
    affiliation: 6
  - name: Mayank Agrawal
    affiliation: 8
  - name: Yu Umegaki
    affiliation: 9
  - name: Peter Foley
    orcid: 0000-0002-0304-7213
    affiliation: 10
  - name: Hugo Fernandes
    orcid: 0000-0002-0168-4104
    affiliation: 11
  - name: Drew Harris
    affiliation: 12
  - name: Beibin Li
    affiliation: 13
  - name: Olivier Pieters
    orcid: 0000-0002-5473-4849
    affiliation: 14, 20
  - name: Scott Otterson
    affiliation: 15
  - name: Giovanni De Toni
    orcid: 0000-0002-8387-9983
    affiliation: 16
  - name: Chris Rodgers
    orcid: 0000-0003-1762-3450
    affiliation: 17
  - name: Eva Dyer
    affiliation: 18
  - name: Matti Hamalainen
    affiliation: "1, 2"
  - name: Konrad Kording
    affiliation: 3
  - name: Pavan Ramkumar
    orcid: 0000-0001-7450-0727
    affiliation: 19
affiliations:
 - name: Massachusetts General Hospital
   index: 1
 - name: Harvard Medical School
   index: 2
 - name: University of Pennsylvania
   index: 3
 - name: Loyola University
   index: 4
 - name: University of Syracuse
   index: 5
 - name: Northwestern University
   index: 6
 - name: Sonat Consulting
   index: 7
 - name: Princeton University
   index: 8
 - name: NTT DATA Mathematical Systems Inc
   index: 9
 - name: 605
   index: 10
 - name: Rockets of Awesome
   index: 11
 - name: Epoch Capital
   index: 12
 - name: University of Washington
   index: 13
 - name: IDLab-AIRO -- Ghent University -- imec
   index: 14
 - name: Clean Power Research
   index: 15
 - name: University of Trento
   index: 16
 - name: Columbia University
   index: 17
 - name: Georgia Tech
   index: 18
 - name: System1 Biosciences Inc
   index: 19
 - name: Research Institute for Agriculture, Fisheries and Food
   index: 20

date: 6 September 2019
bibliography: paper.bib
---

# Summary

[Generalized linear models] (GLMs) are
well-established tools for regression and classification and are widely
applied across the sciences, economics, business, and finance.
Owing to their convex loss, they are easy and efficient to fit.
Moreover, they are relatively easy to interpret because of their well-defined
noise distributions and point-wise nonlinearities.

Mathematically, a GLM is estimated as follows:

$$\min_{\beta_0, \beta} \frac{1}{N} \sum_{i = 1}^N \mathcal{L} (y_i, \beta_0 + \beta^T x_i)
+ \lambda \mathcal{P}(\beta)$$

where $\mathcal{L} (y_i, \beta_0 + \beta^T x_i)$ is the negative log-likelihood of an
observation ($x_i$, $y_i$), and $\lambda \mathcal{P}(\cdot)$ is the penalty that regularizes the solution,
with $\lambda$ being a constant that controls the amount of regularization.

Modern datasets can contain a number of predictor variables, and
data analysis is often exploratory. Under these conditions it is critically
important to regularize the model to avoid overfitting the data.
Regularization works by adding penalty terms that penalize the model parameters in
a variety of ways. This can be used to incorporate prior knowledge
about the parameters in a structured form. In pyglmnet, we offer
users the ability to combine different types of regularization with different noise
distributions in the GLMs.

Despite the attractiveness of regularized GLMs, the available tools in
the Python data science eco-system are highly fragmented. Specifically:

-  [statsmodels] provides a wide range of link functions but no regularization.
-  [scikit-learn] provides elastic net regularization but only limited noise distribution options.
-  [lightning] provides elastic net and group lasso regularization, but only for
   linear and logistic regression.

[Pyglmnet] is a response to this fragmentation.
The table below compares pyglmnet with existing libraries as of this writing.

|                    | [pyglmnet] | [scikit-learn] | [statsmodels] |   [lightning]   |   [py-glm]    | [Matlab]|   [glmnet] in R |
|--------------------|:----------:|:--------------:|:-------------:|:---------------:|:-------------:|:-------:|:---------------:|
| **Distributions**  |            |                |               |                 |               |         |                 |
| Gaussian           |    x       |      x         |      x        |       x         |      x        |    x    |  x              |
| Binomial           |    x       |      x         |      x        |       x         |      x        |    x    |  x              |
| Poisson            |    x       |                |      x        |                 |      x        |    x    |  x              |
| Poisson (softplus)           |    x       |                |               |                 |               |         |                 |
| Probit             |    x       |                |               |                 |               |         |                 |
| Gamma              |    x       |                |      x        |                 |               |    x    |                 |
| **Regularization** |            |                |               |                 |               |         |                 |
| L2                 |    x       |      x         |               |       x         |               |         |                 |
| Lasso              |    x       |      x         |               |       x         |               |         |  x              |
| Group lasso        |    x       |                |               |       x         |               |         |  x              |
| Tikhonov           |    x       |                |               |                 |               |         |                 |

Pyglmnet implements the algorithm described in [Friedman, J., Hastie, T., & Tibshirani, R. (2010)](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) and the accompanying popular R package [glmnet].
As opposed to [python-glmnet] or [glmnet_python], which are wrappers around this package, pyglmnet is written in pure Python and runs on Python 3.5+. The implementation is compatible with the existing data science ecosystem.
Pyglmnet's API is designed to be compatible with scikit-learn [@sklearn_api]. Thus, it is possible to do:


```py
           glm.fit(X, y)
           glm.predict(X)
```

As a result of this compatibility, ``scikit-learn`` tools for building pipelines, cross-validation and grid search can be employed by pyglmnet users. Pyglmnet has already been used in published work
[@bertran2018active; @rybakken2019decoding; @hofling2019probing; @benjamin2017modern]. It contains unit tests and includes documentation in the form of tutorials, docstrings and
examples that are run through continuous integration.

# Acknowledgements

``Pyglmnet`` development is partly supported by NIH NINDS R01-NS104585 and the Special Research Fund (B.O.F.) of Ghent University.

# References

[Generalized linear models]: https://en.wikipedia.org/wiki/Generalized_linear_model
[statsmodel]: https://www.statsmodels.org/
[py-glm]: https://github.com/madrury/py-glm/
[scikit-learn]: https://scikit-learn.org/stable/
[statsmodels]:  http://statsmodels.sourceforge.net/devel/glm.html
[lightning]: https://github.com/scikit-learn-contrib/lightning
[Matlab]: https://www.mathworks.com/help/stats/glmfit.html
[pyglmnet]: http://github.com/glm-tools/pyglmnet/
[glmnet]: https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
[python-glmnet]: https://github.com/civisanalytics/python-glmnet
[glmnet_python]: https://github.com/bbalasub1/glmnet_python
