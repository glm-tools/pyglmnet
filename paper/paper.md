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
    orcid: 0000-0002-2124-2979
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
with $\lambda$ being a hyperparameter that controls the amount of regularization.

Modern datasets can contain a number of predictor variables, and data analysis is often exploratory. To avoid overfitting of the data under these circumstances, it is critically important to regularize the model. Regularization works by adding penalty terms that penalize the model parameters in a variety of ways. It can be used to incorporate our prior knowledge about the parameters' distribution in a structured form.

Despite the attractiveness and importance of regularized GLMs, the available tools in
the Python data science eco-system do not serve all common functionalities. Specifically:

- [statsmodels] provides a wide range of noise distributions but no regularization.
- [scikit-learn] provides elastic net regularization but only limited noise distribution options.
- [lightning] provides elastic net and group lasso regularization, but only for linear (Gaussian) and logistic (binomial) regression.

## Pyglmnet is a response to a fragmented ecosystem

Pyglmnet offers the ability to combine different types of regularization with different GLM noise
distributions. In particular, it implements a broader form of elastic net regularization that include generalized L2 and L1 penalties (Tikhonov regularization and Group Lasso, respectively) with Gaussian, Binomial, Poisson, Probit, and Gamma distributions. The table below compares pyglmnet with existing libraries as of release version 1.1.

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
| L1 (Lasso)              |    x       |      x         |               |       x         |               |         |  x              |
| Generalized L1 (Group Lasso)        |    x       |                |               |       x         |               |         |  x              |
| Generalized L2 (Tikhonov)           |    x       |                |               |                 |               |         |                 |

## Pyglmnet is an extensible pure Python implementation

Pyglmnet implements the algorithm described in [Friedman, J., Hastie, T., & Tibshirani, R. (2010)](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) and its accompanying popular R package [glmnet].
As opposed to [python-glmnet] or [glmnet_python], which are wrappers around this R package, pyglmnet is written in pure Python for Python 3.5+. Therefore, it is easier to extend and more compatible with the existing data science ecosystem.

## Pyglmnet is unit-tested and documented with examples

Pyglmnet has already been used in published work [@bertran2018active; @rybakken2019decoding; @hofling2019probing; @benjamin2017modern]. It contains unit tests and includes [documentation] in the form of tutorials, docstrings and examples that are run through continuous integration.

# Example Usage

Here, we apply pyglmnet to predict incidence of violent crime from the Community and Crime dataset, one of 400+ datasets curated by the UC Irvine Machine Learning Repository [@Dua:2019] which provides a highly curated set of 128 demographic attributes of US counties. The target variable (violent crime per capita) is normalized to the range of $[0, 1]$. Below, we demonstrate the usage of a pyglmnet's binomial-distributed GLM with elastic net regularization.

```py
from sklearn.model_selection import train_test_split
from pyglmnet import GLMCV, simulate_glm, datasets

# Read dataset and split it into train/test
X, y = datasets.fetch_community_crime_data()
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33)

# Instantiate a binomial-distributed GLM with elastic net regularization
glm = GLMCV(distr='binomial', alpha=0.05, score_metric='pseudo_R2', cv=3,
            tol=1e-4)

# Fit the model and then predict
glm.fit(Xtrain, ytrain)
yhat = glm.predict_proba(Xtest)
```

As illustrated above, pyglmnet's API is designed to be compatible with ``scikit-learn`` [@sklearn_api]. Thus, it is possible to use standard idioms such as:

```py
           glm.fit(X, y)
           glm.predict(X)
```

Owing to this compatibility, tools from the ``scikit-learn`` ecosystem for building pipelines, applying cross-validation, and performing grid search over hyperparameters can also be employed with pyglmnet's estimators.

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
[documentation]: https://glm-tools.github.io/pyglmnet/
