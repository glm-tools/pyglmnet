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
  - name: Daniel Acuna
    affiliation: 5
  - name: Ravi Garg
    affiliation: 4
  - name: Peter Foley
    affiliation: 5
  - name: Chris Rodgers
    affiliation: 6
  - name: Eva Dyer
    affiliation: 7
  - name: Hugo Fernandes
    affiliation: 8
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
 - name: University of Pennsylvania
   index: 4
 - name: University of Syracuse
   index: 5
 - name: Columbia University
   index: 6
 - name: Georgia Tech
   index: 7
 - name: Rockets of Awesome
   index: 8

date: 6 September 2019
bibliography: paper.bib
---

# Summary

[Generalized linear models] are
well-established tools for regression and classification and are widely
applied across the sciences, economics, business, and finance. They are
uniquely identifiable due to their convex loss and easy to interpret due
to their point-wise non-linearities and well-defined noise models.

... add two line equation to explain the model and the solvers.

In the era of exploratory data analyses with a large number of predictor
variables, it is important to regularize. Regularization prevents
overfitting by penalizing the negative log likelihood and can be used to
articulate prior knowledge about the parameters in a structured form.

Despite the attractiveness of regularized GLMs, the available tools in
the Python data science eco-system are highly fragmented. More
specifically,

-  [statsmodels] provides a wide range of link functions but no regularization.
-  [scikit-learn] provides elastic net regularization but only for linear models.
-  [lightning] provides elastic net and group lasso regularization, but only for
   linear and logistic regression.

[Pyglmnet] is a response to this fragmentation and here are is a comparison with existing toolboxes.

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
Pyglmnet API was designed to follow scikit-learn as closely as possible.
Specifically, it is possible to do::


```py
           glm.fit(X, y)
           glm.predict(y)
```

The scikit-learn cross-validation and grid search procedure can be used.

Pyglmnet has already been used in a range of domains.
It is unit tested and has documentation (tutorials, docstrings, examples)
and is unit tested.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this: ![Example figure.](figure.png)

[Generalized linear models]: https://en.wikipedia.org/wiki/Generalized_linear_model>`__
[statsmodel]: https://www.statsmodels.org/
[py-glm]: https://github.com/madrury/py-glm/
[scikit-learn]: https://scikit-learn.org/stable/
[statsmodels]:  http://statsmodels.sourceforge.net/devel/glm.html
[lightning]: https://github.com/scikit-learn-contrib/lightning
[Matlab]: https://www.mathworks.com/help/stats/glmfit.html
[pyglmnet]: http://github.com/glm-tools/pyglmnet/
[glmnet]: https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
