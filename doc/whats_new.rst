:orphan:

.. include:: links.inc

.. _whats_new:

.. currentmodule:: pyglmnet

.. _current:

Current
-------

Changelog
~~~~~~~~~

    - Add option for fitting `'probit'` distribution by `Ravi Garg`_.
    - Add option for fitting `'gamma'` distribution with log link by `Mainak Jas`_.
    - Add support for running :class:`sklearn.model_selection.GridSearchCV` on :class:`pyglmnet.GLM` objects by `Mainak Jas`_.
    - Improved testing of gradients for optimization by `Mainak Jas`_ and `Pavan Ramkumar`_.
    - Improved testing of estimated coefficients at convergence by `Mainak Jas`_ and `Pavan Ramkumar`_
    - Add `n_iter_` attribute to :class:`pyglmnet.GLM` by `Mainak Jas`_, `Olivier Pieters`_, `Peter Foley`_ and `Chris Rogers`_
    - Add option to fit intercept or not with a fit_intercept boolean by `Drew Harris`_, `Mainak Jas`_ and `Pavan Ramkumar`_

BUG
~~~

    - Fixed z cache inconsistency in cdfast when alpha and reg_lambda are nonzero by `Peter Foley`_.
    - Fixed incorrect usage of the random_state parameter by `Giovanni De Toni`_.
    - Fixed incorrect proximal operator for group lasso by `Yu Umegaki`_.
    - Changed stopping criteria for convergence (a threshold on the change in objective value) which
      stopped too learly. The new criteria is a threshold on the norm of the gradient, by `Pavan Ramkumar`_.
    - Fixed `group` parameter not being passed for Group Lasso by `Beibin Li`_.
    - Fix URL paths when running on Windows by `Scott Otterson`_.
    - Made temporary file handling OS independent and usable on a cluster by `Scott Otterson`_.
    - Replace StratifiedKFold() with KFold(), as StratifiedKFold() doesn't work for continuous values by `Scott Otterson`_.

API
~~~

    - Refactored `GLM` class into two classes: :class:`pyglmnet.GLM` for fitting a single regularization parameter,
      and :class:`pyglmnet.GLMCV` for fitting along a regularization path with cross-validation and selection of best
      regularization parameter by `Pavan Ramkumar`_ and `Mainak Jas`_.
    - Removed support for fitting `'multinomial'` distribution by `Pavan Ramkumar`_. This option will
      be restored in a future release.


.. _changes_0_14:

Version 0.1
-----------

Changelog
~~~~~~~~~

    - Add GLM class by `Pavan Ramkumar`_
    - Add Tikhonov regularization by `Pavan Ramkumar`_
    - Add group lasso by `Eva Dyer`_
    - Add group lass example by `Matt Antalek`_
    - Add multinomial link function by `Daniel Acuna`_
    - Add benchmarks by `Vinicius Marques`_

.. _Aid Idrizović: https://github.com/the872
.. _Daniel Acuna: https://acuna.io/
.. _Eva Dyer: http://evadyer.github.io/
.. _Hugo Fernandes: https://github.com/hugoguh
.. _Mainak Jas: https://perso.telecom-paristech.fr/mjas/
.. _Matt Antalek: https://github.com/themantalope
.. _Pavan Ramkumar: https://github.com/pavanramkumar
.. _Titipat Achakulvisut: https://github.com/titipata
.. _Vinicius Marques: https://github.com/marquesVF
.. _Ravi Garg: https://github.com/ravigarg27
.. _Yu Umegaki: https://github.com/AnchorBlues
.. _Giovanni De Toni: https://github.com/geektoni
.. _Beibin Li: https://github.com/BeibinLi
.. _Scott Otterson: https://github.com/notuntoward
.. _Peter Foley: https://github.com/peterfoley
.. _Olivier Pieters: https://olivierpieters.be/
.. _Chris Rogers: https://github.com/cxrodgers
.. _Drew Harris: https://github.com/DreHar
