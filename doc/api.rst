.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: pyglmnet

GLM Classes
===========

.. currentmodule:: pyglmnet

.. autosummary::
   :toctree: generated/

   GLM
   GLMCV

Distribution Classes
====================

.. currentmodule:: pyglmnet.distributions

.. autosummary::
   :toctree: generated/

   BaseDistribution
   Poisson
   PoissonSoftplus
   NegBinomialSoftplus
   Binomial
   Probit
   GammaSoftplus


Datasets
========

Functions to download the dataset

.. autofunction:: fetch_community_crime_data
.. autofunction:: fetch_group_lasso_data
.. autofunction:: fetch_tikhonov_data
.. autofunction:: fetch_rgc_spike_trains