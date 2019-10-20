# -*- coding: utf-8 -*-
"""
========================
Tikhonov Regularization
========================

Tikhonov regularization is a generalized form of L2-regularization. It allows
us to articulate our prior knowlege about correlations between
different predictors with a multivariate Gaussian prior. Here, we demonstrate
how pyglmnet's Tikhonov regularizer can be used to estimate spatiotemporal
receptive fields (RFs) from neural data.

Neurons in many brain areas, including the frontal eye fields (FEF) have RFs,
defined as regions in the visual field where visual stimuli are most likely
to result in spiking activity.

These spatial RFs need not be static, they can vary in time in a
systematic way. We want to characterize how such spatiotemporal RFs (STRFs)
remap from one fixation to the next. Remapping is a phenomenon where
the RF of a neuron shifts to process visual information from the subsequent
fixation, prior to the onset of the saccade. The dynamics of this shift
from the "current" to the "future" RF is an active area of research.

With Tikhonov regularization, we can specify a prior covariance matrix
to articulate our belief that parameters encoding neighboring points
in space and time are correlated.

The unpublished data are courtesy of Daniel Wood and Mark Segraves,
Department of Neurobiology, Northwestern University.
"""

########################################################

# Author: Pavan Ramkumar <pavan.ramkumar@gmail.com>
# License: MIT

########################################################
# Imports

import os.path as op
import numpy as np
import pandas as pd

from pyglmnet import GLMCV
from spykes.ml.strf import STRF

import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

########################################################
# Download and fetch data files

from pyglmnet.datasets import fetch_tikhonov_data

with TemporaryDirectory(prefix="tmp_glm-tools") as temp_dir:
    dpath = fetch_tikhonov_data(dpath=temp_dir)
    fixations_df = pd.read_csv(op.join(dpath, 'fixations.csv'))
    probes_df = pd.read_csv(op.join(dpath, 'probes.csv'))
    probes_df = pd.read_csv(op.join(dpath, 'probes.csv'))
    spikes_df = pd.read_csv(op.join(dpath, 'spiketimes.csv'))

spiketimes = np.squeeze(spikes_df.values)

########################################################
# Design spatial basis functions

n_spatial_basis = 36
n_temporal_basis = 7
strf_model = STRF(patch_size=50, sigma=5,
                  n_spatial_basis=n_spatial_basis,
                  n_temporal_basis=n_temporal_basis)
spatial_basis = strf_model.make_gaussian_basis()
strf_model.visualize_gaussian_basis(spatial_basis)

########################################################
# Design temporal basis functions

time_points = np.linspace(-100., 100., 10.)
centers = [-75., -50., -25., 0, 25., 50., 75.]
temporal_basis = strf_model.make_raised_cosine_temporal_basis(
    time_points=time_points,
    centers=centers,
    widths=10. * np.ones(7))
plt.plot(time_points, temporal_basis)
plt.show()

########################################################
# Design parameters

# Spatial extent
n_shape = 50
n_features = n_spatial_basis

# Window of interest
window = [-100, 100]

# Bin size
binsize = 20

# Zero pad bins
n_zero_bins = int(np.floor((window[1] - window[0]) / binsize / 2))

########################################################
# Build design matrix

bin_template = np.arange(window[0], window[1] + binsize, binsize)
n_bins = len(bin_template) - 1

probetimes = probes_df['t_probe'].values
spatial_features = np.zeros((0, n_features))
spike_counts = np.zeros((0,))
fixation_id = np.zeros((0,))

# For each fixation
for fx in fixations_df.index[:1000]:

    # Fixation time
    fixation_time = fixations_df.loc[fx]['t_fix_f']

    this_fixation_spatial_features = np.zeros((n_bins, n_spatial_basis))
    this_fixation_spikecounts = np.zeros(n_bins)
    unique_fixation_id = fixations_df.loc[fx]['trialNum_f']
    unique_fixation_id += 0.01 * fixations_df.loc[fx]['fixNum_f']
    this_fixation_id = unique_fixation_id * np.ones(n_bins)

    # Look for probes in window of interest relative to fixation
    probe_ids = np.searchsorted(probetimes,
                                [fixation_time + window[0] + 0.1,
                                 fixation_time + window[1] - 0.1])

    # For each such probe
    for probe_id in range(probe_ids[0], probe_ids[1]):

        # Check if probe lies within spatial region of interest
        fix_row = fixations_df.loc[fx]['y_curFix_f']
        fix_col = fixations_df.loc[fx]['x_curFix_f']
        probe_row = probes_df.loc[probe_id]['y_probe']
        probe_col = probes_df.loc[probe_id]['x_probe']

        if ((probe_row - fix_row) > -n_shape / 2 and
            (probe_row - fix_row) < n_shape / 2 and
            (probe_col - fix_col) > -n_shape / 2 and
                (probe_col - fix_col) < n_shape / 2):

            # Get probe timestamp relative to fixation
            probe_time = probes_df.loc[probe_id]['t_probe']
            probe_bin = np.where(bin_template < (probe_time - fixation_time))
            probe_bin = probe_bin[0][-1]

            # Define an image based on the relative locations
            img = np.zeros(shape=(n_shape, n_shape))
            row = int(-np.round(probe_row - fix_row) + n_shape / 2 - 1)
            col = int(np.round(probe_col - fix_col) + n_shape / 2 - 1)
            img[row, col] = 1

            # Compute projection
            basis_projection = strf_model.project_to_spatial_basis(
                img, spatial_basis)
            this_fixation_spatial_features[probe_bin, :] = basis_projection

    # Count spikes in window of interest relative to fixation
    bins = fixation_time + bin_template
    searchsorted_idx = np.searchsorted(spiketimes,
                                       [fixation_time + window[0],
                                        fixation_time + window[1]])
    this_fixation_spike_counts = np.histogram(
        spiketimes[searchsorted_idx[0]:searchsorted_idx[1]], bins)[0]

    # Accumulate
    fixation_id = np.concatenate((fixation_id, this_fixation_id), axis=0)
    spatial_features = np.concatenate((spatial_features,
                                       this_fixation_spatial_features), axis=0)
    spike_counts = np.concatenate((spike_counts,
                                   this_fixation_spike_counts), axis=0)

    # Zero pad
    spatial_features = np.concatenate((
        spatial_features, np.zeros((n_zero_bins, n_spatial_basis))))
    fixation_id = np.concatenate((fixation_id, -999. * np.ones(n_zero_bins)))

# Convolve with temporal basis
features = strf_model.convolve_with_temporal_basis(spatial_features,
                                                   temporal_basis)

# Remove zeropad
features = features[fixation_id != -999.]

########################################################
# Visualize the distribution of spike counts

plt.hist(spike_counts, 10)
plt.show()

########################################################
# Plot a few rows of the design matrix

plt.imshow(features[30:150, :], interpolation='none')
plt.show()

#################################################################
# Design prior covariance matrix for Tikhonov regularization
prior_cov = strf_model.design_prior_covariance(
    sigma_temporal=3.,
    sigma_spatial=5.)

plt.imshow(prior_cov, cmap='Greys', interpolation='none')
plt.colorbar()
plt.show()

np.shape(prior_cov)

########################################################
# Fit models
from sklearn.model_selection import train_test_split # noqa

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    features, spike_counts,
    test_size=0.2,
    random_state=42)

########################################################
from pyglmnet import utils # noqa

n_samples = Xtrain.shape[0]
Tau = utils.tikhonov_from_prior(prior_cov, n_samples)

glm = GLMCV(distr='poisson', alpha=0., Tau=Tau, score_metric='pseudo_R2', cv=3)
glm.fit(Xtrain, Ytrain)
print("train score: %f" % glm.score(Xtrain, Ytrain))
print("test score: %f" % glm.score(Xtest, Ytest))
weights = glm.beta_

########################################################
# Visualize

for time_bin_ in range(n_temporal_basis):
    RF = strf_model.make_image_from_spatial_basis(
        spatial_basis,
        weights[range(time_bin_, n_spatial_basis * n_temporal_basis,
                      n_temporal_basis)])
    plt.subplot(1, n_temporal_basis, time_bin_ + 1)
    plt.imshow(RF, cmap='Blues', interpolation='none')
    titletext = str(centers[time_bin_])
    plt.title(titletext)
    plt.axis('off')
plt.show()

########################################################
