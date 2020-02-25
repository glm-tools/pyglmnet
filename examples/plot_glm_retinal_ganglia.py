# -*- coding: utf-8 -*-
"""
=================================================================
GLM for Spike Trains Prediction in Primate Retinal Ganglion Cells
=================================================================

* Original tutorial adapted from Johnathan Pillow, Princeton University
* Dataset provided by E.J. Chichilnisky, Stanford University
* The dataset is granted by the original authors for the educational use in this tutorial
* If you want to use it beyond educational purposes, please contact ``pillow@princeton.edu``

First of all, we would like to thank Professor Johnathan Pillow and Professor E.J. Chichilnisky
for sharing their tutorial dataset. The original MATLAB and Python tutorial can be found from
https://github.com/pillowlab/GLMspiketraintutorial.

These data were collected by Valerie Uzzell in the lab of
E.J. Chichilnisky at the Salk Institute.  For full information see
Uzzell et al. (J Neurophys 04), or Pillow et al. (J Neurosci 2005).
**Again, the dataset in this tutorial is granted by the original authors
for educational use only not for publication**. If you want to use it
beyond the educational purposes,
please contact ``pillow@princeton.edu``

In this tutorial, we will demonstrate how to fit linear GLM and linear-nonlinear GLM 
(a.k.a Poisson GLM) to predict the spike counts recorded from primate retinal ganglion cells.
The dataset contains spike responses from 2 ON and 2 OFF parasol
retinal ganglion cells (RGCs) in primate retina, stimulated with
full-field `binary white noise`. Two experiment performed consisted of a
long (20-minute) binary stochastic (non-repeating) stimulus
which can be used for computing the spike-triggered average
(or characterizing some other model of the response).
"""

# Authors: Jonathan Pillow <pillow@princeton.edu>
# Edited by: Titipat Achakulvisut <my.titipat@gmail.com>, Konrad Kording <koerding@gmail.com>
# License: MIT

########################################################
#
# First, we can import all the relevance libraries.

import os.path as op
import json

import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import hankel

from pyglmnet import GLM

import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

########################################################
#
# Now, we can download JSON file of the dataset.
# The JSON file has the follow keys:  ``stim`` 
# (a list of binary stochastic stimulation where the value is its stimulation intensity),
# ``stim_times`` (time of the stimulation), and ``spike_times`` (recorded time of the spikes)

from pyglmnet.datasets import fetch_RGCs_data

with TemporaryDirectory(prefix="tmp_glm-tools") as temp_dir:
    dpath = fetch_RGCs_data(dpath=temp_dir)
    rgcs_dataset = json.loads(open(op.join(dpath, '.json'), 'r'))

stim = np.array(rgcs_dataset['stim'])
stim_times = np.array(rgcs_dataset['stim_times'])

# spike times for all 4 cells (0-1 are OFF cells, 2-3 are ON cells)
spike_times = [
    np.array(rgcs_dataset['spike_times']['cell_0']),
    np.array(rgcs_dataset['spike_times']['cell_1']),
    np.array(rgcs_dataset['spike_times']['cell_2']),
    np.array(rgcs_dataset['spike_times']['cell_3']),
]

n_cells = len(spike_times) # total number of cells
dt = stim_times[1] - stim_times[0] # time between the stimulation
n_t = len(stim) # total number of the stimulation
f = 1 / dt # frequency of the stimulation

########################################################
#
# You can pick a cell to work with and visualize the spikes for one second.
# In this case, we will pick cell number 2 (ON cell).

cell_num = 2 # pick cell number 2 (ON cell)
spike_time = spike_times[cell_num] # pick spike time for cell_num
n_spikes = len(spike_time) # number of spikes

print('Loaded RGC data: cell {}'.format(cell_num))
print('Number of stim frames: {:d}  ({:.1f} minutes)'.format(n_t, n_t * dt / 60))
print('Time bin size: {:.1f} ms'.format(dt * 1000))
print('Number of spikes: {} (mean rate = {:.1f} Hz)\n'.format(n_spikes, n_spikes / n_t * 60))

# sample indices for visualization
sample_index = np.arange(120)
t_sample = dt * sample_index

plt.subplot(2, 1, 1)
plt.step(t_sample, stim[sample_index])
plt.title('Raw Stimulus (full field flicker) and corresponding spike times')
plt.ylabel('Stimulation Intensity')

plt.subplot(2, 1, 2)
tspplot = spike_time[(spike_time >= t_sample.min()) 
                     & (spike_time < t_sample.max())]
plt.stem(spike_time[sample_index],
         [1] * len(spike_time[sample_index]))
plt.xlim([t_sample.min(), t_sample.max()])
plt.ylim([0, 2])
plt.xlabel('Time (s)')
plt.ylabel('Spikes (0 or 1)')
plt.show()

########################################################
#
# Next, we can bin the spikes according to the stimulation.
# Then, we can create a so-called 'Design Matrix'
# where each row contains the relevant stimulus chunk.
# You can see the stimulus history of length ``n_t_filt`` for each row of 
# the design matrix. This matrix will be used for predicting the spike counts.

# below, we show an orginal way to make design matrix
t_bins = np.arange(n_t + 1) * dt
spikes_binned, _ = np.histogram(spike_time, t_bins)

n_t_filt = 25 # try changing this to see the difference
stim_padded = np.pad(stim, (n_t_filt - 1, 0)) # left pad with zeros

Xdsgn = np.zeros((n_t, n_t_filt))
for j in np.arange(n_t):
    Xdsgn[j] = stim_padded[j: j + n_t_filt]

plt.imshow(Xdsgn[:50, :],
           cmap='binary',
           aspect='auto',
           interpolation='nearest')
plt.xlabel('lags before spike time', fontsize=12)
plt.ylabel('time bin of response', fontsize=12)
plt.title('Sample first 50 rows of design matrix', fontsize=12)
plt.colorbar()
plt.show()

########################################################
#
# We can also use a function from ``scipy`` called ``hankel`` to do the same as above.
# Hankel matrix is a Toeplitz matrix but flipped left to right.
# It's a faster and more elegant way to create a design matrix with no for loop!

Xdsgn = hankel(stim_padded[:-n_t_filt +1], stim[-n_t_filt:])

# Here, it should give the same matrix as the one created above!
plt.imshow(Xdsgn[:50, :],
           cmap='binary',
           aspect='auto',
           interpolation='nearest')
plt.xlabel('lags before spike time', fontsize=12)
plt.ylabel('time bin of response', fontsize=12)
plt.title('Sample first 50 rows of design matrix created using Hankel',
          fontsize=12)
plt.colorbar()
plt.show()

########################################################
# **Compute Spike-Triggered Average (STA)**
#
# When the stimulus is Gaussian white noise, the STA provides an unbiased
# estimator for the filter in a GLM / LNP model (as long as the nonlinearity
# results in an STA whose expectation is not zero; feel free 
# to ignore this parenthetical remark if you're not interested in technical
# details. It just means that if the nonlinearity is symmetric, 
# eg. :math:`x^2`, then this condition won't hold, and the STA won't be useful.
#
# In many cases it's useful to visualize the STA (even if your stimuli are
# not white noise), just because if we don't see any kind of structure then
# this may indicate that we have a problem (e.g., a mismatch between the
# design matrix and binned spike counts.
#
# It's extremely easy to compute the STA now that we have the design matrix.

sta = Xdsgn.T.dot(spikes_binned) / n_spikes

t_lag = dt * np.arange(-n_t_filt + 1, 1) # time bins for STA (in seconds)
plt.plot(t_lag, t_lag * 0, 'k--')
plt.plot(t_lag, sta, 'bo-')
plt.title('Spike-Triggered Average (STA)', fontsize=15)
plt.xlabel('Time before spike (sec)', fontsize=15)
plt.xlim([t_lag.min(), t_lag.max()])
plt.show()

########################################################
# **Fitting and predicting with a linear-Gaussian GLM**
#
# For a general linear model, an observed spikes can be thought of an underlying 
# parameter :math:`\theta` that control the spike output:
# :math:`y = \vec{\theta} \cdot \vec{x} + \epsilon`,
# :math:`y \sim \text{Poiss}(\vec{\theta} \cdot \vec{x})`
#
# In this case, we can rewrite the formula to :math:`Y = X \vec{\theta} + noise`
#
# Then, you can approximate :math:`\vec{\theta} = (X^T X)^{-1} X^T Y`
#
# If the stimuli are non-white, then the STA is generally a biased
# estimator for the linear filter. In this case we may wish to compute the
# "whitened" STA, which is also the maximum-likelihood estimator for the filter of a 
# GLM with "identity" nonlinearity and Gaussian noise (also known as
# least-squares regression).
#
# If the stimuli have correlations this ML estimate may look like garbage
# (more on this later when we come to "regularization").  But for this
# dataset the stimuli are white, so we don't (in general) expect a big
# difference from the STA.  (This is because the whitening matrix
# :math:`(X_{\text{dsng}}^T * X_{\text{dsgn}})^{-1}` is close to a scaled version of the identity.)

# calculate whitened STA, we should expect it to be similar
# to the STA due to the white noise stimulation

wsta = (inv(Xdsgn.T @ Xdsgn) @ sta) * n_spikes
spikes_pred_lgGLM = Xdsgn @ wsta

plt.plot(t_lag, t_lag * 0, 'k--')
plt.plot(t_lag, sta / norm(sta), 'bo-', label="STA")
plt.plot(t_lag, wsta / norm(wsta), 'ro-', label="wSTA")
plt.legend()
plt.title('STA and whitened STA')
plt.xlabel('time before spike (s)')
plt.xlim([t_lag.min(), t_lag.max()])
plt.show()

########################################################
#
# Now, we can add an offset or a "constant" to our design matrix.
# This can be done by concatenating a column of 1 to our previously created
# design matrix.

Xdsgn_offset = np.hstack((np.ones((n_t,1)), Xdsgn))

wsta_offset = inv(Xdsgn_offset.T @ Xdsgn_offset)\
    @ Xdsgn_offset.T.dot(spikes_binned)

const, wsta_offset = wsta_offset[0], wsta_offset[1:]
spikes_pred_lgGLM_offset = const + Xdsgn @ wsta_offset

########################################################
#
# We can also assume that their is a non-linear function governing
# the underlying the firing patterns. Concreately, we can write down as
# :math:`y = f(\vec{\theta} \cdot \vec{x}) + \epsilon`, and
# :math:`y \sim \text{Poiss}(\vec{\theta} \cdot \vec{x})`.
# We call :math:`f^{-1}` a "link function"
#
# Here, we can use Pyglmnet's `GLM` to predict the parameters

# create possion GLM instance
glm_poisson = GLM(distr='poisson',
                  verbose=False, alpha=0.05,
                  max_iter=1000, learning_rate=0.2,
                  score_metric='pseudo_R2',
                  reg_lambda=1e-7, eta=4.0)

# fitting to a design matrix
glm_poisson.fit(Xdsgn, spikes_binned)

# predict spike counts using Poisson GLM
# alternatively, you can also use np.exp(glm_poisson.beta0_ + X.dot(glm_poisson.beta_))
spikes_pred_poissonGLM = glm_poisson.predict(Xdsgn)

#############################################################################
# **Putting all together**
#
# We are plotting the prediction of spike counts from linear Gaussian GLM, 
# linear Gaussian GLM with offset and poisson prediction for one second.

markerline, _, _ = plt.stem(t_sample, spikes_binned[sample_index])
markerline.set_markerfacecolor('none')
plt.plot(t_sample, spikes_pred_lgGLM[sample_index],
         color='red', linewidth=2, label='lgGLM')
plt.plot(t_sample, spikes_pred_lgGLM_offset[sample_index],
         color='gold', linewidth=2, label='lgGLM with offset')
plt.plot(t_sample, spikes_pred_poissonGLM[sample_index],
         color='green', linewidth=2, label='poissonGLM')

plt.xlim([t_sample.min(), t_sample.max()])
plt.title('Spike count prediction using various GLM predictions')
plt.ylabel('Binned Spike Counts')
plt.legend()
plt.show()

# performance of the fitted models
mse_lgGLM = np.mean((spikes_binned - spikes_pred_lgGLM)**2) # mean squared error, GLM no offset
mse_lgGLM_offset = np.mean((spikes_binned - spikes_pred_lgGLM_offset)**2)  # mean squared error, with offset
mse_poissonGLM = np.mean((spikes_binned - spikes_pred_poissonGLM)**2) # mean squared error, poissonGLM
rss = np.mean((spikes_binned - np.mean(spikes_binned))**2) # squared error of spike train

print('Training perf (R^2): lin-gauss GLM, no offset: {:.2f}'.format(1 - mse_lgGLM / rss))
print('Training perf (R^2): lin-gauss GLM, w/ offset: {:.2f}'.format(1 - mse_lgGLM_offset / rss))
print('Training perf (R^2): poisson GLM {:.2f}'.format(1 - mse_poissonGLM / rss))
print('Training perf using Pyglmnet score {:.2f}'.format(glm_poisson.score(Xdsgn, spikes_binned)))

#############################################################################
# **Using spikes history for predicting spike counts**
#
# We can even further predict the spikes by using the spikes history.
# Below, we show how to do it. **Note** the spike-history portion of the design
# matrix had better be shifted so that we aren't allowed to use the spike
# count on this time bin to predict itself!

n_t_filt = 25 # same as before, stimulation history
n_t_hist = 20 # spikes history

# using both stimulation history and spikes history
stim_padded = np.pad(stim, (n_t_filt - 1, 0))
spikes_padded = np.pad(spikes_binned, (n_t_hist - 1, 0))

Xstim = hankel(stim_padded[:-n_t_filt +1], stim[-n_t_filt:])
Xspikes = hankel(spikes_padded[:-n_t_hist +1], stim[-n_t_hist:])

# design matrix with spikes history
Xdsgn = np.hstack((Xstim, Xspikes))

plt.imshow(Xdsgn[:50, :],
           cmap='binary',
           aspect='auto',
           interpolation='nearest')
plt.xlabel('lags before spike time', fontsize=12)
plt.ylabel('time bin of response', fontsize=12)
plt.title('Sample first 50 rows of design matrix', fontsize=12)
plt.colorbar()
plt.show()

#############################################################################
#
# Now, we are ready to fit Poisson GLM with spikes history.

# create possion GLM instance
glm_poisson_hist = GLM(distr='poisson',
                       verbose=False, alpha=0.05,
                       max_iter=1000, learning_rate=0.2,
                       score_metric='pseudo_R2',
                       reg_lambda=1e-7, eta=4.0)

# fitting to a design matrix with spikes history
glm_poisson_hist.fit(Xdsgn, spikes_binned)

# predict spike counts
spikes_pred_poissonGLM_hist = glm_poisson_hist.predict(Xdsgn)

# plot
markerline, _, _ = plt.stem(t_sample, spikes_binned[sample_index])
markerline.set_markerfacecolor('none')
plt.plot(t_sample, spikes_pred_poissonGLM[sample_index],
         color='green', linewidth=2, label='poissonGLM')
plt.plot(t_sample, spikes_pred_poissonGLM_hist[sample_index],
         color='orange', linewidth=2, label='poissonGLM_hist')

plt.xlim([t_sample.min(), t_sample.max()])
plt.title('Spike counts prediction with spikes history using Poisson GLM')
plt.ylabel('Binned Spike Counts')
plt.legend()
plt.show()
print('Training perf using Pyglmnet score {:.2f}'.format(glm_poisson_hist.score(Xdsgn, spikes_binned)))

#############################################################################
# **Using both spikes history and spikes-coupled for predicting spike counts**

n_t_filt = 25 # same as before, stimulation history
n_t_hist = 20 # spikes history

Xdsgn_coupled = [Xstim]
for cell_num in range(n_cells):
    spike_time_cell = spike_times[cell_num]
    spikes_binned_cell, _ = np.histogram(spike_time_cell, t_bins)
    spikes_padded = np.pad(spikes_binned_cell, (n_t_hist - 1, 0))
    Xspikes = hankel(spikes_padded[:-n_t_hist +1], stim[-n_t_hist:])
    Xdsgn_coupled.append(Xspikes)
# this design matrix should have a width of
# n_t_filt + (n_cells * n_t_hist) = 25 + (20 * 4) = 105
Xdsgn_coupled = np.hstack(Xdsgn_coupled)

# create possion GLM instance
glm_poisson_coupled = GLM(distr='poisson',
                          verbose=False, alpha=0.05,
                          max_iter=1000, learning_rate=0.2,
                          score_metric='pseudo_R2',
                          reg_lambda=1e-7, eta=4.0)

# fitting to a design matrix with spikes history
glm_poisson_coupled.fit(Xdsgn_coupled, spikes_binned)

# predict spike counts
spikes_pred_poissonGLM_couple = glm_poisson_coupled.predict(Xdsgn_coupled)

# plot
markerline, _, _ = plt.stem(t_sample, spikes_binned[sample_index])
markerline.set_markerfacecolor('none')
plt.plot(t_sample, spikes_pred_poissonGLM[sample_index],
         color='green', linewidth=2, label='poissonGLM')
plt.plot(t_sample, spikes_pred_poissonGLM_hist[sample_index],
         color='orange', linewidth=2, label='poissonGLM_hist')

plt.xlim([t_sample.min(), t_sample.max()])
plt.title('Spike counts prediction with spikes couple using Poisson GLM')
plt.ylabel('Binned Spike Counts')
plt.legend()
plt.show()
print('Training perf using Pyglmnet score {:.2f}'.\
      format(glm_poisson_coupled.score(Xdsgn_coupled, spikes_binned)))

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# Please cite the following publications if you use the source code based on this tutorials.
#
# * Uzzell, V. J., and E. J. Chichilnisky. `Precision of spike trains in primate retinal ganglion cells.` Journal of Neurophysiology 92.2 (2004)
# * Pillow, Jonathan W., et al. `Prediction and decoding of retinal ganglion cell responses with a probabilistic spiking model.` Journal of Neuroscience 25.47 (2005)
