"""
A set of convenience functions to download datasets for illustrative examples
"""
import urllib
import pandas as pd
import os
import shutil
import numpy as np


def fetch_tikhonov_data(dpath='/tmp/glm-tools'):
    """
    Downloads data for Tikhonov example and returns data frames

    Parameters
    ----------
    dpath: str
        specifies path to which the data files should be downloaded

    Returns
    -------
    fixations_df: DataFrame
        data frame with fixation event data
    probes_df: DataFrame
        data frame with stimulus probe event data
    spikes_df: DataFrame
        data frame with spike count data
    """
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.mkdir(dpath)

    base_url = "https://raw.githubusercontent.com/glm-tools/datasets/master"
    url = os.path.join(base_url, "tikhonov/fixations.csv")
    fname = os.path.join(dpath, 'fixations.csv')
    urllib.urlretrieve(url, fname)
    fixations_df = pd.read_csv(fname)

    url = os.path.join(base_url, "tikhonov/probes.csv")
    fname = os.path.join(dpath, 'probes.csv')
    urllib.urlretrieve(url, fname)
    probes_df = pd.read_csv(fname)

    url = os.path.join(base_url, "tikhonov/spiketimes.csv")
    fname = os.path.join(dpath, 'spiketimes.csv')
    urllib.urlretrieve(url, fname)
    spikes_df = pd.read_csv(fname, header=None)

    return fixations_df, probes_df, spikes_df


def fetch_community_crime_data(dpath='/tmp/glm-tools'):
    """
    Downloads data for the community crime example,
    removes missing values, extracts features, and
    returns numpy arrays

    Parameters
    ----------
    dpath: str
        specifies path to which the data files should be downloaded

    Returns
    -------
    X: numpy array
        (n_samples x n_features)
    y: numpy array
        (n_samples,)
    """
    if os.path.exists(dpath):
        shutil.rmtree(dpath)
    os.mkdir(dpath)

    fname = os.path.join(dpath, 'communities.csv')
    base_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases")
    url = os.path.join(base_url, "communities/communities.data")
    urllib.urlretrieve(url, fname)

    # Read in the file
    df = pd.read_csv('/tmp/glm-tools/communities.csv', header=None)

    # Remove missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True, axis=1)
    df.dropna(inplace=True, axis=0)
    df.reset_index(inplace=True, drop=True)

    # Extract predictors and target from data frame
    X = np.array(df[df.keys()[range(3, 102)]])
    y = np.array(df[127])

    return X, y
