"""
A set of convenience functions to download datasets for illustrative examples
"""
import urllib
import pandas as pd
import os


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
    os.rmdir(dpath)
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
