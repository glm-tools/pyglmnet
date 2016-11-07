"""
A set of convenience functions to download datasets for illustrative examples
"""
import subprocess
import urllib
import pandas as pd
import os


def fetch_tikhonov_data(dpath='/tmp/glm-tools'):
    cmd = 'rm -rf ' + dpath
    subprocess.call(cmd, shell=True)
    cmd = 'mkdir ' + dpath
    subprocess.call(cmd, shell=True)

    fname = os.path.join(dpath, 'fixations.csv')
    url = ("https://raw.githubusercontent.com/"
           "glm-tools/datasets/master/tikhonov/fixations.csv")
    urllib.urlretrieve(url, fname)
    fixations_df = pd.read_csv(fname)

    fname = os.path.join(dpath, 'probes.csv')
    url = ("https://raw.githubusercontent.com/"
           "glm-tools/datasets/master/tikhonov/probes.csv")
    urllib.urlretrieve(url, fname)
    probes_df = pd.read_csv(fname)

    fname = os.path.join(dpath, 'spiketimes.csv')
    url = ("https://raw.githubusercontent.com/"
           "glm-tools/datasets/master/tikhonov/spiketimes.csv")
    urllib.urlretrieve(url, fname)
    spikes_df = pd.read_csv(fname, header=None)

    return fixations_df, probes_df, spikes_df
