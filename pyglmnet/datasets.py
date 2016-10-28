"""
A set of convenience functions to download datasets for illustrative examples
"""
import subprocess
import pandas as pd


def fetch_tikhonov_data(dpath='/tmp/glm-tools'):
    cmd = 'rm -rf ' + dpath
    subprocess.call(cmd, shell=True)
    cmd = 'mkdir ' + dpath
    subprocess.call(cmd, shell=True)

    fname = dpath + '/' + 'fixations.csv'
    url = ("https://raw.githubusercontent.com/"
           "glm-tools/datasets/master/tikhonov/fixations.csv")
    cmd = 'curl -o ' + fname + ' ' + url
    subprocess.call(cmd, shell=True)
    fixations_df = pd.read_csv(fname)

    fname = dpath + '/' + 'probes.csv'
    url = ("https://raw.githubusercontent.com/"
           "glm-tools/datasets/master/tikhonov/probes.csv")
    cmd = 'curl -o ' + fname + ' ' + url
    subprocess.call(cmd, shell=True)
    probes_df = pd.read_csv(fname)

    fname = dpath + '/' + 'spiketimes.csv'
    url = ("https://raw.githubusercontent.com/"
           "glm-tools/datasets/master/tikhonov/spiketimes.csv")
    cmd = 'curl -o ' + fname + ' ' + url
    subprocess.call(cmd, shell=True)
    spikes_df = pd.read_csv(fname, header=None)

    return fixations_df, probes_df, spikes_df
