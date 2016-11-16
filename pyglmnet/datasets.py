"""
A set of convenience functions to download datasets for illustrative examples
"""
import urllib
import pandas as pd
import os
import shutil
import tempfile
import itertools
import numpy as np
from scipy.misc import comb


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


def fetch_group_lasso_datasets():
    """
    Downloads and formats data needed for the group lasso example.

    Returns:
    --------
    design_matrix: pandas.DataFrame
        pandas dataframe with formatted data and labels

    groups: list
        list of group indicies, the value of the ith position in the list
        is the group number for the ith regression coefficient

    """

    # helper functions

    def find_interaction_index(seq, subseq,
                               alphabet="ATGC",
                               all_possible_len_n_interactions=None):
        n = len(subseq)
        alphabet_interactions = \
            [set(p) for
             p in list(itertools.combinations_with_replacement(alphabet, n))]

        num_interactions = len(alphabet_interactions)
        if all_possible_len_n_interactions is None:
            all_possible_len_n_interactions = \
                [set(interaction) for
                 interaction in
                 list(itertools.combinations_with_replacement(seq, n))]

        subseq = set(subseq)

        group_index = num_interactions * \
            all_possible_len_n_interactions.index(subseq)
        value_index = alphabet_interactions.index(subseq)

        final_index = group_index + value_index
        return final_index

    def create_group_indicies_list(seqlength=7,
                                   alphabet="ATGC",
                                   interactions=[1, 2, 3],
                                   include_extra=True):
        alphabet_length = len(alphabet)
        index_groups = []
        if include_extra:
            index_groups.append(0)
        group_count = 1
        for inter in interactions:
            n_interactions = comb(seqlength, inter)
            n_alphabet_combos = comb(alphabet_length,
                                     inter,
                                     repetition=True)

            for x1 in range(int(n_interactions)):
                for x2 in range(int(n_alphabet_combos)):
                    index_groups.append(int(group_count))

                group_count += 1
        return index_groups

    def create_feature_vector_for_sequence(seq,
                                           alphabet="ATGC",
                                           interactions=[1, 2, 3]):
        feature_vector_length = \
            sum([comb(len(seq), inter) *
                 comb(len(alphabet), inter, repetition=True)
                 for inter in interactions]) + 1

        feature_vector = np.zeros(int(feature_vector_length))
        feature_vector[0] = 1.0
        for inter in interactions:
            # interactions at the current level
            cur_interactions = \
                [set(p) for p in list(itertools.combinations(seq, inter))]
            interaction_idxs = \
                [find_interaction_index(
                 seq, cur_inter,
                 all_possible_len_n_interactions=cur_interactions) + 1
                 for cur_inter in cur_interactions]
            feature_vector[interaction_idxs] = 1.0

        return feature_vector

    positive_url = \
        "http://genes.mit.edu/burgelab/maxent/ssdata/MEMset/train5_hs"
    negative_url = \
        "http://genes.mit.edu/burgelab/maxent/ssdata/MEMset/train0_5_hs"

    pos_file = tempfile.NamedTemporaryFile(bufsize=0)
    neg_file = tempfile.NamedTemporaryFile(bufsize=0)

    urllib.urlretrieve(positive_url, pos_file.name)
    urllib.urlretrieve(negative_url, neg_file.name)

    positive_sequences = [str(line.strip().upper()) for idx, line in
                          enumerate(pos_file.readlines())
                          if ">" not in line and idx < 2 * 8000]

    negative_sequences = [str(line.strip().upper()) for idx, line in
                          enumerate(neg_file.readlines())
                          if ">" not in line and
                          idx < 2 * len(positive_sequences)]

    assert len(positive_sequences) == len(negative_sequences), \
        "lengths were not the same: p={pos} n={neg}" \
        .format(pos=len(positive_sequences), neg=len(negative_sequences))

    positive_vector_matrix = np.array([create_feature_vector_for_sequence(s)
                                       for s in positive_sequences])
    negative_vector_matrix = np.array([create_feature_vector_for_sequence(s)
                                       for s in negative_sequences])

    df = pd.DataFrame(data=np.vstack((positive_vector_matrix,
                                      negative_vector_matrix)))
    df.loc[0:positive_vector_matrix.shape[0], "Label"] = 1.0
    df.loc[positive_vector_matrix.shape[0]:, "Label"] = 0.0

    design_matrix = df
    groups = create_group_indicies_list()

    return design_matrix, groups
