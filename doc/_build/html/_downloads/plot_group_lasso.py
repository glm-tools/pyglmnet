"""
=========================
Group Lasso Example
=========================

This is an example demonstrating Pyglmnet with
multinomial the group lasso regularization, typical in regression
problems where it is reasonable to impose penalties to model parameters
in a group-wise fashion based on domain knowledge.


"""

import requests
from pyglmnet import GLM
import itertools
import pandas as pd
import numpy as np
from scipy.special import comb
from tqdm import tqdm

##########################################################
#
#Group Lasso Example
#similar to method found in:
#ftp://ftp.stat.math.ethz.ch/Manuscripts/buhlmann/lukas-sara-peter.pdf
#
#The task here is to determine which base pairs and positions within a 7-mer
#sequence are most important to predicting if the sequence contains a splice
#site or not.
#
##########################################################


#first we need to get the data
#the first dataset is the positive examples
print("Fetching data...")
positives = requests.get(url="http://genes.mit.edu/burgelab/maxent/ssdata/MEMset/train5_hs")
negatives = requests.get(url="http://genes.mit.edu/burgelab/maxent/ssdata/MEMset/train0_5_hs")
print("Data fetched.")

positive_sequences = [str(line.strip().upper()) for idx, line in enumerate(tqdm(positives.text.split("\n"), desc="Getting positive sequences"))
                      if ">" not in line and idx < 2 * 8000]

#we need to make sure that we have balanced set of negative and positive
#training data

negative_sequences = [str(line.strip().upper()) for idx, line in
                      enumerate(tqdm(negatives.text.split("\n"), desc="Getting negative sequences"))
                      if ">" not in line and
                      idx < 2 * len(positive_sequences)]

assert len(positive_sequences) == len(negative_sequences), "Something not right, lengths were not the same: p={pos} n={neg}".format(pos=len(positive_sequences),
                                                                                                                                    neg=len(negative_sequences))

#now to set up the group indicies
#we will need to model all possible 1, 2 and 3 way interactions between
#the base pairs present in the length 7 sequences



def find_interaction_index(seq, subseq, alphabet = "ATGC", all_possible_len_n_interactions = None):
    n = len(subseq)
    alphabet_interactions = [set(p) for p in list(itertools.combinations_with_replacement(alphabet, n))]

    num_interactions = len(alphabet_interactions)
    if all_possible_len_n_interactions is None:
        all_possible_len_n_interactions = [set(interaction) for interaction in list(itertools.combinations_with_replacement(seq, n))]

    subseq = set(subseq)

    group_index = num_interactions * all_possible_len_n_interactions.index(subseq)
    value_index = alphabet_interactions.index(subseq)

    final_index = group_index + value_index
    return final_index


def create_group_indicies_list(seqlength = 7, alphabet = "ATGC", interactions = [1, 2, 3], include_extra=True):
    alphabet_length = len(alphabet)
    index_groups = []
    if include_extra:
        index_groups.append(0)
    group_count = 1
    for inter in interactions:
        n_interactions = comb(seqlength, inter)
        n_alphabet_combos = comb(alphabet_length, inter, repetition = True)

        for x1 in range(int(n_interactions)):
            for x2 in range(int(n_alphabet_combos)):
                index_groups.append(int(group_count))

            group_count += 1
    return index_groups

def create_feature_vector_for_sequence(seq, alphabet = "ATGC", interactions = [1, 2, 3]):
    interactions_seqs = []
    feature_vector_length = sum([comb(len(seq), inter) * comb(len(alphabet), inter, repetition = True) for inter in interactions]) + 1

    feature_vector = np.zeros(int(feature_vector_length))
    feature_vector[0] = 1.0
    for inter in interactions:
        #interactions at the current level
        cur_interactions = [set(p) for p in list(itertools.combinations(seq, inter))]
        interaction_idxs = [find_interaction_index(seq, cur_inter , all_possible_len_n_interactions=cur_interactions)+1 for cur_inter in cur_interactions]
        feature_vector[interaction_idxs] = 1.0

    return feature_vector



group_idxs = create_group_indicies_list()
all_int = np.all([isinstance(g, int) for g in group_idxs])
assert all_int, "Not all values were integer: {pos}, {val}".format(pos=all_int.index(False), val=group_idxs[all_int.index(False)])


#next we need to create the feature vector matricies for our positive and
#negative sequences
positive_vector_matrix = np.array([create_feature_vector_for_sequence(s) for s in tqdm(positive_sequences, desc="Creating positive sequence matrix")])
negative_vector_matrix = np.array([create_feature_vector_for_sequence(s) for s in tqdm(negative_sequences, desc="Creating negative sequence matrix")])


#finally, make a dataframe with all the data in it
df = pd.DataFrame(data=np.vstack((positive_vector_matrix, negative_vector_matrix)))
df.loc[0:positive_vector_matrix.shape[0], "Label"] = 1.0
df.loc[positive_vector_matrix.shape[0]:, "Label"] = 0.0

#set up the group lasso GLM model

gl_glm = GLM(distr="binomial",
             group=group_idxs,
             max_iter=100000,
             learning_rate=1e-2,
             tol=1e-5,
             score_metric="deviance",
             alpha=1.0,
             reg_lambda=np.logspace(np.log(100), np.log(0.01), 10, base=np.exp(1)))


#set up the non group GLM model

glm = GLM(distr="binomial",
          max_iter=100000,
          learning_rate=1e-2,
          tol=1e-5,
          score_metric="deviance",
          alpha=1.0,
          reg_lambda=np.logspace(np.log(100), np.log(0.01), 10, base=np.exp(1)))

print("gl_glm: \n", gl_glm)
print("glm: \n", glm)

X = df[df.columns.difference(["Label"]).values]
y = df.loc[:, "Label"]

print("Fitting models")
gl_glm.fit(X.values, y.values)
glm.fit(X.values, y.values)
print("Model fitting complete.")
print("\n\n")


print("Group lasso post fitting score: ", gl_glm.score(X.values, y.values))
print("Non-group lasso post fitting score: ", glm.score(X.values, y.values))
