import ete3
import os
import numpy as np
import scipy as sp
from joblib import load
from skbio import TreeNode, DistanceMatrix
from scipy.cluster.hierarchy import average, dendrogram


def get_names(source="./CodeSnippets/SnipDat/Total/"):
    directories = os.listdir(source)
    names = []
    for i in range(0, len(directories)):
        names.append(directories[i])
    return names


##

def get_model(source="./BeetleModelResults.joblib"):
    clf = load(source)
    return clf


##

def get_features(dest="./datasets/D1/features/", size="416", strategy="-AVG"):
    l1 = np.load(dest + "X-" + size + "-c1" + strategy + ".npy")
    l2 = np.load(dest + "X-" + size + "-c2" + strategy + ".npy")
    l3 = np.load(dest + "X-" + size + "-c3" + strategy + ".npy")
    l4 = np.load(dest + "X-" + size + "-c4" + strategy + ".npy")
    l5 = np.load(dest + "X-" + size + "-c5" + strategy + ".npy")

    a_all = np.concatenate([l1, l2, l3, l4, l5], 1)

    X = [l1, l2, l3, l4, l5, a_all]

    return X


##

def get_classes(dest="./datasets/D1/features/"):
    Y = np.load(dest + "Y.npy")
    return Y


##

def sum_weights(weights):
    k = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
    for i in range(0, len(weights)):
        for j in range(0, len(weights[i])):
            k[i] = k[i] + weights[i][j]
    return k

##

def lower_triangular_from_weights(s_weights):
    l_triangular = np.ndarray((s_weights.size, s_weights.size), dtype=type(s_weights[0]))
    # Clean the matrix cuz floats give some dirty tiny values
    for i in range(0, len(l_triangular)):                       # from 0 to 10
        for j in range(0, len(l_triangular[i])):                # from 0 to 1->10 growing
            l_triangular[i][j] = 0.

    # get the lower triangular
    for i in range(0, len(l_triangular)):                       # from 0 to 10
        for j in range(0, i + 1):                               # from 0 to 1->10, growing
            l_triangular[i][j] = s_weights[i] - s_weights[j]    # w[i][j] is a lower triangular matrix
            l_triangular[i][j] = abs(l_triangular[i][j])        # only care about absolute distance
    return l_triangular

##

def distance_matrix_from_weights(s_weights):
    # get l_triangular and mirror it
    l_triangular = lower_triangular_from_weights(s_weights)
    d_mat = l_triangular
    for i in range(0, len(d_mat)):      # from 0 to 10
        for j in range(0, i+1):         # from 0 to 1->10, growing
            d_mat[j][i] = d_mat[i][j]   # make a symmetrical matrix u for UPGMA
    return d_mat


##


names = get_names()
clf = get_model()
features = get_features()
classes = get_classes()
weights = clf.coef_
summed_weights = sum_weights(weights)
upper_triangular = lower_triangular_from_weights(summed_weights)
dist_mat = distance_matrix_from_weights(summed_weights)

dm = DistanceMatrix(dist_mat)
dm.ids = names

lms = sp.cluster.hierarchy.average(dm.condensed_form())
tree = TreeNode.from_linkage_matrix(lms, dm.ids)

lm = average(dm.condensed_form())
d = dendrogram(lm, labels=dm.ids, orientation='right',
               link_color_func=lambda x: 'black')

dm.plot(cmap="Greens").show()

ts = ete3.TreeStyle()
ts.show_leaf_name = True
ts.scale = 250
ts.branch_vertical_margin = 15

t = ete3.Tree(str(tree))
t.show()
print(tree.ascii_art())
