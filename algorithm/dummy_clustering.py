import numpy as np

def dummy_clustering(X):
    all = X.shape[0]
    size = all // 2
    return np.concatenate([np.zeros(size), np.ones(all - size)])

def dummy_clustering_2(X):
    all = X.shape[0]
    return np.zeros(all)