import numpy as np
from scipy.special import rel_entr

def psi(train, 
        oos):
    EPS = 1e-7
    train = (train + EPS) / np.sum(train + EPS)
    oos = (oos + EPS) / np.sum(oos + EPS)
    return np.sum(rel_entr(train, oos) + rel_entr(oos, train))