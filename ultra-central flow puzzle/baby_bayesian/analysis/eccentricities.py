"""
Script to calculate weighted eccentricity cumulants.
Author: OptimusThi
"""

import numpy as np

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def epsilon_cumulants(eps, weights):
    eps2 = eps**2
    eps4 = eps**4

    m2 = weighted_mean(eps2, weights)
    m4 = weighted_mean(eps4, weights)

    eps_2 = np.sqrt(m2)
    eps_4 = (2*m2**2 - m4)**0.25

    return eps_2, eps_4
