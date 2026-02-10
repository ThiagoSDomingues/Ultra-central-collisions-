"""
Script to weighted centrality selection
Author: OptimusThi
"""

import numpy as np

def weighted_quantile(values, weights, quantiles):
    """
    Compute weighted quantiles of a distribution
    """
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]

    return np.interp(quantiles, cumulative, values)
