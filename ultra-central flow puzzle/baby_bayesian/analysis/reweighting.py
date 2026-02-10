"""
Script to reweighting procedure.
Author: OptimusThi
"""

import numpy as np

# ---------------------------------------
# TARGET DISTRIBUTIONS
# ---------------------------------------

def gaussian_target(m, mean, sigma):
    return np.exp(-0.5 * ((m - mean) / sigma)**2)

def flat_top_target(m, m_min):
    return (m >= m_min).astype(float)

# ---------------------------------------
# WEIGHT COMPUTATION
# ---------------------------------------

def compute_weights(multiplicity, target="gaussian",
                    mean=None, sigma=None, m_min=None,
                    n_bins=200):
    """
    Compute event weights w = P_target / P_model
    """

    # Model multiplicity PDF
    hist, bin_edges = np.histogram(
        multiplicity, bins=n_bins, density=True
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Interpolate model PDF
    P_model = np.interp(multiplicity, bin_centers, hist)

    # Target PDF
    if target == "gaussian":
        assert mean is not None and sigma is not None
        P_target = gaussian_target(multiplicity, mean, sigma)

    elif target == "flat_top":
        assert m_min is not None
        P_target = flat_top_target(multiplicity, m_min)

    else:
        raise ValueError("Unknown target distribution")

    # Avoid division by zero
    eps = 1e-12
    weights = P_target / (P_model + eps)

    # Normalize weights
    weights /= np.mean(weights)

    return weights
