"""
Script to calculate cumulants from Trento.
"""

import h5py
import numpy as np

# Read TRENTo output
def load_trento(filename):
    with h5py.File(filename, "r") as f:
        entropy = f["entropy"][:]
        eps2 = f["eccentricity_2"][:]
        eps3 = f["eccentricity_3"][:]
    return entropy, eps2, eps3

# Select ultra-central events
def ultra_central_cut(entropy, percentile=1.0):
    idx = np.argsort(entropy)[::-1]
    n_uc = int(len(entropy) * percentile / 100.0)
    return idx[:n_uc]

# Cumulant calculator
def epsilon_2_cumulant(eps):
    return np.sqrt(np.mean(eps**2))

def epsilon_4_cumulant(eps):
    m2 = np.mean(eps**2)
    m4 = np.mean(eps**4)
    val = 2.0 * m2**2 - m4
    return val**0.25 if val > 0 else np.nan

# Full observable set
def eccentricity_observables(eps2, eps3):
    e22 = epsilon_2_cumulant(eps2)
    e24 = epsilon_4_cumulant(eps2)

    e32 = epsilon_2_cumulant(eps3)

    return {
        "eps2{2}": e22,
        "eps2{4}": e24,
        "eps3{2}": e32,
        "eps2{2}/eps3{2}": e22 / e32,
        "eps2{4}/eps2{2}": e24 / e22
    }

# Run everything
entropy, eps2, eps3 = load_trento("trento.h5")

uc_idx = ultra_central_cut(entropy, percentile=1.0)

obs = eccentricity_observables(
    eps2[uc_idx],
    eps3[uc_idx]
)

for k, v in obs.items():
    print(f"{k:15s} = {v:.4f}")
