"""
Script to define ultra-central events
"""

import h5py
import numpy as np

with h5py.File("trento.h5", "r") as f:
    entropy = f["entropy"][:]   # total entropy per event
    eps2 = f["eccentricity_2"][:]
    eps3 = f["eccentricity_3"][:]

# Sort by entropy
idx = np.argsort(entropy)[::-1]

# Ultra-central cut (top 1%)
n_uc = int(0.01 * len(entropy))
uc = idx[:n_uc]

eps2_uc = eps2[uc]
eps3_uc = eps3[uc]

R_eps = eps2_uc.mean() / eps3_uc.mean()

print("ε2/ε3 =", R_eps)
