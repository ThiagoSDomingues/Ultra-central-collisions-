"""
Script to generate arbitrary nuclear configurations for Pb-208.
"""

import numpy as np
import h5py
from scipy.special import sph_harm

# -----------------------------
# Physical constants for Pb-208
# -----------------------------
A = 208
R0 = 6.62       # fm
a = 0.546       # fm
rho0 = 1.0      # irrelevant for rejection sampling

# -----------------------------
# Spherical harmonics (real)
# -----------------------------
def Y20(theta):
    return np.sqrt(5/(16*np.pi)) * (3*np.cos(theta)**2 - 1)

def Y30(theta):
    return np.sqrt(7/(16*np.pi)) * (5*np.cos(theta)**3 - 3*np.cos(theta))

def Y40(theta):
    return (3/16)*np.sqrt(1/np.pi) * (35*np.cos(theta)**4 - 30*np.cos(theta)**2 + 3)

# -----------------------------
# Deformed Woods–Saxon radius
# -----------------------------
def R_theta(theta, beta2, beta3, beta4):
    return R0 * (
        1
        + beta2 * Y20(theta)
        + beta3 * Y30(theta)
        + beta4 * Y40(theta)
    )

# -----------------------------
# Sample one Pb nucleus
# -----------------------------
def sample_pb_nucleus(beta2, beta3, beta4):
    coords = []

    while len(coords) < A:
        r = np.random.uniform(0, R0 + 3*a)
        costh = np.random.uniform(-1, 1)
        theta = np.arccos(costh)
        phi = np.random.uniform(0, 2*np.pi)

        Rdef = R_theta(theta, beta2, beta3, beta4)
        prob = 1 / (1 + np.exp((r - Rdef)/a))

        if np.random.rand() < prob:
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            coords.append([x, y, z])

    return np.array(coords, dtype=np.float32)

# -----------------------------
# Generate many configurations
# -----------------------------
def generate_configs(
    filename,
    n_configs=10000,
    beta2=0.0,
    beta3=0.0,
    beta4=0.0
):
    configs = np.zeros((n_configs, A, 3), dtype=np.float32)

    for i in range(n_configs):
        configs[i] = sample_pb_nucleus(beta2, beta3, beta4)
        if i % 1000 == 0:
            print(f"{i}/{n_configs}")

    with h5py.File(filename, "w") as f:
        f.create_dataset("configs", data=configs)

    print(f"Saved {filename}")
