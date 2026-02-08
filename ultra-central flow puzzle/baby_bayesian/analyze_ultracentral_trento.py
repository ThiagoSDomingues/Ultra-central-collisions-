"""
Script to select ultra-central events from Trento.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================
HDF_FILE = "events.hdf"
UC_PERCENTILE = 1.0     # ultra-central cut (top 1%)
OUTDIR = Path("plots_ultracentral")
OUTDIR.mkdir(exist_ok=True)

# ======================================================
# LOAD EVENTS
# ======================================================
with h5py.File(HDF_FILE, "r") as f:
    events = list(f.values())

# Extract attributes
mult = np.array([ev.attrs["mult"] for ev in events])
eps2 = np.array([ev.attrs["e2"] for ev in events])
eps3 = np.array([ev.attrs["e3"] for ev in events])
b = np.array([ev.attrs["b"] for ev in events])

# ======================================================
# SORT BY MULTIPLICITY
# ======================================================
idx = np.argsort(mult)[::-1]
events_sorted = [events[i] for i in idx]
mult_sorted = mult[idx]
eps2_sorted = eps2[idx]
eps3_sorted = eps3[idx]
b_sorted = b[idx]

# ======================================================
# ULTRA-CENTRAL SELECTION
# ======================================================
n_uc = int(len(events) * UC_PERCENTILE / 100.0)

uc_events = events_sorted[:n_uc]
uc_eps2 = eps2_sorted[:n_uc]
uc_eps3 = eps3_sorted[:n_uc]
uc_b = b_sorted[:n_uc]

print(f"Selected {n_uc} ultra-central events ({UC_PERCENTILE}%)")
