"""
centrality_selection_trento.py
================================

Load TRENTo events (multiple formats), compute centrality,
and return eccentricities in a chosen centrality interval.

Supported formats:
    - .npy / .npz
    - .txt / .dat (standard TRENTo output)
    - .hdf5 / .h5 (dataset must be specified)

Standard TRENTo columns:
    0: event id
    1: impact parameter b
    2: Npart
    3: multiplicity (entropy)
    4: ε2
    5: ε3
    6: ε4
    7: ε5
"""

import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
EVENTS_FILE = "trento_events.dat"  
HDF5_DATASET = "events"            # used only for .hdf5

N_TRENTO_COLS = 8

COL_MULT = 3
COL_E2   = 4
COL_E3   = 5
COL_E4   = 6
COL_E5   = 7


# ─────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────
def load_events(fname):
    fname = Path(fname)
    ext = fname.suffix.lower()

    if ext == ".npy":
        arr = np.load(fname)

    elif ext == ".npz":
        data = np.load(fname)
        # take first array inside
        arr = data[list(data.keys())[0]]

    elif ext in [".txt", ".dat"]:
        rows = []
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < N_TRENTO_COLS:
                    continue

                try:
                    rows.append([float(x) for x in parts[:N_TRENTO_COLS]])
                except ValueError:
                    continue

        if not rows:
            raise RuntimeError("No valid TRENTo events parsed.")

        arr = np.array(rows, dtype=np.float64)

    elif ext in [".hdf5", ".h5"]:
        import h5py
        with h5py.File(fname, "r") as f:
            if HDF5_DATASET not in f:
                raise KeyError(f"Dataset '{HDF5_DATASET}' not found in file.")
            arr = f[HDF5_DATASET][:]

    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return arr


# ─────────────────────────────────────────────────────────────
# CENTRALITY SELECTION
# ─────────────────────────────────────────────────────────────
def select_centrality(arr, cent_min=0.0, cent_max=1.0):
    """
    Select events in a centrality range [cent_min, cent_max).
    """

    mult = arr[:, COL_MULT]

    # Rank-based centrality (standard in TRENTo pipelines)
    rank = np.argsort(np.argsort(-mult))
    cent = rank / len(mult) * 100.0

    mask = (cent >= cent_min) & (cent < cent_max)

    eps2 = np.abs(arr[:, COL_E2])[mask]
    eps3 = np.abs(arr[:, COL_E3])[mask]
    eps4 = np.abs(arr[:, COL_E4])[mask]
    eps5 = np.abs(arr[:, COL_E5])[mask]

    return eps2, eps3, eps4, eps5, cent[mask]


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    arr = load_events(EVENTS_FILE)

    print(f"\nArray shape : {arr.shape}")
    print(f"N_events    : {len(arr):,}")

    # USER CONTROL HERE
    cent_min = 0
    cent_max = 1

    eps2, eps3, eps4, eps5, cent_sel = select_centrality(
        arr, cent_min, cent_max
    )

    print(f"\nCentrality range: {cent_min}%–{cent_max}%")
    print(f"Events in bin  : {len(eps2):,}")

    # sanity checks
    if len(eps2) > 0:
        print(f"<ε2> = {np.mean(eps2):.4f}")
        print(f"<ε3> = {np.mean(eps3):.4f}")
