#!/usr/bin/env python3
"""
Stage 4: Compute eccentricity observables for the PbPb ultracentral scan

Reads Trento event files produced in Stage 3 and computes:

    epsilon2
    epsilon3
    epsilon4
    epsilon5

and the main observables for the ultracentral flow puzzle:

    sqrt(<epsilon_n^2>)
    epsilon2 / epsilon3
    epsilon4 / epsilon3
    epsilon5 / epsilon3

Results are saved for emulator training.
"""

import numpy as np
from pathlib import Path
import logging

# ================================================================
# USER SETTINGS
# ================================================================

WORK_DIR = Path("pbpb_scan")

OUTPUT_FILE = WORK_DIR / "eccentricity_observables.npz"

# number of harmonics we care about
MAX_N = 5

# ================================================================
# LOGGING
# ================================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ================================================================
# LOAD EVENTS
# ================================================================

def load_events(design_dir):

    file = design_dir / "trento_events.npy"

    if not file.exists():
        return None

    data = np.load(file)

    return data


# ================================================================
# COMPUTE ECCENTRICITY STATISTICS
# ================================================================

def compute_statistics(events):

    eps2 = events[:,4]
    eps3 = events[:,5]
    eps4 = events[:,6]
    eps5 = events[:,7]

    stats = {}

    stats["eps2_rms"] = np.sqrt(np.mean(eps2**2))
    stats["eps3_rms"] = np.sqrt(np.mean(eps3**2))
    stats["eps4_rms"] = np.sqrt(np.mean(eps4**2))
    stats["eps5_rms"] = np.sqrt(np.mean(eps5**2))

    stats["eps2_mean"] = np.mean(eps2)
    stats["eps3_mean"] = np.mean(eps3)
    stats["eps4_mean"] = np.mean(eps4)
    stats["eps5_mean"] = np.mean(eps5)

    stats["ratio23"] = stats["eps2_rms"] / stats["eps3_rms"]
    stats["ratio43"] = stats["eps4_rms"] / stats["eps3_rms"]
    stats["ratio53"] = stats["eps5_rms"] / stats["eps3_rms"]

    return stats


# ================================================================
# MAIN LOOP
# ================================================================

def main():

    design_dirs = sorted(WORK_DIR.glob("design_*"))

    results = []

    for d in design_dirs:

        idx = int(d.name.split("_")[1])

        events = load_events(d)

        if events is None:
            log.warning(f"{d} missing events")
            continue

        stats = compute_statistics(events)

        row = [
            idx,
            stats["eps2_rms"],
            stats["eps3_rms"],
            stats["eps4_rms"],
            stats["eps5_rms"],
            stats["ratio23"],
            stats["ratio43"],
            stats["ratio53"]
        ]

        results.append(row)

        log.info(
            f"[{idx:04d}] "
            f"eps2={stats['eps2_rms']:.4f} "
            f"eps3={stats['eps3_rms']:.4f} "
            f"ratio23={stats['ratio23']:.3f}"
        )

    results = np.array(results)

    np.savez(
        OUTPUT_FILE,
        data=results,
        columns=[
            "design_id",
            "eps2_rms",
            "eps3_rms",
            "eps4_rms",
            "eps5_rms",
            "eps2/eps3",
            "eps4/eps3",
            "eps5/eps3"
        ]
    )

    print("\nSaved:", OUTPUT_FILE)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()
