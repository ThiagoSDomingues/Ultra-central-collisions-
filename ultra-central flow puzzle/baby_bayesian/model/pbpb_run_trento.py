#!/usr/bin/env python3
"""
pbpb_run_trento.py
==================

Standalone Trento runner for Pb+Pb ultracentral flow puzzle scans.

This script assumes the following structure already exists:

WORK_DIR/
    lhs_design_matrix.npz
    design_0000/
        WS1.hdf
        WS2.hdf
    design_0001/
        WS1.hdf
        WS2.hdf
    ...

It will:

1. Load the LHS design matrix
2. Loop over design points
3. Run Trento for each design point
4. Save all events to trento_events.npy

Designed to run on:
    - Google Colab
    - clusters
    - local machines

Compatible with the synchronized Trento version:
https://github.com/jppicchetti/trento_sync
"""

import subprocess
import numpy as np
from pathlib import Path
import time
import logging

# =============================================================================
# USER SETTINGS
# =============================================================================

SQRTS = 5.02            # collision energy (2.76 or 5.02)
N_EVENTS = 20000        # events per design point (increase later)

WORK_DIR = Path("pbpb_scan")

TRENTO_BIN = Path("/usr/local/bin/trento")  # change for Colab

FREE_NAMES = ["R","a","beta3","beta4","w"]

# Trento fixed parameters (posterior medians)
TRENTO_FIXED = {
    2.76: {"p":0.0,"k":1.6,"d":1.0,"norm":18.0},
    5.02: {"p":0.0,"k":1.6,"d":1.0,"norm":20.0},
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

log = logging.getLogger(__name__)

# =============================================================================
# LOAD DESIGN
# =============================================================================

def load_design():

    design_file = WORK_DIR / "lhs_design_matrix.npz"

    if not design_file.exists():
        raise RuntimeError("Design matrix not found.")

    data = np.load(design_file)

    return data["design"]


# =============================================================================
# RUN TRENTO
# =============================================================================

def run_trento_for_design(idx, row):

    design_dir = WORK_DIR / f"design_{idx:04d}"

    ws1 = design_dir / "WS1.hdf"
    ws2 = design_dir / "WS2.hdf"

    if not ws1.exists():
        log.warning(f"[{idx:04d}] missing WS1.hdf")
        return

    if not ws2.exists():
        log.warning(f"[{idx:04d}] missing WS2.hdf")
        return

    cache_file = design_dir / "trento_events.npy"

    if cache_file.exists():
        log.info(f"[{idx:04d}] events already exist → skipping")
        return

    params = dict(zip(FREE_NAMES,row))

    fixed = TRENTO_FIXED[SQRTS]

    cmd = [
        str(TRENTO_BIN),
        str(ws1),
        str(ws2),
        str(N_EVENTS),

        "-p", str(fixed["p"]),
        "-k", str(fixed["k"]),
        "-w", f"{params['w']:.6f}",
        "-d", str(fixed["d"]),
        "-n", str(fixed["norm"]),

        "--random-seed", str(1000+idx)
    ]

    log.info(
        f"[{idx:04d}] running Trento "
        f"R={params['R']:.3f} "
        f"a={params['a']:.3f} "
        f"b3={params['beta3']:.3f} "
        f"b4={params['beta4']:.3f} "
        f"w={params['w']:.3f}"
    )

    start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        text=True
    )

    rows = []

    for line in proc.stdout:

        line=line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        parts=line.split()

        if len(parts) < 7:
            continue

        try:
            vals=[float(x) for x in parts[:8]]
            rows.append(vals)
        except:
            pass

    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError("Trento failed.")

    events=np.array(rows,dtype=np.float32)

    np.save(cache_file,events)

    runtime=time.time()-start

    log.info(
        f"[{idx:04d}] saved {len(events)} events "
        f"({runtime:.1f}s)"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\nPbPb Ultracentral Trento Runner")
    print("================================")

    design=load_design()

    print(f"\nDesign points: {len(design)}")
    print(f"Events per point: {N_EVENTS}")
    print()

    t0=time.time()

    for idx,row in enumerate(design):

        run_trento_for_design(idx,row)

    print("\nTotal runtime: %.1f minutes" % ((time.time()-t0)/60))


if __name__=="__main__":
    main()
