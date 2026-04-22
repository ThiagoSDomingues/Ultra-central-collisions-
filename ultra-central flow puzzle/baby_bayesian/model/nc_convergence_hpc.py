#!/usr/bin/env python3
import numpy as np
import subprocess, os, time
from pathlib import Path

# ─────────────────────────────────────────
# USER SETTINGS (EDIT IF NEEDED)
# ─────────────────────────────────────────
EVENT_COUNTS = [10_000, 100_000, 1_000_000]

SIGMA_INEL  = 770.0
SIGMA_NN    = 6.4
B_MAX_10PCT = float(np.sqrt(0.10 * SIGMA_INEL / np.pi))

TRENTO_P    = 0.063
TRENTO_K    = 1.0 / 0.97**2
TRENTO_NORM = 18.12
TRENTO_W    = 0.9
TRENTO_DMIN = 0.52**(1./3.)
TRENTO_SEED = 42

WORK_DIR = Path("./nc_run")
OUTDIR   = WORK_DIR / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRENTO_BIN = os.environ.get("TRENTO_BIN", "trento")

COL_MULT = 3
COL_E2 = 4
COL_E3 = 5
COL_E4 = 6

UC_EDGES = np.arange(0, 11, 1)
UC_CENTRES = 0.5 * (UC_EDGES[:-1] + UC_EDGES[1:])

# ─────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────
def run_trento(n_events):
    print(f"\nGenerating {n_events:,} events...")
    cmd = [
        TRENTO_BIN, "Pb", "Pb", str(n_events),
        "-p", str(TRENTO_P),
        "-k", str(TRENTO_K),
        "-w", str(TRENTO_W),
        "-d", str(TRENTO_DMIN),
        "-n", str(TRENTO_NORM),
        "-x", str(SIGMA_NN),
        "--b-max", str(B_MAX_10PCT),
        "--random-seed", str(TRENTO_SEED),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return parse_output(result.stdout)


def parse_output(text):
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append([float(x) for x in parts[:8]])
    return np.array(rows)


def assign_centrality(arr):
    rank = np.argsort(np.argsort(-arr[:, COL_MULT]))
    return rank / len(arr) * 100.0


def nc(eps):
    m2 = np.mean(eps**2)
    m4 = np.mean(eps**4)
    return m4 / m2**2 - 2.0


def compute_nc(arr):
    cent = assign_centrality(arr)

    results = []
    for lo, hi in zip(UC_EDGES[:-1], UC_EDGES[1:]):
        mask = (cent >= lo) & (cent < hi)
        if mask.sum() < 20:
            results.append((np.nan, np.nan, np.nan))
            continue

        nc2 = nc(arr[mask, COL_E2])
        nc3 = nc(arr[mask, COL_E3])
        nc4 = nc(arr[mask, COL_E4])

        results.append((nc2, nc3, nc4))

    return np.array(results)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    for N in EVENT_COUNTS:
        t0 = time.time()

        arr = run_trento(N)
        nc_vals = compute_nc(arr)

        out_file = OUTDIR / f"nc_{N}.dat"
        np.savetxt(out_file, nc_vals,
                   header="nc2 nc3 nc4 per 1% centrality bin")

        print(f"Saved → {out_file}")
        print(f"Time: {time.time() - t0:.1f} s")
