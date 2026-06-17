import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, time

BASE_DIR       = Path("/home/thiagos/UCC_project/Trento_events")
LHS_DIR        = BASE_DIR / "my_lhs_design_210"
ISO_WORK_DIR   = BASE_DIR / "pbpb_deformed_5.02TeV_new"
TRENTO_BIN     = BASE_DIR / "trento_sync/build/src/trento"
CACHE_DIR      = BASE_DIR / "trento_sync_cache/750k_new"   # separate from std cache
CACHE_DIR.mkdir(parents=True, exist_ok=True)

assert TRENTO_BIN.exists(), "trento_sync binary not found"

os.environ["LD_LIBRARY_PATH"] = "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

# ── tuning ────────────────────────────────────────────────────────────
N_PARALLEL_DESIGNS = 1    # simultaneous design points
CORES_PER_DESIGN   = 60     # event-parallel chunks per design point
EVENTS_PER_DESIGN  = 750_000  # small test first — scale to 1_000_000 after
BASE_SEED          = 42

TRENTO_FIXED = {
    "cross-section":      "7.0",
    "normalization":      "20.0",
    "reduced-thickness":  "0.0",
    "fluctuation":        "0.19",
    "nucleon-min-dist":   "0.81",
}

design_data = np.load(LHS_DIR / "lhs_design_matrix.npz")
X           = design_data["design"]
param_names = list(design_data["param_names"])
idx_w       = param_names.index("w")
print(f"Loaded {len(X)} design points.")

# ── single chunk worker ───────────────────────────────────────────────
def run_chunk(trento_bin, ws1, ws2, n_events, w_val, fixed, seed):
    """Run one trento_sync chunk. Returns numpy array (13 columns) or None."""
    env = {**os.environ,
           "LD_LIBRARY_PATH": "/opt/conda/lib:" + os.environ.get("LD_LIBRARY_PATH", "")}
    cmd = [
        str(trento_bin), str(ws1), str(ws2), str(n_events),
        "-x", fixed["cross-section"],
        "-n", fixed["normalization"],
        "-p", fixed["reduced-thickness"],
        "-k", fixed["fluctuation"],
        "-w", str(w_val),
        "-d", fixed["nucleon-min-dist"],
        "--random-seed", str(42),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print(f"  chunk seed={seed} failed: {proc.stderr[:200]}")
        return None
    lines = [l for l in proc.stdout.splitlines()
             if l.strip() and not l.startswith("#")]
    return np.loadtxt(lines) if lines else None

# ── one design point: inner parallelism over event chunks ─────────────
def run_design_point(idx):
    cache_file = CACHE_DIR / f"events_design_{idx:04d}.npy"
    if cache_file.exists():
        return idx, cache_file, 0.0, "cached"

    design_dir = ISO_WORK_DIR / f"design_{idx:04d}"
    ws1, ws2   = design_dir / "WS1.hdf", design_dir / "WS2.hdf"
    if not (ws1.exists() and ws2.exists()):
        return idx, None, 0.0, "missing HDF5"

    w_val      = X[idx, idx_w]
    chunk_size = EVENTS_PER_DESIGN // CORES_PER_DESIGN

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=CORES_PER_DESIGN) as inner:
        futs = [
            inner.submit(
                run_chunk,
                TRENTO_BIN, ws1, ws2, chunk_size, w_val, TRENTO_FIXED,
                BASE_SEED + idx * 1000 + c   # unique seed per design × chunk
            )
            for c in range(CORES_PER_DESIGN)
        ]
        chunks = [f.result() for f in futs]

    elapsed = time.perf_counter() - t0
    chunks  = [c for c in chunks if c is not None]
    if not chunks:
        return idx, None, elapsed, "all chunks failed"

    arr = np.concatenate(chunks, axis=0)
    np.save(cache_file, arr)
    return idx, cache_file, elapsed, f"{len(arr):,} events | {arr.shape[1]} cols"

# ── outer parallelism: N_PARALLEL_DESIGNS at once ─────────────────────
print(f"Strategy: {N_PARALLEL_DESIGNS} designs × {CORES_PER_DESIGN} cores = "
      f"{N_PARALLEL_DESIGNS * CORES_PER_DESIGN} total")
print(f"Events per design: {EVENTS_PER_DESIGN:,} | "
      f"Total: {EVENTS_PER_DESIGN * len(X):,}")
print(f"Output columns: 13 (trento_sync)")

t_wall = time.perf_counter()
results = {}

with ProcessPoolExecutor(max_workers=N_PARALLEL_DESIGNS) as outer:
    futures = {outer.submit(run_design_point, i): i for i in range(len(X))}
    for fut in as_completed(futures):
        idx, path, elapsed, msg = fut.result()
        results[idx] = path
        status = "✓" if path else "✗"
        print(f"  [{status}] design {idx:02d} | {elapsed:.0f}s | {msg}")

wall = time.perf_counter() - t_wall
good = sum(1 for p in results.values() if p is not None)
print(f"\nDone: {good}/{len(X)} in {wall/60:.1f} min")
print(f"Cache: {CACHE_DIR}")
