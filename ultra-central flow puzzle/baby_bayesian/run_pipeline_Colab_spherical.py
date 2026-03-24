#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Spherical 208Pb+208Pb @ 2.76 TeV — JETSCAPE Prior LHS Scan (SYNC)        ║
# ║  Eccentricities ε₂{2}, ε₃{2}, ε₄{2}, T₃₂, R₄₂, ε₃{2}/ε₃{4}, ε₄{4}⁴   ║
# ║                                                                              ║
# ║  PASTE THIS ENTIRE FILE INTO ONE COLAB CELL AND RUN.                        ║
# ║                                                                              ║
# ║  Prerequisites (run once before this cell):                                 ║
# ║    !apt-get update -qq                                                       ║
# ║    !apt-get install -y -qq cmake g++ libboost-all-dev                       ║
# ║    !rm -rf trento                                                            ║
# ║    !git clone https://github.com/jppicchetti/trento_sync.git                ║
# ║    !cd trento_sync && mkdir build && cd build && cmake .. && make -j$(nproc) && make install ║
# ║  OR simply paste and run -- the script builds TRENTo automatically.         ║
# ║    !pip install -q numpy scipy matplotlib pyyaml h5py                       ║
# ║                                                                              ║
# ║  WHY trento_sync REDUCES NOISE vs standard TRENTo:                          ║
# ║  The synchronized version fixes the same random seeds for both nuclei in    ║
# ║  each event (impact param, orientations, Gamma flucts). This converts       ║
# ║  inter-event fluctuations into correlated cancellations, dramatically        ║
# ║  reducing statistical noise in eccentricity ratios (R₄₂, T₃₂, ε₄{4}⁴)     ║
# ║  which are differences between two large numbers. The noise reduction is    ║
# ║  equivalent to running ~10× more events with standard TRENTo.               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, subprocess, sys, time, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  USER CONFIGURATION  ← edit here
# ─────────────────────────────────────────────────────────────────────────────

# ── TRENTo: auto-detect, or build from source if missing ─────────────────
# This makes the script fully self-contained in a single Colab cell:
# if the binary is not found anywhere it clones and compiles Duke-QCD/trento.

def _scan_for_trento():
    """Search all known locations; return Path if found, else None."""
    candidates = [
        "/usr/local/bin/trento",
        "/usr/bin/trento",
        "/content/trento_sync/build/src/trento",
        "/content/trento_sync/build/trento",
        os.path.expanduser("~/.local/bin/trento"),
        "/root/.local/bin/trento",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return Path(c)
    r = subprocess.run(["which", "trento"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return Path(r.stdout.strip())
    for search_root in ["/content", "/usr"]:
        try:
            r2 = subprocess.run(
                ["find", search_root, "-name", "trento",
                 "-type", "f", "-perm", "/111"],
                capture_output=True, text=True, timeout=15,
            )
            for line in r2.stdout.strip().splitlines():
                line = line.strip()
                if line and os.path.isfile(line) and os.access(line, os.X_OK):
                    return Path(line)
        except Exception:
            pass
    return None


def _build_trento():
    """
    Clone jppicchetti/trento_sync (synchronized TRENTo) and compile it.
    Uses subprocess.run(cwd=) so it works inside a Colab cell
    where shell state is not shared between ! commands.
    Mirrors exactly: cd trento && mkdir build && cd build
                     && cmake .. && make -j4 && make install
    Returns path to the installed/built binary.
    """
    import shutil, multiprocessing
    repo_dir  = Path("/content/trento_sync")
    build_dir = repo_dir / "build"

    print("\n" + "-"*60)
    print("  TRENTo not found -- building jppicchetti/trento_sync from source")
    print("-"*60)

    # 1. System dependencies
    print("[1/4] apt-get update + install cmake g++ libboost-all-dev...")
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    r = subprocess.run(
        ["apt-get", "install", "-y", "-qq",
         "cmake", "g++", "libboost-all-dev"],
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError("apt-get install failed")

    # 2. Clone (rm -rf trento first, exactly as in the user cell)
    print("[2/4] Cloning jppicchetti/trento_sync...")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    r = subprocess.run(
        ["git", "clone",
         "https://github.com/jppicchetti/trento_sync.git",
         str(repo_dir)],
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError("git clone jppicchetti/trento_sync failed")

    # 3. cmake ..  (skip if CMakeCache.txt already present)
    build_dir.mkdir(parents=True, exist_ok=True)
    if (build_dir / "CMakeCache.txt").exists():
        print("[3/4] cmake already done -- skipping (CMakeCache.txt found)")
    else:
        print("[3/4] cmake .. (Release mode for speed)")
        r = subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=str(build_dir),
            check=False,
        )
        if r.returncode != 0:
            raise RuntimeError("cmake failed")

    # 4. make -j4  (mirrors the user cell)
    print("[4/4] make -j4  (takes ~1-2 min)...")
    r = subprocess.run(
        ["make", "-j4"],
        cwd=str(build_dir),
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError("make failed")

    # 5. make install  (to /usr/local/bin/trento)
    print("     make install...")
    r = subprocess.run(
        ["make", "install"],
        cwd=str(build_dir),
        check=False,
    )
    if r.returncode != 0:
        print("     make install failed (non-fatal, using in-tree binary)")

    # Locate binary
    for candidate in [
        Path("/usr/local/bin/trento"),
        build_dir / "src" / "trento",
        build_dir / "trento",
    ]:
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            print(f"  TRENTo ready -> {candidate}")
            return candidate

    found = _scan_for_trento()
    if found:
        print(f"  TRENTo ready -> {found}")
        return found

    raise RuntimeError(
        "Build finished but trento binary not found.\n"
        f"Expected: {build_dir / 'src' / 'trento'} or /usr/local/bin/trento"
    )
def _get_trento_bin():
    """Return the trento binary path, building it if necessary."""
    found = _scan_for_trento()
    if found:
        return found
    return _build_trento()


TRENTO_BIN = _get_trento_bin()

WORK_DIR   = Path("/content/pbpb_spherical_sync")  # all outputs go here
# If using Google Drive: WORK_DIR = Path("/content/drive/MyDrive/pbpb_spherical_sync")

N_DESIGN     = 75         # Latin Hypercube design points
N_EVENTS     = 100_000    # events per design point (100k)
N_LHS_ITER   = 2000       # maximin optimisation trials
SEED_LHS     = 42         # LHS RNG seed
SQRTS        = 2.76       # collision energy [TeV]

# Parallelism: how many TRENTo processes to run simultaneously.
# Colab free tier has 2 vCPUs → N_WORKERS = 2 gives ~1.7x speedup.
# Set to 1 to disable parallelism (safer for debugging).
N_WORKERS    = 2

# ─────────────────────────────────────────────────────────────────────────────
# 2.  JETSCAPE PRIOR RANGES  (Table IV, arXiv:2011.01430)
# ─────────────────────────────────────────────────────────────────────────────
#
# Parameter   TRENTo flag     Prior             Unit / note
# ─────────── ──────────────  ───────────────── ──────────────────────────────
# norm        -n              Uniform[10, 20]   overall multiplicity scale
# p           -p              Uniform[-0.7,0.7] generalised mean exponent
# w           -w              Uniform[0.5, 1.5] nucleon Gaussian width [fm]
# d³_min      --nucleon-      Uniform[0, 1.73]  cube of min nucleon dist [fm³]
#             min-dist                          → d_min = (d³)^(1/3)
# σ_k         -k              Uniform[0.3, 2.0] multiplicity fluct std dev
#                                               → TRENTo k = 1/σ_k²
#
# IMPORTANT NOTES:
#   (a) JETSCAPE floats d³_min with a flat prior in [0, 1.73] fm³, NOT d_min.
#       We sample d³_min uniformly and convert: d_min = (d³_min)^(1/3).
#   (b) TRENTo's -k flag = shape parameter k of the Gamma distribution,
#       k = 1/σ_k².  We sample σ_k and convert.
#   (c) The cross-section σ_NN = 6.4 fm² is fixed for 2.76 TeV.
#   (d) Woods-Saxon for Pb: R = 6.62 fm, a = 0.546 fm (JETSCAPE paper fn. 3)
#       These are fixed (spherical nucleus — no isobar sampler used here).

PARAMS = [
    # name,        lo,    hi,    latex,                       unit
    ("norm",       10.0,  20.0,  r"$N_{2.76}$",               ""),
    ("p",          -0.7,   0.7,  r"$p$",                      ""),
    ("w",           0.5,   1.5,  r"$w$ [fm]",                 "fm"),
    ("d3min",       0.0,   1.73, r"$d^3_\mathrm{min}$ [fm³]", "fm³"),
    ("sigma_k",     0.3,   2.0,  r"$\sigma_k$",               ""),
]
PNAMES  = [p[0] for p in PARAMS]
PLO     = np.array([p[1] for p in PARAMS])
PHI     = np.array([p[2] for p in PARAMS])
PLABELS = [p[3] for p in PARAMS]
N_PAR   = len(PARAMS)

SIGMA_NN = 6.4   # inelastic NN cross section [fm²] @ 2.76 TeV
WS_R     = 6.62  # Pb Woods-Saxon radius [fm]
WS_A     = 0.546 # Pb Woods-Saxon diffuseness [fm]
B_MAX    = 20.0  # maximum impact parameter [fm]

# trento_sync stdout columns (13 columns, vs 8 in stock TRENTo)
# col 0  event_id
# col 1  b          impact parameter [fm]
# col 2  npart
# col 3  mult_tot   total reduced thickness  ← centrality proxy
# col 4  mult_mid   midrapidity multiplicity ← EXTRA vs stock TRENTo
# col 5  e2  |ε₂|   eccentricity magnitude
# col 6  Psi2 Ψ₂    event-plane angle [rad] ← EXTRA (synchronized)
# col 7  e3  |ε₃|
# col 8  Psi3 Ψ₃                            ← EXTRA
# col 9  e4  |ε₄|
# col10  Psi4 Ψ₄                            ← EXTRA
# col11  e5  |ε₅|
# col12  Psi5 Ψ₅                            ← EXTRA
N_SYNC_COLS = 13    # total columns per event line
COL_MULT = 3        # mult_tot  (same position as stock TRENTo)
COL_E2   = 5        # |ε₂|  (shifted +1 vs stock due to mult_mid)
COL_E3   = 7        # |ε₃|  (skip Ψ₂ at col 6)
COL_E4   = 9        # |ε₄|  (skip Ψ₃ at col 8)
COL_E5   = 11       # |ε₅|  (skip Ψ₄ at col 10)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  LATIN HYPERCUBE  (maximin-optimised, pure numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _one_lhs(n, d, rng):
    X = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.uniform(size=n)) / n
    return X

def maximin_lhs(n, d, n_iter, seed):
    rng = np.random.default_rng(seed)
    best, best_dmin = None, -1.0
    for _ in range(n_iter):
        X = _one_lhs(n, d, rng)
        diff = X[:, None] - X[None]
        dist = np.sqrt((diff**2).sum(-1))
        np.fill_diagonal(dist, np.inf)
        dmin = dist.min()
        if dmin > best_dmin:
            best_dmin = dmin
            best = X.copy()
    return best  # unit [0,1]^d

def scale_design(unit, lo, hi):
    return lo + unit * (hi - lo)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRENTO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def trento_flags(row_dict):
    """
    Build TRENTo command flags for one design point.
    row_dict keys: norm, p, w, d3min, sigma_k

    Duke-QCD/trento 2.0 flag reference:
      -p  entropy exponent (generalised mean)
      -k  Gamma fluctuation shape  (= 1/sigma_k^2)
      -w  nucleon Gaussian width [fm]
      -d  minimum nucleon distance [fm]  (= --nucleon-min-dist)
           *** -d and --nucleon-min-dist are THE SAME FLAG ***
           Do NOT pass both — that causes "cannot be specified more than once"
      -n  normalization
      --cross-section  sigma_NN [fm^2]
      --b-max  max impact parameter [fm]

    JETSCAPE prior: d^3_min ~ Uniform[0, 1.73] fm^3
      → d_min = (d3_min)^(1/3), passed as -d
    JETSCAPE prior: sigma_k ~ Uniform[0.3, 2.0]
      → TRENTo k = 1/sigma_k^2, passed as -k
    """
    norm    = row_dict["norm"]
    p_val   = row_dict["p"]
    w_val   = row_dict["w"]
    d3      = row_dict["d3min"]
    sigma_k = row_dict["sigma_k"]

    d_min = d3 ** (1.0 / 3.0)   # convert d^3_min [fm^3] -> d_min [fm]
    k_val = 1.0 / sigma_k ** 2  # convert sigma_k -> TRENTo k = 1/sigma_k^2

    return [
        "-p", f"{p_val:.6f}",
        "-k", f"{k_val:.6f}",
        "-w", f"{w_val:.6f}",
        "-d", f"{d_min:.6f}",    # minimum nucleon distance [fm]
                                  # (same as --nucleon-min-dist; do NOT use both)
        "-n", f"{norm:.6f}",
        "-x", f"{SIGMA_NN:.4f}",       # --cross-section short form
        "--b-max", f"{B_MAX:.1f}",
    ]


def run_trento(idx, row_dict, n_events, work_dir, force=False):
    """
    Run TRENTo for one design point; cache result as .npy.
    Returns float64 array shape (N_events, 8) or None on failure.
    Safe to call from a subprocess worker.
    """
    design_dir = work_dir / f"design_{idx:04d}"
    design_dir.mkdir(parents=True, exist_ok=True)
    cache = design_dir / "trento_events.npy"

    if cache.exists() and not force:
        arr = np.load(cache)
        print(f"  [{idx:04d}] cached ({len(arr):,} events) -- skip",
              flush=True)
        return arr

    flags = trento_flags(row_dict)
    cmd   = [
        str(TRENTO_BIN),
        "Pb", "Pb",
        str(n_events),
        *flags,
        f"--random-seed={1000 + idx}",
    ]

    t0  = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"  [{idx:04d}] TRENTo FAILED (exit {res.returncode})",
              flush=True)
        print(f"    cmd: {' '.join(cmd)}", flush=True)
        for ln in res.stderr.splitlines()[-5:]:
            print(f"    {ln}", flush=True)
        return None

    rows = []
    for ln in res.stdout.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < N_SYNC_COLS:
            continue
        try:
            rows.append([float(x) for x in parts[:N_SYNC_COLS]])
        except ValueError:
            continue

    if not rows:
        print(f"  [{idx:04d}] TRENTo: no parseable events in stdout",
              flush=True)
        return None

    arr = np.array(rows, dtype=np.float64)
    np.save(cache, arr)
    rate = len(arr) / elapsed
    print(f"  [{idx:04d}] {len(arr):,} events | {elapsed:.1f}s | "
          f"{rate:,.0f} evt/s | "
          f"p={row_dict['p']:.3f} w={row_dict['w']:.3f} "
          f"sk={row_dict['sigma_k']:.3f}",
          flush=True)
    return arr


def _worker(args):
    """
    Top-level picklable worker for multiprocessing.Pool.
    args = (idx, row_dict, n_events, work_dir_str, force)
    Returns (idx, elapsed, n_events_got) tuple.
    """
    idx, row_dict, n_events, work_dir_str, force = args
    work_dir = Path(work_dir_str)
    t0  = time.perf_counter()
    arr = run_trento(idx, row_dict, n_events, work_dir, force=force)
    elapsed = time.perf_counter() - t0
    if arr is None:
        return idx, elapsed, 0
    return idx, elapsed, len(arr)


def run_all_parallel(design, n_events, work_dir, n_workers, force=False):
    """
    Run TRENTo for all design points using multiprocessing.Pool.

    On Colab free tier (2 vCPUs), n_workers=2 gives ~1.7x speedup.
    Each TRENTo process is independent, so this is embarrassingly parallel.
    Results are cached to .npy files as they complete.

    Returns list of (idx, elapsed, n_events) tuples in completion order.
    """
    import multiprocessing as mp

    # Build argument list for all design points
    args_list = []
    for idx in range(len(design)):
        row_dict = {p[0]: design[idx, j] for j, p in enumerate(PARAMS)}
        args_list.append((idx, row_dict, n_events, str(work_dir), force))

    # Count how many are already cached (skip those immediately)
    cached = [
        idx for idx in range(len(design))
        if (work_dir / f"design_{idx:04d}" / "trento_events.npy").exists()
        and not force
    ]
    to_run = [a for a in args_list if a[0] not in cached]

    if cached:
        print(f"  {len(cached)} design pts already cached -- skipping")
    print(f"  {len(to_run)} design pts to run  |  {n_workers} workers")
    print()

    results = []

    # Handle cached pts (elapsed=0)
    for idx in cached:
        arr = np.load(work_dir / f"design_{idx:04d}" / "trento_events.npy")
        results.append((idx, 0.0, len(arr)))

    if not to_run:
        return results

    if n_workers == 1:
        # Sequential fallback — easier to debug, same interface
        t_loop = time.perf_counter()
        for i, args in enumerate(to_run):
            idx, elapsed, n_got = _worker(args)
            results.append((idx, elapsed, n_got))
            real_runs = [(e, n) for _, e, n in results if e > 0]
            if real_runs:
                avg_t  = np.mean([e for e,_ in real_runs])
                n_left = len(to_run) - (i + 1)
                wall   = time.perf_counter() - t_loop
                print(f"    -- {i+1}/{len(to_run)} done  |  "
                      f"avg {avg_t:.1f}s/pt  |  "
                      f"ETA {avg_t*n_left/60:.1f} min  |  "
                      f"wall {wall/60:.1f} min", flush=True)
    else:
        # Parallel using imap_unordered so results stream in as they finish
        t_loop = time.perf_counter()
        ctx    = mp.get_context("fork")   # fork is fine on Linux/Colab
        with ctx.Pool(processes=n_workers) as pool:
            done = 0
            for idx, elapsed, n_got in pool.imap_unordered(_worker, to_run):
                results.append((idx, elapsed, n_got))
                done += 1
                real_runs = [(e, n) for _, e, n in results if e > 0]
                avg_t  = np.mean([e for e,_ in real_runs]) if real_runs else 0
                n_left = len(to_run) - done
                wall   = time.perf_counter() - t_loop
                status = "OK" if n_got > 0 else "FAIL"
                print(f"  [{idx:04d}] {status}  {n_got:,} events | "
                      f"{elapsed:.1f}s  --  "
                      f"{done}/{len(to_run)} done  |  "
                      f"ETA {avg_t*n_left/60:.1f} min  |  "
                      f"wall {wall/60:.1f} min", flush=True)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# 5.  CENTRALITY ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_centrality(events):
    """
    Rank-based centrality from descending multiplicity.
    Returns percentile array in [0, 100).
    """
    mult = events[:, COL_MULT]
    rank = np.argsort(np.argsort(-mult))   # 0 = most central
    return rank / len(mult) * 100.0

# ─────────────────────────────────────────────────────────────────────────────
# 6.  ECCENTRICITY CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
#
# TRENTo gives |ε_n| directly per event.  Cumulants are defined as:
#
#   ε_n{2}  = sqrt(<|ε_n|²>)                       2-particle  (RMS)
#   ε_n{4}  = (2<|ε_n|²>² - <|ε_n|⁴>)^(1/4)       4-particle  (suppressed)
#
# Observables:
#   ε₂{2}, ε₃{2}, ε₄{2}             basic RMS eccentricities
#   T₃₂  = ε₃{2} / ε₂{2}            triangularity ratio
#   R₄₂  = ε₂{4} / ε₂{2}            cumulant ratio  (flow-puzzle sensitive)
#   ε₃{2}/ε₃{4}                      ε₃ cumulant ratio
#   ε₄{4}⁴ = 2<ε₄²>² - <ε₄⁴>        raw 4th-order cumulant (sign sensitive)

def _m(eps, order):
    """Raw moment <|ε|^order>."""
    return np.mean(eps**order)

def ecc2_cumulants(eps):
    """
    Compute ε_n{2} and ε_n{4} for a 1-D array of |ε_n| values.
    Returns (e2, e4) or (e2, nan) if the 4-cumulant is negative.
    """
    m2 = _m(eps, 2)
    m4 = _m(eps, 4)
    e2 = np.sqrt(m2)
    inner = 2.0 * m2**2 - m4
    e4 = inner**0.25 if inner >= 0 else np.nan
    return e2, e4

def compute_observables(events, centrality, c_lo, c_hi):
    """
    Compute all observables for events in centrality bin [c_lo, c_hi].
    Returns dict of scalar values.
    """
    mask = (centrality >= c_lo) & (centrality < c_hi)
    sel  = events[mask]
    if len(sel) < 10:
        nans = {k: np.nan for k in
                ["e2_2","e3_2","e4_2","T32","R42","e3_ratio","e4_4c"]}
        nans["n"] = 0
        return nans

    eps2 = sel[:, COL_E2]
    eps3 = sel[:, COL_E3]
    eps4 = sel[:, COL_E4]

    e2_2, e2_4 = ecc2_cumulants(eps2)
    e3_2, e3_4 = ecc2_cumulants(eps3)
    e4_2, e4_4 = ecc2_cumulants(eps4)

    T32    = e3_2 / e2_2 if e2_2 > 0 else np.nan
    R42    = e2_4 / e2_2 if (e2_2 > 0 and not np.isnan(e2_4)) else np.nan
    e3rat  = e3_2 / e3_4 if (e3_4 > 0 and not np.isnan(e3_4)) else np.nan
    # ε₄{4}⁴ raw cumulant — sign sensitive
    e4_4c  = 2.0 * _m(eps4, 2)**2 - _m(eps4, 4)

    return dict(e2_2=e2_2, e3_2=e3_2, e4_2=e4_2,
                T32=T32, R42=R42, e3_ratio=e3rat, e4_4c=e4_4c,
                n=int(mask.sum()))


def compute_centrality_profiles(events, centrality):
    """
    Return dict of arrays over centrality bins for plotting.

    Two sets of bins:
      UC bins : 0-1, 1-2, …, 9-10  (ultracentral, 1% intervals)
      Wide bins: 0-5, 5-10, 10-20, 20-30, …, 80-90  (for RMS overview)
    """
    uc_edges   = np.arange(0, 11, 1)       # 0-10% in 1% steps
    wide_edges = [0,5,10,20,30,40,50,60,70,80,90]

    result = {}
    for tag, edges in [("uc", uc_edges), ("wide", wide_edges)]:
        bins   = list(zip(edges[:-1], edges[1:]))
        mid    = np.array([(lo+hi)/2 for lo,hi in bins])
        keys   = ["e2_2","e3_2","e4_2","T32","R42","e3_ratio","e4_4c"]
        data   = {k: [] for k in keys}
        ns     = []

        for lo, hi in bins:
            obs = compute_observables(events, centrality, lo, hi)
            for k in keys:
                data[k].append(obs[k])
            ns.append(obs["n"])

        result[tag] = dict(mid=mid, bins=bins, n=np.array(ns),
                           **{k: np.array(data[k]) for k in keys})
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 7.  DARK-THEME PLOT STYLE
# ─────────────────────────────────────────────────────────────────────────────

BG    = "#0d0f14"
PANEL = "#13161e"
GRID  = "#1e2230"
TEXT  = "#d8dce8"
TICK  = "#8892a4"

def _dark():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PANEL,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TICK,
        "ytick.color":       TICK,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        TEXT,
        "grid.color":        GRID,
        "grid.linewidth":    0.4,
        "grid.alpha":        0.5,
        "font.size":         9,
        "axes.linewidth":    0.6,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID,
        "legend.labelcolor": TEXT,
        "legend.fontsize":   7.5,
    })

OBS_META = {
    "e2_2":     (r"$\varepsilon_2\{2\}$",    "#4fc3f7"),
    "e3_2":     (r"$\varepsilon_3\{2\}$",    "#ff4081"),
    "e4_2":     (r"$\varepsilon_4\{2\}$",    "#ffd54f"),
    "T32":      (r"$\varepsilon_3\{2\}/\varepsilon_2\{2\}$", "#b39ddb"),
    "R42":      (r"$\varepsilon_2\{4\}/\varepsilon_2\{2\}$", "#80cbc4"),
    "e3_ratio": (r"$\varepsilon_3\{2\}/\varepsilon_3\{4\}$", "#ffcc80"),
    "e4_4c":    (r"$\varepsilon_4\{4\}^4$",  "#f48fb1"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 8.  PLOT: ECCENTRICITIES VS CENTRALITY
# ─────────────────────────────────────────────────────────────────────────────

def plot_vs_centrality(all_profiles, out_dir):
    """
    Two panels per observable:
      Left  — ultracentral 0-10%, 1% bins, all design points as faint lines
               + median band
      Right — RMS over 0-90%, wide bins, all design points + band
    """
    _dark()
    obs_keys = list(OBS_META.keys())
    n_obs    = len(obs_keys)

    fig, axes = plt.subplots(n_obs, 2,
                             figsize=(13, n_obs * 2.6),
                             squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        r"$^{208}$Pb+$^{208}$Pb $\sqrt{s_{NN}}=2.76$ TeV — Spherical nucleus (trento\_sync)"
        f"\nJETSCAPE prior LHS scan  |  {N_DESIGN} design pts  |  {N_EVENTS:,} events / pt",
        fontsize=11, color=TEXT, y=1.005, fontweight="bold")

    for row, key in enumerate(obs_keys):
        label, color = OBS_META[key]

        for col, (tag, xtitle) in enumerate([
                ("uc",   r"Centrality $c$ (%)"),
                ("wide", r"Centrality $c$ (%)")]):

            ax = axes[row, col]
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
            ax.grid(True, ls="--", alpha=0.35)

            # Collect all design-point curves
            curves = []
            for prof in all_profiles:
                p   = prof[tag]
                y   = p[key]
                mid = p["mid"]
                valid = np.isfinite(y)
                if valid.sum() < 2:
                    continue
                ax.plot(mid[valid], y[valid], color=color,
                        alpha=0.12, lw=0.7, zorder=2)
                curves.append(y)

            # Median ± 1-sigma band across design points
            if curves:
                mat    = np.array(curves)          # (N_pts, N_bins)
                x_mid  = all_profiles[0][tag]["mid"]
                med    = np.nanmedian(mat, axis=0)
                lo16   = np.nanpercentile(mat, 16, axis=0)
                hi84   = np.nanpercentile(mat, 84, axis=0)
                valid  = np.isfinite(med)
                ax.plot(x_mid[valid], med[valid], color=color,
                        lw=2.0, zorder=4, label="median")
                ax.fill_between(x_mid[valid], lo16[valid], hi84[valid],
                                color=color, alpha=0.25, zorder=3,
                                label=r"16–84%")

            ax.set_ylabel(label, fontsize=9.5, color=color)
            ax.set_xlabel(xtitle, fontsize=8.5)
            if col == 0:
                ax.set_xlim(0, 10)
                ax.set_title("Ultracentral  (1% bins, 0–10%)",
                             fontsize=8, color=TICK, pad=3)
            else:
                ax.set_xlim(0, 90)
                ax.set_title("Wide centrality  (0–90%)",
                             fontsize=8, color=TICK, pad=3)

            if row == 0:
                ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    path = out_dir / "ecc_vs_centrality.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOT: ECCENTRICITIES VS MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_vs_params(design, scalar_obs, out_dir):
    """
    One figure per centrality class × observable combination.
    Shows each observable vs each of the 5 LHS parameters as scatter.
    Centrality classes: 0-1%, 0-5%, 20-30%.
    """
    _dark()

    cent_classes = [
        ("0-1%",   0,  1),
        ("0-5%",   0,  5),
        ("20-30%", 20, 30),
    ]
    obs_keys = list(OBS_META.keys())

    for c_label, c_lo, c_hi in cent_classes:
        # scalar_obs[idx][c_key] = dict of observables
        c_key = f"{c_lo}-{c_hi}"

        fig, axes = plt.subplots(
            len(obs_keys), N_PAR,
            figsize=(N_PAR * 2.8, len(obs_keys) * 2.4),
            squeeze=False)
        fig.patch.set_facecolor(BG)
        fig.suptitle(
            r"$^{208}$Pb+$^{208}$Pb $\sqrt{s_{NN}}=2.76$ TeV — "
            f"Centrality {c_label}\n"
            "Eccentricities vs JETSCAPE prior parameters",
            fontsize=10.5, color=TEXT, y=1.004, fontweight="bold")

        for r_obs, okey in enumerate(obs_keys):
            obs_label, color = OBS_META[okey]

            for c_par, (pname, _, _, plabel, _) in enumerate(PARAMS):
                ax = axes[r_obs, c_par]
                ax.set_facecolor(PANEL)
                for sp in ax.spines.values():
                    sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
                ax.grid(True, ls="--", alpha=0.3)

                x_vals, y_vals = [], []
                for idx in range(N_DESIGN):
                    obs_dict = scalar_obs[idx].get(c_key)
                    if obs_dict is None:
                        continue
                    y = obs_dict[okey]
                    if not np.isfinite(y):
                        continue
                    x_vals.append(design[idx, c_par])
                    y_vals.append(y)

                if len(x_vals) > 3:
                    x_arr = np.array(x_vals)
                    y_arr = np.array(y_vals)
                    # colour by density
                    try:
                        xy   = np.vstack([x_arr, y_arr])
                        dens = gaussian_kde(xy)(xy)
                        sort = np.argsort(dens)
                        ax.scatter(x_arr[sort], y_arr[sort],
                                   c=dens[sort], cmap="plasma",
                                   s=14, linewidths=0, alpha=0.85, zorder=3)
                    except Exception:
                        ax.scatter(x_arr, y_arr, color=color,
                                   s=10, alpha=0.6, zorder=3)

                    # Spearman-like trend line (running median)
                    if len(x_arr) >= 10:
                        order   = np.argsort(x_arr)
                        xs      = x_arr[order]
                        ys      = y_arr[order]
                        w_size  = max(3, len(xs) // 8)
                        xs_sm, ys_sm = [], []
                        for start in range(0, len(xs)-w_size, w_size//2):
                            xs_sm.append(np.median(xs[start:start+w_size]))
                            ys_sm.append(np.median(ys[start:start+w_size]))
                        ax.plot(xs_sm, ys_sm, color="white",
                                lw=1.2, alpha=0.7, zorder=4)

                if c_par == 0:
                    ax.set_ylabel(obs_label, fontsize=9, color=color)
                if r_obs == len(obs_keys) - 1:
                    ax.set_xlabel(plabel, fontsize=8.5)
                if r_obs == 0:
                    ax.set_title(plabel, fontsize=8.5, color=TEXT, pad=3)

        fig.tight_layout(h_pad=0.3, w_pad=0.4)
        safe = c_label.replace("%","pct").replace("-","_")
        path = out_dir / f"ecc_vs_params_{safe}.pdf"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. PLOT: LHS CORNER PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_lhs_corner(unit_lhs, out_dir):
    """5×5 corner plot of the LHS design in unit space."""
    _dark()
    n = N_PAR
    COLORS = ["#4fc3f7", "#ffd54f", "#ff4081", "#b39ddb", "#80cbc4"]

    fig, axes = plt.subplots(n, n, figsize=(n*2.6, n*2.6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        r"$^{208}$Pb+$^{208}$Pb LHS Design — JETSCAPE Prior Ranges"
        f"\n{N_DESIGN} pts × {N_PAR} params   "
        r"($\sqrt{s_{NN}}=2.76$ TeV, spherical)",
        fontsize=10, color=TEXT, y=1.01, fontweight="bold")

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.5)

            if row == col:
                c = COLORS[col]
                ax.hist(unit_lhs[:, col], bins=12, range=(0,1),
                        color=c, alpha=0.75, edgecolor=BG, lw=0.3)
                ax.set_xlim(0, 1)
                ax.set_yticks([])
                if row == n-1:
                    lo, hi = PLO[col], PHI[col]
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xticklabels([f"{lo:.2g}",
                                        f"{(lo+hi)/2:.3g}",
                                        f"{hi:.2g}"],
                                       fontsize=7, color=TICK)
                else:
                    ax.set_xticks([])
            elif row > col:
                c = COLORS[col]
                ax.scatter(unit_lhs[:, col], unit_lhs[:, row],
                           c=c, s=8, alpha=0.5, linewidths=0)
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                for v in [0.25, 0.5, 0.75]:
                    ax.axhline(v, color=GRID, lw=0.3, ls="--", alpha=0.5)
                    ax.axvline(v, color=GRID, lw=0.3, ls="--", alpha=0.5)
                if row == n-1:
                    lo, hi = PLO[col], PHI[col]
                    ax.set_xticks([0, 0.5, 1])
                    ax.set_xticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.3g}",
                                        f"{hi:.2g}"],
                                       fontsize=7, color=TICK)
                else:
                    ax.set_xticks([])
                if col == 0:
                    lo, hi = PLO[row], PHI[row]
                    ax.set_yticks([0, 0.5, 1])
                    ax.set_yticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.3g}",
                                        f"{hi:.2g}"],
                                       fontsize=7, color=TICK)
                else:
                    ax.set_yticks([])
            else:
                ax.set_visible(False)

    for j in range(n):
        axes[n-1, j].set_xlabel(PLABELS[j], fontsize=9.5,
                                 color=COLORS[j], labelpad=5)
        if j > 0:
            axes[j, 0].set_ylabel(PLABELS[j], fontsize=9.5,
                                   color=COLORS[j], labelpad=5)

    fig.tight_layout(h_pad=0.15, w_pad=0.15)
    path = out_dir / "lhs_corner.pdf"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = WORK_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 11.1  Check TRENTo ───────────────────────────────────────────────────
    # TRENTO_BIN was resolved at import time (built from source if necessary).
    # Do a final sanity-check here.
    if not TRENTO_BIN.exists() or not os.access(str(TRENTO_BIN), os.X_OK):
        raise RuntimeError(
            f"TRENTo binary still not executable after build attempt: {TRENTO_BIN}"
        )
    ver = subprocess.run([str(TRENTO_BIN), "--version"],
                         capture_output=True, text=True)
    ver_str = (ver.stdout + ver.stderr).strip().splitlines()
    print(f"TRENTo binary  → {TRENTO_BIN}")
    print(f"TRENTo version → {ver_str[0] if ver_str else '(unknown)'}")

    # ── 11.2  Generate LHS ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generating {N_DESIGN}-pt maximin LHS ({N_LHS_ITER} trials)…")
    t0       = time.perf_counter()
    unit_lhs = maximin_lhs(N_DESIGN, N_PAR, N_LHS_ITER, SEED_LHS)
    design   = scale_design(unit_lhs, PLO, PHI)
    print(f"Done in {time.perf_counter()-t0:.1f} s")

    # Print design summary
    print(f"\n{'':>8}", end="")
    for p in PARAMS:
        print(f"  {p[0]:>8}", end="")
    print()
    for i in range(min(5, N_DESIGN)):
        print(f"  pt {i:04d}", end="")
        for j in range(N_PAR):
            print(f"  {design[i,j]:>8.4f}", end="")
        print()
    print(f"  … ({N_DESIGN} total design points)")

    # Save design
    np.savez_compressed(
        WORK_DIR / "lhs_design.npz",
        design=design, unit_lhs=unit_lhs,
        param_names=np.array(PNAMES), lo=PLO, hi=PHI,
    )
    print(f"Design saved → {WORK_DIR/'lhs_design.npz'}")

    # LHS corner plot
    print("\nPlotting LHS corner…")
    plot_lhs_corner(unit_lhs, plots_dir)

    # ── 11.3  Run TRENTo + Compute Eccentricities ────────────────────────────
    print(f"\n{'='*60}")
    print(f"Running TRENTo: {N_DESIGN} pts × {N_EVENTS:,} events")
    print(f"  Pb+Pb  √s={SQRTS} TeV  σ_NN={SIGMA_NN} fm²  b_max={B_MAX} fm")
    print(f"  WS: R={WS_R} fm, a={WS_A} fm  (spherical, no deformation)")
    print(f"{'='*60}")

    # Centrality classes for scalar observables
    cent_classes_scalar = [
        ("0-1",  0,  1),
        ("0-5",  0,  5),
        ("5-10", 5, 10),
        ("0-10", 0, 10),
        ("20-30",20, 30),
        ("40-50",40, 50),
    ]

    all_profiles  = [None] * N_DESIGN
    scalar_obs    = {}
    timing        = []

    # ── Run TRENTo (parallel or sequential) ──────────────────────────────────
    print(f"  Parallel workers : {N_WORKERS}  "
          f"({'parallel' if N_WORKERS > 1 else 'sequential'})")
    print(f"  Resume-safe      : existing trento_events.npy are skipped")
    print()

    t_loop = time.perf_counter()
    run_results = run_all_parallel(design, N_EVENTS, WORK_DIR,
                                   N_WORKERS, force=False)
    t_run_total = time.perf_counter() - t_loop
    print(f"\n  TRENTo wall time : {t_run_total/60:.1f} min")

    # ── Compute eccentricities from cached .npy files ─────────────────────────
    print("\n  Computing eccentricities from cached events...")
    for idx, elapsed, n_got in sorted(run_results, key=lambda x: x[0]):
        cache = WORK_DIR / f"design_{idx:04d}" / "trento_events.npy"
        if not cache.exists():
            scalar_obs[idx] = {}
            continue

        arr  = np.load(cache)
        cent = assign_centrality(arr)

        prof = compute_centrality_profiles(arr, cent)
        all_profiles[idx] = prof

        scalar_obs[idx] = {}
        for c_label, c_lo, c_hi in cent_classes_scalar:
            c_key = f"{c_lo}-{c_hi}"
            scalar_obs[idx][c_key] = compute_observables(arr, cent, c_lo, c_hi)

        if elapsed > 0:
            timing.append(elapsed)

    # ── 11.4  Save all eccentricities ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Saving eccentricity results…")

    obs_keys = list(OBS_META.keys())
    save_dict = dict(
        design      = design,
        unit_lhs    = unit_lhs,
        param_names = np.array(PNAMES),
        param_lo    = PLO,
        param_hi    = PHI,
        obs_keys    = np.array(obs_keys),
    )

    # Save scalar observables as structured arrays
    for c_label, c_lo, c_hi in cent_classes_scalar:
        c_key = f"{c_lo}-{c_hi}"
        for ok in obs_keys:
            arr_obs = np.array([
                scalar_obs[idx].get(c_key, {}).get(ok, np.nan)
                for idx in range(N_DESIGN)
            ])
            save_dict[f"{ok}_c{c_key}"] = arr_obs

    # Save full centrality profiles
    for tag in ["uc", "wide"]:
        valid = [p for p in all_profiles if p is not None]
        if valid:
            mid = valid[0][tag]["mid"]
            save_dict[f"cent_mid_{tag}"] = mid
            for ok in obs_keys:
                mat = np.array([
                    p[tag][ok] if p is not None else np.full(len(mid), np.nan)
                    for p in all_profiles
                ])
                save_dict[f"{ok}_{tag}"] = mat

    out_npz = WORK_DIR / "eccentricities.npz"
    np.savez_compressed(out_npz, **save_dict)
    print(f"Eccentricities saved → {out_npz}")

    # ── 11.5  Plots ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating plots…")

    valid_profiles = [p for p in all_profiles if p is not None]
    if valid_profiles:
        plot_vs_centrality(valid_profiles, plots_dir)
        plot_vs_params(design, scalar_obs, plots_dir)
    else:
        print("  No valid profiles to plot.")

    # ── 11.6  Timing & budget report ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TIMING & BUDGET REPORT")
    print(f"{'='*60}")
    if timing:
        t_avg  = np.mean(timing)
        t_std  = np.std(timing)
        t_min  = np.min(timing)
        t_max  = np.max(timing)
        t_tot  = np.sum(timing)
        rate   = N_EVENTS / t_avg

        def fmt(s):
            if s < 60: return f"{s:.1f} s"
            if s < 3600: return f"{s/60:.1f} min"
            return f"{s/3600:.2f} h"

        print(f"  Events/pt          : {N_EVENTS:,}")
        print(f"  Design pts run     : {len(timing)} / {N_DESIGN}")
        print(f"  Min  time/pt       : {fmt(t_min)}")
        print(f"  Max  time/pt       : {fmt(t_max)}")
        print(f"  Mean time/pt       : {fmt(t_avg)}  ± {fmt(t_std)}")
        print(f"  Total  (ran)       : {fmt(t_tot)}")
        print(f"  Throughput         : {rate:,.0f} events/s")
        print()
        t_seq_100    = t_avg * 100
        t_par_100    = t_seq_100 / max(N_WORKERS, 1) / 0.85  # ~85% efficiency
        print(f"  BUDGET ESTIMATE (extrapolation to N=100 pts, {N_EVENTS:,} events)")
        print(f"  ── Sequential  (1 worker)  : {fmt(t_seq_100)}")
        print(f"  ── Parallel    ({N_WORKERS} workers)   : {fmt(t_par_100)}")
        print(f"  Colab free tier: 2 vCPUs, 13 GB RAM, 12h session limit")
        print(f"  Resume-safe: re-run skips already-cached design pts")
    print(f"{'='*60}")
    print(f"\nAll outputs in:  {WORK_DIR.resolve()}")
    print(f"  lhs_design.npz")
    print(f"  eccentricities.npz")
    print(f"  design_NNNN/trento_events.npy   (one per design pt)")
    print(f"  plots/lhs_corner.pdf")
    print(f"  plots/ecc_vs_centrality.pdf")
    print(f"  plots/ecc_vs_params_0_1pct.pdf")
    print(f"  plots/ecc_vs_params_0_5pct.pdf")
    print(f"  plots/ecc_vs_params_20_30pct.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 12. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__" or "get_ipython" in dir():
    main()
