#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Deformed 208Pb+208Pb @ 2.76 TeV — Isobar Sampler + trento_sync Scan      ║
# ║                                                                              ║
# ║  5 FREE PARAMETERS (LHS-sampled):                                            ║
# ║    R      WS radius        [6.50, 6.80] fm   → Isobar Sampler               ║
# ║    a      WS diffuseness   [0.44, 0.65] fm   → Isobar Sampler               ║
# ║    β₃     octupole         [0.00, 0.12]      → Isobar Sampler               ║
# ║    β₄     hexadecapole     [-0.02, 0.06]     → Isobar Sampler (fork)        ║
# ║    w      nucleon width    [0.40, 1.20] fm   → trento_sync -w flag           ║
# ║                                                                              ║
# ║  FIXED PARAMETERS (JETSCAPE Grad MAP, arXiv:2011.01430 Table VII):           ║
# ║    TRENTo:   p=0.063  k=1.063  norm=18.12  d_min=0.804 fm                   ║
# ║    Isobar:   β₂=0  γ=0  C_l=0.4 fm  C_s=-1                                 ║
# ║                                                                              ║
# ║  PIPELINE (one Colab cell):                                                  ║
# ║    Stage 0 — pip install deps + build tools                                  ║
# ║    Stage 1 — clone + build jppicchetti/trento_sync                           ║
# ║    Stage 2 — clone ThiagoSDomingues/Isobar-Sampler                          ║
# ║    Stage 3 — generate LHS design (50 pts × 5 params)                        ║
# ║    Stage 4 — make_seeds.py  (ONCE, shared seed bank)                        ║
# ║    Stage 5 — for each design pt:                                             ║
# ║                build_isobars.py → WS1.hdf, WS2.hdf                          ║
# ║                trento_sync      → trento_events.npy                          ║
# ║                compute eccentricities + save                                 ║
# ║    Stage 6 — all plots (vs centrality + vs model parameters)                 ║
# ║                                                                              ║
# ║  PASTE THIS ENTIRE FILE INTO ONE COLAB CELL AND RUN.                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, subprocess, sys, time, shutil, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  USER CONFIGURATION  ← edit here
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR      = Path("/content/pbpb_deformed")
# Google Drive: WORK_DIR = Path("/content/drive/MyDrive/pbpb_deformed")

N_DESIGN      = 30        # LHS design points
N_CONFIGS     = 1_00     # nuclear configs per design pt (Isobar Sampler)
N_EVENTS      = 50_000   # TRENTo events per design pt
N_LHS_ITER    = 2000      # maximin LHS optimisation trials
SEED_LHS      = 42        # RNG seed — identical to spherical scan for comparison
N_WORKERS     = 2         # parallel TRENTo workers (Colab free: 2 vCPU)
N_ISO_PARALLEL = -1       # Isobar Sampler parallel jobs (-1 = all CPUs)

# ── 5 FREE parameters ─────────────────────────────────────────────────────────
# (name, lo, hi, latex_label, which_stage)
#   IS  = Isobar Sampler param
#   TR  = trento_sync param (passed as flag)
PARAMS = [
    # name    lo      hi     latex                     stage
    ("R",     6.50,   6.80,  r"$R$ [fm]",              "IS"),
    ("a",     0.44,   0.65,  r"$a$ [fm]",              "IS"),
    ("beta3", 0.00,   0.12,  r"$\beta_3$",             "IS"),
    ("beta4", -0.02,  0.06,  r"$\beta_4$",             "IS"),
    ("w",     0.40,   1.20,  r"$w$ [fm]",              "TR"),
]
PNAMES  = [p[0] for p in PARAMS]
PLO     = np.array([p[1] for p in PARAMS])
PHI     = np.array([p[2] for p in PARAMS])
PLABELS = [p[3] for p in PARAMS]
N_PAR   = len(PARAMS)

# ── FIXED TRENTo params (JETSCAPE Grad MAP, arXiv:2011.01430 Table VII) ───────
# p=0.063, sigma_k=0.97 → k=1/0.97²≈1.063, norm=18.12
# d³_min=0.52 fm³ → d_min=0.804 fm
FIX_P     = 0.063          # reduced thickness exponent
FIX_K     = 1.0 / 0.97**2  # Gamma shape  (σ_k = 0.97 → k ≈ 1.063)
FIX_NORM  = 18.12           # multiplicity normalization @ 2.76 TeV
FIX_DMIN  = 0.52**(1/3)    # d_min [fm]  (d³=0.52 fm³)
SIGMA_NN  = 6.4             # σ_NN [fm²] @ 2.76 TeV
B_MAX     = 20.0            # max impact parameter [fm]

# ── FIXED Isobar Sampler params ───────────────────────────────────────────────
FIX_BETA2   = 0.0   # 208Pb is doubly magic → spherical ground state
FIX_GAMMA   = 0.0   # triaxiality angle [rad]  (irrelevant when β₂=0)
FIX_CORR_L  = 0.4   # correlation length  [fm]  hard-core NN scale
FIX_CORR_S  = -1.0  # correlation strength       full Pauli exclusion

A_PB = 208           # mass number

# ── trento_sync stdout column indices ─────────────────────────────────────────
# 13 columns per event (vs 8 in stock TRENTo):
# 0:ev  1:b  2:npart  3:mult_tot  4:mult_mid
# 5:|ε₂|  6:Ψ₂  7:|ε₃|  8:Ψ₃  9:|ε₄|  10:Ψ₄  11:|ε₅|  12:Ψ₅
N_SYNC_COLS = 13
COL_MULT    = 3    # total mult → centrality
COL_E2      = 5    # |ε₂|
COL_E3      = 7    # |ε₃|
COL_E4      = 9    # |ε₄|
COL_E5      = 11   # |ε₅|

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt(s):
    if s < 60:    return f"{s:.1f} s"
    if s < 3600:  return f"{s/60:.1f} min"
    return f"{s/3600:.2f} h"

def run_cmd(cmd, label="", cwd=None, capture=False):
    """Run subprocess; stream output live unless capture=True."""
    cmd = [str(c) for c in cmd]
    r = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        capture_output=capture, text=capture,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"{label or cmd[0]} failed (exit {r.returncode})"
            + (f"\n{r.stderr[-500:]}" if capture else "")
        )
    return r

def write_yaml(path, data):
    import yaml
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 64)
print("  Stage 0 — Dependencies")
print("=" * 64)

# Python packages
t0 = time.perf_counter()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "numpy", "h5py", "joblib", "scipy", "pyyaml", "matplotlib"],
    check=True,
)
print(f"  pip packages OK  ({fmt(time.perf_counter()-t0)})")

# System build tools
subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(
    ["apt-get", "install", "-y", "-qq",
     "build-essential", "cmake", "git", "libboost-all-dev"],
    check=True,
)
print(f"  apt packages OK")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BUILD trento_sync
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  Stage 1 — trento_sync")
print("=" * 64)

TRENTO_REPO  = Path("/content/trento_sync")
TRENTO_BUILD = TRENTO_REPO / "build"

def find_trento():
    candidates = [
        "/usr/local/bin/trento",
        str(TRENTO_BUILD / "src" / "trento"),
        str(TRENTO_BUILD / "trento"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return Path(c)
    r = subprocess.run(["which", "trento"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return Path(r.stdout.strip())
    return None

TRENTO_BIN = find_trento()
if TRENTO_BIN:
    print(f"  trento_sync already built → {TRENTO_BIN}")
else:
    print("  Building trento_sync from source...")
    if TRENTO_REPO.exists():
        shutil.rmtree(TRENTO_REPO)
    run_cmd(["git", "clone", "--depth=1",
              "https://github.com/jppicchetti/trento_sync.git",
              str(TRENTO_REPO)],
             label="git clone trento_sync")
    TRENTO_BUILD.mkdir(parents=True, exist_ok=True)
    run_cmd(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
             cwd=TRENTO_BUILD, label="cmake trento_sync")
    import multiprocessing
    run_cmd(["make", f"-j{multiprocessing.cpu_count()}"],
             cwd=TRENTO_BUILD, label="make trento_sync")
    subprocess.run(["make", "install"], cwd=str(TRENTO_BUILD), check=False)
    TRENTO_BIN = find_trento()
    if not TRENTO_BIN:
        raise RuntimeError("trento_sync binary not found after build")

ver = subprocess.run([str(TRENTO_BIN), "--version"],
                      capture_output=True, text=True)
print(f"  trento_sync → {TRENTO_BIN}")
print(f"  version     → {(ver.stdout+ver.stderr).strip().splitlines()[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — CLONE ISOBAR SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  Stage 2 — Isobar Sampler (ThiagoSDomingues fork)")
print("=" * 64)

ISO_REPO = Path("/content/Isobar-Sampler")
if ISO_REPO.exists() and (ISO_REPO / "exec" / "make_seeds.py").exists():
    print(f"  Repo already at {ISO_REPO} — skipping clone")
else:
    if ISO_REPO.exists():
        shutil.rmtree(ISO_REPO)
    run_cmd(["git", "clone", "--depth=1",
              "https://github.com/ThiagoSDomingues/Isobar-Sampler",
              str(ISO_REPO)],
             label="git clone Isobar-Sampler")
print(f"  make_seeds.py    ✓")
print(f"  build_isobars.py ✓")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — LHS DESIGN
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  Stage 3 — LHS Design")
print("=" * 64)

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
            best_dmin, best = dmin, X.copy()
    return best

lhs_path = WORK_DIR / "lhs_design.npz"
if lhs_path.exists():
    data    = np.load(lhs_path, allow_pickle=True)
    unit_lhs = data["unit_lhs"]
    design   = data["design"]
    print(f"  LHS loaded from cache  {design.shape}")
else:
    print(f"  Generating {N_DESIGN}-pt maximin LHS ({N_LHS_ITER} trials)…",
          end="", flush=True)
    t0 = time.perf_counter()
    unit_lhs = maximin_lhs(N_DESIGN, N_PAR, N_LHS_ITER, SEED_LHS)
    design   = PLO + unit_lhs * (PHI - PLO)
    print(f" done ({fmt(time.perf_counter()-t0)})")
    np.savez_compressed(lhs_path, design=design, unit_lhs=unit_lhs,
                        param_names=np.array(PNAMES), lo=PLO, hi=PHI)

print(f"\n  {'idx':>4}  {'R':>6}  {'a':>6}  {'β₃':>8}  {'β₄':>8}  {'w':>6}")
print("  " + "-" * 46)
for i in range(min(5, N_DESIGN)):
    r = design[i]
    print(f"  {i:4d}  {r[0]:6.4f}  {r[1]:6.4f}  "
          f"{r[2]:8.5f}  {r[3]:8.5f}  {r[4]:6.4f}")
print(f"  … ({N_DESIGN} total)")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SEED BANK  (once, shared by all design points)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  Stage 4 — Nucleon Seed Bank (make_seeds.py)")
print("=" * 64)

seeds_dir = WORK_DIR / "seeds"
seeds_dir.mkdir(exist_ok=True)
seeds_hdf  = seeds_dir / "nucleon-seeds.hdf"
seeds_conf = seeds_dir / "seeds-conf.yaml"

if seeds_hdf.exists() and seeds_hdf.stat().st_size > 0:
    size_mb = seeds_hdf.stat().st_size / 1e6
    print(f"  Seed bank exists ({size_mb:.1f} MB, {N_CONFIGS:,} configs) — skip")
else:
    write_yaml(seeds_conf, {
        "isobar_seeds": {
            "description": f"208Pb seed bank, {N_CONFIGS} configs.",
            "number_nucleons":  {"description": "A", "value": A_PB},
            "number_configs":   {"description": "N_configs",
                                 "value": N_CONFIGS},
            "output_file":      {"description": "HDF5 output",
                                 "filename": str(seeds_hdf.resolve())},
            "number_of_parallel_processes":
                                {"description": "-1=all CPUs",
                                 "value": N_ISO_PARALLEL},
        }
    })
    print(f"  Running make_seeds.py  ({N_CONFIGS:,} configs, {A_PB} nucleons)…")
    t0 = time.perf_counter()
    run_cmd([sys.executable,
              str(ISO_REPO / "exec" / "make_seeds.py"),
              str(seeds_conf)],
             label="make_seeds.py", cwd=ISO_REPO)
    t_seeds = time.perf_counter() - t0
    if not seeds_hdf.exists():
        raise FileNotFoundError(f"Seed bank not found after make_seeds: {seeds_hdf}")
    size_mb = seeds_hdf.stat().st_size / 1e6
    print(f"  Done: {seeds_hdf.name}  {size_mb:.1f} MB  ({fmt(t_seeds)})")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — PER-DESIGN-POINT LOOP
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print(f"  Stage 5 — Isobar + TRENTo loop  ({N_DESIGN} pts × {N_EVENTS:,} events)")
print("=" * 64)

# ── Isobar YAML builder ───────────────────────────────────────────────────────

def _isobar_block(name, R, a, beta3, beta4):
    """One isobar entry for isobar-conf.yaml (ThiagoSDomingues fork schema)."""
    return {
        "isobar_name": name,
        "WS_radius":            {"description": "R [fm]",  "value": float(R)},
        "WS_diffusiveness":     {"description": "a [fm]",  "value": float(a)},
        "beta_2":               {"description": "β₂=0 (208Pb doubly magic)",
                                 "value": FIX_BETA2},
        "gamma":                {"description": "γ [rad]", "value": FIX_GAMMA},
        "beta_3":               {"description": "octupole β₃",
                                 "value": float(beta3)},
        "correlation_length":   {"description": "C_l [fm]",
                                 "value": FIX_CORR_L},
        "correlation_strength": {"description": "C_s",
                                 "value": FIX_CORR_S},
        "beta_4":               {"description": "hexadecapole β₄ (fork param)",
                                 "value": float(beta4)},
    }

def build_isobars_for_pt(idx, R, a, beta3, beta4, design_dir):
    """
    Run build_isobars.py for design point idx.
    Returns (ws1_path, ws2_path) or raises on failure.
    """
    nuclei_dir   = design_dir / "nuclei"
    nuclei_dir.mkdir(parents=True, exist_ok=True)
    ws1 = nuclei_dir / "WS1.hdf"
    ws2 = nuclei_dir / "WS2.hdf"

    if ws1.exists() and ws2.exists() \
            and ws1.stat().st_size > 0 and ws2.stat().st_size > 0:
        return ws1, ws2   # already built

    isobar_conf = design_dir / "isobar-conf.yaml"
    write_yaml(isobar_conf, {
        "isobar_samples": {
            "description": f"208Pb configs, design {idx:04d}",
            "number_configs":   {"description": "N", "value": N_CONFIGS},
            "number_nucleons":  {"description": "A", "value": A_PB},
            "seeds_file":       {"description": "seed bank",
                                 "filename": str(seeds_hdf.resolve())},
            "output_path":      {"description": "HDF5 output dir",
                                 "dirname": str(nuclei_dir.resolve())},
            "number_of_parallel_processes":
                                {"description": "-1=all",
                                 "value": N_ISO_PARALLEL},
        },
        "isobar_properties": {
            "description": "WS1=projectile, WS2=target (identical params)",
            "isobar1": _isobar_block("WS1", R, a, beta3, beta4),
            "isobar2": _isobar_block("WS2", R, a, beta3, beta4),
        },
    })
    run_cmd([sys.executable,
              str(ISO_REPO / "exec" / "build_isobars.py"),
              str(isobar_conf)],
             label=f"build_isobars [{idx:04d}]", cwd=ISO_REPO)

    # build_isobars.py may write to CWD instead of output_path — relocate
    for name in ["WS1.hdf", "WS2.hdf"]:
        for possible in [nuclei_dir / name, ISO_REPO / name,
                         Path("/content") / name]:
            if possible.exists() and possible.stat().st_size > 0:
                dest = nuclei_dir / name
                if possible != dest:
                    shutil.move(str(possible), str(dest))
                break

    if not (ws1.exists() and ws2.exists()):
        raise FileNotFoundError(
            f"WS1/WS2 not found after build_isobars [{idx:04d}]"
        )
    return ws1, ws2

# ── TRENTo runner ─────────────────────────────────────────────────────────────

def run_trento_sync(idx, w_val, ws1, ws2, design_dir, force=False):
    """
    Run trento_sync with HDF nuclear input.
    Uses -q (quiet) + stdout parsing — same column layout as pbpb_spherical_sync.

    Seed = 1000 + idx  (reproducible, identical to spherical scan convention).
    Returns float64 array shape (N_events, N_SYNC_COLS).
    """
    cache = design_dir / "trento_events.npy"
    if cache.exists() and not force:
        arr = np.load(cache)
        print(f"  [{idx:04d}] cached ({len(arr):,} events) — skip", flush=True)
        return arr

    cmd = [
        str(TRENTO_BIN),
        str(ws1), str(ws2),     # HDF nuclear configs (synchronized mode)
        str(N_EVENTS),
        "-p", f"{FIX_P:.6f}",  # fixed JETSCAPE Grad MAP value
        "-k", f"{FIX_K:.6f}",  # fixed
        "-w", f"{w_val:.6f}",  # FREE (LHS-sampled nucleon width)
        "-d", f"{FIX_DMIN:.6f}",   # fixed d_min [fm]
        "-n", f"{FIX_NORM:.6f}",   # fixed norm
        "-x", f"{SIGMA_NN:.4f}",   # σ_NN [fm²]
        "--b-max", f"{B_MAX:.1f}",
        "--random-seed", str(1000 + idx),
    ]

    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if res.returncode != 0:
        print(f"  [{idx:04d}] TRENTo FAILED  (exit {res.returncode})", flush=True)
        print(f"    cmd: {' '.join(cmd)}", flush=True)
        for ln in res.stderr.splitlines()[-4:]:
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
        print(f"  [{idx:04d}] TRENTo: no events parsed", flush=True)
        return None

    arr = np.array(rows, dtype=np.float64)
    np.save(cache, arr)
    rate = len(arr) / elapsed
    print(f"  [{idx:04d}] {len(arr):,} evt | {fmt(elapsed)} | "
          f"{rate:,.0f} evt/s | R={design[idx,0]:.3f} a={design[idx,1]:.3f} "
          f"β₃={design[idx,2]:.4f} β₄={design[idx,3]:.4f} w={w_val:.3f}",
          flush=True)
    return arr

# ── Eccentricity calculations ─────────────────────────────────────────────────

def assign_centrality(events):
    mult = events[:, COL_MULT]
    rank = np.argsort(np.argsort(-mult))
    return rank / len(mult) * 100.0

def _m(eps, order): return np.mean(eps**order)

def ecc2_cumulants(eps):
    m2, m4 = _m(eps, 2), _m(eps, 4)
    e2     = np.sqrt(m2)
    inner  = 2.0 * m2**2 - m4
    e4     = inner**0.25 if inner >= 0 else np.nan
    return e2, e4

def compute_observables(events, centrality, c_lo, c_hi):
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
    T32   = e3_2 / e2_2  if e2_2 > 0 else np.nan
    R42   = e2_4 / e2_2  if (e2_2 > 0 and not np.isnan(e2_4)) else np.nan
    e3rat = e3_2 / e3_4  if (e3_4 > 0 and not np.isnan(e3_4)) else np.nan
    e4_4c = 2.0 * _m(eps4, 2)**2 - _m(eps4, 4)
    return dict(e2_2=e2_2, e3_2=e3_2, e4_2=e4_2,
                T32=T32, R42=R42, e3_ratio=e3rat, e4_4c=e4_4c,
                n=int(mask.sum()))

def compute_centrality_profiles(events, centrality):
    uc_edges   = np.arange(0, 11, 1)
    wide_edges = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    result = {}
    obs_keys = ["e2_2","e3_2","e4_2","T32","R42","e3_ratio","e4_4c"]
    for tag, edges in [("uc", uc_edges), ("wide", wide_edges)]:
        bins = list(zip(edges[:-1], edges[1:]))
        mid  = np.array([(lo+hi)/2 for lo, hi in bins])
        data = {k: [] for k in obs_keys}
        ns   = []
        for lo, hi in bins:
            obs = compute_observables(events, centrality, lo, hi)
            for k in obs_keys:
                data[k].append(obs[k])
            ns.append(obs["n"])
        result[tag] = dict(mid=mid, bins=bins, n=np.array(ns),
                           **{k: np.array(data[k]) for k in obs_keys})
    return result

# ── Main per-design-point loop ────────────────────────────────────────────────

CENT_CLASSES = [
    ("0-1",  0,  1), ("0-5",  0,  5), ("5-10",  5, 10),
    ("0-10", 0, 10), ("20-30",20, 30), ("40-50",40, 50),
]
OBS_KEYS  = ["e2_2","e3_2","e4_2","T32","R42","e3_ratio","e4_4c"]

all_profiles = [None] * N_DESIGN
scalar_obs   = {}
timing_iso   = []
timing_trento= []
t_loop_start = time.perf_counter()

for idx in range(N_DESIGN):
    row   = design[idx]
    R_val, a_val, b3_val, b4_val, w_val = row

    design_dir = WORK_DIR / f"design_{idx:04d}"
    design_dir.mkdir(parents=True, exist_ok=True)

    # ── 5a. Isobar Sampler ────────────────────────────────────────────────────
    t_iso0 = time.perf_counter()
    ws1, ws2 = build_isobars_for_pt(idx, R_val, a_val, b3_val, b4_val,
                                     design_dir)
    t_iso = time.perf_counter() - t_iso0
    if (design_dir / "nuclei" / "WS1.hdf").stat().st_size > 0:
        # Only count if it actually ran (not cached)
        cache_existed = t_iso < 0.5
        if not cache_existed:
            timing_iso.append(t_iso)

    # ── 5b. TRENTo ────────────────────────────────────────────────────────────
    t_tr0 = time.perf_counter()
    arr = run_trento_sync(idx, w_val, ws1, ws2, design_dir)
    t_tr = time.perf_counter() - t_tr0
    if arr is not None and t_tr > 0.5:
        timing_trento.append(t_tr)

    if arr is None:
        scalar_obs[idx] = {}
        continue

    # ── 5c. Eccentricities ────────────────────────────────────────────────────
    cent = assign_centrality(arr)
    all_profiles[idx] = compute_centrality_profiles(arr, cent)
    scalar_obs[idx] = {}
    for c_label, c_lo, c_hi in CENT_CLASSES:
        c_key = f"{c_lo}-{c_hi}"
        scalar_obs[idx][c_key] = compute_observables(arr, cent, c_lo, c_hi)

    # Live ETA
    real_tr = [t for t in timing_trento if t > 0]
    if real_tr:
        avg_t  = np.mean(real_tr)
        n_left = N_DESIGN - (idx + 1)
        wall   = time.perf_counter() - t_loop_start
        print(f"    ── {idx+1}/{N_DESIGN} done  |  "
              f"avg-trento {fmt(avg_t)}/pt  |  "
              f"ETA {fmt(avg_t*n_left)}  |  wall {fmt(wall)}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  Saving eccentricity results")
print("=" * 64)

save_dict = dict(
    design=design, unit_lhs=unit_lhs,
    param_names=np.array(PNAMES), param_lo=PLO, param_hi=PHI,
    obs_keys=np.array(OBS_KEYS),
)
for c_label, c_lo, c_hi in CENT_CLASSES:
    c_key = f"{c_lo}-{c_hi}"
    for ok in OBS_KEYS:
        save_dict[f"{ok}_c{c_key}"] = np.array([
            scalar_obs[idx].get(c_key, {}).get(ok, np.nan)
            for idx in range(N_DESIGN)
        ])
for tag in ["uc", "wide"]:
    valid = [p for p in all_profiles if p is not None]
    if valid:
        mid = valid[0][tag]["mid"]
        save_dict[f"cent_mid_{tag}"] = mid
        for ok in OBS_KEYS:
            save_dict[f"{ok}_{tag}"] = np.array([
                p[tag][ok] if p is not None else np.full(len(mid), np.nan)
                for p in all_profiles
            ])

out_npz = WORK_DIR / "eccentricities.npz"
np.savez_compressed(out_npz, **save_dict)
print(f"  Saved → {out_npz}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  Stage 6 — Plots")
print("=" * 64)

plots_dir = WORK_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# ── Dark theme ────────────────────────────────────────────────────────────────
BG = "#0d0f14"; PANEL = "#13161e"; GRID = "#1e2230"
TEXT = "#d8dce8"; TICK = "#8892a4"

def _dark():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
        "xtick.color": TICK,   "ytick.color": TICK,
        "xtick.labelsize": 8,  "ytick.labelsize": 8,
        "text.color": TEXT,    "grid.color": GRID,
        "grid.linewidth": 0.4, "grid.alpha": 0.5,
        "font.size": 9,        "axes.linewidth": 0.6,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID,
        "legend.labelcolor": TEXT, "legend.fontsize": 7.5,
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

PARAM_COLORS = ["#4fc3f7", "#ffd54f", "#ff4081", "#b39ddb", "#80cbc4"]

# ── LHS corner plot ───────────────────────────────────────────────────────────
_dark()
n = N_PAR
fig, axes = plt.subplots(n, n, figsize=(n*2.6, n*2.6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    r"$^{208}$Pb+$^{208}$Pb Deformed Scan — LHS Design"
    f"\n{N_DESIGN} pts × {N_PAR} params  "
    r"($\sqrt{s_{NN}}=2.76$ TeV)",
    fontsize=10, color=TEXT, y=1.01, fontweight="bold")

for row in range(n):
    for col in range(n):
        ax = axes[row, col]
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
        if row == col:
            c = PARAM_COLORS[col]
            ax.hist(unit_lhs[:, col], bins=12, range=(0,1),
                    color=c, alpha=0.75, edgecolor=BG, lw=0.3)
            ax.set_xlim(0,1); ax.set_yticks([])
            if row == n-1:
                lo, hi = PLO[col], PHI[col]
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.3g}",
                                     f"{hi:.2g}"], fontsize=7, color=TICK)
            else:
                ax.set_xticks([])
        elif row > col:
            ax.scatter(unit_lhs[:, col], unit_lhs[:, row],
                       c=PARAM_COLORS[col], s=8, alpha=0.5, linewidths=0)
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
            for v in [0.25, 0.5, 0.75]:
                ax.axhline(v, color=GRID, lw=0.3, ls="--", alpha=0.5)
                ax.axvline(v, color=GRID, lw=0.3, ls="--", alpha=0.5)
            if row == n-1:
                lo, hi = PLO[col], PHI[col]
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.3g}",
                                     f"{hi:.2g}"], fontsize=7, color=TICK)
            else:
                ax.set_xticks([])
            if col == 0:
                lo, hi = PLO[row], PHI[row]
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.3g}",
                                     f"{hi:.2g}"], fontsize=7, color=TICK)
            else:
                ax.set_yticks([])
        else:
            ax.set_visible(False)

for j in range(n):
    axes[n-1, j].set_xlabel(PLABELS[j], fontsize=9, color=PARAM_COLORS[j],
                              labelpad=5)
    if j > 0:
        axes[j, 0].set_ylabel(PLABELS[j], fontsize=9, color=PARAM_COLORS[j],
                               labelpad=5)
fig.tight_layout(h_pad=0.15, w_pad=0.15)
p = plots_dir / "lhs_corner.pdf"
fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Saved: {p}")

# ── Eccentricities vs centrality ──────────────────────────────────────────────
_dark()
valid_prof = [p for p in all_profiles if p is not None]
if valid_prof:
    obs_list = list(OBS_META.keys())
    n_obs = len(obs_list)
    fig, axes = plt.subplots(n_obs, 2, figsize=(13, n_obs*2.6), squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        r"$^{208}$Pb+$^{208}$Pb Deformed — Eccentricities vs Centrality"
        f"\n{N_DESIGN} design pts  |  {N_EVENTS:,} events/pt  |  "
        r"trento\_sync  |  Isobar-Sampler ($\beta_3,\beta_4$ free)",
        fontsize=10, color=TEXT, y=1.005, fontweight="bold")

    for row, key in enumerate(obs_list):
        label, color = OBS_META[key]
        for col, (tag, xlim, xtitle) in enumerate([
                ("uc",   (0, 10),  r"Centrality $c$ (%)"),
                ("wide", (0, 90),  r"Centrality $c$ (%)")]):
            ax = axes[row, col]
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
            ax.grid(True, ls="--", alpha=0.35)

            curves = []
            for prof in valid_prof:
                y   = prof[tag][key]
                mid = prof[tag]["mid"]
                ok  = np.isfinite(y)
                if ok.sum() < 2: continue
                ax.plot(mid[ok], y[ok], color=color, alpha=0.12,
                        lw=0.7, zorder=2)
                curves.append(y)

            if curves:
                mat = np.array(curves)
                x_m = valid_prof[0][tag]["mid"]
                med = np.nanmedian(mat, axis=0)
                lo16= np.nanpercentile(mat, 16, axis=0)
                hi84= np.nanpercentile(mat, 84, axis=0)
                ok  = np.isfinite(med)
                ax.plot(x_m[ok], med[ok], color=color, lw=2.0, zorder=4,
                        label="median")
                ax.fill_between(x_m[ok], lo16[ok], hi84[ok],
                                color=color, alpha=0.25, zorder=3,
                                label="16–84%")

            ax.set_ylabel(label, fontsize=9, color=color)
            ax.set_xlabel(xtitle, fontsize=8)
            ax.set_xlim(*xlim)
            if col == 0:
                ax.set_title("Ultracentral (1% bins, 0–10%)",
                             fontsize=8, color=TICK, pad=3)
            else:
                ax.set_title("Wide centrality (0–90%)",
                             fontsize=8, color=TICK, pad=3)
            if row == 0:
                ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    p = plots_dir / "ecc_vs_centrality.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")

# ── Eccentricities vs model parameters ───────────────────────────────────────
_dark()
cent_plot_classes = [("0-1%",0,1), ("0-5%",0,5), ("20-30%",20,30)]

for c_label, c_lo, c_hi in cent_plot_classes:
    c_key = f"{c_lo}-{c_hi}"
    obs_list = list(OBS_META.keys())
    fig, axes = plt.subplots(len(obs_list), N_PAR,
                              figsize=(N_PAR*2.8, len(obs_list)*2.4),
                              squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        r"$^{208}$Pb+$^{208}$Pb Deformed — "
        f"Eccentricities vs Parameters  (centrality {c_label})\n"
        r"4 Isobar-Sampler params ($R$,$a$,$\beta_3$,$\beta_4$) + TRENTo $w$",
        fontsize=10, color=TEXT, y=1.004, fontweight="bold")

    for r_obs, okey in enumerate(obs_list):
        obs_label, color = OBS_META[okey]
        for c_par in range(N_PAR):
            ax = axes[r_obs, c_par]
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
            ax.grid(True, ls="--", alpha=0.3)

            xs, ys = [], []
            for i in range(N_DESIGN):
                obs_d = scalar_obs[i].get(c_key, {})
                y = obs_d.get(okey, np.nan)
                if not np.isfinite(y): continue
                xs.append(design[i, c_par])
                ys.append(y)

            if len(xs) > 3:
                xa, ya = np.array(xs), np.array(ys)
                try:
                    dens = gaussian_kde(np.vstack([xa, ya]))(np.vstack([xa, ya]))
                    srt  = np.argsort(dens)
                    ax.scatter(xa[srt], ya[srt], c=dens[srt], cmap="plasma",
                               s=14, linewidths=0, alpha=0.85, zorder=3)
                except Exception:
                    ax.scatter(xa, ya, color=color, s=10, alpha=0.6, zorder=3)

                # Running median trend
                if len(xa) >= 10:
                    order  = np.argsort(xa)
                    xs_s, ys_s = xa[order], ya[order]
                    ws = max(3, len(xs_s)//8)
                    xsm, ysm = [], []
                    for st in range(0, len(xs_s)-ws, ws//2):
                        xsm.append(np.median(xs_s[st:st+ws]))
                        ysm.append(np.median(ys_s[st:st+ws]))
                    ax.plot(xsm, ysm, color="white", lw=1.2,
                            alpha=0.7, zorder=4)

            if c_par == 0:
                ax.set_ylabel(obs_label, fontsize=9, color=color)
            if r_obs == len(obs_list)-1:
                ax.set_xlabel(PLABELS[c_par], fontsize=8.5)
            if r_obs == 0:
                ax.set_title(PLABELS[c_par], fontsize=8.5,
                             color=TEXT, pad=3)

    fig.tight_layout(h_pad=0.3, w_pad=0.4)
    safe = c_label.replace("%","pct").replace("-","_")
    p = plots_dir / f"ecc_vs_params_{safe}.pdf"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {p}")

# ─────────────────────────────────────────────────────────────────────────────
# TIMING + BUDGET REPORT
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("  TIMING & BUDGET REPORT")
print("=" * 64)

if timing_trento:
    t_avg_tr  = np.mean(timing_trento)
    t_avg_iso = np.mean(timing_iso) if timing_iso else 0.0
    t_avg_pt  = t_avg_iso + t_avg_tr
    t_total   = time.perf_counter() - t_loop_start

    print(f"  Isobar Sampler / pt  : {fmt(t_avg_iso)}  ({N_CONFIGS:,} configs)")
    print(f"  TRENTo sync    / pt  : {fmt(t_avg_tr)}  ({N_EVENTS:,} events)")
    print(f"  Combined       / pt  : {fmt(t_avg_pt)}")
    print(f"  Total wall time      : {fmt(t_total)}")
    print()
    print(f"  BUDGET (extrapolation)")
    for n_pts in [50, 100]:
        t_seq = t_avg_pt * n_pts
        t_par = t_seq / 1.7
        fits  = "✓" if t_par < 12*3600*0.9 else "✗  exceeds 12h"
        print(f"  {n_pts} pts sequential : {fmt(t_seq)}  "
              f"|  parallel×2 : {fmt(t_par)}  {fits}")
    print()
    print(f"  Colab free: 2 vCPU | 13 GB RAM | 12h session")
    print(f"  Seeds run ONCE and reused across all design pts")

print("\n" + "=" * 64)
print(f"  All outputs in: {WORK_DIR.resolve()}")
print(f"    lhs_design.npz")
print(f"    eccentricities.npz")
print(f"    design_NNNN/nuclei/WS1.hdf + WS2.hdf")
print(f"    design_NNNN/trento_events.npy")
print(f"    plots/lhs_corner.pdf")
print(f"    plots/ecc_vs_centrality.pdf")
print(f"    plots/ecc_vs_params_0_1pct.pdf")
print(f"    plots/ecc_vs_params_0_5pct.pdf")
print(f"    plots/ecc_vs_params_20_30pct.pdf")
print("=" * 64)

if __name__ == "__main__" or "get_ipython" in dir():
    pass   # script already ran as a module-level sequence above
