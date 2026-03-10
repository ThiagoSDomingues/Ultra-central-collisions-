#!/usr/bin/env python3
"""
pbpb_ultracentral_scan.py
=========================
Full automated pipeline for the 208Pb+208Pb ultracentral flow puzzle study.

PIPELINE STAGES (each can be run independently via STAGE_* flags)
  Stage 0 — Generate LHS design (5 free params + 3 fixed nuclear + 5 fixed Trento)
  Stage 1 — Write seed config + run make_seeds.py     (once, shared bank)
  Stage 2 — Write isobar configs + run build_isobars.py (once per design point)
  Stage 3 — Run Trento (once per design point, reads WS1.hdf / WS2.hdf)
  Stage 4 — Compute eccentricities + bootstrap bands (once per design point)
  Stage 5 — Produce summary plots vs design-point index and vs each parameter

COLLISION SYSTEM
  208Pb + 208Pb  at sqrt(s_NN) = 2.76 TeV  OR  5.02 TeV  (choose via SQRTS)
  Both beams are identical: same WS1.hdf and WS2.hdf per design point.

FREE PARAMETERS (LHS-sampled, 5 parameters)
  R         WS radius [fm]      [6.50, 6.80]
  a         WS diffuseness [fm] [0.44, 0.65]
  beta_3    octupole            [0.00, 0.12]   KEY puzzle param (Carzon 2020)
  beta_4    hexadecapole        [-0.02, 0.06]  puzzle param (v4{4}^4 sign)
  w         Trento nucleon width [fm] [0.4, 1.2]

FIXED PARAMETERS (set to Bayesian-calibrated central values)
  Nuclear:
    beta_2           = 0.0      (208Pb doubly magic, near-spherical)
    gamma            = 0.0      (axially symmetric)
    correlation_length  = 0.4   (fm, hard-core scale, Luzum et al. 2023)
    correlation_strength = -1.0 (full Pauli exclusion)
  Trento (from Bernhard 2019 / Trajectum Nijs 2021 Pb+Pb posterior medians):
    p    = 0.0    (geometric mean, KLN-like)
    k    = 1.6    (nucleon fluctuation shape)
    d    = 1.0    (fm, min nucleon distance)
    norm = 18.0   (multiplicity normalisation at 2.76 TeV)
         = 20.0   (at 5.02 TeV, ~11% higher charged multiplicity)

DESIGN POINTS
  N_DESIGN = 100.  In 5 dimensions a 100-point LHS gives ~8-10 pts per
  marginal bin (10 strata), which is sufficient for a GP emulator with 5
  active inputs.  200 points would be better if run-time budget allows.

OBSERVABLES COMPUTED PER DESIGN POINT (ultracentral 0-1% bin)
  e2{2}, e2{4}, e2{6}, e2{8}  — ε₂ cumulant hierarchy
  e3{2}                        — triangularity
  e4{2}                        — hexadecapole
  e2{4}/e2{2} = R42            — fluctuation suppression ratio
  e3{2}/e2{2}                  — triangularity-to-ellipticity ratio (PUZZLE)
  Gamma_2                      — excess kurtosis of p(ε₂)
  e4{4}^4                      — sign-change observable (PUZZLE)
  <e2^8>/<e2^4>^2              — sign-change trigger

DIRECTORY STRUCTURE
  {WORK_DIR}/
    seeds/
      seeds-conf.yaml
      nucleon-seeds.hdf
    design_{NNN:04d}/
      isobar-conf.yaml
      WS1.hdf   (projectile 208Pb config)
      WS2.hdf   (target 208Pb config — identical nuclear params)
      trento_events.npy
    results/
      observables.npz        (all observables × all design points)
      summary_table.txt
      plots/
        corner_lhs.pdf
        obs_vs_beta3.pdf
        obs_vs_beta4.pdf
        obs_vs_R.pdf
        obs_vs_a.pdf
        obs_vs_w.pdf
        obs_all_panels.pdf

DEPENDENCIES
  python3, numpy, matplotlib, pyyaml
  Isobar-Sampler  (mluzum/Isobar-Sampler on GitHub)
  Trento          (https://github.com/Duke-QCD/trento)
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yaml

# =============================================================================
# ── TOP-LEVEL CONFIGURATION  <-- edit here
# =============================================================================

# ── Collision energy ──────────────────────────────────────────────────────────
SQRTS = 2.76   # TeV  |  choose 2.76 or 5.02

# ── Executables ───────────────────────────────────────────────────────────────
ISOBAR_SAMPLER_DIR = Path("/path/to/Isobar-Sampler")   # <-- set this
MAKE_SEEDS_EXEC    = ISOBAR_SAMPLER_DIR / "exec" / "make_seeds.py"
BUILD_ISOBARS_EXEC = ISOBAR_SAMPLER_DIR / "exec" / "build_isobars.py"
TRENTO_BIN         = Path("/root/.local/bin/trento")    # <-- set this

# ── Working directory ─────────────────────────────────────────────────────────
WORK_DIR = Path(f"pbpb_scan_{int(SQRTS*100):d}GeV")

# ── Stage control (set False to skip a completed stage) ──────────────────────
RUN_STAGE_0_LHS      = True   # generate LHS design matrix
RUN_STAGE_1_SEEDS    = True   # create nucleon-seed bank (once)
RUN_STAGE_2_ISOBARS  = True   # build nuclear configurations (per design pt)
RUN_STAGE_3_TRENTO   = True   # run Trento (per design pt)
RUN_STAGE_4_ECCEN    = True   # compute eccentricities (per design pt)
RUN_STAGE_5_PLOTS    = True   # make summary plots

# ── LHS design ────────────────────────────────────────────────────────────────
N_DESIGN   = 100    # design points  (100 is the minimum; 200 is recommended)
N_LHS_ITER = 2000   # maximin optimisation iterations
SEED_LHS   = 0

# ── Isobar sampler settings ───────────────────────────────────────────────────
N_NUCLEONS   = 208     # 208Pb
N_CONFIGS    = 10000   # nucleon-position configurations (seed bank size)
N_PARALLEL   = -1      # -1 = all available CPUs

# ── Trento settings ───────────────────────────────────────────────────────────
N_EVENTS     = 500_000   # events per design point

# ── Bootstrap settings ────────────────────────────────────────────────────────
N_BOOT = 300   # bootstrap resamples per centrality bin

# =============================================================================
# ── PARAMETER SPACE
# =============================================================================

# ── Free parameters (LHS-sampled) ────────────────────────────────────────────
# (name, lo, hi, latex_label)
FREE_PARAMS = [
    ("R",      6.50, 6.80, r"$R$ [fm]"),
    ("a",      0.44, 0.65, r"$a$ [fm]"),
    ("beta3",  0.00, 0.12, r"$\beta_3$"),      # KEY: v2-to-v3 puzzle
    ("beta4", -0.02, 0.06, r"$\beta_4$"),      # KEY: v4{4}^4 sign
    ("w",      0.40, 1.20, r"$w$ [fm]"),       # Trento nucleon width
]
FREE_NAMES  = [p[0] for p in FREE_PARAMS]
FREE_LO     = np.array([p[1] for p in FREE_PARAMS])
FREE_HI     = np.array([p[2] for p in FREE_PARAMS])
FREE_LABELS = [p[3] for p in FREE_PARAMS]
N_FREE      = len(FREE_PARAMS)

# ── Fixed nuclear parameters (Bayesian calibration central values) ────────────
# Ref: Bernhard 2019 (arXiv:1901.07808); Nijs et al. 2021 (arXiv:2010.15134)
#      de Vries et al., ADNDT 36, 495 (1987) — WS proton distribution of Pb
#      Luzum et al., arXiv:2302.14026 — correlation defaults
FIXED_NUCLEAR = {
    "beta_2":             0.0,   # 208Pb is doubly magic → near-spherical
    "gamma":              0.0,   # axially symmetric
    "correlation_length": 0.4,   # fm, NN hard-core scale
    "correlation_strength": -1.0,  # full Pauli exclusion
}

# ── Fixed Trento parameters (energy-dependent) ───────────────────────────────
# norm scales with dN_ch/deta; 2.76 TeV: ~1600, 5.02 TeV: ~1970 (ALICE data)
# Posterior medians from Nijs et al. (Trajectum) Pb+Pb fit
_TRENTO_FIXED_BY_ENERGY = {
    2.76: {"p": 0.0, "k": 1.6, "d": 1.0, "norm": 18.0},
    5.02: {"p": 0.0, "k": 1.6, "d": 1.0, "norm": 20.0},
}

def get_fixed_trento():
    if SQRTS not in _TRENTO_FIXED_BY_ENERGY:
        sys.exit(f"[ERROR] SQRTS={SQRTS} not in {list(_TRENTO_FIXED_BY_ENERGY.keys())}. "
                 "Choose 2.76 or 5.02.")
    return _TRENTO_FIXED_BY_ENERGY[SQRTS]

# ── Centrality configuration ──────────────────────────────────────────────────
CENTRALITY_EDGES = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7.5,
                    10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
UC_BIN_EDGE = 1.0   # "ultracentral" = 0–1%

# ── Observables to compute and store ─────────────────────────────────────────
OBS_KEYS = [
    "e2_2", "e2_4", "e2_6", "e2_8",
    "e3_2", "e4_2",
    "R42",          # e2{4}/e2{2}
    "T32",          # e3{2}/e2{2}  — THE PUZZLE RATIO
    "Gamma2",       # excess kurtosis
    "e4_4c",        # e4{4}^4 = 2<e4^2>^2 - <e4^4>  (can go negative)
    "nl_r8",        # <e2^8>/<e2^4>^2  (sign-change trigger > 2 → neg v4{4}^4)
]
N_OBS = len(OBS_KEYS)

# Logging
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# ── STAGE 0: MAXIMIN LATIN HYPERCUBE
# =============================================================================

def _lhs_unit(n, d, rng):
    X = np.empty((n, d))
    for j in range(d):
        perm    = rng.permutation(n)
        X[:, j] = (perm + rng.uniform(size=n)) / n
    return X


def generate_lhs(n, d, rng, n_iter=N_LHS_ITER):
    """Maximin-optimised LHS in [0,1]^d — no external packages required."""
    best, best_dmin = None, -1.0
    for _ in range(n_iter):
        cand  = _lhs_unit(n, d, rng)
        diff  = cand[:, None, :] - cand[None, :, :]
        dists = np.sqrt((diff**2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)
        dmin = dists.min()
        if dmin > best_dmin:
            best_dmin = dmin
            best      = cand.copy()
    return best


def stage0_lhs(work_dir: Path):
    """Generate the LHS design matrix and save it."""
    rng      = np.random.default_rng(SEED_LHS)
    log.info(f"Generating {N_DESIGN}-pt maximin LHS over {N_FREE} params "
             f"(n_iter={N_LHS_ITER}) ...")
    t0       = time.perf_counter()
    unit_lhs = generate_lhs(N_DESIGN, N_FREE, rng)
    design   = FREE_LO + unit_lhs * (FREE_HI - FREE_LO)
    log.info(f"LHS done in {time.perf_counter()-t0:.1f}s")

    # Print summary
    print("\n" + "=" * 70)
    print(f"LHS  —  {N_DESIGN} pts × {N_FREE} free params  "
          f"(sqrt(s)={SQRTS} TeV, 208Pb+208Pb)")
    print("=" * 70)
    print(f"{'Param':>8}  {'Lo':>6}  {'Hi':>6}  {'Mean':>8}  {'Std':>8}")
    print("-" * 70)
    for j, (name, lo, hi, lbl) in enumerate(FREE_PARAMS):
        col = design[:, j]
        print(f"{name:>8}  {lo:>6.3f}  {hi:>6.3f}  "
              f"{col.mean():>8.4f}  {col.std():>8.4f}")
    # Space-filling
    diff  = unit_lhs[:, None, :] - unit_lhs[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    print(f"\n  Min pairwise dist (unit cube): {dists.min():.4f}")
    print(f"  Max pairwise dist (unit cube): {dists[dists<np.inf].max():.4f}")
    print("=" * 70 + "\n")

    # Save
    archive = work_dir / "lhs_design_matrix.npz"
    np.savez_compressed(archive,
        design=design, unit_lhs=unit_lhs,
        param_names=np.array(FREE_NAMES), param_lo=FREE_LO, param_hi=FREE_HI)
    log.info(f"Design matrix saved  ->  {archive}")
    return design


def load_design(work_dir: Path) -> np.ndarray:
    return np.load(work_dir / "lhs_design_matrix.npz")["design"]


# =============================================================================
# ── STAGE 1: NUCLEON SEED BANK
# =============================================================================

def _write_seeds_conf(seeds_dir: Path) -> Path:
    """Write seeds-conf.yaml for make_seeds.py."""
    cfg = {
        "isobar_seeds": {
            "description": "Nucleon-position seeds for 208Pb (LHC Pb+Pb study).",
            "number_nucleons": {"description": "Mass number A.", "value": N_NUCLEONS},
            "number_configs":  {"description": "Number of seed configurations.",
                                "value": N_CONFIGS},
            "output_file": {
                "description": "Output path for seeds HDF file.",
                "filename": str(seeds_dir / "nucleon-seeds.hdf"),
            },
            "number_of_parallel_processes": {
                "description": "Parallel processes (-1 = all CPUs).",
                "value": N_PARALLEL,
            },
        }
    }
    conf_path = seeds_dir / "seeds-conf.yaml"
    with open(conf_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return conf_path


def stage1_seeds(work_dir: Path):
    """Generate the nucleon-seed bank (once, shared across all design points)."""
    seeds_dir = work_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seeds_hdf = seeds_dir / "nucleon-seeds.hdf"

    if seeds_hdf.exists():
        log.info(f"Seed bank already exists, skipping: {seeds_hdf}")
        return seeds_hdf

    conf_path = _write_seeds_conf(seeds_dir)
    log.info(f"Running make_seeds.py  ...")
    _run(["python3", str(MAKE_SEEDS_EXEC), str(conf_path)],
         label="make_seeds")
    if not seeds_hdf.exists():
        sys.exit(f"[ERROR] Expected seed file not found: {seeds_hdf}")
    log.info(f"Seed bank saved  ->  {seeds_hdf}")
    return seeds_hdf


# =============================================================================
# ── STAGE 2: BUILD NUCLEAR CONFIGURATIONS (per design point)
# =============================================================================

def _write_isobar_conf(design_dir: Path, row: np.ndarray,
                       seeds_hdf: Path) -> Path:
    """
    Write isobar-conf.yaml for build_isobars.py.

    Both isobars (WS1 = projectile, WS2 = target) use identical parameters
    for Pb+Pb collisions.
    """
    d = dict(zip(FREE_NAMES, row))
    f = lambda x: round(float(x), 6)

    def _pb_isobar(name):
        return {
            "isobar_name": name,
            "WS_radius": {
                "description": "Woods-Saxon radius R [fm]",
                "value": f(d["R"]),
            },
            "WS_diffusiveness": {
                "description": "Woods-Saxon diffuseness a [fm]",
                "value": f(d["a"]),
            },
            "beta_2": {
                "description": "Quadrupole deformation (fixed 0 for doubly-magic Pb).",
                "value": f(FIXED_NUCLEAR["beta_2"]),
            },
            "gamma": {
                "description": "Triaxiality angle [rad] (fixed 0 = axially symmetric).",
                "value": f(FIXED_NUCLEAR["gamma"]),
            },
            "beta_3": {
                "description": "Octupole deformation (KEY: v2-to-v3 puzzle).",
                "value": f(d["beta3"]),
            },
            "beta_4": {
                "description": "Hexadecapole deformation (v4{4}^4 sign puzzle).",
                "value": f(d["beta4"]),
            },
            "correlation_length": {
                "description": "NN short-range correlation radius [fm] (fixed).",
                "value": f(FIXED_NUCLEAR["correlation_length"]),
            },
            "correlation_strength": {
                "description": "NN correlation depth (fixed, -1 = full exclusion).",
                "value": f(FIXED_NUCLEAR["correlation_strength"]),
            },
        }

    cfg = {
        "isobar_samples": {
            "description": "Isobar nucleon-position samples for 208Pb.",
            "number_configs": {
                "description": "Number of configurations.",
                "value": N_CONFIGS,
            },
            "number_nucleons": {
                "description": "Mass number A.",
                "value": N_NUCLEONS,
            },
            "seeds_file": {
                "description": "Nucleon-position seeds.",
                "filename": str(seeds_hdf),
            },
            "output_path": {
                "description": "Output directory for WS*.hdf files.",
                "dirname": str(design_dir),
            },
            "number_of_parallel_processes": {
                "description": "Parallel processes (-1 = all CPUs).",
                "value": N_PARALLEL,
            },
        },
        "isobar_properties": {
            "description": "Nuclear properties of 208Pb (both beams identical).",
            "isobar1": _pb_isobar("WS1"),
            "isobar2": _pb_isobar("WS2"),
        },
    }

    conf_path = design_dir / "isobar-conf.yaml"
    with open(conf_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    return conf_path


def stage2_isobars(work_dir: Path, design: np.ndarray, seeds_hdf: Path):
    """Build nuclear configurations for every design point."""
    for idx, row in enumerate(design):
        design_dir = work_dir / f"design_{idx:04d}"
        design_dir.mkdir(parents=True, exist_ok=True)

        ws1 = design_dir / "WS1.hdf"
        ws2 = design_dir / "WS2.hdf"
        if ws1.exists() and ws2.exists():
            log.info(f"  [{idx:04d}] HDF files exist, skipping build_isobars.")
            continue

        conf_path = _write_isobar_conf(design_dir, row, seeds_hdf)
        log.info(f"  [{idx:04d}] build_isobars  "
                 f"R={row[0]:.3f} a={row[1]:.3f} "
                 f"b3={row[2]:.4f} b4={row[3]:.4f} w={row[4]:.3f}")
        _run(["python3", str(BUILD_ISOBARS_EXEC), str(conf_path)],
             label=f"build_isobars [{idx:04d}]")

        if not ws1.exists() or not ws2.exists():
            log.warning(f"  [{idx:04d}] Expected WS1.hdf/WS2.hdf not found "
                        f"after build_isobars — check isobar sampler output.")


# =============================================================================
# ── STAGE 3: RUN TRENTO (per design point)
# =============================================================================

def stage3_trento(work_dir: Path, design: np.ndarray):
    """Run Trento for every design point; cache events as .npy."""
    fixed = get_fixed_trento()

    for idx, row in enumerate(design):
        design_dir = work_dir / f"design_{idx:04d}"
        cache_file = design_dir / "trento_events.npy"

        if cache_file.exists():
            log.info(f"  [{idx:04d}] Trento cache exists, skipping.")
            continue

        ws1 = design_dir / "WS1.hdf"
        ws2 = design_dir / "WS2.hdf"
        if not ws1.exists() or not ws2.exists():
            log.warning(f"  [{idx:04d}] Missing WS*.hdf — skipping Trento.")
            continue

        d_row = dict(zip(FREE_NAMES, row))
        w_val = d_row["w"]

        cmd = [
            str(TRENTO_BIN),
            str(ws1), str(ws2),
            str(N_EVENTS),
            "-p", str(fixed["p"]),
            "-k", str(fixed["k"]),
            "-w", f"{w_val:.6f}",
            "-d", str(fixed["d"]),
            "-n", str(fixed["norm"]),
            "--random-seed", str(1000 + idx),
            # No -o flag: events summary only via stdout
        ]

        log.info(f"  [{idx:04d}] Trento  w={w_val:.3f}  "
                 f"(R={row[0]:.3f} a={row[1]:.3f} "
                 f"b3={row[2]:.4f} b4={row[3]:.4f})")
        result = _run(cmd, label=f"trento [{idx:04d}]",
                      capture_output=True)

        rows = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    rows.append([float(x) for x in parts[:8]])
                except ValueError:
                    pass

        if not rows:
            log.warning(f"  [{idx:04d}] Trento produced no events — "
                        "check binary and HDF files.")
            continue

        arr = np.array(rows, dtype=np.float64)
        np.save(cache_file, arr)
        log.info(f"  [{idx:04d}] {len(arr):,} events  ->  {cache_file.name}")


# =============================================================================
# ── STAGE 4: COMPUTE ECCENTRICITIES (per design point)
# =============================================================================
# Trento stdout columns: ev  b  npart  mult  e2  e3  e4  e5
COL_MULT = 3
COL_E2   = 4
COL_E3   = 5
COL_E4   = 6


def _assign_centrality(events: np.ndarray) -> np.ndarray:
    rank = np.argsort(np.argsort(-events[:, COL_MULT]))
    return rank / len(events) * 100.0


def _moments(eps: np.ndarray, orders=(2, 4, 6, 8)) -> dict:
    return {n: np.mean(eps**n) for n in orders}


def _e2_cumulants(m: dict):
    m2, m4, m6, m8 = m[2], m[4], m[6], m[8]
    e2 = np.sqrt(m2)
    inner4 = 2.0 * m2**2 - m4
    e4     = inner4**0.25 if inner4 >= 0 else np.nan
    c6     = m6 - 9.0 * m4 * m2 + 12.0 * m2**3
    e6     = (-c6/4.0)**(1.0/6.0) if -c6/4.0 >= 0 else np.nan
    c8     = (m8 - 16.0*m6*m2 - 18.0*m4**2
              + 144.0*m4*m2**2 - 144.0*m2**4)
    e8     = (c8/33.0)**(1.0/8.0) if c8/33.0 >= 0 else np.nan
    return e2, e4, e6, e8


def _compute_obs(sel: np.ndarray) -> np.ndarray:
    """Compute all observables for a subset of events; return 1-D array."""
    eps2 = sel[:, COL_E2]
    eps3 = sel[:, COL_E3]
    eps4 = sel[:, COL_E4] if sel.shape[1] > COL_E4 else np.zeros(len(sel))

    m2         = _moments(eps2, (2, 4, 6, 8))
    e22, e24, e26, e28 = _e2_cumulants(m2)
    e32        = float(np.sqrt(np.mean(eps3**2)))
    e42        = float(np.sqrt(np.mean(eps4**2)))
    R42        = e24/e22 if (e22 > 0 and not np.isnan(e24)) else np.nan
    T32        = e32/e22 if e22 > 0 else np.nan
    Gamma2     = float((m2[4] - 2*m2[2]**2)/m2[2]**2) if m2[2] > 0 else np.nan
    m4_e4      = np.mean(eps4**4)
    e4_4c      = float(2*e42**4 - m4_e4)  # = 2<e4^2>^2 - <e4^4>
    nl_r8      = float(m2[8]/m2[4]**2) if m2[4] > 0 else np.nan

    return np.array([e22, e24, e26, e28, e32, e42, R42, T32,
                     Gamma2, e4_4c, nl_r8])


def _bootstrap_uc(events: np.ndarray, centrality: np.ndarray,
                  uc_max: float = UC_BIN_EDGE, n_boot: int = N_BOOT):
    """
    Bootstrap uncertainty for the ultracentral bin (0–uc_max%).
    Returns dict key -> (central, lo_1sigma, hi_1sigma).
    """
    sel     = events[centrality < uc_max]
    if len(sel) < 20:
        return {k: (np.nan, np.nan, np.nan) for k in OBS_KEYS}

    central = _compute_obs(sel)
    rng     = np.random.default_rng(seed=42)
    boot    = np.empty((n_boot, N_OBS))
    n       = len(sel)
    for i in range(n_boot):
        idx     = rng.integers(0, n, size=n)
        boot[i] = _compute_obs(sel[idx])

    lo = np.nanpercentile(boot, 16, axis=0)
    hi = np.nanpercentile(boot, 84, axis=0)
    return {k: (central[j], lo[j], hi[j]) for j, k in enumerate(OBS_KEYS)}


def stage4_eccentricities(work_dir: Path, design: np.ndarray) -> dict:
    """
    Compute ultracentral observables for every design point.
    Returns results dict: key -> array of shape (N_DESIGN,) for central values,
                                 and key+'_lo', key+'_hi' for ±1σ bands.
    """
    results = {k:    np.full(N_DESIGN, np.nan) for k in OBS_KEYS}
    results.update({k+"_lo": np.full(N_DESIGN, np.nan) for k in OBS_KEYS})
    results.update({k+"_hi": np.full(N_DESIGN, np.nan) for k in OBS_KEYS})
    results["n_uc"] = np.zeros(N_DESIGN, dtype=int)

    for idx in range(N_DESIGN):
        cache_file = work_dir / f"design_{idx:04d}" / "trento_events.npy"
        if not cache_file.exists():
            log.warning(f"  [{idx:04d}] No trento_events.npy — skipping.")
            continue

        events     = np.load(cache_file)
        centrality = _assign_centrality(events)
        uc_mask    = centrality < UC_BIN_EDGE
        n_uc       = uc_mask.sum()
        results["n_uc"][idx] = n_uc

        if n_uc < 50:
            log.warning(f"  [{idx:04d}] Only {n_uc} UC events — skipping.")
            continue

        obs = _bootstrap_uc(events, centrality)
        for k in OBS_KEYS:
            results[k][idx]       = obs[k][0]
            results[k+"_lo"][idx] = obs[k][1]
            results[k+"_hi"][idx] = obs[k][2]

        log.info(f"  [{idx:04d}] n_UC={n_uc:,}  "
                 f"e2{{2}}={results['e2_2'][idx]:.4f}  "
                 f"e3{{2}}={results['e3_2'][idx]:.4f}  "
                 f"T32={results['T32'][idx]:.3f}  "
                 f"e4{{4}}^4={results['e4_4c'][idx]:+.4e}")

    # Save full results
    results_dir = work_dir / "results"
    results_dir.mkdir(exist_ok=True)
    np.savez_compressed(results_dir / "observables.npz",
        design=design, param_names=np.array(FREE_NAMES),
        param_lo=FREE_LO, param_hi=FREE_HI,
        obs_keys=np.array(OBS_KEYS),
        **results)
    log.info(f"Observables saved  ->  {results_dir/'observables.npz'}")

    # Print summary table
    _print_summary_table(design, results, results_dir)
    return results


def _print_summary_table(design, results, results_dir):
    hdr = (f"{'idx':>5}  {'R':>6}  {'a':>6}  "
           f"{'b3':>7}  {'b4':>7}  {'w':>5}  "
           f"{'e2{{2}}':>7}  {'e3{{2}}':>7}  "
           f"{'T32':>6}  {'R42':>6}  "
           f"{'G2':>7}  {'e4{{4}}^4':>10}  {'nl_r8':>7}")
    sep = "=" * len(hdr)
    lines = [sep, hdr, sep]
    for idx in range(N_DESIGN):
        if np.isnan(results["e2_2"][idx]):
            continue
        row = design[idx]
        lines.append(
            f"{idx:>5}  {row[0]:>6.3f}  {row[1]:>6.3f}  "
            f"{row[2]:>7.4f}  {row[3]:>7.4f}  {row[4]:>5.3f}  "
            f"{results['e2_2'][idx]:>7.4f}  {results['e3_2'][idx]:>7.4f}  "
            f"{results['T32'][idx]:>6.3f}  {results['R42'][idx]:>6.3f}  "
            f"{results['Gamma2'][idx]:>7.3f}  "
            f"{results['e4_4c'][idx]:>+10.4e}  "
            f"{results['nl_r8'][idx]:>7.3f}"
        )
    lines.append(sep)
    text = "\n".join(lines)
    print(text)
    (results_dir / "summary_table.txt").write_text(text)


# =============================================================================
# ── STAGE 5: PLOTS
# =============================================================================

def stage5_plots(work_dir: Path, design: np.ndarray, results: dict):
    """
    Two plot types:
      A) LHS corner plot  (space-filling quality)
      B) Observables vs each free parameter  (5 × N_OBS panel grid)
    """
    plots_dir = work_dir / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_lhs_corner(design, plots_dir)
    _plot_obs_panels(design, results, plots_dir)
    log.info(f"Plots saved  ->  {plots_dir}/")


# ── Corner plot ───────────────────────────────────────────────────────────────

def _plot_lhs_corner(design: np.ndarray, plots_dir: Path):
    BG = "#0d0f14"; PANEL = "#13161e"; GRID = "#1e2230"; TEXT = "#d8dce8"
    COLORS = ["#4fc3f7", "#4fc3f7", "#ff4081", "#ff80ab", "#ffd54f"]
    # R, a = cyan (WS); beta3, beta4 = magenta (PUZZLE); w = gold (Trento)

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT,    "ytick.color": TEXT,
        "text.color": TEXT,     "grid.color": GRID,
        "grid.linewidth": 0.35, "font.family": "sans-serif",
        "font.size": 8,
    })

    unit = (design - FREE_LO) / (FREE_HI - FREE_LO)
    n    = N_FREE
    fig, axes = plt.subplots(n, n, figsize=(n*2.8, n*2.8))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        r"$^{208}$Pb+$^{208}$Pb Ultracentral Flow Puzzle — LHS Corner Plot"
        f"\n{N_DESIGN} design pts × {N_FREE} free params  "
        f"(sqrt(s)={SQRTS} TeV)\n"
        r"WS: cyan   |   $\beta_3$/$\beta_4$ puzzle: magenta   |   "
        r"$w$ Trento: gold",
        fontsize=10, color=TEXT, y=1.02, fontweight="bold",
    )

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.4)

            if row == col:
                c = COLORS[row]
                ax.hist(unit[:, col], bins=14, color=c, alpha=0.80,
                        edgecolor=BG, linewidth=0.25)
                ax.set_xlim(0, 1); ax.set_yticks([])
                ax.plot(unit[:, col], np.zeros(N_DESIGN) - 0.5,
                        "|", color=c, alpha=0.4, markersize=3.5)
            elif row > col:
                c = COLORS[col]
                ax.scatter(unit[:, col], unit[:, row],
                           c=c, s=9, alpha=0.55, linewidths=0)
                ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
            else:
                ax.set_visible(False)
                continue

            ax.grid(True, linestyle="--", alpha=0.18)

            if row == n - 1:
                ax.set_xlabel(FREE_LABELS[col], fontsize=9, color=TEXT,
                              labelpad=2)
                lo, hi = FREE_LO[col], FREE_HI[col]
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.2g}",
                                    f"{hi:.2g}"], fontsize=6.5)
            else:
                ax.set_xticks([]); ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(FREE_LABELS[row], fontsize=9, color=TEXT,
                              labelpad=2)
                lo, hi = FREE_LO[row], FREE_HI[row]
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.2g}",
                                    f"{hi:.2g}"], fontsize=6.5)
            else:
                ax.set_yticks([]); ax.set_yticklabels([])

    # Highlight puzzle-param diagonal cells
    for pi in [2, 3]:  # beta3, beta4
        for sp in axes[pi, pi].spines.values():
            sp.set_edgecolor("#ff4081"); sp.set_linewidth(1.3)
        axes[pi, pi].set_title("PUZZLE", color="#ff4081",
                               fontsize=7, pad=2, fontweight="bold")

    fig.tight_layout(h_pad=0.06, w_pad=0.06)
    fig.savefig(plots_dir / "corner_lhs.pdf", dpi=150,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("  Saved corner_lhs.pdf")


# ── Observables vs parameters ─────────────────────────────────────────────────

# Observable metadata: (key, label, is_puzzle_obs, allow_negative)
OBS_META = [
    ("e2_2",   r"$\varepsilon_2\{2\}$",          False, False),
    ("e2_4",   r"$\varepsilon_2\{4\}$",          False, False),
    ("e3_2",   r"$\varepsilon_3\{2\}$",          True,  False),
    ("e4_2",   r"$\varepsilon_4\{2\}$",          False, False),
    ("R42",    r"$\varepsilon_2\{4\}/\varepsilon_2\{2\}$",  True, False),
    ("T32",    r"$\varepsilon_3\{2\}/\varepsilon_2\{2\}$",  True, False),
    ("Gamma2", r"$\Gamma_2$",                    True,  True),
    ("e4_4c",  r"$\varepsilon_4\{4\}^4$",        True,  True),
    ("nl_r8",  r"$\langle\varepsilon_2^8\rangle/"
                r"\langle\varepsilon_2^4\rangle^2$", True, False),
]

def _plot_obs_panels(design: np.ndarray, results: dict, plots_dir: Path):
    """
    For each free parameter: one column.
    For each observable: one row.
    Each panel: scatter of obs central value vs param, with ±1σ error bars,
    colour-coded by whether the observable is a "puzzle observable".
    """
    BG    = "#0d0f14"; PANEL = "#13161e"; GRID  = "#1e2230"; TEXT = "#d8dce8"
    NORM_C  = "#4fc3f7"   # non-puzzle observables: cyan
    PUZ_C   = "#ff4081"   # puzzle observables: magenta
    REF_C   = "#ffffff"   # reference line: white

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "grid.color": GRID,
        "grid.linewidth": 0.35, "font.family": "sans-serif", "font.size": 8,
    })

    n_obs  = len(OBS_META)
    n_par  = N_FREE
    fw     = n_par * 3.2
    fh     = n_obs * 2.5

    fig, axes = plt.subplots(n_obs, n_par, figsize=(fw, fh), squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        r"$^{208}$Pb+$^{208}$Pb  Ultracentral (0–1%) Eccentricities vs Parameters"
        f"\nsqrt(s)={SQRTS} TeV  |  {N_DESIGN} design pts  |  ±1σ bootstrap",
        fontsize=12, color=TEXT, y=1.002, fontweight="bold",
    )

    valid = ~np.isnan(results["e2_2"])  # mask for design pts with results

    for i_obs, (key, lbl, is_puzzle, allow_neg) in enumerate(OBS_META):
        c_obs = PUZ_C if is_puzzle else NORM_C
        yvals = results[key]
        ylo   = results[key + "_lo"]
        yhi   = results[key + "_hi"]
        yerr_lo = np.where(valid, yvals - ylo, np.nan)
        yerr_hi = np.where(valid, yhi - yvals, np.nan)

        for i_par, (pname, plo, phi, plbl) in enumerate(FREE_PARAMS):
            ax  = axes[i_obs, i_par]
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.4)

            xvals = design[valid, i_par]
            yv    = yvals[valid]
            ye_lo = yerr_lo[valid]
            ye_hi = yerr_hi[valid]

            # Error bar plot
            ax.errorbar(xvals, yv,
                        yerr=[ye_lo, ye_hi],
                        fmt="o", color=c_obs,
                        markersize=4, alpha=0.65,
                        elinewidth=0.8, capsize=2,
                        markerfacecolor=BG, markeredgewidth=1.0,
                        markeredgecolor=c_obs)

            # Reference lines
            if key == "T32":
                ax.axhline(1.0, color=REF_C, lw=0.8, ls="--", alpha=0.5,
                           label="T32=1 (data target)")
            if key == "e4_4c":
                ax.axhline(0.0, color=REF_C, lw=0.8, ls=":", alpha=0.5)
            if key == "nl_r8":
                ax.axhline(2.0, color=REF_C, lw=0.8, ls="--", alpha=0.5,
                           label="trigger=2")
            if key == "Gamma2":
                ax.axhline(-2.0, color=REF_C, lw=0.8, ls="--", alpha=0.5,
                           label=r"$\Gamma_2=-2$ limit")

            ax.grid(True, linestyle="--", alpha=0.20)
            if not allow_neg:
                ax.set_ylim(bottom=0)

            # Labels
            if i_obs == n_obs - 1:
                ax.set_xlabel(plbl, fontsize=9, labelpad=3)
                ax.set_xlim(plo - 0.01*(phi-plo),
                            phi + 0.01*(phi-plo))
            else:
                ax.set_xticklabels([])

            if i_par == 0:
                ax.set_ylabel(lbl, fontsize=9, labelpad=3)
            else:
                ax.set_yticklabels([])

            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

        # Row label on the left spine
        axes[i_obs, 0].annotate(
            "PUZZLE" if is_puzzle else "",
            xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=7, color=PUZ_C, ha="center", va="center",
            rotation=90, alpha=0.7,
        )

    # Column headers
    for i_par, (_, _, _, plbl) in enumerate(FREE_PARAMS):
        axes[0, i_par].set_title(plbl, color=TEXT, fontsize=10,
                                  fontweight="bold", pad=6)

    fig.tight_layout(h_pad=0.25, w_pad=0.15,
                     rect=[0.03, 0, 1, 1])
    out = plots_dir / "obs_all_panels.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info(f"  Saved obs_all_panels.pdf")

    # Also save individual per-parameter plots for easy inspection
    for i_par, (pname, plo, phi, plbl) in enumerate(FREE_PARAMS):
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor(BG); ax2.set_facecolor(PANEL)
        for sp in ax2.spines.values():
            sp.set_edgecolor(GRID)
        xvals = design[valid, i_par]
        for key, lbl, is_puzzle, allow_neg in OBS_META:
            c = PUZ_C if is_puzzle else NORM_C
            yv = results[key][valid]
            ax2.plot(xvals, yv, "o", color=c, markersize=4,
                     alpha=0.6, label=lbl,
                     markerfacecolor=BG, markeredgewidth=0.8)
        ax2.set_xlabel(plbl, color=TEXT); ax2.set_ylabel("observable", color=TEXT)
        ax2.set_title(f"Observables vs {pname}  (sqrt(s)={SQRTS} TeV, UC 0–1%)",
                      color=TEXT)
        ax2.grid(True, ls="--", alpha=0.2)
        ax2.legend(fontsize=6, ncol=3, facecolor=PANEL,
                   edgecolor=GRID, labelcolor=TEXT)
        fig2.tight_layout()
        fig2.savefig(plots_dir / f"obs_vs_{pname}.pdf", dpi=150,
                     bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
    log.info("  Saved individual obs_vs_*.pdf plots")


# =============================================================================
# ── UTILITY: run subprocess
# =============================================================================

def _run(cmd, label="", capture_output=False):
    log.debug(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=capture_output,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"{label} failed (exit {result.returncode}).")
        if capture_output:
            log.error(f"stderr: {result.stderr[:400]}")
        sys.exit(1)
    return result


# =============================================================================
# ── MAIN
# =============================================================================

def main():
    # Print configuration header
    print("=" * 68)
    print(f"  208Pb+208Pb Ultracentral Flow Puzzle Scan")
    print(f"  sqrt(s_NN) = {SQRTS} TeV")
    print(f"  N_DESIGN = {N_DESIGN}   N_EVENTS = {N_EVENTS:,}/pt   "
          f"N_CONFIGS = {N_CONFIGS:,}")
    print(f"  Work directory: {WORK_DIR}")
    print("=" * 68)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    design = None

    # ── Stage 0: LHS ─────────────────────────────────────────────────────────
    design_archive = WORK_DIR / "lhs_design_matrix.npz"
    if RUN_STAGE_0_LHS or not design_archive.exists():
        design = stage0_lhs(WORK_DIR)
    else:
        design = load_design(WORK_DIR)
        log.info(f"Loaded existing design ({len(design)} pts).")

    # ── Stage 1: Seeds ────────────────────────────────────────────────────────
    seeds_hdf = WORK_DIR / "seeds" / "nucleon-seeds.hdf"
    if RUN_STAGE_1_SEEDS:
        seeds_hdf = stage1_seeds(WORK_DIR)
    elif not seeds_hdf.exists():
        log.warning("Stage 1 skipped but seeds HDF not found. "
                    "Set RUN_STAGE_1_SEEDS=True.")

    # ── Stage 2: Isobar configurations ───────────────────────────────────────
    if RUN_STAGE_2_ISOBARS:
        stage2_isobars(WORK_DIR, design, seeds_hdf)

    # ── Stage 3: Trento ───────────────────────────────────────────────────────
    if RUN_STAGE_3_TRENTO:
        stage3_trento(WORK_DIR, design)

    # ── Stage 4: Eccentricities ───────────────────────────────────────────────
    results_archive = WORK_DIR / "results" / "observables.npz"
    if RUN_STAGE_4_ECCEN or not results_archive.exists():
        results = stage4_eccentricities(WORK_DIR, design)
    else:
        log.info(f"Loading cached observables from {results_archive}")
        data = np.load(results_archive, allow_pickle=True)
        results = {k: data[k] for k in list(data.keys())
                   if k not in ("design", "param_names", "param_lo",
                                "param_hi", "obs_keys")}

    # ── Stage 5: Plots ────────────────────────────────────────────────────────
    if RUN_STAGE_5_PLOTS:
        stage5_plots(WORK_DIR, design, results)

    print("\n[INFO] All stages complete.")


if __name__ == "__main__":
    main()
