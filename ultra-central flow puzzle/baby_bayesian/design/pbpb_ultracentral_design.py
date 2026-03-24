#!/usr/bin/env python3
"""
pbpb_ultracentral_design.py
=========================
PART 1 of the 208Pb+208Pb ultracentral flow puzzle pipeline. Generates a maximin-optimised Latin Hypercube Sample (LHS) over 
the N_FREE free nuclear/Trento parameters and produces a publication-quality corner plot for visual inspection of the 
space-filling quality.

OUTPUTS (written to OUTPUT_DIR)
  lhs_design_matrix.npz   -- design matrix + full metadata
                             load with: np.load(..., allow_pickle=True)
                             keys: design (N×N_FREE), unit_lhs (N×N_FREE),
                                   param_names, param_lo, param_hi
  lhs_corner_plot.pdf     -- N_FREE×N_FREE corner plot
  lhs_design_table.txt    -- human-readable table of all design points

USAGE
  python pbpb_lhs_design.py            # runs with defaults
  python pbpb_lhs_design.py --n 200    # 200 design points
  python pbpb_lhs_design.py --sqrts 5.02 # collision energy 5.02 [TeV]

FREE PARAMETERS (LHS-sampled)
  WS_R       WS radius [fm]          [6.50, 6.80]   de Vries 1987
  WS_A       WS diffuseness [fm]     [0.44, 0.65]   de Vries 1987 / PREX-II
  beta3   octupole deformation    [-0.12, 0.12]   Carzon 2020 / Xu 2025
  beta4   hexadecapole deformation    [-0.02, 0.06]  Bally 2022
  w       Trento nucleon width [fm]    [0.50, 1.50]   JETSCAPE 2021

FIXED PARAMETERS (not sampled — set to JETSCAPE Grad MAP parameters)
  Nuclear (Isobar sampler):  beta2=0, gamma=0, C_l=0.4 fm, C_s=-1
  Trento:   p=0, k=1.6, d=1.0 fm, norm=18.0 (2.76 TeV) / 20.0 (5.02 TeV)

DEPENDENCIES
  numpy, matplotlib  (standard; no special packages needed)
"""

import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================================
# ── CONFIGURATION  <-- edit here or pass via CLI flags
# =============================================================================

# ── Collision energy ──────────────────────────────────────────────────────────
SQRTS = 2.76 # collision energy [TeV] | choose 2.76 or 5.02
#SQRTS = 5.02 (option)

# ── Working directory ─────────────────────────────────────────────────────────
WORK_DIR = Path(f"pbpb_scan_{int(SQRTS*1000):d}GeV")

# ── LHS design ────────────────────────────────────────────────────────────────
N_DESIGN = 100 # design points (100 recommended for GP emulator; 200 better)
N_LHS_ITER = 2000 # maximin optimisation iterations or trials (more = better spacing)
SEED_LHS = 0 # # RNG seed for reproducibility.

# =============================================================================
# ── PARAMETER SPACE
# =============================================================================

# ── Free parameters (LHS-sampled) ──────────────────────────────────────────── 
# (name, lo, hi, latex_label, group)
# group: "WS" = Woods-Saxon nuclear, "PUZ" = puzzle-critical, "TR" = Trento
FREE_PARAMS = [
    ("WS_R", 6.50, 6.80, r"$R_0$ [fm]", "WS"), # Pb Woods-Saxon radius [fm]. Standard value: 6.62
    ("WS_A", 0.44, 0.65, r"$a$ [fm]", "WS"), # Pb Woods-Saxon diffuseness [fm]. Standard value: 0.546
    ("beta3", -0.12, 0.12, r"$\beta_3$", "PUZ"), # Octupole deformation coefficient
    ("beta4", -0.02, 0.06, r"$\beta_4$", "PUZ"), # Hexadecapole deformation coefficient
    ("w", 0.50, 1.50, r"$w$ [fm]", "TR"), # Trento nucleon width [fm] (JETSCAPE prior range: https://arxiv.org/pdf/2011.01430)
]

FREE_NAMES = [p[0] for p in FREE_PARAMS]
FREE_LO = np.array([p[1] for p in FREE_PARAMS])
FREE_HI = np.array([p[2] for p in FREE_PARAMS])
FREE_LABELS = [p[3] for p in FREE_PARAMS]
FREE_GROUPS = [p[4] for p in FREE_PARAMS]
N_FREE = len(FREE_PARAMS)

# Fixed parameters (for documentation in the output table)
FIXED_NUCLEAR = dict(beta2=0.0, gamma=0.0, corr_length=0.4, corr_strength=-1.0)
FIXED_TRENTO  = {2.76: dict(p=0.0, k=1.6, d=1.0, norm=18.0),
                 5.02: dict(p=0.0, k=1.6, d=1.0, norm=20.0)}

# Colour scheme for groups
GROUP_COLORS = {"WS": "#4fc3f7",   # cyan   — Woods-Saxon structure
                "PUZ": "#ff4081",  # magenta — puzzle-critical
                "TR": "#ffd54f"}   # gold   — Trento
# 
# =============================================================================
# ── STAGE 0: MAXIMIN LATIN HYPERCUBE
# =============================================================================

def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """One random LHS realisation in [0, 1]^d."""
    X = np.empty((n, d)) 
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.uniform(size=n)) / n
    return X

def generate_lhs(n: int, d: int, rng: np.random.Generator,
                 n_iter: int = N_LHS_ITER) -> np.ndarray:
    """Maximin-optimised LHS in [0,1]^d - no external packages required.
    Generates n_iter random LHS candidates and returns the one that
    maximises the minimum pairwise Euclidean distance between points.
    No external packages required.
    """
    best, best_dmin = None, -1.0
    for _ in range(n_iter):
        cand = _lhs_unit(n, d, rng)
        diff = cand[:, None, :] - cand[None, :, :] # (n, n, d)
        dists = np.sqrt((diff**2).sum(axis=-1)) # (n, n)
        np.fill_diagonal(dists, np.inf)
        dmin = dists.min()
        if dmin > best_dmin:
            best_dmin = dmin
            best = cand.copy()         
    return best
    
def scale_lhs(unit_lhs: np.ndarray) -> np.ndarray:
    """Affine map from [0, 1]^d to physical parameter ranges."""
    return FREE_LO + unit_lhs * (FREE_HI - FREE_LO)

# =============================================================================
# ── PRINT & SAVE
# =============================================================================

def print_summary(design: np.ndarray, unit_lhs: np.ndarray) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 72)
    print(f"  208Pb+208Pb LHS Design  |  sqrt(s) = {SQRTS} TeV")
    print(f"  {N_DESIGN} design points  ×  {N_FREE} free parameters")
    print("=" * 72)
    print(f"  {'Param':>7}  {'Group':>5}  "
          f"{'Lo':>6}  {'Hi':>6}  {'Mean':>8}  {'Std':>8}  {'CV%':>6}")
    print("  " + "-" * 56)
    for j, (name, lo, hi, lbl, grp) in enumerate(FREE_PARAMS):
        col = design[:, j]
        cv  = 100 * col.std() / col.mean() if col.mean() != 0 else float("nan")
        print(f"  {name:>7}  {grp:>5}  "
              f"{lo:>6.3f}  {hi:>6.3f}  "
              f"{col.mean():>8.4f}  {col.std():>8.4f}  {cv:>6.1f}")

    # Space-filling metrics
    diff = unit_lhs[:, None, :] - unit_lhs[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    dmin = dist.min()
    dmax = dist[dist < np.inf].max()

    print()
    print(f"  Space-filling (unit hypercube):")
    print(f"    Min pairwise distance : {dmin:.4f}   (larger = better)")
    print(f"    Max pairwise distance : {dmax:.4f}")
    print(f"    Max / Min ratio       : {dmax/dmin:.2f}")
    print("=" * 72)
    print()


def save_archive(design: np.ndarray, unit_lhs: np.ndarray,
                 out_dir: Path) -> Path:
    """Save design matrix and metadata to compressed .npz archive."""
    path = out_dir / "lhs_design_matrix.npz"
    np.savez_compressed(
        path,
        design      = design,
        unit_lhs    = unit_lhs,
        param_names = np.array(FREE_NAMES),
        param_lo    = FREE_LO,
        param_hi    = FREE_HI,
        param_labels= np.array(FREE_LABELS),
        param_groups= np.array(FREE_GROUPS),
        sqrts_tev   = np.float64(SQRTS),
    )
    print(f"[INFO] Design matrix  ->  {path}   shape: {design.shape}")
    return path


def save_table(design: np.ndarray, out_dir: Path) -> Path:
    """Save human-readable design table as plain text."""
    fixed_tr = FIXED_TRENTO.get(SQRTS, FIXED_TRENTO[2.76])
    lines = [
        "=" * 90,
        f"208Pb+208Pb LHS Design Table  |  sqrt(s) = {SQRTS} TeV  "
        f"|  {N_DESIGN} design points",
        "",
        "  Free parameters (LHS-sampled):",
        f"  {'idx':>4}  {'R [fm]':>8}  {'a [fm]':>8}  "
        f"{'beta3':>8}  {'beta4':>8}  {'w [fm]':>8}",
        "  " + "-" * 56,
    ]
    for i, row in enumerate(design):
        lines.append(
            f"  {i:>4}  {row[0]:>8.5f}  {row[1]:>8.5f}  "
            f"{row[2]:>8.5f}  {row[3]:>8.5f}  {row[4]:>8.5f}"
        )
    lines += [
        "",
        "  Fixed nuclear parameters:",
        f"    beta2 = {FIXED_NUCLEAR['beta2']}   gamma = {FIXED_NUCLEAR['gamma']} rad",
        f"    correlation_length = {FIXED_NUCLEAR['corr_length']} fm   "
        f"correlation_strength = {FIXED_NUCLEAR['corr_strength']}",
        "",
        f"  Fixed Trento parameters (sqrt(s) = {SQRTS} TeV):",
        f"    p = {fixed_tr['p']}   k = {fixed_tr['k']}   "
        f"d = {fixed_tr['d']} fm   norm = {fixed_tr['norm']}",
        "=" * 90,
    ]
    text = "\n".join(lines)
    path = out_dir / "lhs_design_table.txt"
    path.write_text(text)
    print(f"[INFO] Design table   ->  {path}")
    print(text)
    return path

def stage0_lhs(work_dir: Path):

    return design 
    
def load_design(work_dir: Path) -> np.ndarray:
    return np.load(work_dir / "lhs_design_matrix.npz")["design"]
    
# ── Corner plot ───────────────────────────────────────────────────────────────          

def _plot_lhs_corner(design: np.ndarray, plots_dir: Path):
    BG = 
    COLORS = 
    
    plt.rcParams.update


# =============================================================================
# ── MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Pb+Pb LHS design generator")
    p.add_argument("--n",      type=int,   default=N_DESIGN,
                   help=f"Number of design points (default {N_DESIGN})")
    p.add_argument("--iter",   type=int,   default=N_LHS_ITER,
                   help=f"Maximin optimisation iterations (default {N_LHS_ITER})")
    p.add_argument("--seed",   type=int,   default=SEED_LHS,
                   help=f"RNG seed (default {SEED_LHS})")
    p.add_argument("--sqrts",  type=float, default=SQRTS,
                   choices=[2.76, 5.02],
                   help=f"Collision energy in TeV (default {SQRTS})")
    p.add_argument("--outdir", type=str,   default=str(OUTPUT_DIR),
                   help=f"Output directory (default {OUTPUT_DIR})")
    return p.parse_args()

def main():
    # Print configuration header
    print("=" * 68)
    print(f "208Pb+208Pb Ultracentral Flow Puzzle Scan")
    print(f" sqrt(s_NN)" = {SQRTS} TeV")
    print()
    print()
    print("=" * 68)
    
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    design = None
    
    # ── Stage 0: LHS ─────────────────────────────────────────────────────────
