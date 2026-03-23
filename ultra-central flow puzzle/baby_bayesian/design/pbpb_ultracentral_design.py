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
# (name, lo, hi, latex_label)
FREE_PARAMS = [
    ("WS_R", 6.50, 6.80, r"$R_0$ [fm]"), # Pb Woods-Saxon radius [fm]. Standard value: 6.62
    ("WS_A", 0.44, 0.65, r"$a$ [fm]"), # Pb Woods-Saxon diffuseness [fm]. Standard value: 0.546
    ("beta3", -0.12, 0.12, r"$\beta_3$"), # Octupole deformation coefficient
    ("beta4", -0.02, 0.06, r"$\beta_4$"), # Hexadecapole deformation coefficient
    ("w", 0.50, 1.50, r"$w$ [fm]"), # Trento nucleon width [fm] (JETSCAPE prior range: https://arxiv.org/pdf/2011.01430)
]

FREE_NAMES = [p[0] for p in FREE_PARAMS]
FREE_LO = np.array([p[1] for p in FREE_PARAMS])
FREE_HI = np.array([p[2] for p in FREE_PARAMS])
FREE_LABELS = [p[3] for p in FREE_PARAMS]
N_FREE = len(FREE_PARAMS)

# =============================================================================
# ── STAGE 0: MAXIMIN LATIN HYPERCUBE
# =============================================================================

def _lhs_unit(n, d, rng):
    X = np.empty((n, d)) 
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.uniform(size=n)) / n
    return X

def generate_lhs(n, d, rng, n_iter=N_LHS_ITER):
    """Maximin-optimised LHS in [0,1]^d - no external packages required."""
    best, best_dmin = None, -1.0
    for _ in range(n_iter):
        cand = _lhs_unit(n, d, rng)
        diff = cand[:, None, :] - cand[None, :, :]
        dists = np.sqrt((diff**2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)
        dmin = dists.min()
        if dmin > best_dmin:
            best_dmin = dmin
            best = cand.copy()         
    return best
    
 
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
