#!/usr/bin/env python3
"""
pbpb_ultracentral_design.py
=========================
Latin Hypercube Design for the 208Pb+208Pb ultracentral flow puzzle study.

the maximin criterion
Nd: number of design points 
d: dimension of the parameter space
Parameter priors

"""

import numpy as np
from pathlib import Path

# ── Collision energy ──────────────────────────────────────────────────────────
SQRTS = 2.76 # TeV | choose 2.76 or 5.02

# ── Working directory ─────────────────────────────────────────────────────────
WORK_DIR = Path(f"pbpb_scan_{int(SQRTS*100):d}GeV")

# ── LHS design ────────────────────────────────────────────────────────────────
N_DESIGN = 100 # design points
N_LHS_ITER = 2000 # maximin optimisation (should we increase this?) iterations
SEED_LHS = 0 

# ── Free parameters (LHS-sampled) ──────────────────────────────────────────── 
# (name, lo, hi, latex_label)
FREE_PARAMS = [
    ("R", 6.50, 6.80, r"$R$ [fm]"), # find better references
    ("a", 0.44, 0.65, r"$a$ [fm]"), # find better references 
    ("beta3", 0.00, 0.12, r"$\beta_3$"), # KEY: v2-to-v3 puzzle
    ("beta4", -0.02, 0.06, r"$\beta_4$"), # kEY: v4{4}^4 sign 
    ("w", 0.50, 1.50, r"$w$ [fm]"), # Trento nucleon width (JETSCAPE prior range: https://arxiv.org/pdf/2011.01430)
]
FREE_NAMES = [p[0] for p in FREE_PARAMS]
FREE_LO = np.array([p[1] for in FREE_PARAMS])
FREE_HI = np.array([p[2] for in FREE_PARAMS])
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
