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

# ── LHS design ────────────────────────────────────────────────────────────────
N_DESIGN = 100 # design points
N_LHS_ITER = 2000 # maximin optimisation (should we increase this?) iterations
SEED_LHS = 0 

# ── Free parameters (LHS-sampled) ──────────────────────────────────────────── 
# (name, lo, hi, latex_label)
FREE_PARAMS = [
    ("R", 6.50, 6.80, r"$R$ [fm]"),
    ("a", 0.44, 0.65, r"$a$ [fm]"),
    ("beta3", 0.00, 0.12, r"$\beta_3$"), # KEY: v2-to-v3 puzzle
    ("beta4", -0.02, 0.06, r"$\beta_4$"), # kEY: v4{4}^4 sign
    ("w", 0.40, 1.20, r"$w$ [fm]"), # Trento nucleon width
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
