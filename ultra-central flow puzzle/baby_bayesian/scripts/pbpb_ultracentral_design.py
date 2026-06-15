#!/usr/bin/env python3
"""
pbpb_ultracentral_design.py
===========================
PART 1 of the 208Pb+208Pb ultracentral flow puzzle pipeline.
Generates a maximin-optimised Latin Hypercube Sample (LHS) over the N_FREE
free nuclear/Trento parameters and produces a publication-quality corner plot.

OUTPUTS (written to WORK_DIR)
  lhs_design_matrix.npz   -- design matrix + full metadata
  lhs_corner_plot.pdf/png -- N_FREE×N_FREE corner plot
  lhs_design_table.txt    -- human-readable table of all design points

USAGE
  python pbpb_ultracentral_design.py              # defaults
  python pbpb_ultracentral_design.py --n 200      # 200 design points
  python pbpb_ultracentral_design.py --sqrts 5.02 # PbPb 5.02 TeV

FREE PARAMETERS (LHS-sampled)
  WS_R   : Woods-Saxon radius       [fm]    [6.50, 6.80]
  WS_A   : Woods-Saxon diffuseness  [fm]    [0.44, 0.65]
  beta2  : quadrupole deformation           [-0.15, 0.15]
  beta3  : octupole deformation             [-0.30, 0.30]
  beta4  : hexadecapole deformation         [-0.15, 0.15]
  w      : Trento nucleon width     [fm]    [0.50, 1.50]

FIXED PARAMETERS
  Nuclear: gamma=0, C_l=0.4 fm, C_s=-1
  Trento : p=0, k=1.6, d=1.0 fm, norm=18.0 (2.76 TeV) / 20.0 (5.02 TeV)

DEPENDENCIES
  numpy, matplotlib  (no special packages needed)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# =============================================================================
# ── CONFIGURATION  (edit here or use CLI flags)
# =============================================================================

SQRTS    : float = 2.76       # collision energy [TeV]  — 2.76 or 5.02
N_DESIGN : int   = 100        # design points
N_LHS_ITER: int  = 2000       # maximin optimisation trials
SEED_LHS : int   = 0          # RNG seed

WORK_DIR = Path(f"pbpb_scan_{int(SQRTS*1000):d}GeV")

# =============================================================================
# ── PARAMETER SPACE
# =============================================================================

# (name, lo, hi, latex_label, group)
FREE_PARAMS = [
    ("WS_R",  6.50,  6.80, r"$R_0\ [\mathrm{fm}]$",   "WS"),
    ("WS_A",  0.44,  0.65, r"$a\ [\mathrm{fm}]$",      "WS"),
    ("beta2", -0.15, 0.15, r"$\beta_2$",            "PUZ"),
    ("beta3", -0.30, 0.30, r"$\beta_3$",               "PUZ"),
    ("beta4", -0.15, 0.15, r"$\beta_4$",               "PUZ"),
    ("w",     0.50,  1.50, r"$w\ [\mathrm{fm}]$",      "TR"),
]

FREE_NAMES  = [p[0] for p in FREE_PARAMS]
FREE_LO     = np.array([p[1] for p in FREE_PARAMS])
FREE_HI     = np.array([p[2] for p in FREE_PARAMS])
FREE_LABELS = [p[3] for p in FREE_PARAMS]
FREE_GROUPS = [p[4] for p in FREE_PARAMS]
N_FREE      = len(FREE_PARAMS)

FIXED_NUCLEAR = dict(gamma=0.0, corr_length=0.4, corr_strength=-1.0)
FIXED_TRENTO  = {
    2.76: dict(p=0.0, k=1.6, d=1.0, norm=18.0),
    5.02: dict(p=0.0, k=1.6, d=1.0, norm=20.0),
}

# Colour scheme per group
GROUP_COLORS = {
    "WS" : "#4fc3f7",   # cyan    — Woods-Saxon
    "PUZ": "#ff4081",   # magenta — puzzle-critical
    "TR" : "#ffd54f",   # gold    — Trento
}

# =============================================================================
# ── STAGE 0: MAXIMIN LATIN HYPERCUBE
# =============================================================================

def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Single random LHS realisation in [0, 1]^d."""
    X = np.empty((n, d))
    for j in range(d):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.uniform(size=n)) / n
    return X


def generate_lhs(n: int, d: int, rng: np.random.Generator,
                 n_iter: int = N_LHS_ITER) -> np.ndarray:
    """
    Maximin-optimised LHS in [0, 1]^d.

    Generates ``n_iter`` random LHS candidates and returns the one that
    maximises the minimum pairwise Euclidean distance between points.
    No external packages required.
    """
    best: Optional[np.ndarray] = None
    best_dmin = -1.0

    for _ in range(n_iter):
        cand = _lhs_unit(n, d, rng)
        diff  = cand[:, None, :] - cand[None, :, :]   # (n, n, d)
        dists = np.sqrt((diff ** 2).sum(axis=-1))      # (n, n)
        np.fill_diagonal(dists, np.inf)
        dmin = dists.min()
        if dmin > best_dmin:
            best_dmin = dmin
            best = cand.copy()

    return best   # type: ignore[return-value]


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
    print(f"  208Pb+208Pb LHS Design  |  √s = {SQRTS} TeV")
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

    diff = unit_lhs[:, None, :] - unit_lhs[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    dmin = dist.min()
    dmax = dist[dist < np.inf].max()

    print()
    print("  Space-filling (unit hypercube):")
    print(f"    Min pairwise distance : {dmin:.4f}   (larger = better)")
    print(f"    Max pairwise distance : {dmax:.4f}")
    print(f"    Max / Min ratio       : {dmax / dmin:.2f}")
    print("=" * 72)
    print()


def save_archive(design: np.ndarray, unit_lhs: np.ndarray,
                 out_dir: Path) -> Path:
    """Save design matrix and metadata to a compressed .npz archive."""
    path = out_dir / "lhs_design_matrix.npz"
    np.savez_compressed(
        path,
        design       = design,
        unit_lhs     = unit_lhs,
        param_names  = np.array(FREE_NAMES),
        param_lo     = FREE_LO,
        param_hi     = FREE_HI,
        param_labels = np.array(FREE_LABELS),
        param_groups = np.array(FREE_GROUPS),
        sqrts_tev    = np.float64(SQRTS),
    )
    print(f"[INFO] Design matrix  →  {path}   shape: {design.shape}")
    return path


def save_table(design: np.ndarray, out_dir: Path) -> Path:
    """Save human-readable design table as plain text."""
    fixed_tr = FIXED_TRENTO.get(SQRTS, FIXED_TRENTO[2.76])
    lines = [
        "=" * 90,
        f"208Pb+208Pb LHS Design Table  |  √s = {SQRTS} TeV  "
        f"|  {N_DESIGN} design points",
        "",
        "  Free parameters (LHS-sampled):",
        f"  {'idx':>4}  {'R [fm]':>8}  {'a [fm]':>8}  "
        f"{'beta2':>8}  {'beta3':>8}  {'beta4':>8}  {'w [fm]':>8}",
        "  " + "-" * 56,
    ]
    for i, row in enumerate(design):
        lines.append(
            f"  {i:>4}  {row[0]:>8.5f}  {row[1]:>8.5f}  "
            f"{row[2]:>8.5f}  {row[3]:>8.5f}  "
            f"{row[4]:>8.5f}  {row[5]:>8.5f}"
        )
    lines += [
        "",
        "  Fixed nuclear parameters:",
        f"    gamma = {FIXED_NUCLEAR['gamma']} rad",
        f"    corr_length = {FIXED_NUCLEAR['corr_length']} fm   "
        f"corr_strength = {FIXED_NUCLEAR['corr_strength']}",
        "",
        f"  Fixed Trento parameters (√s = {SQRTS} TeV):",
        f"    p = {fixed_tr['p']}   k = {fixed_tr['k']}   "
        f"d = {fixed_tr['d']} fm   norm = {fixed_tr['norm']}",
        "=" * 90,
    ]
    text = "\n".join(lines)
    path = out_dir / "lhs_design_table.txt"
    path.write_text(text)
    print(f"[INFO] Design table   →  {path}")
    print(text)
    return path

# =============================================================================
# ── CORNER PLOT
# =============================================================================

def _plot_lhs_corner(design: np.ndarray, plots_dir: Path) -> Path:
    """
    Produce a publication-quality N_FREE × N_FREE corner plot.

    Diagonal panels   : 1-D histogram of each parameter, coloured by group.
    Off-diagonal panels: 2-D scatter of every parameter pair, coloured by group
                         of the *column* parameter.
    """
    # ── Aesthetic config ──────────────────────────────────────────────────────
    BG = "#0d1117"          # near-black background
    FG = "#e6edf3"          # near-white foreground text / ticks
    GRID_C = "#21262d"      # subtle grid lines

    COLORS = {grp: c for grp, c in GROUP_COLORS.items()}
    # Point colour for each parameter (indexed by column)
    pt_colors = [COLORS[FREE_GROUPS[j]] for j in range(N_FREE)]

    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   FG,
        "xtick.color":       FG,
        "ytick.color":       FG,
        "text.color":        FG,
        "grid.color":        GRID_C,
        "grid.linewidth":    0.5,
        "font.family":       "DejaVu Sans",
        "font.size":         9,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    })

    fig_size = 2.2 * N_FREE
    fig, axes = plt.subplots(N_FREE, N_FREE,
                             figsize=(fig_size, fig_size),
                             facecolor=BG)
    fig.subplots_adjust(hspace=0.08, wspace=0.08,
                        left=0.08, right=0.97,
                        top=0.96, bottom=0.06)

    for row in range(N_FREE):
        for col in range(N_FREE):
            ax = axes[row, col]
            ax.set_facecolor(BG)

            # ── above diagonal: hide ─────────────────────────────────────────
            if col > row:
                ax.axis("off")
                continue

            xdata = design[:, col]
            color = pt_colors[col]

            # ── diagonal: 1-D histogram ──────────────────────────────────────
            if col == row:
                n_bins = max(10, N_DESIGN // 8)
                ax.hist(xdata, bins=n_bins,
                        color=color, alpha=0.85,
                        edgecolor=BG, linewidth=0.4)
                ax.set_xlim(FREE_LO[col], FREE_HI[col])
                ax.grid(True, axis="y", alpha=0.4)
                # Vertical lines at lo / hi
                for v in (FREE_LO[col], FREE_HI[col]):
                    ax.axvline(v, color=FG, lw=0.6, alpha=0.4, ls="--")

            # ── below diagonal: 2-D scatter ──────────────────────────────────
            else:
                ydata = design[:, row]
                ax.scatter(xdata, ydata,
                           c=color, s=14, alpha=0.72,
                           linewidths=0, rasterized=True)
                ax.set_xlim(FREE_LO[col], FREE_HI[col])
                ax.set_ylim(FREE_LO[row], FREE_HI[row])
                ax.grid(True, alpha=0.35)

            # ── axis labels (outer edges only) ────────────────────────────────
            if row == N_FREE - 1:
                ax.set_xlabel(FREE_LABELS[col], fontsize=10, labelpad=4)
            else:
                ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(FREE_LABELS[row], fontsize=10, labelpad=4)
            else:
                ax.set_yticklabels([])

            # Reduce tick clutter
            ax.xaxis.set_major_locator(mticker.MaxNLocator(3, prune="both"))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(3, prune="both"))

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"208Pb+208Pb LHS Design  |  "
        f"$\\sqrt{{s_{{\\rm NN}}}} = {SQRTS}$ TeV  |  "
        f"{N_DESIGN} points × {N_FREE} parameters",
        fontsize=11, color=FG, y=0.99,
    )

    # ── Group legend ──────────────────────────────────────────────────────────
    group_labels = {
        "WS":  "Woods-Saxon (nuclear)",
        "PUZ": "Puzzle-critical (deformation)",
        "TR":  "Trento (initial state)",
    }
    handles = [
        mpatches.Patch(facecolor=COLORS[g], label=group_labels[g])
        for g in ("WS", "PUZ", "TR")
    ]
    fig.legend(handles=handles,
               loc="upper right",
               fontsize=8,
               framealpha=0.25,
               edgecolor="#30363d",
               facecolor="#161b22",
               labelcolor=FG,
               bbox_to_anchor=(0.97, 0.97))

    # ── Save ──────────────────────────────────────────────────────────────────
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = plots_dir / "lhs_corner_plot"

    fig.savefig(f"{stem}.pdf", dpi=150, bbox_inches="tight", facecolor=BG)
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)

    print(f"[INFO] Corner plot    →  {stem}.pdf / .png")
    return Path(f"{stem}.pdf")

# =============================================================================
# ── STAGE 0 ENTRY POINT  (used by main and by the notebook)
# =============================================================================

def stage0_lhs(
    n_design:  int   = N_DESIGN,
    n_iter:    int   = N_LHS_ITER,
    seed:      int   = SEED_LHS,
    sqrts:     float = SQRTS,
    work_dir:  Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    verbose:   bool  = True,
) -> np.ndarray:
    """
    Run the maximin-LHS design stage.

    Parameters
    ----------
    n_design  : number of design points
    n_iter    : maximin optimisation trials (more → better space-filling)
    seed      : RNG seed for reproducibility
    sqrts     : collision energy in TeV (affects fixed Trento norm)
    work_dir  : directory for .npz and .txt outputs  (None → skip saving)
    plots_dir : directory for corner plot             (None → skip plotting)
    verbose   : print progress and summary

    Returns
    -------
    design : ndarray, shape (n_design, N_FREE)
        Physical parameter values for each design point.
    """
    global N_DESIGN, N_LHS_ITER, SEED_LHS, SQRTS, WORK_DIR
    N_DESIGN   = n_design
    N_LHS_ITER = n_iter
    SEED_LHS   = seed
    SQRTS      = sqrts

    if verbose:
        print("=" * 68)
        print(f"  208Pb+208Pb Ultracentral Flow Puzzle — LHS Design")
        print(f"  √s_NN = {SQRTS} TeV  |  {N_DESIGN} points  |  seed = {SEED_LHS}")
        print("=" * 68)

    t0 = time.perf_counter()
    rng      = np.random.default_rng(seed)
    unit_lhs = generate_lhs(n_design, N_FREE, rng, n_iter)
    design   = scale_lhs(unit_lhs)
    elapsed  = time.perf_counter() - t0

    if verbose:
        print(f"[INFO] LHS generated in {elapsed:.1f} s")
        print_summary(design, unit_lhs)

    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        save_archive(design, unit_lhs, work_dir)
        save_table(design, work_dir)

    if plots_dir is not None:
        _plot_lhs_corner(design, Path(plots_dir))

    return design


def load_design(work_dir: Path) -> np.ndarray:
    """Load a previously saved design matrix from work_dir."""
    path = Path(work_dir) / "lhs_design_matrix.npz"
    data = np.load(path, allow_pickle=True)
    print(f"[INFO] Loaded design from {path}  shape: {data['design'].shape}")
    return data["design"]


# =============================================================================
# ── CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pb+Pb LHS design generator — Part 1 of the ultracentral pipeline"
    )
    p.add_argument("--n",      type=int,   default=N_DESIGN,
                   help=f"Design points (default {N_DESIGN})")
    p.add_argument("--iter",   type=int,   default=N_LHS_ITER,
                   help=f"Maximin iterations (default {N_LHS_ITER})")
    p.add_argument("--seed",   type=int,   default=SEED_LHS,
                   help=f"RNG seed (default {SEED_LHS})")
    p.add_argument("--sqrts",  type=float, default=SQRTS,
                   choices=[2.76, 5.02],
                   help=f"Collision energy [TeV] (default {SQRTS})")
    p.add_argument("--outdir", type=str,   default=str(WORK_DIR),
                   help=f"Output directory (default {WORK_DIR})")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip corner plot generation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    work_dir  = Path(args.outdir)
    plots_dir = None if args.no_plot else work_dir / "plots"

    stage0_lhs(
        n_design  = args.n,
        n_iter    = args.iter,
        seed      = args.seed,
        sqrts     = args.sqrts,
        work_dir  = work_dir,
        plots_dir = plots_dir,
        verbose   = True,
    )


if __name__ == "__main__":
    main()
