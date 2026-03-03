#!/usr/bin/env python3
"""
combined_lhs_scan.py
====================
Latin-Hypercube Sampling (LHS) over the combined parameter space of:

  - Nuclear structure parameters for BOTH isobars (Woods-Saxon + deformations
    + short-range correlations), matching the isobar_sampler YAML schema exactly
  - Trento initial-condition parameters (p, k, w, d, norm)

OUTPUTS (written inside lhs_design/)
  lhs_design_matrix.npz      compressed design matrix + metadata
  yaml_configs/              one YAML file per design point, ready for the
                             isobar-sampler + Trento pipeline
  lhs_corner_plot.pdf        space-filling corner plot (see below)

HOW TO USE
  1. pip install pyyaml   (only external dependency)
  2. python combined_lhs_scan.py
  3. Inspect lhs_corner_plot.pdf for LHS quality
  4. For each design point idx:
       a) run isobar_sampler with  yaml_configs/isobar_trento_design_NNNN.yaml
       b) run Trento with the generated WS1.hdf / WS2.hdf and the
          p, k, w, d, norm values from the same YAML

PRIOR RANGES AND REFERENCES
────────────────────────────────────────────────────────────────────────────
Woods-Saxon radius  R  [fm]
  Ru: [4.80, 5.40]  central 5.09 fm
  Zr: [4.70, 5.30]  central 5.02 fm
  ±0.3 fm spans all DFT and electron-scattering parametrisations.
  Ref: Zhang & Jia, PRL 128, 022301 (2022)  [arXiv:2109.01631]
       Xu et al., PRC 105, 014905 (2022)    [arXiv:2111.14812]

Woods-Saxon diffuseness  a  [fm]
  Ru & Zr: [0.40, 0.65]  (Ru: 0.46, Zr: 0.52 central)
  Lower bound from quasi-elastic back-scattering data;
  upper bound from fusion-anomaly studies.
  Ref: Jia, PRC 105, 014905 (2022);  Hinde et al., PRC (2002)

beta_2  (quadrupole deformation)
  Ru: [-0.05, 0.35]  central 0.16  (prolate)
  Zr: [-0.05, 0.25]  central 0.06  (near-spherical to mildly deformed)
  Negative values allow oblate shapes.
  Ref: Zhang & Jia, PRL 128, 022301 (2022)
       Giacalone, Jia & Soma, PRC 104, L041903 (2021) [arXiv:2102.08158]

gamma  (triaxiality, radians) — FIXED, not sampled
  Ru: pi/6 = 0.5236  (maximal triaxiality, isobar_sampler default)
  Zr: 0.0            (axially symmetric)

beta_3  (octupole deformation)
  Ru: [0.00, 0.20]  central 0.0   (Ru near-spherical in octupole)
  Zr: [0.00, 0.35]  central 0.20  (Zr has significant octupole collectivity)
  Ref: Mach et al., NPA 523, 197 (1991)
       Burrello et al., PRC 107, 044315 (2023)

beta_4  (hexadecapole deformation)
  Ru: [-0.05, 0.15]  central 0.0
  Zr: [-0.05, 0.15]  central 0.10
  DFT calculations give |beta_4| ~ 0.04-0.10 for A~96 nuclei.
  Ref: Moller, Nix & Kratz, ADNDT 66, 131 (1997)
       Bally et al., PRL 128, 082301 (2022)  [arXiv:2108.09578]

correlation_length  C_l  [fm]
  Both: [0.20, 0.80]  central 0.4 fm  (hard-core scale from NN scattering)
  Ref: Luzum et al., arXiv:2312.10129 (2023)

correlation_strength  C_s
  Both: [-1.00, 0.00]
  -1 = full Pauli exclusion (hard-core),  0 = no short-range correlation
  Ref: Luzum et al., arXiv:2312.10129 (2023)
       Broniowski & Bozek, PRC 101, 024901 (2020)

Trento parameters  (all flat/uniform priors)
  p    [-1.0,  1.0]  entropy deposition: -1=WN, 0=geo-mean(KLN), +1=arithmetic
  k    [ 0.3,  2.5]  nucleon gamma-fluctuation shape parameter
  w    [ 0.4,  1.6]  nucleon Gaussian width [fm]
  d    [ 0.0,  2.0]  minimum nucleon-nucleon distance [fm]
  norm [ 8.0, 22.0]  multiplicity normalisation
  Ref: Bernhard, Moreland & Bass, Nature Phys. 15, 1113 (2019)
       Nijs et al./Trajectum, PRC 103, 054909 (2021) [arXiv:2010.15134]
────────────────────────────────────────────────────────────────────────────
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # safe for headless / Colab / HPC
import matplotlib.pyplot as plt
import yaml                    # pip install pyyaml


# =============================================================================
# CONFIGURATION  <-- edit here
# =============================================================================
N_DESIGN     = 100      # number of LHS design points
N_LHS_ITER   = 2000     # maximin optimisation trials (more = better spacing)
SEED_LHS     = 0        # RNG seed for reproducibility
OUTPUT_DIR   = Path("lhs_design")
PLOT_FILE    = OUTPUT_DIR / "lhs_corner_plot.pdf"
ARCHIVE_FILE = OUTPUT_DIR / "lhs_design_matrix.npz"

# Fixed per-isobar triaxiality angles (NOT sampled -- kept at defaults)
GAMMA_RU = 0.5236   # pi/6 = maximal triaxiality (isobar_sampler default)
GAMMA_ZR = 0.0      # axially symmetric

# Fixed isobar_samples global settings (not model parameters)
N_CONFIGS  = 4      # configurations per design point
N_NUCLEONS = 96     # mass number A
N_PARALLEL = -1     # -1 = use all CPUs (joblib.Parallel)


# =============================================================================
# PARAMETER SPACE  (name, lo, hi, latex_label, unit)
# =============================================================================
PARAMS = [
    # -- Isobar 1 : Ru  (96-Ru, Z=44) ----------------------------------------
    ("ru_R",         4.80, 5.40,  r"$R_\mathrm{Ru}$",        "fm"),
    ("ru_a",         0.40, 0.65,  r"$a_\mathrm{Ru}$",        "fm"),
    ("ru_beta2",    -0.05, 0.35,  r"$\beta_2^\mathrm{Ru}$",  ""),
    ("ru_beta3",     0.00, 0.20,  r"$\beta_3^\mathrm{Ru}$",  ""),
    ("ru_beta4",    -0.05, 0.15,  r"$\beta_4^\mathrm{Ru}$",  ""),
    ("ru_corr_len",  0.20, 0.80,  r"$C_\ell^\mathrm{Ru}$",   "fm"),
    ("ru_corr_str", -1.00, 0.00,  r"$C_s^\mathrm{Ru}$",      ""),
    # -- Isobar 2 : Zr  (96-Zr, Z=40) ----------------------------------------
    ("zr_R",         4.70, 5.30,  r"$R_\mathrm{Zr}$",        "fm"),
    ("zr_a",         0.40, 0.65,  r"$a_\mathrm{Zr}$",        "fm"),
    ("zr_beta2",    -0.05, 0.25,  r"$\beta_2^\mathrm{Zr}$",  ""),
    ("zr_beta3",     0.00, 0.35,  r"$\beta_3^\mathrm{Zr}$",  ""),
    ("zr_beta4",    -0.05, 0.15,  r"$\beta_4^\mathrm{Zr}$",  ""),
    ("zr_corr_len",  0.20, 0.80,  r"$C_\ell^\mathrm{Zr}$",   "fm"),
    ("zr_corr_str", -1.00, 0.00,  r"$C_s^\mathrm{Zr}$",      ""),
    # -- Trento ----------------------------------------------------------------
    ("trento_p",    -1.00, 1.00,  r"$p$",                    ""),
    ("trento_k",     0.30, 2.50,  r"$k$",                    ""),
    ("trento_w",     0.40, 1.60,  r"$w$",                    "fm"),
    ("trento_d",     0.00, 2.00,  r"$d$",                    "fm"),
    ("trento_norm",  8.00,22.00,  r"norm",                   ""),
]

PARAM_NAMES  = [p[0] for p in PARAMS]
PARAM_LO     = np.array([p[1] for p in PARAMS])
PARAM_HI     = np.array([p[2] for p in PARAMS])
PARAM_LABELS = [p[3] for p in PARAMS]
N_PARAMS     = len(PARAMS)

# Group indices -- used for corner-plot colour coding
IDX_RU     = list(range(0,  7))
IDX_ZR     = list(range(7,  14))
IDX_TRENTO = list(range(14, 19))


# =============================================================================
# STEP 1 -- MAXIMIN LATIN HYPERCUBE
# =============================================================================

def _make_lhs_unit(n, d, rng):
    """Single random LHS realisation in [0, 1]^d."""
    X = np.empty((n, d))
    for j in range(d):
        perm    = rng.permutation(n)
        X[:, j] = (perm + rng.uniform(size=n)) / n
    return X


def generate_lhs(n, d, rng, n_iter=N_LHS_ITER):
    """
    Maximin-optimised LHS.

    Generates n_iter random LHS candidates and returns the one that maximises
    the minimum pairwise Euclidean distance (maximin criterion).
    No external packages required.
    """
    best, best_dmin = None, -1.0
    for _ in range(n_iter):
        cand  = _make_lhs_unit(n, d, rng)
        diff  = cand[:, None, :] - cand[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        np.fill_diagonal(dists, np.inf)
        dmin = dists.min()
        if dmin > best_dmin:
            best_dmin = dmin
            best      = cand.copy()
    return best


def scale_to_physical(unit_lhs):
    """Affine map from [0, 1]^d to physical parameter ranges."""
    return PARAM_LO + unit_lhs * (PARAM_HI - PARAM_LO)


# =============================================================================
# STEP 2 -- YAML CONFIG FILES
# =============================================================================

def _row_to_config(row, idx):
    """One design-point row -> YAML-serialisable dict (isobar_sampler schema)."""
    d = dict(zip(PARAM_NAMES, row))
    f = lambda x: round(float(x), 6)

    return {
        "design_index": int(idx),

        "isobar_samples": {
            "description": "Options for the isobar nucleon-position samples",
            "number_configs": {
                "description": "Number of configurations to be sampled.",
                "value": N_CONFIGS,
            },
            "number_nucleons": {
                "description": "Mass number A of the nuclei.",
                "value": N_NUCLEONS,
            },
            "seeds_file": {
                "description": "Input file with list of seeds for nucleon positions.",
                "filename": "nucleon-seeds.hdf",
            },
            "output_path": {
                "description": "Output directory where to save",
                "dirname": f"nuclei/design_{idx:04d}",
            },
            "number_of_parallel_processes": {
                "description": (
                    "Number of processes to compute in parallel. "
                    "A value of -1 automatically selects the number of CPUs present. "
                    "See joblib.Parallel class."
                ),
                "value": N_PARALLEL,
            },
        },

        "isobar_properties": {
            "description": (
                "Nuclear properties of isobars to be sampled. "
                "Results are saved to isobar_name.hdf"
            ),
            "isobar1": {
                "isobar_name": "WS1",
                "WS_radius":            {"description": "Woods-Saxon radius parameter R",
                                         "value": f(d["ru_R"])},
                "WS_diffusiveness":     {"description": "Woods-Saxon diffusiveness parameter a",
                                         "value": f(d["ru_a"])},
                "beta_2":               {"description": "Quadrupolar deformation beta_2 of isobar.",
                                         "value": f(d["ru_beta2"])},
                "gamma":                {"description": "Quadrupolar deformation angle (in radians).",
                                         "value": GAMMA_RU},
                "beta_3":               {"description": "Octupolar deformation beta_3 of isobar.",
                                         "value": f(d["ru_beta3"])},
                "beta_4":               {"description": "Hexadecapolar deformation beta_4 of isobar.",
                                         "value": f(d["ru_beta4"])},
                "correlation_length":   {"description": "Radius of step-function correlation function C(r) in fm.",
                                         "value": f(d["ru_corr_len"])},
                "correlation_strength": {"description": "Depth of correlation function for r < correlation_length. Should be >= -1.",
                                         "value": f(d["ru_corr_str"])},
            },
            "isobar2": {
                "isobar_name": "WS2",
                "WS_radius":            {"description": "Woods-Saxon radius parameter R",
                                         "value": f(d["zr_R"])},
                "WS_diffusiveness":     {"description": "Woods-Saxon diffusiveness parameter a",
                                         "value": f(d["zr_a"])},
                "beta_2":               {"description": "Quadrupolar deformation beta_2 of isobar.",
                                         "value": f(d["zr_beta2"])},
                "gamma":                {"description": "Quadrupolar deformation angle (in radians).",
                                         "value": GAMMA_ZR},
                "beta_3":               {"description": "Octupolar deformation beta_3 of isobar.",
                                         "value": f(d["zr_beta3"])},
                "beta_4":               {"description": "Hexadecapolar deformation beta_4 of isobar.",
                                         "value": f(d["zr_beta4"])},
                "correlation_length":   {"description": "Radius of step-function correlation function C(r) in fm.",
                                         "value": f(d["zr_corr_len"])},
                "correlation_strength": {"description": "Depth of correlation function for r < correlation_length. Should be >= -1.",
                                         "value": f(d["zr_corr_str"])},
            },
        },

        # hdf paths point to the nuclei generated for this design point.
        # random_seed is offset by design index for independent Trento runs.
        "trento": {
            "hdf1":        f"nuclei/design_{idx:04d}/WS1.hdf",
            "hdf2":        f"nuclei/design_{idx:04d}/WS2.hdf",
            "p":           f(d["trento_p"]),
            "k":           f(d["trento_k"]),
            "w":           f(d["trento_w"]),
            "d":           f(d["trento_d"]),
            "norm":        f(d["trento_norm"]),
            "random_seed": 100 + int(idx),
        },
    }


def write_yaml_configs(design):
    yaml_dir = OUTPUT_DIR / "yaml_configs"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(design):
        cfg  = _row_to_config(row, idx)
        path = yaml_dir / f"isobar_trento_design_{idx:04d}.yaml"
        with open(path, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
    print(f"[INFO] {len(design)} YAML configs  ->  {yaml_dir}/")


# =============================================================================
# STEP 3 -- SAVE ARCHIVE
# =============================================================================

def save_archive(design):
    """
    Save design matrix and metadata to a compressed .npz archive.

    Loading example:
        data   = np.load("lhs_design/lhs_design_matrix.npz", allow_pickle=True)
        design = data["design"]       # shape (N_DESIGN, N_PARAMS)
        names  = data["param_names"]  # string array, length N_PARAMS
        lo, hi = data["param_lo"], data["param_hi"]
    """
    np.savez_compressed(
        ARCHIVE_FILE,
        design       = design,
        param_names  = np.array(PARAM_NAMES),
        param_lo     = PARAM_LO,
        param_hi     = PARAM_HI,
        param_labels = np.array(PARAM_LABELS),
        idx_ru       = np.array(IDX_RU),
        idx_zr       = np.array(IDX_ZR),
        idx_trento   = np.array(IDX_TRENTO),
    )
    print(f"[INFO] Archive  ->  {ARCHIVE_FILE}   shape: {design.shape}")


# =============================================================================
# STEP 4 -- CORNER PLOT
# =============================================================================

def make_corner_plot(design):
    """
    Full N_PARAMS x N_PARAMS corner plot.

    Diagonal  : marginal histogram + rug plot
    Lower tri : 2-D scatter projection (coloured by parameter group)
    Upper tri : blank (standard corner-plot convention)

    Colours:  Ru nuclear (cyan) | Zr nuclear (coral) | Trento (gold)
    Tick labels show physical values (lo, mid, hi) of each prior range.
    """
    BG    = "#0d0f14";  PANEL = "#13161e";  GRID = "#1e2230"
    TEXT  = "#d8dce8"
    RU_C  = "#4fc3f7"   # cyan
    ZR_C  = "#ef5350"   # coral
    TR_C  = "#ffd54f"   # gold

    def _gc(j):
        if j in IDX_RU:     return RU_C
        if j in IDX_ZR:     return ZR_C
        return TR_C

    plt.rcParams.update({
        "figure.facecolor": BG,    "axes.facecolor":  PANEL,
        "axes.edgecolor":   GRID,  "axes.labelcolor": TEXT,
        "xtick.color":      TEXT,  "ytick.color":     TEXT,
        "text.color":       TEXT,  "grid.color":      GRID,
        "grid.linewidth":   0.35,  "font.family":     "sans-serif",
        "font.size":        7,     "xtick.labelsize": 5.5,
        "ytick.labelsize":  5.5,
    })

    n  = N_PARAMS
    fs = max(24, n * 1.3)
    fig, axes = plt.subplots(n, n, figsize=(fs, fs))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Combined Nuclear Structure + Trento LHS  "
        f"|  {N_DESIGN} design pts x {N_PARAMS} parameters\n"
        "  Ru nuclear (cyan)   |   Zr nuclear (coral)   |   Trento (gold)",
        fontsize=13, color=TEXT, y=1.001, fontweight="bold",
    )

    # Normalise to unit cube for uniform axis scaling across all parameters
    unit = (design - PARAM_LO) / (PARAM_HI - PARAM_LO)

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.4)

            if row == col:
                c = _gc(row)
                ax.hist(unit[:, col], bins=12, color=c, alpha=0.80,
                        edgecolor=BG, linewidth=0.25)
                ax.set_xlim(0, 1); ax.set_yticks([])
                # Rug plot
                ax.plot(unit[:, col], np.zeros(N_DESIGN) - 0.5,
                        "|", color=c, alpha=0.40, markersize=3,
                        markeredgewidth=0.5)

            elif row > col:
                c = _gc(col)
                ax.scatter(unit[:, col], unit[:, row],
                           c=c, s=7, alpha=0.55, linewidths=0)
                ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)

            else:
                ax.set_visible(False)
                continue

            ax.grid(True, linestyle="--", alpha=0.18)

            # Axis tick labels: outer edges only
            if row == n - 1:
                ax.set_xlabel(PARAM_LABELS[col], fontsize=7.5,
                              color=TEXT, labelpad=2)
                lo, hi = PARAM_LO[col], PARAM_HI[col]
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels(
                    [f"{lo:.2g}", f"{(lo+hi)/2:.2g}", f"{hi:.2g}"],
                    fontsize=5.5)
            else:
                ax.set_xticks([]); ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(PARAM_LABELS[row], fontsize=7.5,
                              color=TEXT, labelpad=2)
                lo, hi = PARAM_LO[row], PARAM_HI[row]
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels(
                    [f"{lo:.2g}", f"{(lo+hi)/2:.2g}", f"{hi:.2g}"],
                    fontsize=5.5)
            else:
                ax.set_yticks([]); ax.set_yticklabels([])

    # Faint group-boundary lines
    for br in [len(IDX_RU), len(IDX_RU) + len(IDX_ZR)]:
        if 0 < br < n:
            for c in range(br):
                if axes[br, c].get_visible():
                    axes[br, c].axhline(0, color=TEXT, lw=0.5, alpha=0.25)
            for r in range(n):
                if r >= br and axes[r, br].get_visible():
                    axes[r, br].axvline(0, color=TEXT, lw=0.5, alpha=0.25)

    # Group labels on selected diagonal cells
    for di, c, lbl in [(3, RU_C, "Ru nuclear"),
                       (10, ZR_C, "Zr nuclear"),
                       (16, TR_C, "Trento")]:
        axes[di, di].set_title(lbl, color=c, fontsize=9,
                               fontweight="bold", pad=3)

    fig.tight_layout(h_pad=0.05, w_pad=0.05)
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[INFO] Corner plot  ->  {PLOT_FILE}")


# =============================================================================
# STEP 5 -- PRINT DESIGN SUMMARY
# =============================================================================

def print_summary(design):
    print("\n" + "=" * 82)
    print(f"LHS Design Summary  --  {N_DESIGN} pts x {N_PARAMS} parameters")
    print("=" * 82)
    print(f"{'Parameter':>22}  {'Prior lo':>9}  {'Prior hi':>9}  "
          f"{'Sample mean':>12}  {'Sample std':>11}")
    print("-" * 82)
    for j, name in enumerate(PARAM_NAMES):
        col = design[:, j]
        print(f"{name:>22}  {PARAM_LO[j]:>9.4f}  {PARAM_HI[j]:>9.4f}  "
              f"{col.mean():>12.4f}  {col.std():>11.4f}")

    unit  = (design - PARAM_LO) / (PARAM_HI - PARAM_LO)
    diff  = unit[:, None, :] - unit[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    d_min = dists.min()
    d_max = dists[dists < np.inf].max()

    print("=" * 82)
    print(f"\n  Space-filling quality (unit hypercube):")
    print(f"    Min pairwise Euclidean distance  :  {d_min:.4f}")
    print(f"    Max pairwise Euclidean distance  :  {d_max:.4f}")
    print(f"    max/min ratio                    :  {d_max/d_min:.2f}")
    print(f"    (Larger ratio = better maximin space-filling)\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_LHS)

    print(f"[INFO] Generating {N_DESIGN}-pt maximin LHS over "
          f"{N_PARAMS} parameters  (n_iter={N_LHS_ITER}) ...")
    t0       = time.perf_counter()
    unit_lhs = generate_lhs(N_DESIGN, N_PARAMS, rng)
    design   = scale_to_physical(unit_lhs)
    print(f"[INFO] LHS done in {time.perf_counter()-t0:.1f} s")

    print_summary(design)
    save_archive(design)
    write_yaml_configs(design)
    print("[INFO] Building corner plot ...")
    make_corner_plot(design)
    print("[INFO] All done.")


if __name__ == "__main__":
    main()
