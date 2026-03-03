#!/usr/bin/env python3

"""
pbpb_lhs_ultracentral.py
========================
Latin-Hypercube Sampling (LHS) over the combined parameter space relevant
to the ULTRACENTRAL Pb+Pb FLOW PUZZLE at the LHC (ALICE/ATLAS/CMS data).

TARGET OBSERVABLES (the puzzle)
--------------------------------
  v2{2} ~ e2{2}  ]  In 0-1% Pb+Pb, data show v2{2} ≈ v3{2}, but almost
  v3{2} ~ e3{2}  ]  all models predict v2{2} > v3{2}.  "v2-to-v3 puzzle."

  v3{4}/v3{2}    ]  Models underpredict this ratio in 0-1%, i.e. they
                 ]  OVERESTIMATE the width of the e3 distribution.

  v4{4}^4        ]  Measured NEGATIVE in 0-1% (ALICE). Models generally
                 ]  predict positive values.

  R42 = v2{4}/v2{2}  → 0 in 0-1%.  Models predict too-large values.

  Gamma_2              → -2 in 0-1%.  Kurtosis of p(e2).

This LHS samples the nuclear-structure + Trento parameter space to enable
a Bayesian emulator study of which parameters resolve the puzzle.

PARAMETER SPACE (13 total: 8 nuclear + 5 Trento)
-------------------------------------------------
Since both nuclei in Pb+Pb are identical (208Pb, Z=82), a single set of
nuclear-structure parameters describes BOTH colliding nuclei.  This halves
the nuclear-structure dimension compared to the isobar case.

The gamma (triaxiality) angle is NOT sampled:
  - 208Pb is doubly magic and near-spherical; triaxiality has no physical
    motivation.  gamma is fixed at 0 (axially symmetric).

PRIOR RANGES AND REFERENCES
-----------------------------
WS radius  R  [fm]:  [6.50, 6.80]   central 6.62 fm
  Standard value R0 = 1.12 × A^(1/3) = 6.624 fm for A=208.
  The Giacalone–Nijs–van der Schee (2023) Bayesian analysis from LHC data
  constrains the neutron skin Δr_np = 0.217 ± 0.058 fm, implying
  R_n ~ 6.70 fm, R_p ~ 6.62 fm.  We span the combined proton/neutron range.
  Ref: de Vries et al., ADNDT 36, 495 (1987) — R_p = 6.624, a_p = 0.549 fm
       Giacalone, Nijs & van der Schee, PRL 131, 202302 (2023) [arXiv:2305.00015]

WS diffuseness  a  [fm]:  [0.44, 0.65]   central 0.549 fm
  Proton distribution well-measured by electron scattering: a_p = 0.549 fm.
  Neutron distribution from PREX-II/pion photoproduction: a_n ~ 0.55-0.62 fm.
  We span the proton-to-neutron range to cover theoretical uncertainty.
  Ref: Adhikari et al. (PREX), PRL 126, 172502 (2021) [arXiv:2102.10767]
       Tarbert et al., PRL 112, 242502 (2014) [neutron distribution]

beta_2  (quadrupole):  [-0.02, 0.10]   central 0.0
  208Pb is doubly magic (Z=82, N=126): the ground state is near-spherical.
  Low-energy data: beta_2 ≈ 0 (collective quadrupole excitation at 4.09 MeV).
  HI collisions: Giacalone et al. allow small beta_2 to scan.  Small positive
  values slightly increase e2{2}.  Upper 0.10 is a generous upper bound.
  Ref: Henderson et al., PRL 134, 062502 (2025) — "Deformation and
       Collectivity in Doubly Magic Pb208"
       Roca-Maza & Paar, Prog. Part. Nucl. Phys. 101, 96 (2018)

beta_3  (octupole):  [0.00, 0.07]   central 0.0
  KEY PARAMETER FOR THE PUZZLE.  208Pb is doubly magic => beta_3 ≈ 0 in
  the static ground state.  But:
    - DFT (minimization-after-projection): beta_3 ≈ 0.0375 in ground state
    - Carzon et al. (2020): LHC data constrain beta_3 ≲ 0.0375
    - Xu et al. (2025) "breathing octupole": effective beta_3 ~ 0.05-0.07
      could resolve the v2-to-v3 puzzle
  We sample [0, 0.07] to cover the full range of static + dynamic proposals.
  Ref: Carzon et al., PRC 102, 054905 (2020) [arXiv:2007.00780]
       Xu et al., arXiv:2504.19644 (2025) — breathing octupole puzzle paper

beta_4  (hexadecapole):  [-0.02, 0.05]   central 0.0
  Very small for a doubly-magic nucleus.  DFT gives |beta_4| ~ 0.01-0.02.
  Included because it couples to v4{4}^4 via the nonlinear channel.
  Ref: Bally et al., PRL 128, 082301 (2022) [arXiv:2108.09578]
       Moller, Nix & Kratz, ADNDT 66, 131 (1997)

correlation_length  C_l  [fm]:  [0.20, 0.80]   central 0.4 fm
  Step-function hard-core scale of short-range nucleon-nucleon correlations.
  Well-motivated by nucleon NN scattering data.
  Ref: Luzum et al., arXiv:2312.10129 (2023)

correlation_strength  C_s:  [-1.00, 0.00]   central -1.0
  -1 = full Pauli hard-core exclusion, 0 = no short-range correlation.
  Ref: Luzum et al., arXiv:2312.10129 (2023)
       Broniowski & Bozek, PRC 101, 024901 (2020)

neutron_skin  delta_r  [fm]:  [0.10, 0.35]   central 0.217 fm
  Difference in rms radii between neutron and proton distributions.
  Implemented as an additive shift to R for the neutron distribution:
    R_n = R + delta_r  (to leading order; propagated in the YAML output)
  PREX-II: Δr_np = 0.283 ± 0.071 fm
  LHC Bayesian: Δr_np = 0.217 ± 0.058 fm (Giacalone et al. 2023)
  We sample [0.10, 0.35] to span both measurements.
  Ref: Adhikari et al. (PREX), PRL 126, 172502 (2021) [arXiv:2102.10767]
       Giacalone, Nijs & van der Schee, PRL 131, 202302 (2023) [arXiv:2305.00015]

Trento parameters  (flat/uniform priors from Pb+Pb Bayesian analyses)
  p    [-1.0,  1.0]  entropy deposition: -1=WN, 0=geo-mean(KLN), +1=arithmetic
  k    [ 0.3,  2.5]  nucleon gamma-fluctuation shape
  w    [ 0.4,  1.2]  nucleon Gaussian width [fm]  (narrower upper bound for Pb)
  d    [ 0.0,  1.6]  minimum nucleon-nucleon distance [fm]
  norm [14.0, 24.0]  multiplicity normalisation (higher for Pb vs Ru/Zr)
  Ref: Bernhard, Moreland & Bass, Nature Phys. 15, 1113 (2019) [arXiv:1901.07808]
       Nijs et al./Trajectum, PRC 103, 054909 (2021) [arXiv:2010.15134]
       Giacalone, Nijs & van der Schee, PRL 131, 202302 (2023) [arXiv:2305.00015]

OUTPUTS (all inside lhs_pbpb_design/)
  lhs_pbpb_design_matrix.npz     design matrix + metadata
  yaml_configs/                  one YAML per design point, ready for
                                 isobar_sampler (single isobar: Pb) + Trento
  lhs_pbpb_corner_plot.pdf       space-filling corner plot

HOW TO USE
  pip install pyyaml
  python pbpb_lhs_ultracentral.py

  Then for each design point idx:
    a) run isobar_sampler with yaml_configs/pbpb_lhs_design_NNNN.yaml
       => generates nuclei/design_NNNN/Pb208.hdf
    b) run Trento:
         trento Pb208.hdf Pb208.hdf N_EVENTS \
           -p YAML[trento.p]  -k YAML[trento.k]  -w YAML[trento.w] \
           -d YAML[trento.d]  -n YAML[trento.norm]
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
N_DESIGN     = 150      # design points (more needed: 13-D vs 19-D isobar case)
N_LHS_ITER   = 2000     # maximin optimisation trials (more = better spacing)
SEED_LHS     = 0        # RNG seed for reproducibility
OUTPUT_DIR   = Path("lhs_pbpb_design")
PLOT_FILE    = OUTPUT_DIR / "lhs_pbpb_corner_plot.pdf"
ARCHIVE_FILE = OUTPUT_DIR / "lhs_pbpb_design_matrix.npz"

# Fixed nuclear settings (not sampled)
A_PB       = 208    # mass number
Z_PB       = 82     # atomic number
GAMMA_PB   = 0.0    # axially symmetric — physically motivated for doubly-magic Pb
N_CONFIGS  = 4      # nucleon-position configurations per design point
N_PARALLEL = -1     # -1 = use all CPUs


# =============================================================================
# PARAMETER SPACE  (name, lo, hi, latex_label, unit)
#
# Single set of Pb nuclear parameters — both beams are identical 208Pb.
# =============================================================================
PARAMS = [
    # -- 208Pb nuclear structure ----------------------------------------------
    # The puzzle is sensitive to ALL of these via the initial eccentricities.
    ("pb_R",         6.50, 6.80,  r"$R_\mathrm{Pb}$",       "fm"),
    ("pb_a",         0.44, 0.65,  r"$a_\mathrm{Pb}$",       "fm"),
    ("pb_beta2",    -0.02, 0.10,  r"$\beta_2$",              ""),
    ("pb_beta3",     0.00, 0.07,  r"$\beta_3$",              ""),  # KEY: puzzle param
    ("pb_beta4",    -0.02, 0.05,  r"$\beta_4$",              ""),
    ("pb_corr_len",  0.20, 0.80,  r"$C_\ell$",               "fm"),
    ("pb_corr_str", -1.00, 0.00,  r"$C_s$",                  ""),
    ("pb_delta_r",   0.10, 0.35,  r"$\Delta r_{np}$",        "fm"),  # neutron skin
    # -- Trento initial conditions --------------------------------------------
    ("trento_p",    -1.00, 1.00,  r"$p$",                    ""),
    ("trento_k",     0.30, 2.50,  r"$k$",                    ""),
    ("trento_w",     0.40, 1.20,  r"$w$",                    "fm"),
    ("trento_d",     0.00, 1.60,  r"$d$",                    "fm"),
    ("trento_norm", 14.00,24.00,  r"norm",                   ""),
]

PARAM_NAMES  = [p[0] for p in PARAMS]
PARAM_LO     = np.array([p[1] for p in PARAMS])
PARAM_HI     = np.array([p[2] for p in PARAMS])
PARAM_LABELS = [p[3] for p in PARAMS]
N_PARAMS     = len(PARAMS)

# Group indices for corner-plot colour coding
IDX_NUC    = list(range(0, 8))   # 208Pb nuclear (8 params)
IDX_TRENTO = list(range(8, 13))  # Trento (5 params)

# Puzzle-critical parameters (highlighted differently in the corner plot)
# beta3, beta4, and delta_r are the key nuclear-structure handles on the puzzle
IDX_PUZZLE = [3, 4, 7]  # pb_beta3, pb_beta4, pb_delta_r


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
    Generates n_iter random candidates; keeps the one maximising the minimum
    pairwise Euclidean distance between design points.
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
    """
    One design-point row -> YAML dict (isobar_sampler schema).

    Both isobars are identical 208Pb.
    The neutron skin delta_r shifts R for the neutron distribution:
      R_neutron = R + delta_r
    This is stored in the YAML as a comment; the actual per-nucleon sampling
    uses R for proton positions and R + delta_r for neutron positions
    if the isobar_sampler supports separate proton/neutron distributions.
    If not, use R_effective = R + delta_r/2 (mean) for the single WS distribution,
    and note the skin as a derived quantity.
    """
    d = dict(zip(PARAM_NAMES, row))
    f = lambda x: round(float(x), 6)

    # Effective WS radius accounting for neutron skin
    # (proton-distribution R + half neutron-skin shift as a single-fluid approx)
    R_eff = f(d["pb_R"] + 0.5 * d["pb_delta_r"])

    return {
        "design_index": int(idx),
        "nucleus": "208Pb",
        "neutron_skin_delta_r_fm": f(d["pb_delta_r"]),   # informational
        "R_proton_fm":   f(d["pb_R"]),
        "R_neutron_fm":  f(d["pb_R"] + d["pb_delta_r"]),
        "R_effective_fm": R_eff,

        "isobar_samples": {
            "description": "Options for the isobar nucleon-position samples (208Pb)",
            "number_configs": {
                "description": "Number of configurations to be sampled.",
                "value": N_CONFIGS,
            },
            "number_nucleons": {
                "description": "Mass number A of the nucleus.",
                "value": A_PB,
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
                "Nuclear properties of 208Pb. "
                "Both beams use identical nuclei. "
                "Result saved to Pb208.hdf"
            ),
            "isobar1": {
                "isobar_name": "Pb208",
                "WS_radius":            {"description": "Woods-Saxon radius parameter R (effective, mean of p/n)",
                                         "value": R_eff},
                "WS_diffusiveness":     {"description": "Woods-Saxon diffusiveness parameter a",
                                         "value": f(d["pb_a"])},
                "beta_2":               {"description": "Quadrupolar deformation beta_2.",
                                         "value": f(d["pb_beta2"])},
                "gamma":                {"description": "Quadrupolar deformation angle (radians). Fixed = 0 for doubly-magic spherical Pb.",
                                         "value": GAMMA_PB},
                "beta_3":               {"description": "Octupolar deformation beta_3. KEY parameter for v2-to-v3 puzzle.",
                                         "value": f(d["pb_beta3"])},
                "beta_4":               {"description": "Hexadecapolar deformation beta_4. Coupled to v4{4}^4 puzzle.",
                                         "value": f(d["pb_beta4"])},
                "correlation_length":   {"description": "Radius of step-function short-range correlation function C(r) in fm.",
                                         "value": f(d["pb_corr_len"])},
                "correlation_strength": {"description": "Depth of correlation function for r < correlation_length. Should be >= -1.",
                                         "value": f(d["pb_corr_str"])},
            },
        },

        # Both beams use the same Pb208.hdf
        "trento": {
            "hdf1":        f"nuclei/design_{idx:04d}/Pb208.hdf",
            "hdf2":        f"nuclei/design_{idx:04d}/Pb208.hdf",
            "p":           f(d["trento_p"]),
            "k":           f(d["trento_k"]),
            "w":           f(d["trento_w"]),
            "d":           f(d["trento_d"]),
            "norm":        f(d["trento_norm"]),
            "random_seed": 200 + int(idx),   # offset from isobar design
        },
    }


def write_yaml_configs(design):
    yaml_dir = OUTPUT_DIR / "yaml_configs"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(design):
        cfg  = _row_to_config(row, idx)
        path = yaml_dir / f"pbpb_lhs_design_{idx:04d}.yaml"
        with open(path, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)
    print(f"[INFO] {len(design)} YAML configs  ->  {yaml_dir}/")


# =============================================================================
# STEP 3 -- SAVE ARCHIVE
# =============================================================================

def save_archive(design):
    """
    Save design matrix and full metadata to a compressed .npz archive.

    Loading:
        data        = np.load("lhs_pbpb_design/lhs_pbpb_design_matrix.npz",
                              allow_pickle=True)
        design      = data["design"]          # shape (N_DESIGN, N_PARAMS)
        param_names = data["param_names"]
        lo, hi      = data["param_lo"], data["param_hi"]
        idx_puzzle  = data["idx_puzzle"]      # nuclear-structure puzzle params
    """
    np.savez_compressed(
        ARCHIVE_FILE,
        design       = design,
        param_names  = np.array(PARAM_NAMES),
        param_lo     = PARAM_LO,
        param_hi     = PARAM_HI,
        param_labels = np.array(PARAM_LABELS),
        idx_nuc      = np.array(IDX_NUC),
        idx_trento   = np.array(IDX_TRENTO),
        idx_puzzle   = np.array(IDX_PUZZLE),
    )
    print(f"[INFO] Archive  ->  {ARCHIVE_FILE}   shape: {design.shape}")


# =============================================================================
# STEP 4 -- CORNER PLOT
# =============================================================================

def make_corner_plot(design):
    """
    Full N_PARAMS x N_PARAMS corner plot.

    Diagonal  : marginal histogram + rug
    Lower tri : 2-D scatter projection
    Upper tri : blank (standard corner-plot convention)

    Colour scheme:
      Nuclear structure params  -- cyan
      Puzzle-critical params    -- magenta/orange (beta3, beta4, delta_r)
      Trento params             -- gold
    """
    BG    = "#0d0f14";  PANEL = "#13161e";  GRID = "#1e2230"
    TEXT  = "#d8dce8"
    NUC_C = "#4fc3f7"   # cyan  — nuclear structure
    PUZ_C = "#ff4081"   # magenta — puzzle-critical (beta3, beta4, delta_r)
    TR_C  = "#ffd54f"   # gold  — Trento

    def _gc(j):
        if j in IDX_PUZZLE: return PUZ_C
        if j in IDX_NUC:    return NUC_C
        return TR_C

    plt.rcParams.update({
        "figure.facecolor": BG,    "axes.facecolor":  PANEL,
        "axes.edgecolor":   GRID,  "axes.labelcolor": TEXT,
        "xtick.color":      TEXT,  "ytick.color":     TEXT,
        "text.color":       TEXT,  "grid.color":      GRID,
        "grid.linewidth":   0.35,  "font.family":     "sans-serif",
        "font.size":        7.5,   "xtick.labelsize": 6,
        "ytick.labelsize":  6,
    })

    n  = N_PARAMS
    fs = max(20, n * 1.35)
    fig, axes = plt.subplots(n, n, figsize=(fs, fs))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        r"$^{208}$Pb+$^{208}$Pb LHS — Ultracentral Flow Puzzle"
        f"  |  {N_DESIGN} pts × {N_PARAMS} params\n"
        r"  $^{208}$Pb nuclear (cyan)   |   Puzzle-critical: $\beta_3$, $\beta_4$, $\Delta r_{np}$ (magenta)"
        "   |   Trento (gold)",
        fontsize=12, color=TEXT, y=1.001, fontweight="bold",
    )

    unit = (design - PARAM_LO) / (PARAM_HI - PARAM_LO)

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.4)

            if row == col:
                c = _gc(row)
                ax.hist(unit[:, col], bins=14, color=c, alpha=0.82,
                        edgecolor=BG, linewidth=0.25)
                ax.set_xlim(0, 1); ax.set_yticks([])
                ax.plot(unit[:, col], np.zeros(N_DESIGN) - 0.5,
                        "|", color=c, alpha=0.40, markersize=3,
                        markeredgewidth=0.5)

            elif row > col:
                # Colour by the more interesting of the two axes
                c = _gc(col) if col in IDX_PUZZLE else _gc(row) if row in IDX_PUZZLE else _gc(col)
                ax.scatter(unit[:, col], unit[:, row],
                           c=c, s=7, alpha=0.50, linewidths=0)
                ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)

            else:
                ax.set_visible(False)
                continue

            ax.grid(True, linestyle="--", alpha=0.18)

            if row == n - 1:
                ax.set_xlabel(PARAM_LABELS[col], fontsize=8, color=TEXT, labelpad=2)
                lo, hi = PARAM_LO[col], PARAM_HI[col]
                ax.set_xticks([0, 0.5, 1])
                ax.set_xticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.2g}",
                                    f"{hi:.2g}"], fontsize=6)
            else:
                ax.set_xticks([]); ax.set_xticklabels([])

            if col == 0 and row > 0:
                ax.set_ylabel(PARAM_LABELS[row], fontsize=8, color=TEXT, labelpad=2)
                lo, hi = PARAM_LO[row], PARAM_HI[row]
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels([f"{lo:.2g}", f"{(lo+hi)/2:.2g}",
                                    f"{hi:.2g}"], fontsize=6)
            else:
                ax.set_yticks([]); ax.set_yticklabels([])

    # Group boundary line between nuclear and Trento blocks
    br = len(IDX_NUC)
    for c in range(br):
        if axes[br, c].get_visible():
            axes[br, c].axhline(0, color=TEXT, lw=0.5, alpha=0.25)
    for r in range(n):
        if r >= br and axes[r, br].get_visible():
            axes[r, br].axvline(0, color=TEXT, lw=0.5, alpha=0.25)

    # Puzzle-highlight borders on diagonal cells for beta3, beta4, delta_r
    for pi in IDX_PUZZLE:
        for sp in axes[pi, pi].spines.values():
            sp.set_edgecolor(PUZ_C); sp.set_linewidth(1.2)

    # Group title annotations
    axes[3, 3].set_title(r"$\beta_3$ KEY", color=PUZ_C, fontsize=8,
                         fontweight="bold", pad=3)
    axes[7, 7].set_title(r"$\Delta r_{np}$ KEY", color=PUZ_C, fontsize=8,
                         fontweight="bold", pad=3)
    axes[10, 10].set_title("Trento", color=TR_C, fontsize=8,
                            fontweight="bold", pad=3)

    fig.tight_layout(h_pad=0.05, w_pad=0.05)
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[INFO] Corner plot  ->  {PLOT_FILE}")


# =============================================================================
# STEP 5 -- PRINT DESIGN SUMMARY
# =============================================================================

def print_summary(design):
    print("\n" + "=" * 84)
    print(f"Pb+Pb LHS Design Summary  --  {N_DESIGN} pts x {N_PARAMS} parameters")
    print("=" * 84)
    print(f"{'Parameter':>22}  {'Prior lo':>9}  {'Prior hi':>9}  "
          f"{'Mean':>9}  {'Std':>9}  {'Puzzle?':>8}")
    print("-" * 84)
    puzzle_set = set(IDX_PUZZLE)
    for j, name in enumerate(PARAM_NAMES):
        col    = design[:, j]
        flag   = "  <<< " if j in puzzle_set else ""
        print(f"{name:>22}  {PARAM_LO[j]:>9.4f}  {PARAM_HI[j]:>9.4f}  "
              f"{col.mean():>9.4f}  {col.std():>9.4f}{flag}")

    unit  = (design - PARAM_LO) / (PARAM_HI - PARAM_LO)
    diff  = unit[:, None, :] - unit[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    d_min = dists.min()
    d_max = dists[dists < np.inf].max()

    print("=" * 84)
    print(f"\n  Space-filling (unit hypercube):")
    print(f"    Min pairwise distance  :  {d_min:.4f}")
    print(f"    Max pairwise distance  :  {d_max:.4f}")
    print(f"    max/min ratio          :  {d_max/d_min:.2f}  (larger = better)")
    print(f"\n  Puzzle-critical parameters:  {[PARAM_NAMES[i] for i in IDX_PUZZLE]}")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_LHS)

    print(f"[INFO] 208Pb+208Pb ultracentral flow puzzle LHS")
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
