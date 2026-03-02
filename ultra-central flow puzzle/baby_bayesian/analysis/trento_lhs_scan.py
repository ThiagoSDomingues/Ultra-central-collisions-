#!/usr/bin/env python3
"""
trento_lhs_scan.py
-------------------
Latin-Hypercube (LHS) parameter-space exploration for Trento Pb+Pb
initial conditions.

What this script does
----------------------
1.  Draws N_DESIGN points from a maximin-optimised LHS over the 5
    Trento parameters (p, k, w, d, norm) using the established Bayesian
    prior ranges from the literature (Bernhard 2018 / Trajectum 2021).
2.  Runs one Trento call per design point (stdout-only, no .dat files).
3.  For each design point it computes the full set of eccentricity
    observables in every centrality bin: e₂{2}, e₂{4}, e₃{2}, e₄{2},
    e₂{4}/e₂{2}, Γ₂ plus the v₄{4}⁴ puzzle set (ε₄{4}⁴, ⟨ε₂⁸⟩/⟨ε₂⁴⟩²,
    ε₄{Ψ₂}).
4.  Saves all results to a structured NumPy archive (.npz) for downstream
    emulator training / GP regression.
5.  Produces a 3-panel diagnostic figure:
      - LHS corner plot (pair-wise projections of the design)
      - e₂{2} vs centrality coloured by p-parameter
      - Γ₂ in the 0–1% bin vs each varied parameter (sensitivity)

Prior ranges used
------------------
These are flat (uniform) priors following Bernhard et al. (Phys. Rev. C 94,
2016; PhD thesis 2018) and Nijs et al./Trajectum (Phys. Rev. C 103, 2021):

  Parameter  │ Symbol │  Prior range  │ Physical meaning
  ───────────┼────────┼───────────────┼──────────────────────────────────
  p          │   p    │  [-1.0, 1.0]  │ entropy deposition: p=-1 wounded
             │        │               │ nucleon, p=0 geometric mean (KLN/
             │        │               │ saturation), p=1 arithmetic mean
  k (fluct.) │   k    │  [0.3, 2.5]   │ gamma shape for nucleon fluct.;
             │        │               │ k→∞ = no fluct., small k = large
  w (width)  │   w    │  [0.4, 1.6]   │ nucleon Gaussian width [fm]
  d (dmin)   │   d    │  [0.0, 2.0]   │ min. nucleon–nucleon distance [fm]
  norm       │   n    │  [8.0, 22.0]  │ overall multiplicity normalisation

References: Bernhard 2016 (arXiv:1605.03954), Bernhard PhD 2018
            (arXiv:1804.06469), Nijs et al. 2021 (arXiv:2010.15134).
"""

import subprocess
import sys
import itertools
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these
# ─────────────────────────────────────────────────────────────────────────────
HDF1       = "WS1.hdf"
HDF2       = "WS2.hdf"
TRENTO_BIN = "/root/.local/bin/trento"

N_DESIGN   = 50       # number of LHS design points (50–200 is typical for emulators)
N_EVENTS   = 20_000   # events per design point (trade accuracy vs runtime)
SEED_LHS   = 0        # seed for LHS generation
SEED_BASE  = 100      # Trento --random-seed = SEED_BASE + design_index

OUTPUT_NPZ  = Path("lhs_scan_results.npz")
PLOT_FILE   = Path("lhs_scan_diagnostic.pdf")
LOG_FILE    = Path("lhs_scan.log")

# ── Prior ranges (flat uniform) ───────────────────────────────────────────────
#  param_name : (lo, hi)
PRIORS = {
    "p":    (-1.0,  1.0),   # entropy deposition exponent
    "k":    ( 0.3,  2.5),   # nucleon fluctuation shape (gamma)
    "w":    ( 0.4,  1.6),   # nucleon width [fm]
    "d":    ( 0.0,  2.0),   # min nucleon–nucleon distance [fm]
    "norm": ( 8.0, 22.0),   # multiplicity normalisation
}
PARAM_NAMES = list(PRIORS.keys())

# Centrality bin edges [%]
CENTRALITY_EDGES = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7.5,
                    10, 15, 20, 25, 30, 40, 50, 60, 70, 80]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LATIN HYPERCUBE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def latin_hypercube(n_samples: int, n_params: int, rng: np.random.Generator,
                    n_iter: int = 1000) -> np.ndarray:
    """
    Generate a maximin-optimised LHS in [0, 1]^n_params.

    Strategy: generate `n_iter` random LHS realisations, keep the one with
    the largest minimum inter-point distance (maximin criterion).  This is
    the simplest criterion and works well for n_params ≤ 10.
    """
    best_lhs, best_dmin = None, -1.0

    for _ in range(n_iter):
        # Standard LHS: stratified random sample in each dimension
        lhs = np.empty((n_samples, n_params))
        for j in range(n_params):
            perm = rng.permutation(n_samples)
            lhs[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples

        # Maximin: compute minimum pairwise Euclidean distance
        diffs  = lhs[:, None, :] - lhs[None, :, :]        # (n, n, p)
        dists  = np.sqrt((diffs**2).sum(axis=-1))          # (n, n)
        np.fill_diagonal(dists, np.inf)
        dmin   = dists.min()

        if dmin > best_dmin:
            best_dmin = dmin
            best_lhs  = lhs.copy()

    return best_lhs   # shape (n_samples, n_params) in [0,1]^p


def lhs_to_params(unit_lhs: np.ndarray) -> np.ndarray:
    """Map unit-cube LHS to physical parameter ranges."""
    params = np.empty_like(unit_lhs)
    for j, name in enumerate(PARAM_NAMES):
        lo, hi = PRIORS[name]
        params[:, j] = lo + unit_lhs[:, j] * (hi - lo)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — RUN TRENTO FOR ONE DESIGN POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_trento_point(p, k, w, d, norm, seed: int) -> np.ndarray | None:
    """
    Run Trento for one parameter combination; return (N_events, 8) array
    or None on failure.
    """
    cmd = [
        TRENTO_BIN, HDF1, HDF2, str(N_EVENTS),
        "-p", f"{p:.6f}",
        "-k", f"{k:.6f}",
        "-w", f"{w:.6f}",
        "-d", f"{d:.6f}",
        "-n", f"{norm:.6f}",
        "--random-seed", str(seed),
        # No -o → no per-event files
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return None

    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 6:
            try:
                rows.append([float(x) for x in parts[:8]])
            except ValueError:
                continue

    return np.array(rows, dtype=np.float64) if rows else None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — OBSERVABLE FUNCTIONS  (same as in the eccentricity scripts)
# ─────────────────────────────────────────────────────────────────────────────

def _moments(eps: np.ndarray) -> dict:
    return {n: float(np.mean(eps**n)) for n in (2, 4, 6, 8)}


def e2_cumulants(m: dict):
    m2, m4 = m[2], m[4]
    e22    = float(np.sqrt(m2))
    inner4 = 2.0 * m2**2 - m4
    e24    = float(inner4**0.25) if inner4 >= 0 else np.nan
    return e22, e24


def e4_4_cumulant(eps4: np.ndarray) -> float:
    if len(eps4) < 4:
        return np.nan
    m2 = np.mean(eps4**2)
    m4 = np.mean(eps4**4)
    return float(2.0 * m2**2 - m4)


def nl_r8(m: dict) -> float:
    m4, m8 = m[4], m[8]
    return float(m8 / m4**2) if m4 > 0 else np.nan


def e4_psi2(eps2: np.ndarray, eps4: np.ndarray) -> float:
    if len(eps2) < 4:
        return np.nan
    denom = float(np.sqrt(np.mean(eps2**4)))
    return float(np.mean(eps4 * eps2**2) / denom) if denom > 0 else np.nan


def gamma2(m: dict) -> float:
    m2, m4 = m[2], m[4]
    return float((m4 - 2.0 * m2**2) / m2**2) if m2 > 0 else np.nan


# Observable key list (must match order in _obs_vector)
OBS_KEYS = [
    "e2_2", "e2_4", "e3_2", "e4_2",
    "ratio_e24_e22",   # e2{4}/e2{2}
    "gamma2",          # Γ₂
    "e4_4c",           # ε₄{4}⁴
    "nl_r8",           # ⟨ε₂⁸⟩/⟨ε₂⁴⟩²
    "e4psi2",          # ε₄{Ψ₂}
]
N_OBS = len(OBS_KEYS)


def _obs_vector(sel: np.ndarray) -> np.ndarray:
    """Compute all observables for an event subset; return 1-D array."""
    if len(sel) < 10:
        return np.full(N_OBS, np.nan)

    eps2 = sel[:, 4]
    eps3 = sel[:, 5]
    eps4 = sel[:, 6] if sel.shape[1] > 6 else np.zeros(len(sel))

    m2 = _moments(eps2)

    e22, e24 = e2_cumulants(m2)
    e32      = float(np.sqrt(np.mean(eps3**2)))
    e42      = float(np.sqrt(np.mean(eps4**2)))
    e44c     = e4_4_cumulant(eps4)
    r8       = nl_r8(m2)
    ep2      = e4_psi2(eps2, eps4)
    g2       = gamma2(m2)
    r24      = float(e24 / e22) if (e22 > 0 and not np.isnan(e24)) else np.nan

    return np.array([e22, e24, e32, e42, r24, g2, e44c, r8, ep2])


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — COMPUTE OBSERVABLES IN ALL CENTRALITY BINS
# ─────────────────────────────────────────────────────────────────────────────

def assign_centrality(events: np.ndarray) -> np.ndarray:
    rank = np.argsort(np.argsort(-events[:, 3]))
    return rank / len(events) * 100.0


def observables_vs_centrality(events: np.ndarray) -> np.ndarray:
    """
    Return array of shape (N_BINS, N_OBS) with observables per centrality bin.
    """
    cent   = assign_centrality(events)
    edges  = CENTRALITY_EDGES
    n_bins = len(edges) - 1
    out    = np.empty((n_bins, N_OBS))

    for i in range(n_bins):
        mask   = (cent >= edges[i]) & (cent < edges[i+1])
        out[i] = _obs_vector(events[mask])

    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MAIN LOOP OVER DESIGN POINTS
# ─────────────────────────────────────────────────────────────────────────────

def run_scan(design_params: np.ndarray) -> np.ndarray:
    """
    Run Trento for each row of design_params (shape N_DESIGN × 5).
    Returns results array of shape (N_DESIGN, N_BINS, N_OBS).
    N_BINS = len(CENTRALITY_EDGES) - 1
    """
    n_bins   = len(CENTRALITY_EDGES) - 1
    n_design = len(design_params)
    results  = np.full((n_design, n_bins, N_OBS), np.nan)

    log = open(LOG_FILE, "w")
    log.write("idx  p        k        w        d        norm     status  t[s]\n")
    log.write("-" * 72 + "\n")

    for idx, row in enumerate(design_params):
        p, k, w, d, norm = row
        t0     = time.perf_counter()
        seed   = SEED_BASE + idx
        events = run_trento_point(p, k, w, d, norm, seed)
        dt     = time.perf_counter() - t0

        if events is None or len(events) < 100:
            status = "FAIL"
        else:
            results[idx] = observables_vs_centrality(events)
            status       = "OK"

        msg = (f"{idx:3d}  {p:+.4f}  {k:.4f}  {w:.4f}  {d:.4f}  "
               f"{norm:6.2f}  {status:4s}  {dt:.1f}s")
        print(f"[{idx+1:3d}/{n_design}] {msg}")
        log.write(msg + "\n")
        log.flush()

    log.close()
    print(f"\n[INFO] Log → {LOG_FILE}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def save_results(design_params: np.ndarray, results: np.ndarray):
    """
    Save everything to a single .npz archive.

    Contents:
      design_params  – (N_DESIGN, 5)  raw parameter values
      param_names    – list of 5 strings
      obs_keys       – list of N_OBS strings
      centrality_edges – (N_EDGES,)
      centrality_centres – (N_BINS,)
      results        – (N_DESIGN, N_BINS, N_OBS)
    """
    edges   = np.array(CENTRALITY_EDGES)
    centres = 0.5 * (edges[:-1] + edges[1:])

    np.savez_compressed(
        OUTPUT_NPZ,
        design_params       = design_params,
        param_names         = np.array(PARAM_NAMES),
        obs_keys            = np.array(OBS_KEYS),
        centrality_edges    = edges,
        centrality_centres  = centres,
        results             = results,
    )
    print(f"[INFO] Results saved → {OUTPUT_NPZ}")
    print(f"       Shape: design_params {design_params.shape}, "
          f"results {results.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — DIAGNOSTIC PLOT
# ─────────────────────────────────────────────────────────────────────────────

def make_diagnostic_plot(design_params: np.ndarray, results: np.ndarray):
    """
    Three-panel diagnostic figure:
      Panel A – LHS corner plot (pair projections)
      Panel B – e₂{2} vs centrality, coloured by p-parameter
      Panel C – sensitivity: Γ₂ in 0–1% bin vs each parameter
    """
    BG     = "#0d0f14";  PANEL  = "#13161e";  GRID   = "#1e2230"
    TEXT   = "#d8dce8";  CYAN   = "#4fc3f7";  CORAL  = "#ef5350"
    GREEN  = "#66bb6a";  GOLD   = "#ffd54f";  VIOLET = "#ce93d8"
    ORANGE = "#ffa726"

    plt.rcParams.update({
        "figure.facecolor": BG,    "axes.facecolor":  PANEL,
        "axes.edgecolor":   GRID,  "axes.labelcolor": TEXT,
        "xtick.color":      TEXT,  "ytick.color":     TEXT,
        "text.color":       TEXT,  "grid.color":      GRID,
        "grid.linewidth":   0.5,   "font.family":     "sans-serif",
        "font.size":        9,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID,
    })

    n_params = len(PARAM_NAMES)
    edges    = np.array(CENTRALITY_EDGES)
    centres  = 0.5 * (edges[:-1] + edges[1:])

    # ── build figure with 3 logical panels ──────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        f"Trento LHS Parameter Scan  |  {len(design_params)} design points  ×  "
        f"{N_EVENTS:,} events/point",
        fontsize=13, color=TEXT, fontweight="bold", y=1.01,
    )

    # Left block: corner plot (n_params × n_params minus diagonal)
    gs_left  = GridSpec(n_params, n_params, figure=fig,
                        left=0.04, right=0.46, top=0.95, bottom=0.06,
                        hspace=0.05, wspace=0.05)
    # Right block: 2 panels stacked
    gs_right = GridSpec(2, 1, figure=fig,
                        left=0.54, right=0.97, top=0.95, bottom=0.06,
                        hspace=0.38)

    # ── Panel A: corner plot ─────────────────────────────────────────────────
    lo_arr = np.array([PRIORS[n][0] for n in PARAM_NAMES])
    hi_arr = np.array([PRIORS[n][1] for n in PARAM_NAMES])
    labels  = [r"$p$", r"$k$", r"$w$ [fm]", r"$d_{\min}$ [fm]", r"norm"]
    colors  = [CYAN, CORAL, GREEN, GOLD, VIOLET]

    for row in range(n_params):
        for col in range(n_params):
            ax = fig.add_subplot(gs_left[row, col])
            ax.set_facecolor(PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID)

            if row == col:
                # Marginal histogram
                ax.hist(design_params[:, col], bins=12,
                        color=colors[col], alpha=0.75, edgecolor=BG)
                ax.set_xlim(lo_arr[col], hi_arr[col])
                ax.set_yticks([])
            elif row > col:
                # Scatter
                ax.scatter(design_params[:, col], design_params[:, row],
                           c=colors[row], s=18, alpha=0.7, linewidths=0)
                ax.set_xlim(lo_arr[col], hi_arr[col])
                ax.set_ylim(lo_arr[row], hi_arr[row])
            else:
                ax.set_visible(False)
                continue

            # Axis labels on edges only
            if row == n_params - 1:
                ax.set_xlabel(labels[col], fontsize=8, color=TEXT)
            else:
                ax.set_xticklabels([])
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=8, color=TEXT)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7, colors=TEXT)
            ax.grid(True, linestyle="--", alpha=0.25)

    # ── Panel B: e₂{2} vs centrality, coloured by p ──────────────────────────
    ax_b   = fig.add_subplot(gs_right[0])
    obs_idx = OBS_KEYS.index("e2_2")
    p_vals  = design_params[:, PARAM_NAMES.index("p")]
    cmap    = plt.cm.coolwarm
    p_norm  = plt.Normalize(PRIORS["p"][0], PRIORS["p"][1])

    for i in range(len(design_params)):
        y     = results[i, :, obs_idx]
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue
        ax_b.plot(centres[valid], y[valid],
                  color=cmap(p_norm(p_vals[i])),
                  linewidth=1.0, alpha=0.65)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=p_norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_b, fraction=0.04, pad=0.02)
    cb.set_label("$p$ parameter", color=TEXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=TEXT)

    ax_b.set_xlabel("Centrality (%)", labelpad=5)
    ax_b.set_ylabel("ε₂{2}", labelpad=5)
    ax_b.set_title("ε₂{2} vs centrality  —  coloured by $p$",
                   color=CYAN, fontsize=10)
    ax_b.set_xlim(left=0)
    ax_b.set_ylim(bottom=0)
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.axvspan(0, 1, alpha=0.08, color=CYAN, zorder=0)
    ax_b.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ── Panel C: Γ₂ in 0–1% bin vs each parameter ───────────────────────────
    # Find the 0–1% bin index (edges 0→1%)
    uc_idx = next(
        i for i in range(len(edges)-1)
        if edges[i] <= 0.5 and edges[i+1] <= 1.5 and edges[i+1] > 0.4
    )
    # Use the 0–1% accumulated bin: pool all bins inside [0,1]
    uc_mask  = (centres < 1.0) & (centres >= 0.0)
    g2_idx   = OBS_KEYS.index("gamma2")
    # Average Γ₂ over 0-centrality bins
    g2_uc    = np.nanmean(results[:, uc_mask, g2_idx], axis=1)  # (N_DESIGN,)

    ax_c = fig.add_subplot(gs_right[1])
    param_colors = [CYAN, CORAL, GREEN, GOLD, VIOLET]
    param_labels = [r"$p$", r"$k$", r"$w$", r"$d_{\min}$", r"norm"]

    for j, (name, color, lbl) in enumerate(
            zip(PARAM_NAMES, param_colors, param_labels)):
        x = design_params[:, j]
        # Normalise x to [0,1] for overlay
        lo, hi = PRIORS[name]
        xn     = (x - lo) / (hi - lo)
        valid  = ~np.isnan(g2_uc)
        ax_c.scatter(xn[valid], g2_uc[valid],
                     c=color, s=20, alpha=0.65, linewidths=0,
                     label=lbl, zorder=3)

    ax_c.axhline(0.0, color=TEXT, linewidth=0.8, linestyle=":", alpha=0.5)
    ax_c.set_xlabel("Normalised parameter value  [0 = prior min, 1 = prior max]",
                    labelpad=5)
    ax_c.set_ylabel("Γ₂  (0–1% centrality)", labelpad=5)
    ax_c.set_title(
        "Parameter sensitivity of Γ₂ in ultra-central (0–1%) bin",
        color=ORANGE, fontsize=10,
    )
    ax_c.legend(fontsize=8, ncol=5, loc="upper center",
                bbox_to_anchor=(0.5, 1.0))
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.savefig(PLOT_FILE, dpi=160, bbox_inches="tight", facecolor=BG)
    print(f"[INFO] Diagnostic plot → {PLOT_FILE}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — PRINT DESIGN TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_design_table(design_params: np.ndarray):
    hdr = (f"{'idx':>4}  " +
           "  ".join(f"{n:>8}" for n in PARAM_NAMES))
    sep = "─" * len(hdr)
    print(f"\n{sep}\nLHS Design Points\n{sep}\n{hdr}\n{sep}")
    for i, row in enumerate(design_params):
        vals = "  ".join(f"{v:>8.4f}" for v in row)
        print(f"{i:>4}  {vals}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — PRINT OBSERVABLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_obs_summary(design_params: np.ndarray, results: np.ndarray):
    """Print mean ± std across design points for each observable at 5 bins."""
    edges   = np.array(CENTRALITY_EDGES)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Highlight bins: find index closest to each centrality centre
    highlight_cents = [0.05, 0.25, 0.75, 3.0, 12.5]
    bin_idxs = [int(np.argmin(np.abs(centres - hc))) for hc in highlight_cents]
    bin_lbls = [f"{edges[i]:.1f}–{edges[i+1]:.1f}%" for i in bin_idxs]

    print("\n" + "=" * 90)
    print(f"Observable summary across {len(design_params)} design points")
    print("Format: mean ± std\n")

    hdr = f"{'Obs':>16}" + "".join(f"  {lbl:>14}" for lbl in bin_lbls)
    print(hdr)
    print("─" * len(hdr))

    for j, key in enumerate(OBS_KEYS):
        row = f"{key:>16}"
        for b in bin_idxs:
            vals  = results[:, b, j]
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                row += f"  {np.mean(valid):>6.4f}±{np.std(valid):>6.4f}"
            else:
                row += f"  {'---':>14}"
        print(row)

    print("=" * 90 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED_LHS)

    print(f"[INFO] Generating {N_DESIGN}-point maximin LHS over "
          f"{len(PARAM_NAMES)} parameters…")
    unit_lhs     = latin_hypercube(N_DESIGN, len(PARAM_NAMES), rng)
    design_params = lhs_to_params(unit_lhs)

    print_design_table(design_params)

    print(f"[INFO] Running {N_DESIGN} Trento jobs  "
          f"({N_EVENTS:,} events each)…\n")
    t_start = time.perf_counter()
    results  = run_scan(design_params)
    t_total  = time.perf_counter() - t_start
    print(f"\n[INFO] Total wall time: {t_total/60:.1f} min")

    save_results(design_params, results)
    print_obs_summary(design_params, results)
    make_diagnostic_plot(design_params, results)


if __name__ == "__main__":
    main()
