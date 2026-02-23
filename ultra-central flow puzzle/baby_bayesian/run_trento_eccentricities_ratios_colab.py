#!/usr/bin/env python3
"""
run_trento_eccentricities_ratios_colab.py
-----------------------------
Self-contained script to:
  1. Run Trento heavy-ion collision events using two isobar HDF files
  2. Compute eccentricities and ultracentral flow observables from stdout
  3. Plot two rows:
       Row 1 — e2{2}, e2{4}, e3{2}
       Row 2 — e2{4}/e2{2}  (fluctuation ratio),
               e3{2}/e2{2}  (triangularity/ellipticity),
               e4{2}        (hexadecapole eccentricity),
               Gamma_2      (normalized 4th cumulant, ultra-central probe)

Ultra-central flow puzzle context
----------------------------------
In ultra-central collisions (0-1%) geometry is dominated by fluctuations.
Key observables:
  - e2{4}/e2{2} < 1: measures non-Gaussian fluctuations; sensitive to
    nucleon correlations and deformation. Drops sharply at 0%.
  - e3{2}/e2{2}: triangularity relative to ellipticity; purely fluctuation-
    driven, expected ~1 in very central collisions.
  - e4{2}: hexadecapole eccentricity; sensitive to beta4 deformation and
    higher-order fluctuations.
  - Gamma_2 = (e2{4}^4 - e2{6}^4) / (e2{2}^2 * e2{4}^2):
    normalized cumulant difference probing the eccentricity distribution
    shape, particularly sensitive to geometry in the 0-1% bin.
    (Here approximated via 4th standardized cumulant of eps2 distribution.)
"""

import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION  <- edit these
# ─────────────────────────────────────────────
HDF1        = "WS1.hdf"
HDF2        = "WS2.hdf"
SKIP_TRENTO = False    # set True to reuse saved events

TRENTO_BIN  = "/root/.local/bin/trento"
N_EVENTS    = 500
RANDOM_SEED = 42

# Physics parameters
P_PARAM = 0.063
K_PARAM = 1 / 1.05**2
W_PARAM = 1.12
D_PARAM = 2.97
N_PARAM = 14.2

OUTPUT_DIR = Path("trento_events")
CACHE_FILE = OUTPUT_DIR / "events_summary.npy"
PLOT_FILE  = "eccentricities_vs_centrality.pdf"

# Centrality bin edges for the plot (percentile)
CENTRALITY_EDGES = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7.5,
                    10, 15, 20, 25, 30, 40, 50, 60, 70, 80]

# Bins printed in the summary table
HIGHLIGHT_BINS = [(0, 0.1), (0, 0.5), (0, 1), (1, 5),
                  (5, 10), (10, 20), (20, 30)]


# ─────────────────────────────────────────────
# STEP 1 — RUN TRENTO
# ─────────────────────────────────────────────

def run_trento(hdf1, hdf2):
    """Run Trento, parse summary lines from stdout, return (N,8) array."""
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)

    cmd = [
        TRENTO_BIN, hdf1, hdf2, str(N_EVENTS),
        "-p", str(P_PARAM),
        "-k", str(K_PARAM),
        "-w", str(W_PARAM),
        "-d", str(D_PARAM),
        "-n", str(N_PARAM),
        "--random-seed", str(RANDOM_SEED),
        "-o", str(OUTPUT_DIR),
        # NOTE: no -q so summary lines are printed to stdout
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Trento failed (return code {result.returncode}).\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )

    # Stdout format: event_number  b  npart  mult  e2  e3  e4  e5
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

    if not rows:
        raise RuntimeError(
            f"Trento ran but no summary lines found in stdout.\n"
            f"stdout preview:\n{result.stdout[:500]}\n"
            f"stderr:\n{result.stderr[:500]}"
        )

    arr = np.array(rows)
    np.save(CACHE_FILE, arr)
    print(f"[INFO] {len(arr)} events loaded. Cached to {CACHE_FILE}")
    return arr


# ─────────────────────────────────────────────
# STEP 2 — CENTRALITY
# ─────────────────────────────────────────────

def assign_centrality(events):
    """Assign centrality percentile by ranking on multiplicity (col 3)."""
    mult = events[:, 3]
    rank = np.argsort(np.argsort(-mult))   # highest mult -> rank 0
    return rank / len(events) * 100        # percentile


def select_bin(events, centrality, cmin, cmax):
    mask = (centrality >= cmin) & (centrality < cmax)
    return events[mask]


# ─────────────────────────────────────────────
# STEP 3 — CUMULANTS & UC OBSERVABLES
# ─────────────────────────────────────────────

def cumulants(eps):
    """Two- and four-particle cumulant eccentricities."""
    if len(eps) < 4:
        return np.nan, np.nan
    eps2  = eps**2
    eps4  = eps**4
    e_2   = np.sqrt(np.mean(eps2))
    inner = 2 * np.mean(eps2)**2 - np.mean(eps4)
    e_4   = inner**0.25 if inner >= 0 else np.nan
    return e_2, e_4


def gamma2(eps):
    """
    Normalized 4th cumulant of the eps2 distribution:
        Gamma_2 = (<eps^4> - 2<eps^2>^2) / <eps^2>^2
    This is the 4th standardized cumulant (excess kurtosis analog).
    Gamma_2 = 0 for a Gaussian distribution.
    Negative Gamma_2 signals non-Gaussian fluctuations prominent in UC.
    Related to e2{4}^4/e2{2}^4 - 2.
    """
    if len(eps) < 4:
        return np.nan
    m2 = np.mean(eps**2)
    m4 = np.mean(eps**4)
    if m2 == 0:
        return np.nan
    return (m4 - 2 * m2**2) / m2**2


# ─────────────────────────────────────────────
# STEP 4 — COMPUTE ALL OBSERVABLES VS CENTRALITY
# ─────────────────────────────────────────────

def compute_eccentricities(events, centrality):
    edges   = CENTRALITY_EDGES
    centres = np.array([(edges[i] + edges[i+1]) / 2
                        for i in range(len(edges)-1)])

    results = {k: [] for k in [
        "e2_2", "e2_4", "e3_2", "e4_2",
        "ratio_e2_4_e2_2",    # e2{4}/e2{2}  — fluctuation suppression
        "ratio_e3_e2",         # e3{2}/e2{2}  — triangularity/ellipticity
        "gamma2",              # normalized 4th cumulant
        "n",
    ]}

    for i in range(len(edges)-1):
        sel = select_bin(events, centrality, edges[i], edges[i+1])
        results["n"].append(len(sel))

        if len(sel) < 10:
            for k in results:
                if k != "n":
                    results[k].append(np.nan)
            continue

        e2_2, e2_4 = cumulants(sel[:, 4])
        e3_2, _    = cumulants(sel[:, 5])
        e4_2, _    = cumulants(sel[:, 6]) if sel.shape[1] > 6 else (np.nan, np.nan)
        g2         = gamma2(sel[:, 4])

        results["e2_2"].append(e2_2)
        results["e2_4"].append(e2_4)
        results["e3_2"].append(e3_2)
        results["e4_2"].append(e4_2)
        results["ratio_e2_4_e2_2"].append(
            e2_4 / e2_2 if (e2_2 > 0 and not np.isnan(e2_4)) else np.nan)
        results["ratio_e3_e2"].append(
            e3_2 / e2_2 if e2_2 > 0 else np.nan)
        results["gamma2"].append(g2)

    return {"centres": centres,
            **{k: np.array(v) for k, v in results.items()}}


# ─────────────────────────────────────────────
# STEP 5 — PRINT TABLE
# ─────────────────────────────────────────────

def print_table(events, centrality):
    hdr = (f"{'Centrality':>12}  {'N':>6}  {'e2{2}':>7}  {'e2{4}':>7}  "
           f"{'e3{2}':>7}  {'e4{2}':>7}  {'e24/e22':>8}  {'e3/e2':>7}  {'Gamma2':>8}")
    print("\n" + "="*len(hdr))
    print(hdr)
    print("="*len(hdr))
    for cmin, cmax in HIGHLIGHT_BINS:
        sel = select_bin(events, centrality, cmin, cmax)
        if len(sel) < 4:
            print(f"{f'{cmin}-{cmax}%':>12}  {len(sel):>6}  {'---':>7}  {'---':>7}  "
                  f"{'---':>7}  {'---':>7}  {'---':>8}  {'---':>7}  {'---':>8}")
            continue
        e2_2, e2_4 = cumulants(sel[:, 4])
        e3_2, _    = cumulants(sel[:, 5])
        e4_2, _    = cumulants(sel[:, 6]) if sel.shape[1] > 6 else (np.nan, np.nan)
        g2         = gamma2(sel[:, 4])
        r24        = e2_4/e2_2 if (e2_2 > 0 and not np.isnan(e2_4)) else np.nan
        r32        = e3_2/e2_2 if e2_2 > 0 else np.nan
        print(f"{f'{cmin}-{cmax}%':>12}  {len(sel):>6}  {e2_2:>7.4f}  {e2_4:>7.4f}  "
              f"{e3_2:>7.4f}  {e4_2:>7.4f}  {r24:>8.4f}  {r32:>7.4f}  {g2:>8.4f}")
    print("="*len(hdr) + "\n")


# ─────────────────────────────────────────────
# STEP 6 — PLOT (2 rows x 3 columns)
# ─────────────────────────────────────────────

def make_plot(res):
    BG     = "#0d0f14"
    PANEL  = "#13161e"
    GRID   = "#1e2230"
    TEXT   = "#d8dce8"
    CYAN   = "#4fc3f7"
    CORAL  = "#ef5350"
    GREEN  = "#66bb6a"
    GOLD   = "#ffd54f"
    VIOLET = "#ce93d8"
    ORANGE = "#ffa726"

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   PANEL,
        "axes.edgecolor":   GRID,
        "axes.labelcolor":  TEXT,
        "xtick.color":      TEXT,
        "ytick.color":      TEXT,
        "text.color":       TEXT,
        "grid.color":       GRID,
        "grid.linewidth":   0.6,
        "font.family":      "monospace",
        "font.size":        10,
        "axes.titlesize":   11,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
    })

    c = res["centres"]

    # ── Layout: 2 rows x 3 cols ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Eccentricities & Ultra-Central Flow Observables  |  Trento",
                 fontsize=14, color=TEXT, y=1.01, fontweight="bold")

    # ── Row 1: standard eccentricities ──
    row1 = [
        (axes[0, 0], res["e2_2"], CYAN,   "e2{2}",
         "two-particle cumulant",   False),
        (axes[0, 1], res["e2_4"], CORAL,  "e2{4}",
         "four-particle cumulant",  False),
        (axes[0, 2], res["e3_2"], GREEN,  "e3{2}",
         "triangularity 2-part.",   False),
    ]

    # ── Row 2: ultra-central puzzle observables ──
    row2 = [
        (axes[1, 0], res["ratio_e2_4_e2_2"], GOLD,
         "e2{4} / e2{2}",
         "fluctuation suppression\n(< 1 from non-Gaussianity)",
         True),
        (axes[1, 1], res["ratio_e3_e2"], VIOLET,
         "e3{2} / e2{2}",
         "triangularity / ellipticity\n(-> 1 in ultra-central)",
         False),
        (axes[1, 2], res["gamma2"], ORANGE,
         "Gamma_2",
         "4th cumulant of eps2\n(< 0 = non-Gaussian fluct.)",
         None),   # None = allow negative y-axis
    ]

    def _plot_panel(ax, y, color, label, sublabel, force_ylim_positive):
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            ax.text(0.5, 0.5, "insufficient data",
                    transform=ax.transAxes, ha="center",
                    va="center", color=TEXT, alpha=0.5)
        else:
            ax.fill_between(c[valid], y[valid]*0.95, y[valid]*1.05,
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(c[valid], y[valid], color=color, linewidth=2.2,
                    marker="o", markersize=4,
                    markerfacecolor=BG, markeredgecolor=color,
                    markeredgewidth=1.5)

        ax.set_xlabel("Centrality (%)", labelpad=6)
        ax.set_ylabel(label, labelpad=6, fontsize=11)
        ax.set_title(f"{label}\n{sublabel}", color=color, pad=8, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_xlim(left=0)

        if force_ylim_positive is True:
            ax.set_ylim(bottom=0)
        elif force_ylim_positive is False:
            ax.set_ylim(bottom=0)
        # None -> auto (allow negative, e.g. Gamma2)

        # Ultra-central shading
        ax.axvspan(0, 1, alpha=0.07, color=color)
        ylo, yhi = ax.get_ylim()
        ax.text(0.5, yhi - (yhi - ylo)*0.06, "UC",
                ha="center", va="top", fontsize=8,
                color=color, alpha=0.8)

        # Reference lines for ratio panels
        if "/ e2" in label or "/ e2" in sublabel:
            ax.axhline(1.0, color=TEXT, linewidth=0.8,
                       linestyle=":", alpha=0.5)
        if "Gamma" in label:
            ax.axhline(0.0, color=TEXT, linewidth=0.8,
                       linestyle=":", alpha=0.5)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    for (ax, y, color, label, sublabel, ylim_flag) in row1 + row2:
        _plot_panel(ax, y, color, label, sublabel, ylim_flag)

    # Row labels on the left
    axes[0, 0].annotate("ROW 1\nEccentricities",
                         xy=(-0.28, 0.5), xycoords="axes fraction",
                         fontsize=9, color=TEXT, alpha=0.6,
                         ha="center", va="center", rotation=90)
    axes[1, 0].annotate("ROW 2\nUC Puzzle",
                         xy=(-0.28, 0.5), xycoords="axes fraction",
                         fontsize=9, color=TEXT, alpha=0.6,
                         ha="center", va="center", rotation=90)

    fig.tight_layout(rect=[0.03, 0, 1, 1])
    fig.savefig(PLOT_FILE, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"[INFO] Plot saved -> {PLOT_FILE}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if SKIP_TRENTO:
        if not CACHE_FILE.exists():
            raise RuntimeError(
                f"Cache file {CACHE_FILE} not found. "
                "Run with SKIP_TRENTO=False first."
            )
        events = np.load(CACHE_FILE)
        print(f"[INFO] Loaded {len(events)} events from cache.")
    else:
        events = run_trento(HDF1, HDF2)

    centrality = assign_centrality(events)
    print_table(events, centrality)
    res = compute_eccentricities(events, centrality)
    make_plot(res)

main()
