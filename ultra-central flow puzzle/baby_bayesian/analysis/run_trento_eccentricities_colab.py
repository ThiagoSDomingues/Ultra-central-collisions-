#!/usr/bin/env python3
"""
run_trento_eccentricities_collab.py
-----------------------------
Self-contained script to:
  1. Run Trento heavy-ion collision events using two isobar HDF files using Google Collab
  2. Compute eccentricities e2 and e3 from stdout summary lines
  3. Calculate cumulants e2{2}, e2{4}, e3{2} as a function of centrality
  4. Plot eccentricities vs centrality
"""

import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit these
# ─────────────────────────────────────────────
HDF1        = "WS1.hdf"
HDF2        = "WS2.hdf"
SKIP_TRENTO = False       # set True to reuse saved events (skips Trento run)

TRENTO_BIN  = "/root/.local/bin/trento"
N_EVENTS    = 5000
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
    rank = np.argsort(np.argsort(-mult))       # highest mult -> rank 0
    return rank / len(events) * 100            # percentile


def select_bin(events, centrality, cmin, cmax):
    mask = (centrality >= cmin) & (centrality < cmax)
    return events[mask]


# ─────────────────────────────────────────────
# STEP 3 — CUMULANTS
# ─────────────────────────────────────────────

def cumulants(eps):
    """Return e{2} and e{4} cumulant eccentricities."""
    if len(eps) < 4:
        return np.nan, np.nan
    eps2 = eps**2
    eps4 = eps**4
    e_2   = np.sqrt(np.mean(eps2))
    inner = 2 * np.mean(eps2)**2 - np.mean(eps4)
    e_4   = inner**0.25 if inner >= 0 else np.nan
    return e_2, e_4


# ─────────────────────────────────────────────
# STEP 4 — COMPUTE VS CENTRALITY
# ─────────────────────────────────────────────

def compute_eccentricities(events, centrality):
    edges   = CENTRALITY_EDGES
    centres = np.array([(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)])
    e2_2_arr, e2_4_arr, e3_2_arr, n_arr = [], [], [], []

    for i in range(len(edges)-1):
        sel = select_bin(events, centrality, edges[i], edges[i+1])
        n_arr.append(len(sel))
        if len(sel) < 10:
            e2_2_arr.append(np.nan)
            e2_4_arr.append(np.nan)
            e3_2_arr.append(np.nan)
            continue
        e2_2, e2_4 = cumulants(sel[:, 4])
        e3_2, _    = cumulants(sel[:, 5])
        e2_2_arr.append(e2_2)
        e2_4_arr.append(e2_4)
        e3_2_arr.append(e3_2)

    return {
        "centres": centres,
        "e2_2":    np.array(e2_2_arr),
        "e2_4":    np.array(e2_4_arr),
        "e3_2":    np.array(e3_2_arr),
        "n":       np.array(n_arr),
    }


# ─────────────────────────────────────────────
# STEP 5 — PRINT TABLE
# ─────────────────────────────────────────────

def print_table(events, centrality):
    print("\n" + "="*68)
    print(f"{'Centrality':>12}  {'N':>7}  {'e2{2}':>8}  {'e2{4}':>8}  {'e3{2}':>8}")
    print("="*68)
    for cmin, cmax in HIGHLIGHT_BINS:
        sel = select_bin(events, centrality, cmin, cmax)
        if len(sel) < 4:
            print(f"{f'{cmin}-{cmax}%':>12}  {len(sel):>7}  {'---':>8}  {'---':>8}  {'---':>8}")
            continue
        e2_2, e2_4 = cumulants(sel[:, 4])
        e3_2, _    = cumulants(sel[:, 5])
        print(f"{f'{cmin}-{cmax}%':>12}  {len(sel):>7}  {e2_2:>8.4f}  {e2_4:>8.4f}  {e3_2:>8.4f}")
    print("="*68 + "\n")


# ─────────────────────────────────────────────
# STEP 6 — PLOT
# ─────────────────────────────────────────────

def make_plot(res):
    BG      = "#0d0f14"
    PANEL   = "#13161e"
    GRID    = "#1e2230"
    TEXT    = "#d8dce8"
    CYAN    = "#4fc3f7"
    CORAL   = "#ef5350"
    GREEN   = "#66bb6a"

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
        "font.size":        11,
        "axes.titlesize":   13,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
    })

    c = res["centres"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Eccentricities vs Centrality  -  Trento",
                 fontsize=15, color=TEXT, y=1.02, fontweight="bold")

    datasets = [
        (axes[0], res["e2_2"], CYAN,  "e2{2}", "two-particle cumulant"),
        (axes[1], res["e2_4"], CORAL, "e2{4}", "four-particle cumulant"),
        (axes[2], res["e3_2"], GREEN, "e3{2}", "two-particle cumulant"),
    ]

    for ax, y, color, label, sublabel in datasets:
        valid = ~np.isnan(y)
        ax.fill_between(c[valid], y[valid]*0.95, y[valid]*1.05,
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(c[valid], y[valid], color=color, linewidth=2.2,
                marker="o", markersize=4,
                markerfacecolor=BG, markeredgecolor=color, markeredgewidth=1.5)
        ax.set_xlabel("Centrality (%)", labelpad=8)
        ax.set_ylabel(label, labelpad=8, fontsize=13)
        ax.set_title(f"{label}\n({sublabel})", color=color, pad=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.axvspan(0, 1, alpha=0.07, color=color)
        ax.text(0.5, ax.get_ylim()[1]*0.95, "UC",
                ha="center", va="top", fontsize=8, color=color, alpha=0.8)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"[INFO] Plot saved -> {PLOT_FILE}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if SKIP_TRENTO:
        if not CACHE_FILE.exists():
            raise RuntimeError(
                f"Cache file {CACHE_FILE} not found. Run with SKIP_TRENTO=False first."
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
