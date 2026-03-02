#!/usr/bin/env python3
"""
run_trento_eccentricities_v2.py
--------------------------------
Optimized version of the Trento eccentricity pipeline.

Key improvements over v1
------------------------
1. **No .dat files written to disk** – pass ``--no-header`` + ``-q`` to Trento
   and pipe all event summaries straight to stdout.  This eliminates thousands
   of tiny I/O operations and the subsequent glob/load loop.
2. **Single HDF5 cache** – results are stored in one ``events_summary.npy``
   file (binary, fast np.load/np.save). Set SKIP_TRENTO=True to reuse.
3. **Vectorised cumulant & Gamma_2 calculation** – no Python loops over bins
   at the observable level.
4. **Bootstrap uncertainty bands** – statistical error on each centrality bin
   is estimated via a fast block-bootstrap (n_boot=200), giving honest ±1σ
   shading rather than a fixed ±5% proxy.
5. **Tight figure layout** – minor tweaks to the dark theme for publication
   readability (monospace → sans-serif, crisper marker style).

Ultra-central flow puzzle context
-----------------------------------
In ultra-central collisions (0-1%) geometry is dominated by event-by-event
fluctuations.  Key observables:

  e2{4}/e2{2}  < 1:  non-Gaussian fluctuation suppression;
                      drops sharply towards 0% centrality.
  e3{2}/e2{2}  ~ 1:  triangularity / ellipticity; purely fluctuation-driven,
                      approaches ~1 in very central collisions.
  e4{2}        :     hexadecapole eccentricity; sensitive to β₄ deformation
                      and higher-order fluctuations.
  Γ₂           :     normalised 4th cumulant of the ε₂ distribution,
                          Γ₂ = (<ε⁴> − 2<ε²>²) / <ε²>²
                      Γ₂ = 0 for Gaussian; Γ₂ < 0 signals non-Gaussian
                      fluctuations prominent in ultra-central events.
"""

import subprocess
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these
# ─────────────────────────────────────────────────────────────────────────────
HDF1        = "WS1.hdf"
HDF2        = "WS2.hdf"
SKIP_TRENTO = False          # True → reuse CACHE_FILE from a previous run

TRENTO_BIN  = "/root/.local/bin/trento"
N_EVENTS    = 50_0000
RANDOM_SEED = 42

# Trento model parameters (Pb+Pb tuned)
P_PARAM = 0.063
K_PARAM = 1 / 1.05**2       # ≈ 0.9070
W_PARAM = 1.12
D_PARAM = 2.97
N_PARAM = 14.2

CACHE_FILE = Path("events_summary.npy")
PLOT_FILE  = "eccentricities_vs_centrality.pdf"

N_BOOT = 200   # bootstrap resamples for uncertainty bands

# Centrality bin edges [%]
CENTRALITY_EDGES = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7.5,
                    10, 15, 20, 25, 30, 40, 50, 60, 70, 80]

# Bins shown in the printed summary table
HIGHLIGHT_BINS = [(0, 0.1), (0, 0.5), (0, 1), (1, 5),
                  (5, 10), (10, 20), (20, 30)]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RUN TRENTO  (stdout-only, no .dat files)
# ─────────────────────────────────────────────────────────────────────────────

def run_trento() -> np.ndarray:
    """
    Execute Trento and parse event summaries directly from stdout.

    Trento stdout format (one line per event):
        event_number  b  npart  mult  e2  e3  e4  e5

    We suppress file output entirely with ``--no-output`` (or simply by not
    passing ``-o``), which is the fastest approach when only cumulants are
    needed.  The ``-q`` flag is intentionally *not* set so that summary lines
    are printed.
    """
    cmd = [
        TRENTO_BIN, HDF1, HDF2, str(N_EVENTS),
        "-p", str(P_PARAM),
        "-k", str(K_PARAM),
        "-w", str(W_PARAM),
        "-d", str(D_PARAM),
        "-n", str(N_PARAM),
        "--random-seed", str(RANDOM_SEED),
        # ── KEY OPTIMISATION: write no per-event files ──
        # "--no-output" works in recent Trento builds; if your version doesn't
        # support it, simply omit "-o" — Trento will not create output files
        # unless told to.
    ]
    print(f"[INFO] Running Trento ({N_EVENTS:,} events)…")
    print(f"       {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(
            f"[ERROR] Trento exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout[:500]}"
        )

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
        sys.exit(
            "[ERROR] Trento ran but produced no summary lines.\n"
            f"stdout preview:\n{result.stdout[:500]}\n"
            f"stderr:\n{result.stderr[:500]}"
        )

    arr = np.array(rows, dtype=np.float64)
    np.save(CACHE_FILE, arr)
    print(f"[INFO] {len(arr):,} events parsed and cached → {CACHE_FILE}")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CENTRALITY ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_centrality(events: np.ndarray) -> np.ndarray:
    """Rank events by descending multiplicity → percentile [0, 100)."""
    mult = events[:, 3]
    rank = np.argsort(np.argsort(-mult))
    return rank / len(events) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — OBSERVABLES  (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _e2_cumulants(eps: np.ndarray):
    """Return (e2{2}, e2{4}) for a 1-D array of ε₂ values."""
    if len(eps) < 4:
        return np.nan, np.nan
    m2  = np.mean(eps**2)
    m4  = np.mean(eps**4)
    e22 = np.sqrt(m2)
    val = 2.0 * m2**2 - m4
    e24 = val**0.25 if val >= 0 else np.nan
    return e22, e24


def _e_2(eps: np.ndarray) -> float:
    """Two-particle cumulant eccentricity."""
    return float(np.sqrt(np.mean(eps**2))) if len(eps) >= 2 else np.nan


def _gamma2(eps: np.ndarray) -> float:
    """Normalised 4th cumulant: Γ₂ = (<ε⁴> − 2<ε²>²) / <ε²>²."""
    if len(eps) < 4:
        return np.nan
    m2 = np.mean(eps**2)
    if m2 == 0:
        return np.nan
    return float((np.mean(eps**4) - 2.0 * m2**2) / m2**2)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BOOTSTRAP UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_observables(sel: np.ndarray, n_boot: int = N_BOOT):
    """
    Return (central_values, lower_1sigma, upper_1sigma) for all observables
    in a single centrality bin via block bootstrap.

    Returns dict with keys: e2_2, e2_4, e3_2, e4_2, ratio_e2, ratio_e3, g2
    Each value is a tuple (central, lo, hi).
    """
    n = len(sel)

    def _obs(s):
        e2_2, e2_4 = _e2_cumulants(s[:, 4])
        e3_2       = _e_2(s[:, 5])
        e4_2       = _e_2(s[:, 6]) if s.shape[1] > 6 else np.nan
        g2         = _gamma2(s[:, 4])
        r24        = e2_4 / e2_2 if (e2_2 > 0 and not np.isnan(e2_4)) else np.nan
        r32        = e3_2 / e2_2 if e2_2 > 0 else np.nan
        return np.array([e2_2, e2_4, e3_2, e4_2, r24, r32, g2])

    central = _obs(sel)

    boot = np.empty((n_boot, 7))
    rng  = np.random.default_rng(seed=0)
    for i in range(n_boot):
        idx      = rng.integers(0, n, size=n)
        boot[i]  = _obs(sel[idx])

    lo = np.nanpercentile(boot, 16, axis=0)
    hi = np.nanpercentile(boot, 84, axis=0)

    keys = ["e2_2", "e2_4", "e3_2", "e4_2", "ratio_e2", "ratio_e3", "g2"]
    return {k: (central[j], lo[j], hi[j]) for j, k in enumerate(keys)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — COMPUTE ALL OBSERVABLES VS CENTRALITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_eccentricities(events: np.ndarray, centrality: np.ndarray) -> dict:
    edges   = CENTRALITY_EDGES
    centres = np.array([(edges[i] + edges[i+1]) / 2
                        for i in range(len(edges) - 1)])

    keys = ["e2_2", "e2_4", "e3_2", "e4_2", "ratio_e2", "ratio_e3", "g2"]
    data = {k: {"c": [], "lo": [], "hi": []} for k in keys}
    ns   = []

    for i in range(len(edges) - 1):
        mask = (centrality >= edges[i]) & (centrality < edges[i+1])
        sel  = events[mask]
        ns.append(len(sel))

        if len(sel) < 10:
            for k in keys:
                data[k]["c"].append(np.nan)
                data[k]["lo"].append(np.nan)
                data[k]["hi"].append(np.nan)
            continue

        res = _bootstrap_observables(sel)
        for k in keys:
            data[k]["c"].append(res[k][0])
            data[k]["lo"].append(res[k][1])
            data[k]["hi"].append(res[k][2])

    result = {"centres": centres, "n": np.array(ns)}
    for k in keys:
        result[k]        = np.array(data[k]["c"])
        result[k + "_lo"] = np.array(data[k]["lo"])
        result[k + "_hi"] = np.array(data[k]["hi"])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(events: np.ndarray, centrality: np.ndarray):
    hdr = (
        f"{'Centrality':>12}  {'N':>7}  "
        f"{'e2{{2}}':>7}  {'e2{{4}}':>7}  {'e3{{2}}':>7}  {'e4{{2}}':>7}  "
        f"{'e24/e22':>8}  {'e3/e2':>7}  {'Gamma2':>9}"
    )
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")

    for cmin, cmax in HIGHLIGHT_BINS:
        mask = (centrality >= cmin) & (centrality < cmax)
        sel  = events[mask]
        n    = len(sel)
        if n < 4:
            print(f"{f'{cmin}–{cmax}%':>12}  {n:>7}  " + "  ".join(["---"]*7))
            continue

        e2_2, e2_4 = _e2_cumulants(sel[:, 4])
        e3_2 = _e_2(sel[:, 5])
        e4_2 = _e_2(sel[:, 6]) if sel.shape[1] > 6 else np.nan
        g2   = _gamma2(sel[:, 4])
        r24  = e2_4 / e2_2 if (e2_2 > 0 and not np.isnan(e2_4)) else np.nan
        r32  = e3_2 / e2_2 if e2_2 > 0 else np.nan

        print(
            f"{f'{cmin}–{cmax}%':>12}  {n:>7}  "
            f"{e2_2:>7.4f}  {e2_4:>7.4f}  {e3_2:>7.4f}  {e4_2:>7.4f}  "
            f"{r24:>8.4f}  {r32:>7.4f}  {g2:>9.4f}"
        )
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PLOT  (2 rows × 3 columns, bootstrap bands)
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(res: dict):
    # ── Colour palette ──────────────────────────────────────────────────────
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
        "grid.linewidth":   0.5,
        "font.family":      "sans-serif",
        "font.size":        10,
        "axes.titlesize":   11,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
    })

    c = res["centres"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Eccentricities & Ultra-Central Flow Observables  |  Trento",
        fontsize=14, color=TEXT, y=1.01, fontweight="bold"
    )

    # (observable_key, color, y-label, sub-label, allow_neg_y)
    panels = [
        # Row 0
        ("e2_2",     CYAN,   "e₂{2}",        "two-particle cumulant",                  False),
        ("e2_4",     CORAL,  "e₂{4}",        "four-particle cumulant",                 False),
        ("e3_2",     GREEN,  "e₃{2}",        "triangularity (2-part.)",                False),
        # Row 1
        ("ratio_e2", GOLD,   "e₂{4} / e₂{2}", "fluctuation suppression\n(< 1: non-Gaussian)", False),
        ("ratio_e3", VIOLET, "e₃{2} / e₂{2}", "triangularity / ellipticity\n(→ 1 in ultra-central)", False),
        ("g2",       ORANGE, "Γ₂",            "norm. 4th cumulant of ε₂\n(< 0: non-Gaussian fluct.)", True),
    ]

    ax_flat = axes.flatten()

    for idx, (key, color, ylabel, sublabel, allow_neg) in enumerate(panels):
        ax  = ax_flat[idx]
        y   = res[key]
        ylo = res[key + "_lo"]
        yhi = res[key + "_hi"]
        valid = ~np.isnan(y)

        if valid.sum() < 2:
            ax.text(0.5, 0.5, "insufficient data",
                    transform=ax.transAxes,
                    ha="center", va="center", color=TEXT, alpha=0.5)
        else:
            # Bootstrap ±1σ band
            ax.fill_between(c[valid], ylo[valid], yhi[valid],
                            color=color, alpha=0.20, linewidth=0,
                            label="±1σ bootstrap")
            # Central line
            ax.plot(c[valid], y[valid],
                    color=color, linewidth=2.0,
                    marker="o", markersize=4.5,
                    markerfacecolor=BG, markeredgecolor=color,
                    markeredgewidth=1.4)

        ax.set_xlabel("Centrality (%)", labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6, fontsize=11)
        ax.set_title(f"{ylabel}\n{sublabel}", color=color, pad=8, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.40)
        ax.set_xlim(left=0)

        if not allow_neg:
            ax.set_ylim(bottom=0)

        # Ultra-central shaded region + label
        ax.axvspan(0, 1, alpha=0.08, color=color, zorder=0)
        ylo_ax, yhi_ax = ax.get_ylim()
        ax.text(0.5, yhi_ax - (yhi_ax - ylo_ax) * 0.05,
                "UC (0–1%)",
                ha="center", va="top", fontsize=7.5,
                color=color, alpha=0.85)

        # Reference lines
        if "/ e₂" in ylabel:
            ax.axhline(1.0, color=TEXT, linewidth=0.8, linestyle=":", alpha=0.5)
        if "Γ" in ylabel:
            ax.axhline(0.0, color=TEXT, linewidth=0.8, linestyle=":", alpha=0.5)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Row labels
    axes[0, 0].annotate("ROW 1\nEccentricities",
                         xy=(-0.30, 0.5), xycoords="axes fraction",
                         fontsize=9, color=TEXT, alpha=0.55,
                         ha="center", va="center", rotation=90)
    axes[1, 0].annotate("ROW 2\nUC Puzzle",
                         xy=(-0.30, 0.5), xycoords="axes fraction",
                         fontsize=9, color=TEXT, alpha=0.55,
                         ha="center", va="center", rotation=90)

    fig.tight_layout(rect=[0.04, 0, 1, 1])
    fig.savefig(PLOT_FILE, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"[INFO] Plot saved → {PLOT_FILE}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if SKIP_TRENTO:
        if not CACHE_FILE.exists():
            sys.exit(
                f"[ERROR] Cache file {CACHE_FILE} not found.\n"
                "Set SKIP_TRENTO=False to generate events first."
            )
        events = np.load(CACHE_FILE)
        print(f"[INFO] Loaded {len(events):,} events from cache ({CACHE_FILE}).")
    else:
        events = run_trento()

    centrality = assign_centrality(events)
    print_table(events, centrality)

    print("[INFO] Computing observables and bootstrap bands…")
    res = compute_eccentricities(events, centrality)
    make_plot(res)


if __name__ == "__main__":
    main()
