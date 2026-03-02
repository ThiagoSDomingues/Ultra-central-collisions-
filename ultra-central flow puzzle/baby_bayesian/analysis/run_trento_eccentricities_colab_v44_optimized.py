#!/usr/bin/env python3
"""
run_trento_eccentricities_v4_v2.py
------------------------------------
Optimised version of the extended Trento eccentricity pipeline
(ε₄{4}⁴ / ultra-central puzzle).

Fixes in this version
----------------------
1. ValueError in print_table: removed '+' sign specifier from string header
   format ('>+10' → '>10'). The '+' flag is only valid for numeric types.
2. SKIP_TRENTO=True now works correctly: loads events_summary.npy, skips
   Trento entirely, and goes straight to print_table + plot.
   Set SKIP_TRENTO = True in CONFIGURATION to reuse a cached run.
"""

import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these
# ─────────────────────────────────────────────────────────────────────────────
HDF1        = "WS1.hdf"
HDF2        = "WS2.hdf"
SKIP_TRENTO = False           # ← set True to reuse events_summary.npy

TRENTO_BIN  = "/root/.local/bin/trento"
N_EVENTS    = 1_000_000 # 1 million events
RANDOM_SEED = 42

P_PARAM = 0.063
K_PARAM = 1 / 1.05**2       # ≈ 0.9070
W_PARAM = 1.12
D_PARAM = 2.97
N_PARAM = 14.2

CACHE_FILE = Path("events_summary.npy")
PLOT_FILE  = "eccentricities_vs_centrality.pdf"

N_BOOT = 200   # bootstrap resamples per centrality bin

CENTRALITY_EDGES = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7.5,
                    10, 15, 20, 25, 30, 40, 50, 60, 70, 80]

HIGHLIGHT_BINS = [(0, 0.1), (0, 0.5), (0, 1), (1, 5),
                  (5, 10), (10, 20), (20, 30)]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RUN TRENTO  (stdout-only, no .dat files)
# ─────────────────────────────────────────────────────────────────────────────

def run_trento() -> np.ndarray:
    """
    Execute Trento and parse event summaries directly from stdout.
    Omitting -o means Trento writes no per-event files.
    """
    cmd = [
        TRENTO_BIN, HDF1, HDF2, str(N_EVENTS),
        "-p", str(P_PARAM),
        "-k", str(K_PARAM),
        "-w", str(W_PARAM),
        "-d", str(D_PARAM),
        "-n", str(N_PARAM),
        "--random-seed", str(RANDOM_SEED),
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
    rank = np.argsort(np.argsort(-events[:, 3]))
    return rank / len(events) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — OBSERVABLE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _moments(eps: np.ndarray, orders=(2, 4, 6, 8)) -> dict:
    return {n: np.mean(eps**n) for n in orders}


def e2_cumulants(m: dict):
    m2, m4, m6, m8 = m[2], m[4], m[6], m[8]

    e2 = np.sqrt(m2)

    inner4 = 2.0 * m2**2 - m4
    e4     = inner4**0.25 if inner4 >= 0 else np.nan

    c6 = m6 - 9.0 * m4 * m2 + 12.0 * m2**3
    e6 = (-c6 / 4.0)**(1.0/6.0) if -c6 / 4.0 >= 0 else np.nan

    c8 = m8 - 16.0*m6*m2 - 18.0*m4**2 + 144.0*m4*m2**2 - 144.0*m2**4
    e8 = (c8 / 33.0)**(1.0/8.0) if c8 / 33.0 >= 0 else np.nan

    return e2, e4, e6, e8


def e4_4_cumulant(eps4: np.ndarray) -> float:
    if len(eps4) < 4:
        return np.nan
    m2 = np.mean(eps4**2)
    m4 = np.mean(eps4**4)
    return float(2.0 * m2**2 - m4)


def nl_sign_term(m2_eps2: dict):
    m4 = m2_eps2[4]
    m8 = m2_eps2[8]
    raw   = float(2.0 * m4**2 - m8)
    ratio = float(m8 / m4**2) if m4 > 0 else np.nan
    return raw, ratio


def e4_psi2(eps2: np.ndarray, eps4: np.ndarray) -> float:
    if len(eps2) < 4:
        return np.nan
    denom = np.sqrt(np.mean(eps2**4))
    return float(np.mean(eps4 * eps2**2) / denom) if denom > 0 else np.nan


def gamma2_obs(m: dict) -> float:
    m2, m4 = m[2], m[4]
    return float((m4 - 2.0 * m2**2) / m2**2) if m2 > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BOOTSTRAP UNCERTAINTY
# ─────────────────────────────────────────────────────────────────────────────

_OBS_KEYS = [
    "e2_2", "e2_4", "e2_6", "e2_8",
    "e3_2", "e4_2", "e4_4c",
    "ratio_e2", "ratio_e3",
    "e4p2", "nl_raw", "nl_r8", "g2",
]


def _compute_all(sel: np.ndarray) -> np.ndarray:
    eps2 = sel[:, 4]
    eps3 = sel[:, 5]
    eps4 = sel[:, 6] if sel.shape[1] > 6 else np.zeros(len(sel))

    m2  = _moments(eps2, (2, 4, 6, 8))
    e22, e24, e26, e28 = e2_cumulants(m2)
    e32 = float(np.sqrt(np.mean(eps3**2))) if len(eps3) >= 2 else np.nan
    e42 = float(np.sqrt(np.mean(eps4**2))) if len(eps4) >= 2 else np.nan
    e44c        = e4_4_cumulant(eps4)
    nl_raw, nl_r8 = nl_sign_term(m2)
    e4p2        = e4_psi2(eps2, eps4)
    g2          = gamma2_obs(m2)
    r24         = e24 / e22 if (e22 > 0 and not np.isnan(e24)) else np.nan
    r32         = e32 / e22 if e22 > 0 else np.nan

    return np.array([e22, e24, e26, e28,
                     e32, e42, e44c,
                     r24, r32,
                     e4p2, nl_raw, nl_r8, g2])


def _bootstrap_bin(sel: np.ndarray, n_boot: int = N_BOOT):
    central = _compute_all(sel)
    n       = len(sel)
    rng     = np.random.default_rng(seed=0)
    boot    = np.empty((n_boot, len(_OBS_KEYS)))
    for i in range(n_boot):
        idx     = rng.integers(0, n, size=n)
        boot[i] = _compute_all(sel[idx])

    lo = np.nanpercentile(boot, 16, axis=0)
    hi = np.nanpercentile(boot, 84, axis=0)

    return {k: (central[j], lo[j], hi[j])
            for j, k in enumerate(_OBS_KEYS)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — COMPUTE ALL OBSERVABLES VS CENTRALITY
# ─────────────────────────────────────────────────────────────────────────────

def compute_eccentricities(events: np.ndarray, centrality: np.ndarray) -> dict:
    edges   = CENTRALITY_EDGES
    centres = np.array([(edges[i] + edges[i+1]) / 2
                        for i in range(len(edges) - 1)])

    store = {k: {"c": [], "lo": [], "hi": []} for k in _OBS_KEYS}
    ns    = []

    for i in range(len(edges) - 1):
        mask = (centrality >= edges[i]) & (centrality < edges[i+1])
        sel  = events[mask]
        ns.append(len(sel))

        if len(sel) < 10:
            for k in _OBS_KEYS:
                store[k]["c"].append(np.nan)
                store[k]["lo"].append(np.nan)
                store[k]["hi"].append(np.nan)
            continue

        res = _bootstrap_bin(sel)
        for k in _OBS_KEYS:
            store[k]["c"].append(res[k][0])
            store[k]["lo"].append(res[k][1])
            store[k]["hi"].append(res[k][2])

    result = {"centres": centres, "n": np.array(ns)}
    for k in _OBS_KEYS:
        result[k]         = np.array(store[k]["c"])
        result[k + "_lo"] = np.array(store[k]["lo"])
        result[k + "_hi"] = np.array(store[k]["hi"])

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_table(events: np.ndarray, centrality: np.ndarray):
    # FIX: removed '+' from string format specifiers (only valid for numerics)
    hdr = (
        f"{'Centrality':>12}  {'N':>7}  "
        f"{'e2{2}':>7}  {'e2{4}':>7}  {'e2{6}':>7}  {'e2{8}':>7}  "
        f"{'e4{4}^4':>10}  {'e4{Psi2}':>9}  "
        f"{'m8/m4^2':>10}  {'Gamma2':>9}"
    )
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")

    for cmin, cmax in HIGHLIGHT_BINS:
        mask = (centrality >= cmin) & (centrality < cmax)
        sel  = events[mask]
        n    = len(sel)
        if n < 8:
            print(f"{f'{cmin}-{cmax}%':>12}  {n:>7}  (insufficient statistics)")
            continue

        obs = _compute_all(sel)
        d   = dict(zip(_OBS_KEYS, obs))
        print(
            f"{f'{cmin}-{cmax}%':>12}  {n:>7}  "
            f"{d['e2_2']:>7.4f}  {d['e2_4']:>7.4f}  "
            f"{d['e2_6']:>7.4f}  {d['e2_8']:>7.4f}  "
            f"{d['e4_4c']:>+10.6f}  {d['e4p2']:>9.4f}  "
            f"{d['nl_r8']:>10.4f}  {d['g2']:>9.4f}"
        )
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — PLOT  (3 rows × 3 columns, bootstrap bands)
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(res: dict):
    BG     = "#0d0f14";  PANEL  = "#13161e";  GRID   = "#1e2230"
    TEXT   = "#d8dce8";  CYAN   = "#4fc3f7";  CORAL  = "#ef5350"
    GREEN  = "#66bb6a";  GOLD   = "#ffd54f";  VIOLET = "#ce93d8"
    ORANGE = "#ffa726";  RED    = "#ff1744";  TEAL   = "#26c6da"

    plt.rcParams.update({
        "figure.facecolor": BG,    "axes.facecolor":  PANEL,
        "axes.edgecolor":   GRID,  "axes.labelcolor": TEXT,
        "xtick.color":      TEXT,  "ytick.color":     TEXT,
        "text.color":       TEXT,  "grid.color":      GRID,
        "grid.linewidth":   0.5,   "font.family":     "sans-serif",
        "font.size":        10,    "axes.titlesize":  10,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID,
    })

    c = res["centres"]

    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    fig.suptitle(
        "Eccentricities & v₄{4}⁴ Observables  |  Trento\n"
        "Row 1: standard  |  Row 2: ε₂ cumulant hierarchy  |  Row 3: v₄{4}⁴ puzzle",
        fontsize=12, color=TEXT, y=1.01, fontweight="bold"
    )

    panels = [
        # Row 1
        ("e2_2",  CYAN,   "ε₂{2}",           "ellipticity 2-part.",                   False),
        ("e3_2",  GREEN,  "ε₃{2}",           "triangularity 2-part.",                 False),
        ("e4_2",  CORAL,  "ε₄{2}",           "hexadecapole 2-part.",                  False),
        # Row 2
        ("e2_4",  GOLD,   "ε₂{4}",           "4-particle cumulant",                   False),
        ("e2_6",  VIOLET, "ε₂{6}",           "6-particle cumulant",                   False),
        ("e2_8",  ORANGE, "ε₂{8}",           "8-particle cumulant",                   False),
        # Row 3
        ("e4_4c", RED,    "ε₄{4}⁴",          "2⟨ε₄²⟩²−⟨ε₄⁴⟩  (can go negative!)",  True),
        ("nl_r8", TEAL,   "⟨ε₂⁸⟩/⟨ε₂⁴⟩²",  "sign-change trigger\n(> 2 → neg. v₄{4}⁴)", False),
        ("e4p2",  ORANGE, "ε₄{Ψ₂}",          "nonlinear coupling\nε₄ → ε₂²",          False),
    ]

    ax_flat = axes.flatten()

    for idx, (key, color, ylabel, sublabel, allow_neg) in enumerate(panels):
        ax  = ax_flat[idx]
        y   = res[key]
        ylo = res[key + "_lo"]
        yhi = res[key + "_hi"]
        valid = ~np.isnan(y)

        if valid.sum() >= 2:
            ax.fill_between(c[valid], ylo[valid], yhi[valid],
                            color=color, alpha=0.20, linewidth=0,
                            label="±1σ bootstrap")
            ax.plot(c[valid], y[valid],
                    color=color, linewidth=2.0,
                    marker="o", markersize=4.5,
                    markerfacecolor=BG, markeredgecolor=color,
                    markeredgewidth=1.4)
        else:
            ax.text(0.5, 0.5, "insufficient data",
                    transform=ax.transAxes,
                    ha="center", va="center", color=TEXT, alpha=0.5)

        ax.set_xlabel("Centrality (%)", labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6, fontsize=11)
        ax.set_title(f"{ylabel}  —  {sublabel}", color=color, pad=8, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.40)
        ax.set_xlim(left=0)

        if not allow_neg:
            ax.set_ylim(bottom=0)

        if "ε₂⁸" in ylabel:
            ax.axhline(2.0, color=TEXT, linewidth=1.2,
                       linestyle="--", alpha=0.7, label="threshold = 2")
            ax.legend(fontsize=8)
        if "ε₄{4}" in ylabel:
            ax.axhline(0.0, color=TEXT, linewidth=1.0,
                       linestyle=":", alpha=0.6)
            ax.text(0.98, 0.05, "sign change\nbelow this line",
                    transform=ax.transAxes, ha="right", fontsize=8,
                    color=TEXT, alpha=0.6)

        ax.axvspan(0, 1, alpha=0.08, color=color, zorder=0)
        ylo_ax, yhi_ax = ax.get_ylim()
        ax.text(0.5, yhi_ax - (yhi_ax - ylo_ax) * 0.05,
                "UC (0–1%)",
                ha="center", va="top", fontsize=7.5,
                color=color, alpha=0.85)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    for row_idx, label in enumerate([
        "ROW 1\nStandard",
        "ROW 2\nε₂ cumulants",
        "ROW 3\nv₄{4}⁴ puzzle",
    ]):
        axes[row_idx, 0].annotate(
            label,
            xy=(-0.30, 0.5), xycoords="axes fraction",
            fontsize=8, color=TEXT, alpha=0.55,
            ha="center", va="center", rotation=90,
        )

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
