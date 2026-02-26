#!/usr/bin/env python3
"""
run_trento_eccentricities.py  — extended for v4{4}^4 / ultracentral puzzle
"""

import subprocess, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
HDF1        = "WS1.hdf"
HDF2        = "WS2.hdf"
SKIP_TRENTO = False

TRENTO_BIN  = "/root/.local/bin/trento"
N_EVENTS    = 50000
RANDOM_SEED = 42

P_PARAM = 0.063
K_PARAM = 1 / 1.05**2
W_PARAM = 1.12
D_PARAM = 2.97
N_PARAM = 14.2

OUTPUT_DIR = Path("trento_events")
CACHE_FILE = OUTPUT_DIR / "events_summary.npy"
PLOT_FILE  = "eccentricities_vs_centrality.pdf"

CENTRALITY_EDGES = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 7.5,
                    10, 15, 20, 25, 30, 40, 50, 60, 70, 80]

HIGHLIGHT_BINS = [(0, 0.1), (0, 0.5), (0, 1), (1, 5),
                  (5, 10), (10, 20), (20, 30)]


# ─────────────────────────────────────────────
# STEP 1 — RUN TRENTO
# ─────────────────────────────────────────────

def run_trento(hdf1, hdf2):
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)
    cmd = [
        TRENTO_BIN, hdf1, hdf2, str(N_EVENTS),
        "-p", str(P_PARAM), "-k", str(K_PARAM),
        "-w", str(W_PARAM), "-d", str(D_PARAM),
        "-n", str(N_PARAM),
        "--random-seed", str(RANDOM_SEED),
        "-o", str(OUTPUT_DIR),
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Trento failed.\nstderr:\n{result.stderr}")

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
        raise RuntimeError("No summary lines found in Trento stdout.")

    arr = np.array(rows)
    np.save(CACHE_FILE, arr)
    print(f"[INFO] {len(arr)} events loaded. Cached to {CACHE_FILE}")
    return arr


# ─────────────────────────────────────────────
# STEP 2 — CENTRALITY
# ─────────────────────────────────────────────

def assign_centrality(events):
    mult = events[:, 3]
    rank = np.argsort(np.argsort(-mult))
    return rank / len(events) * 100

def select_bin(events, centrality, cmin, cmax):
    return events[(centrality >= cmin) & (centrality < cmax)]


# ─────────────────────────────────────────────
# STEP 3 — CUMULANTS (2, 4, 6, 8-particle)
# ─────────────────────────────────────────────

def cumulants_2_4(eps):
    """Standard e{2} and e{4}."""
    if len(eps) < 4:
        return np.nan, np.nan
    m2 = np.mean(eps**2)
    m4 = np.mean(eps**4)
    e2 = np.sqrt(m2)
    inner = 2*m2**2 - m4
    e4 = inner**0.25 if inner >= 0 else np.nan
    return e2, e4

def cumulant_6(eps):
    """e{6}: sixth-order cumulant eccentricity."""
    if len(eps) < 6:
        return np.nan
    m2 = np.mean(eps**2)
    m4 = np.mean(eps**4)
    m6 = np.mean(eps**6)
    # c6 = <e^6> - 9<e^4><e^2> + 12<e^2>^3
    c6 = m6 - 9*m4*m2 + 12*m2**3
    return (-c6/4)**( 1/6) if -c6/4 >= 0 else np.nan

def cumulant_8(eps):
    """e{8}: eighth-order cumulant eccentricity."""
    if len(eps) < 8:
        return np.nan
    m2 = np.mean(eps**2)
    m4 = np.mean(eps**4)
    m6 = np.mean(eps**6)
    m8 = np.mean(eps**8)
    # c8 = <e^8> - 16<e^6><e^2> - 18<e^4>^2 + 144<e^4><e^2>^2 - 144<e^2>^4
    c8 = m8 - 16*m6*m2 - 18*m4**2 + 144*m4*m2**2 - 144*m2**4
    return (c8/33)**(1/8) if c8/33 >= 0 else np.nan

def e4_fourth_cumulant(eps4):
    """
    ε₄{4}⁴ = 2<ε₄²>² - <ε₄⁴>
    Direct eccentricity analog of v₄{4}⁴.
    Can be negative — that's the observable of interest!
    """
    if len(eps4) < 4:
        return np.nan
    m2 = np.mean(eps4**2)
    m4 = np.mean(eps4**4)
    return 2*m2**2 - m4

def nonlinear_sign_term(eps2):
    """
    2<ε₂⁴>² - <ε₂⁸>  (from Eq. 3 in the paper, with ε instead of v)
    This is the term that goes negative when ε₂ fluctuations are large.
    Normalized as: (2<ε₂⁴>² - <ε₂⁸>) / <ε₂⁴>²  = 2 - <ε₂⁸>/<ε₂⁴>²
    """
    if len(eps2) < 8:
        return np.nan, np.nan
    m4 = np.mean(eps2**4)   # = <ε₂²²> = <ε₂⁴>
    m8 = np.mean(eps2**8)   # = <ε₂⁸>
    raw       = 2*m4**2 - m8
    ratio_m8  = m8 / m4**2 if m4 > 0 else np.nan   # should be < 2 for positive
    return raw, ratio_m8

def nonlinear_eccentricity_e4psi2(eps2, eps4):
    """
    ε₄{Ψ₂} = |<ε₄ · ε₂*²>| / sqrt(<ε₂⁴>)
    Eccentricity analog of v₄{Ψ₂}: isolates the nonlinear coupling.
    Here we use real parts only since Trento outputs magnitudes.
    Approximation: <ε₄ ε₂²> ≈ <ε₄ ε₂²> (treating magnitudes).
    """
    if len(eps2) < 4:
        return np.nan
    num   = np.mean(eps4 * eps2**2)
    denom = np.sqrt(np.mean(eps2**4))
    return num / denom if denom > 0 else np.nan

def gamma2(eps):
    """Normalized 4th cumulant: Γ₂ = (<ε²⁴> - 2<ε₂²>²) / <ε₂²>²"""
    if len(eps) < 4:
        return np.nan
    m2 = np.mean(eps**2)
    m4 = np.mean(eps**4)
    if m2 == 0:
        return np.nan
    return (m4 - 2*m2**2) / m2**2


# ─────────────────────────────────────────────
# STEP 4 — COMPUTE ALL OBSERVABLES VS CENTRALITY
# ─────────────────────────────────────────────

def compute_eccentricities(events, centrality):
    edges   = CENTRALITY_EDGES
    centres = np.array([(edges[i]+edges[i+1])/2 for i in range(len(edges)-1)])

    keys = [
        "e2_2", "e2_4", "e2_6", "e2_8",
        "e3_2", "e4_2", "e4_4_cumulant",
        "ratio_e2_4_e2_2", "ratio_e3_e2",
        "e4psi2",
        "nl_raw", "nl_ratio_m8",   # 2<e2^4>^2 - <e2^8> and <e2^8>/<e2^4>^2
        "gamma2", "n",
    ]
    results = {k: [] for k in keys}

    for i in range(len(edges)-1):
        sel = select_bin(events, centrality, edges[i], edges[i+1])
        results["n"].append(len(sel))

        if len(sel) < 10:
            for k in keys:
                if k != "n":
                    results[k].append(np.nan)
            continue

        eps2 = sel[:, 4]
        eps3 = sel[:, 5]
        eps4 = sel[:, 6] if sel.shape[1] > 6 else np.full(len(sel), np.nan)

        e2_2, e2_4      = cumulants_2_4(eps2)
        e2_6            = cumulant_6(eps2)
        e2_8            = cumulant_8(eps2)
        e3_2, _         = cumulants_2_4(eps3)
        e4_2, _         = cumulants_2_4(eps4)
        e4_4c           = e4_fourth_cumulant(eps4)
        nl_raw, nl_r8   = nonlinear_sign_term(eps2)
        e4p2            = nonlinear_eccentricity_e4psi2(eps2, eps4)
        g2              = gamma2(eps2)

        results["e2_2"].append(e2_2)
        results["e2_4"].append(e2_4)
        results["e2_6"].append(e2_6)
        results["e2_8"].append(e2_8)
        results["e3_2"].append(e3_2)
        results["e4_2"].append(e4_2)
        results["e4_4_cumulant"].append(e4_4c)
        results["ratio_e2_4_e2_2"].append(
            e2_4/e2_2 if (e2_2 > 0 and not np.isnan(e2_4)) else np.nan)
        results["ratio_e3_e2"].append(
            e3_2/e2_2 if e2_2 > 0 else np.nan)
        results["e4psi2"].append(e4p2)
        results["nl_raw"].append(nl_raw)
        results["nl_ratio_m8"].append(nl_r8)
        results["gamma2"].append(g2)

    return {"centres": centres,
            **{k: np.array(v) for k, v in results.items()}}


# ─────────────────────────────────────────────
# STEP 5 — PRINT TABLE
# ─────────────────────────────────────────────

def print_table(events, centrality):
    hdr = (f"{'Centrality':>12}  {'N':>6}  {'e2{2}':>7}  {'e2{4}':>7}  "
           f"{'e2{6}':>7}  {'e2{8}':>7}  {'e4{4}^4':>9}  "
           f"{'e4{Ψ2}':>8}  {'<e2^8>/<e2^4>^2':>16}  {'Gamma2':>8}")
    print("\n" + "="*len(hdr))
    print(hdr)
    print("="*len(hdr))
    for cmin, cmax in HIGHLIGHT_BINS:
        sel = select_bin(events, centrality, cmin, cmax)
        if len(sel) < 8:
            print(f"{f'{cmin}-{cmax}%':>12}  {len(sel):>6}  (insufficient statistics)")
            continue
        eps2 = sel[:, 4]
        eps4 = sel[:, 6] if sel.shape[1] > 6 else np.full(len(sel), np.nan)
        e2_2, e2_4 = cumulants_2_4(eps2)
        e2_6       = cumulant_6(eps2)
        e2_8       = cumulant_8(eps2)
        e4_4c      = e4_fourth_cumulant(eps4)
        _, nl_r8   = nonlinear_sign_term(eps2)
        e4p2       = nonlinear_eccentricity_e4psi2(eps2, eps4)
        g2         = gamma2(eps2)
        print(f"{f'{cmin}-{cmax}%':>12}  {len(sel):>6}  "
              f"{e2_2:>7.4f}  {e2_4:>7.4f}  {e2_6:>7.4f}  {e2_8:>7.4f}  "
              f"{e4_4c:>+9.6f}  {e4p2:>8.4f}  {nl_r8:>16.4f}  {g2:>8.4f}")
    print("="*len(hdr) + "\n")


# ─────────────────────────────────────────────
# STEP 6 — PLOT (3 rows x 3 columns)
# ─────────────────────────────────────────────

def make_plot(res):
    BG     = "#0d0f14";  PANEL = "#13161e";  GRID  = "#1e2230"
    TEXT   = "#d8dce8";  CYAN  = "#4fc3f7";  CORAL = "#ef5350"
    GREEN  = "#66bb6a";  GOLD  = "#ffd54f";  VIOLET= "#ce93d8"
    ORANGE = "#ffa726";  RED   = "#ff1744";  TEAL  = "#26c6da"

    plt.rcParams.update({
        "figure.facecolor": BG,   "axes.facecolor": PANEL,
        "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT,      "ytick.color": TEXT,
        "text.color":  TEXT,      "grid.color":  GRID,
        "grid.linewidth": 0.6,    "font.family": "monospace",
        "font.size": 10,          "axes.titlesize": 10,
        "legend.facecolor": PANEL,"legend.edgecolor": GRID,
    })

    c = res["centres"]

    fig, axes = plt.subplots(3, 3, figsize=(17, 14))
    fig.suptitle(
        "Eccentricities & v₄{4}⁴ Observables  |  Trento\n"
        "Row 1: standard  |  Row 2: higher-order e₂ cumulants  |  Row 3: v₄{4}⁴ puzzle",
        fontsize=12, color=TEXT, y=1.01, fontweight="bold")

    panels = [
        # Row 1 — standard
        (axes[0,0], res["e2_2"],             CYAN,   "e2{2}",
         "ellipticity 2-part.", False),
        (axes[0,1], res["e3_2"],             GREEN,  "e3{2}",
         "triangularity 2-part.", False),
        (axes[0,2], res["e4_2"],             CORAL,  "e4{2}",
         "hexadecapole 2-part.", False),

        # Row 2 — higher e2 cumulants (control sign of v4{4}^4)
        (axes[1,0], res["e2_4"],             GOLD,   "e2{4}",
         "4-part cumulant eccentricity", False),
        (axes[1,1], res["e2_6"],             VIOLET, "e2{6}",
         "6-part cumulant eccentricity", False),
        (axes[1,2], res["e2_8"],             ORANGE, "e2{8}",
         "8-part cumulant eccentricity", False),

        # Row 3 — v4{4}^4 puzzle observables
        (axes[2,0], res["e4_4_cumulant"],    RED,    "ε₄{4}⁴",
         "2<ε₄²>²−<ε₄⁴>  (can go negative!)", None),
        (axes[2,1], res["nl_ratio_m8"],      TEAL,   "<ε₂⁸>/<ε₂⁴>²",
         "sign-change trigger\n(negative v₄{4}⁴ when > 2)", False),
        (axes[2,2], res["e4psi2"],           ORANGE, "ε₄{Ψ₂}",
         "nonlinear coupling\nε₄ → ε₂²", False),
    ]

    for (ax, y, color, label, sublabel, ylim_flag) in panels:
        valid = ~np.isnan(y)
        if valid.sum() >= 2:
            ax.fill_between(c[valid], y[valid]*0.95, y[valid]*1.05,
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(c[valid], y[valid], color=color, linewidth=2.2,
                    marker="o", markersize=4,
                    markerfacecolor=BG, markeredgecolor=color,
                    markeredgewidth=1.5)
        else:
            ax.text(0.5, 0.5, "insufficient data",
                    transform=ax.transAxes, ha="center",
                    va="center", color=TEXT, alpha=0.5)

        ax.set_xlabel("Centrality (%)", labelpad=6)
        ax.set_ylabel(label, labelpad=6, fontsize=11)
        ax.set_title(f"{label}  —  {sublabel}", color=color, pad=8, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_xlim(left=0)
        if ylim_flag is False:
            ax.set_ylim(bottom=0)
        # ylim_flag=None: allow negative (ε₄{4}⁴ can go negative)

        # Key reference lines
        if "<ε₂⁸>" in label or "sign" in sublabel:
            ax.axhline(2.0, color=TEXT, linewidth=1.2,
                       linestyle="--", alpha=0.7,
                       label="threshold = 2")
            ax.legend(fontsize=8)
        if "ε₄{4}" in label:
            ax.axhline(0.0, color=TEXT, linewidth=1.0,
                       linestyle=":", alpha=0.6)
            ax.text(0.98, 0.05, "sign change\nbelow this line",
                    transform=ax.transAxes, ha="right", fontsize=8,
                    color=TEXT, alpha=0.6)

        # Ultra-central shading
        ax.axvspan(0, 1, alpha=0.07, color=color)
        ylo, yhi = ax.get_ylim()
        ax.text(0.5, yhi - (yhi-ylo)*0.06, "UC",
                ha="center", va="top", fontsize=8, color=color, alpha=0.8)

        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Row labels
    for row_idx, label in enumerate(["ROW 1\nStandard", "ROW 2\nε₂ cumulants", "ROW 3\nv₄{4}⁴ puzzle"]):
        axes[row_idx, 0].annotate(label,
            xy=(-0.30, 0.5), xycoords="axes fraction",
            fontsize=8, color=TEXT, alpha=0.6,
            ha="center", va="center", rotation=90)

    fig.tight_layout(rect=[0.04, 0, 1, 1])
    fig.savefig(PLOT_FILE, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"[INFO] Plot saved -> {PLOT_FILE}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if SKIP_TRENTO:
        if not CACHE_FILE.exists():
            raise RuntimeError(f"Cache {CACHE_FILE} not found.")
        events = np.load(CACHE_FILE)
        print(f"[INFO] Loaded {len(events)} events from cache.")
    else:
        events = run_trento(HDF1, HDF2)

    centrality = assign_centrality(events)
    print_table(events, centrality)
    res = compute_eccentricities(events, centrality)
    make_plot(res)

main()
