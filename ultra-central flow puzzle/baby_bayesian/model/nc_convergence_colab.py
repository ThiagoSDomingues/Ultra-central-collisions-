#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  nc_convergence_colab.py                                                    ║
# ║                                                                              ║
# ║  Convergence study for the normalised 4-particle cumulant                   ║
# ║                                                                              ║
# ║    nc_n{4,ε} = c_n{4,ε}/c_n{2,ε}^2 = <εₙ⁴>/<εₙ²>² − 2                   ║
# ║                                                                              ║
# ║  for n = 3 and n = 4 in ultra-central Pb+Pb @ 2.76 TeV,                    ║
# ║  using STANDARD TRENTo (no Isobar Sampler, no trento_sync).                 ║
# ║                                                                              ║
# ║  Strategy                                                                    ║
# ║  ────────                                                                    ║
# ║  1. Build TRENTo from Duke-QCD/trento (Release mode).                       ║
# ║  2. Choose b_max = b_{10%} ≈ 4.95 fm so that ALL generated events           ║
# ║     land in the 0–10% centrality range — 100 % efficiency.                  ║
# ║  3. Generate N_TOTAL = 5 000 000 events in one shot.                        ║
# ║  4. Rank-sort by multiplicity → centrality; split into 1 %-wide UC bins.    ║
# ║  5. Plot nc_3 and nc_4 vs centrality (0–10 %, 1 % intervals).               ║
# ║  6. Convergence study: compute nc_3 and nc_4 in the 0–1 % bin as a          ║
# ║     function of N_events to find the minimum statistics needed.              ║
# ║                                                                              ║
# ║  Paste this entire file into ONE Colab cell and run.                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import subprocess, sys, os, shutil, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Physics ───────────────────────────────────────────────────────────────────
N_TOTAL     = 1_000_000      # total events to generate
SQRTS_TEV   = 2.76           # collision energy [TeV]
SIGMA_NN    = 6.4            # inelastic NN cross section [fm²] @ 2.76 TeV
SIGMA_INEL  = 770.0          # Pb+Pb total inelastic cross section [fm²]

# b_max = b_{10%}: the impact parameter that corresponds to the 10% centrality cut
# pi * b_{10%}^2 = 0.10 * sigma_inel  →  b_{10%} = sqrt(0.10 * sigma_inel / pi)
# This means ALL events generated with b ∈ [0, b_{10%}] are UC (0–10%),
# giving 100% efficiency. Using b_max = 20 fm would waste ~94% of events.
B_MAX_10PCT = float(np.sqrt(0.10 * SIGMA_INEL / np.pi))   # ≈ 4.951 fm

# TRENTo parameters (JETSCAPE Grad MAP, arXiv:2011.01430 Table VII)
TRENTO_P    = 0.063
TRENTO_K    = 1.0 / 0.97**2   # k = 1/σ_k²,  σ_k = 0.97
TRENTO_NORM = 18.12
TRENTO_W    = 0.9             # nucleon width [fm] (typical MAP value)
TRENTO_DMIN = 0.52**(1./3.)   # d_min from d³=0.52 fm³

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR   = Path("/content/nc_convergence")
TRENTO_REPO= Path("/content/trento")
TRENTO_BIN_CANDIDATES = [
    Path("/usr/local/bin/trento"),
    TRENTO_REPO / "build" / "src" / "trento",
    TRENTO_REPO / "build" / "trento",
]
EVENTS_FILE = WORK_DIR / "events_5M.npy"
OUTDIR      = WORK_DIR / "plots"

# ── Convergence study ─────────────────────────────────────────────────────────
# Sample sizes tested in the convergence study (number of events in 0–1% bin)
N_CONV_SAMPLES = [
    200, 500, 1_000, 2_000, 5_000,
    10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
]
N_BOOT = 300   # bootstrap replicates per sample size
SEED   = 42

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt(s):
    if s < 60:   return f"{s:.1f} s"
    if s < 3600: return f"{s/60:.1f} min"
    return f"{s/3600:.2f} h"

def run(cmd, cwd=None, check=True):
    cmd = [str(c) for c in cmd]
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed (exit {r.returncode})")
    return r

def find_trento():
    for p in TRENTO_BIN_CANDIDATES:
        if p.exists() and os.access(p, os.X_OK):
            return p
    r = subprocess.run(["which", "trento"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return Path(r.stdout.strip())
    return None

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 64)
print("  Stage 0 — Dependencies")
print("=" * 64)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "numpy", "matplotlib", "scipy"],
    check=True,
)
subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(
    ["apt-get", "install", "-y", "-qq",
     "build-essential", "cmake", "git", "libboost-all-dev"],
    check=True,
)
print("  OK")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import CubicSpline

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BUILD TRENTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 1 — TRENTo (Duke-QCD/trento)")
print("=" * 64)

TRENTO_BIN = find_trento()
if TRENTO_BIN:
    print(f"  Already built → {TRENTO_BIN}")
else:
    print("  Cloning and building Duke-QCD/trento …")
    if TRENTO_REPO.exists():
        shutil.rmtree(TRENTO_REPO)
    run(["git", "clone", "--depth=1",
         "https://github.com/Duke-QCD/trento.git",
         str(TRENTO_REPO)])
    build_dir = TRENTO_REPO / "build"
    build_dir.mkdir(exist_ok=True)
    # cmake: source dir as positional arg, Release for maximum performance
    run(["cmake", str(TRENTO_REPO), "-DCMAKE_BUILD_TYPE=Release"],
        cwd=build_dir)
    import multiprocessing
    t0 = time.perf_counter()
    run(["make", f"-j{multiprocessing.cpu_count()}"], cwd=build_dir)
    print(f"  Compiled in {time.perf_counter()-t0:.1f} s")
    subprocess.run(["make", "install"], cwd=str(build_dir), check=False)
    TRENTO_BIN = find_trento()
    if not TRENTO_BIN:
        raise RuntimeError("TRENTo binary not found after build")

ver = subprocess.run([str(TRENTO_BIN), "--version"], capture_output=True, text=True)
print(f"  Binary  : {TRENTO_BIN}")
print(f"  Version : {(ver.stdout + ver.stderr).strip().splitlines()[0]}")
print(f"\n  b_max = {B_MAX_10PCT:.4f} fm  (= b_{{10%}}  →  100% efficiency in 0–10%)")
print(f"  Generating {N_TOTAL:,} events …")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — GENERATE EVENTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 2 — Generate events")
print("=" * 64)

# Standard TRENTo output format (8 columns):
# ev  b  npart  mult  e2  e3  e4  e5
N_TRENTO_COLS = 8
COL_MULT = 3
COL_E2   = 4
COL_E3   = 5
COL_E4   = 6
COL_E5   = 7

if EVENTS_FILE.exists() and EVENTS_FILE.stat().st_size > 0:
    print(f"  Cached: {EVENTS_FILE}  ({EVENTS_FILE.stat().st_size/1e9:.2f} GB)")
    arr = np.load(EVENTS_FILE)
    if len(arr) != N_TOTAL:
        print(f"  Wrong size ({len(arr):,} ≠ {N_TOTAL:,}) — re-generating")
        EVENTS_FILE.unlink()

if not EVENTS_FILE.exists():
    cmd = [
        str(TRENTO_BIN), "Pb", "Pb", str(N_TOTAL),
        "-p", f"{TRENTO_P:.6f}",
        "-k", f"{TRENTO_K:.6f}",
        "-w", f"{TRENTO_W:.4f}",
        "-d", f"{TRENTO_DMIN:.6f}",
        "-n", f"{TRENTO_NORM:.6f}",
        "-x", f"{SIGMA_NN:.4f}",
        "--b-max",       f"{B_MAX_10PCT:.6f}",
        "--random-seed", "42",
    ]
    print(f"  Command: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    t_run  = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"TRENTo failed:\n{result.stderr[-500:]}")

    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < N_TRENTO_COLS:
            continue
        try:
            rows.append([float(x) for x in parts[:N_TRENTO_COLS]])
        except ValueError:
            continue

    if not rows:
        raise RuntimeError("No events parsed — check TRENTo output")

    arr = np.array(rows, dtype=np.float64)
    np.save(EVENTS_FILE, arr)
    rate = len(arr) / t_run
    print(f"  Generated: {len(arr):,} events in {fmt(t_run)}  ({rate:,.0f} evt/s)")
    print(f"  Saved → {EVENTS_FILE}")

print(f"\n  Array shape : {arr.shape}")
print(f"  N_events    : {len(arr):,}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CENTRALITY + CUMULANTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 3 — Centrality and cumulant ratios")
print("=" * 64)

mult = arr[:, COL_MULT]
eps2 = np.abs(arr[:, COL_E2])
eps3 = np.abs(arr[:, COL_E3])
eps4 = np.abs(arr[:, COL_E4])

# Rank-based centrality (same convention as the scan pipeline)
cent = np.argsort(np.argsort(-mult)) / len(mult) * 100.0

# Cumulant ratio: nc_n = <ε^4>/<ε²>² - 2
# = (2<ε²>² - <ε⁴>) / <ε²>²  with overall sign flip vs our old convention
# Physical meaning:
#   nc_n > 0  →  super-Gaussian (fluctuation-dominated), e.g. UC
#   nc_n < 0  →  sub-Gaussian  (geometry-dominated), e.g. semi-central
def nc(eps):
    """nc_n = <ε⁴>/<ε²>² − 2  (user's sign convention, eq. in the prompt)."""
    m2 = np.mean(eps**2)
    m4 = np.mean(eps**4)
    return m4 / m2**2 - 2.0

def nc_with_err(eps, n_boot=N_BOOT, rng_=None):
    """Returns (value, bootstrap_std)."""
    if rng_ is None:
        rng_ = np.random.default_rng(SEED)
    val  = nc(eps)
    N    = len(eps)
    boot = [nc(rng_.choice(eps, N, replace=True)) for _ in range(n_boot)]
    return val, float(np.std(boot))

UC_EDGES   = np.arange(0, 11, 1)
UC_CENTRES = 0.5 * (UC_EDGES[:-1] + UC_EDGES[1:])
N_UC_BINS  = len(UC_CENTRES)

print(f"\n  {'bin':7s}  {'N_ev':>8s}  {'nc_2':>10s} ± {'err':>7s}  "
      f"{'nc_3':>10s} ± {'err':>7s}  {'nc_4':>10s} ± {'err':>7s}")

nc2_vals = np.zeros(N_UC_BINS)
nc3_vals = np.zeros(N_UC_BINS)
nc4_vals = np.zeros(N_UC_BINS)
nc2_errs = np.zeros(N_UC_BINS)
nc3_errs = np.zeros(N_UC_BINS)
nc4_errs = np.zeros(N_UC_BINS)
n_per_bin= np.zeros(N_UC_BINS, dtype=int)

rng_global = np.random.default_rng(SEED)

for k, (lo, hi) in enumerate(zip(UC_EDGES[:-1], UC_EDGES[1:])):
    mask = (cent >= lo) & (cent < hi)
    n    = mask.sum()
    n_per_bin[k] = n

    v2, e2 = nc_with_err(eps2[mask], rng_=rng_global)
    v3, e3 = nc_with_err(eps3[mask], rng_=rng_global)
    v4, e4 = nc_with_err(eps4[mask], rng_=rng_global)

    nc2_vals[k] = v2;  nc2_errs[k] = e2
    nc3_vals[k] = v3;  nc3_errs[k] = e3
    nc4_vals[k] = v4;  nc4_errs[k] = e4

    print(f"  {lo:.0f}-{hi:.0f}%  {n:>8,}  "
          f"{v2:>+10.5f} ± {e2:>7.5f}  "
          f"{v3:>+10.5f} ± {e3:>7.5f}  "
          f"{v4:>+10.5f} ± {e4:>7.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — CONVERGENCE STUDY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 4 — Convergence study (0–1% bin)")
print("=" * 64)

mask_01   = (cent >= 0) & (cent < 1)
eps2_01   = eps2[mask_01]
eps3_01   = eps3[mask_01]
eps4_01   = eps4[mask_01]
N_01_avail= len(eps2_01)
print(f"  Available events in 0–1% bin: {N_01_avail:,}")

# True values at full statistics
nc2_true = nc(eps2_01)
nc3_true = nc(eps3_01)
nc4_true = nc(eps4_01)
print(f"  nc2 (true, N={N_01_avail:,}) = {nc2_true:+.6f}")
print(f"  nc3 (true, N={N_01_avail:,}) = {nc3_true:+.6f}")
print(f"  nc4 (true, N={N_01_avail:,}) = {nc4_true:+.6f}")

# Use only sample sizes up to what is available
sample_sizes = [ns for ns in N_CONV_SAMPLES if ns <= N_01_avail]

conv_nc2 = {"mean": [], "std": [], "bias": []}
conv_nc3 = {"mean": [], "std": [], "bias": []}
conv_nc4 = {"mean": [], "std": [], "bias": []}

print(f"\n  {'N':>8s}  {'nc2':>10s} ± {'σ':>7s}  "
      f"{'nc3':>10s} ± {'σ':>7s}  {'nc4':>10s} ± {'σ':>7s}")

rng_conv = np.random.default_rng(SEED + 1)

for ns in sample_sizes:
    # Draw N_BOOT bootstrap samples each of size ns
    idx_pool = rng_conv.integers(0, N_01_avail, size=(N_BOOT, ns))

    b2 = [nc(eps2_01[idx_pool[b]]) for b in range(N_BOOT)]
    b3 = [nc(eps3_01[idx_pool[b]]) for b in range(N_BOOT)]
    b4 = [nc(eps4_01[idx_pool[b]]) for b in range(N_BOOT)]

    for conv, boot, true in [
            (conv_nc2, b2, nc2_true),
            (conv_nc3, b3, nc3_true),
            (conv_nc4, b4, nc4_true),
    ]:
        conv["mean"].append(np.mean(boot))
        conv["std"].append(np.std(boot))
        conv["bias"].append(np.mean(boot) - true)

    print(f"  {ns:>8,}  "
          f"{conv_nc2['mean'][-1]:>+10.5f} ± {conv_nc2['std'][-1]:>7.5f}  "
          f"{conv_nc3['mean'][-1]:>+10.5f} ± {conv_nc3['std'][-1]:>7.5f}  "
          f"{conv_nc4['mean'][-1]:>+10.5f} ± {conv_nc4['std'][-1]:>7.5f}")

# Fit 1/sqrt(N) scaling for each observable
sample_arr = np.array(sample_sizes, dtype=float)
def fit_sqrt(n_arr, std_arr):
    """Fit σ = A / sqrt(N)  →  A = σ * sqrt(N) at largest available N."""
    A = std_arr[-1] * np.sqrt(n_arr[-1])
    return A

A2 = fit_sqrt(sample_arr, conv_nc2["std"])
A3 = fit_sqrt(sample_arr, conv_nc3["std"])
A4 = fit_sqrt(sample_arr, conv_nc4["std"])
n_fit = np.logspace(np.log10(sample_sizes[0]), np.log10(N_01_avail*10), 200)

print(f"\n  1/√N scaling coefficients:  nc2={A2:.4f}  nc3={A3:.4f}  nc4={A4:.4f}")
print(f"  (σ ≈ A/√N_bin)")

# Find N needed for various precision targets
print(f"\n  Events needed in 0–1% bin for target precision:")
print(f"  {'σ_target':>10s}  {'N(nc2)':>10s}  {'N(nc3)':>10s}  {'N(nc4)':>10s}")
for sig_t in [0.10, 0.05, 0.02, 0.01, 0.005]:
    n2 = int((A2 / sig_t)**2)
    n3 = int((A3 / sig_t)**2)
    n4 = int((A4 / sig_t)**2)
    print(f"  {sig_t:>10.3f}  {n2:>10,}  {n3:>10,}  {n4:>10,}")
print(f"\n  (Multiply by 100 to get total events with b_max=b_{{10%}})")
print(f"  (Multiply by 100/0.061 ≈ 1640 for total events with b_max=20 fm)")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 5 — Plots")
print("=" * 64)

BG    = "#0d0f14"
PANEL = "#13161e"
GRID  = "#1e2230"
TEXT  = "#d8dce8"
TICK  = "#7f8c9a"
C2    = "#4fc3f7"    # blue   n=2
C3    = "#ff4081"    # pink   n=3
C4    = "#ffd54f"    # yellow n=4
FIT   = "#b0b8c8"   # grey   fit line

def _style(ax, ylabel, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelcolor=TICK, labelsize=9)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID); sp.set_linewidth(0.6)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.6, ls="--")
    ax.set_xlabel("Centrality (%)" if "Centrality" in ylabel or not ylabel else
                  r"$N_\mathrm{events}$ in 0–1% bin", fontsize=10, color=TEXT)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT)
    ax.set_title(title, fontsize=9.5, color=TEXT, pad=5)

fig = plt.figure(figsize=(16, 10.5), facecolor=BG)
gs  = gridspec.GridSpec(
    2, 3, figure=fig,
    hspace=0.42, wspace=0.34,
    left=0.07, right=0.97, top=0.92, bottom=0.09
)

# ── [0,0] nc vs centrality — n=2 ──────────────────────────────────────────────
ax00 = fig.add_subplot(gs[0, 0])
_style(ax00,
       r"$\mathrm{nc}_2\{4,\varepsilon\}$",
       r"$\mathrm{nc}_2 = \langle\varepsilon_2^4\rangle/\langle\varepsilon_2^2\rangle^2-2$"
       "\nUC 0–10%, 1% bins")

ax00.axhline(0, color=TICK, lw=1.1, ls="-", alpha=0.7, zorder=1)
ax00.errorbar(UC_CENTRES, nc2_vals, yerr=nc2_errs,
              fmt="o-", ms=7, lw=1.8, color=C2,
              ecolor=C2, elinewidth=1.5, capsize=4,
              zorder=4, label=f"TRENTo 2.76 TeV\n$N_\\mathrm{{tot}}={N_TOTAL/1e6:.0f}$M")
ax00.set_xlabel("Centrality (%)", fontsize=10, color=TEXT)
ax00.set_xticks(UC_CENTRES)
ax00.set_xticklabels([f"{int(c)}" for c in UC_CENTRES], fontsize=8)
leg = ax00.legend(fontsize=8, framealpha=0.35, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TEXT, loc="best")
ax00.text(0.97, 0.97,
          f"~{int(n_per_bin[0]/1000)}k evt/bin",
          transform=ax00.transAxes, ha="right", va="top",
          fontsize=8, color=TICK,
          bbox=dict(facecolor=PANEL, edgecolor=GRID, alpha=0.7, pad=2))

# ── [0,1] nc vs centrality — n=3 ──────────────────────────────────────────────
ax01 = fig.add_subplot(gs[0, 1])
_style(ax01,
       r"$\mathrm{nc}_3\{4,\varepsilon\}$",
       r"$\mathrm{nc}_3 = \langle\varepsilon_3^4\rangle/\langle\varepsilon_3^2\rangle^2-2$"
       "\nUC 0–10%, 1% bins")

ax01.axhline(0, color=TICK, lw=1.1, ls="-", alpha=0.7, zorder=1)
ax01.errorbar(UC_CENTRES, nc3_vals, yerr=nc3_errs,
              fmt="s-", ms=7, lw=1.8, color=C3,
              ecolor=C3, elinewidth=1.5, capsize=4,
              zorder=4, label=f"TRENTo 2.76 TeV\n$N_\\mathrm{{tot}}={N_TOTAL/1e6:.0f}$M")
ax01.set_xlabel("Centrality (%)", fontsize=10, color=TEXT)
ax01.set_xticks(UC_CENTRES)
ax01.set_xticklabels([f"{int(c)}" for c in UC_CENTRES], fontsize=8)
leg = ax01.legend(fontsize=8, framealpha=0.35, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TEXT, loc="best")

# ── [0,2] nc vs centrality — n=4 ──────────────────────────────────────────────
ax02 = fig.add_subplot(gs[0, 2])
_style(ax02,
       r"$\mathrm{nc}_4\{4,\varepsilon\}$",
       r"$\mathrm{nc}_4 = \langle\varepsilon_4^4\rangle/\langle\varepsilon_4^2\rangle^2-2$"
       "\nUC 0–10%, 1% bins")

ax02.axhline(0, color=TICK, lw=1.1, ls="-", alpha=0.7, zorder=1)
ax02.errorbar(UC_CENTRES, nc4_vals, yerr=nc4_errs,
              fmt="^-", ms=7, lw=1.8, color=C4,
              ecolor=C4, elinewidth=1.5, capsize=4,
              zorder=4, label=f"TRENTo 2.76 TeV\n$N_\\mathrm{{tot}}={N_TOTAL/1e6:.0f}$M")
ax02.set_xlabel("Centrality (%)", fontsize=10, color=TEXT)
ax02.set_xticks(UC_CENTRES)
ax02.set_xticklabels([f"{int(c)}" for c in UC_CENTRES], fontsize=8)
leg = ax02.legend(fontsize=8, framealpha=0.35, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TEXT, loc="best")

# ── [1,0] Convergence nc3 ─────────────────────────────────────────────────────
ax10 = fig.add_subplot(gs[1, 0])
_style(ax10,
       r"$\sigma(\mathrm{nc}_3)$  in 0–1% bin",
       r"Convergence of $\mathrm{nc}_3$  (0–1% bin)")

ax10.loglog(sample_arr, conv_nc3["std"],
            "s-", ms=7, lw=1.8, color=C3, label=r"$\sigma(\mathrm{nc}_3)$", zorder=4)
ax10.loglog(n_fit, A3 / np.sqrt(n_fit),
            "--", lw=1.4, color=FIT, alpha=0.7, label=r"$\propto 1/\sqrt{N}$")
ax10.axhline(0.05, color="#80cbc4", lw=1.0, ls=":", alpha=0.8)
ax10.axhline(0.01, color="#b39ddb", lw=1.0, ls=":", alpha=0.8)
ax10.text(sample_arr[-1]*1.1, 0.05, "σ=0.05", color="#80cbc4", fontsize=7.5, va="center")
ax10.text(sample_arr[-1]*1.1, 0.01, "σ=0.01", color="#b39ddb", fontsize=7.5, va="center")
ax10.set_xlabel(r"$N_\mathrm{events}$ in 0–1% bin", fontsize=10, color=TEXT)
leg = ax10.legend(fontsize=8.5, framealpha=0.35, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TEXT)

# ── [1,1] Convergence nc4 ─────────────────────────────────────────────────────
ax11 = fig.add_subplot(gs[1, 1])
_style(ax11,
       r"$\sigma(\mathrm{nc}_4)$  in 0–1% bin",
       r"Convergence of $\mathrm{nc}_4$  (0–1% bin)")

ax11.loglog(sample_arr, conv_nc4["std"],
            "^-", ms=7, lw=1.8, color=C4, label=r"$\sigma(\mathrm{nc}_4)$", zorder=4)
ax11.loglog(n_fit, A4 / np.sqrt(n_fit),
            "--", lw=1.4, color=FIT, alpha=0.7, label=r"$\propto 1/\sqrt{N}$")
ax11.axhline(0.05, color="#80cbc4", lw=1.0, ls=":", alpha=0.8)
ax11.axhline(0.01, color="#b39ddb", lw=1.0, ls=":", alpha=0.8)
ax11.set_xlabel(r"$N_\mathrm{events}$ in 0–1% bin", fontsize=10, color=TEXT)
leg = ax11.legend(fontsize=8.5, framealpha=0.35, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TEXT)

# ── [1,2] Convergence of nc3 VALUE (bias) ─────────────────────────────────────
ax12 = fig.add_subplot(gs[1, 2])
_style(ax12,
       r"$\langle\mathrm{nc}_3\rangle$  in 0–1% bin",
       r"Convergence of $\mathrm{nc}_3$ value  (bias check)")

ax12.semilogx(sample_arr, conv_nc3["mean"],
              "s-", ms=7, lw=1.8, color=C3,
              label=r"$\langle\mathrm{nc}_3\rangle$", zorder=4)
ax12.fill_between(
    sample_arr,
    np.array(conv_nc3["mean"]) - np.array(conv_nc3["std"]),
    np.array(conv_nc3["mean"]) + np.array(conv_nc3["std"]),
    color=C3, alpha=0.2, zorder=3,
)
ax12.axhline(nc3_true, color=FIT, lw=1.5, ls="--", alpha=0.8,
             label=f"True  ({N_01_avail:,} events): {nc3_true:+.4f}")
ax12.set_xlabel(r"$N_\mathrm{events}$ in 0–1% bin", fontsize=10, color=TEXT)
ax12.axhline(0, color=TICK, lw=0.8, ls="-", alpha=0.5)
leg = ax12.legend(fontsize=8, framealpha=0.35, facecolor=PANEL,
                  edgecolor=GRID, labelcolor=TEXT)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.97,
    r"$\mathrm{nc}_n\{4,\varepsilon\} = \langle\varepsilon_n^4\rangle/"
    r"\langle\varepsilon_n^2\rangle^2 - 2$"
    r"  |  Standard TRENTo  |  $^{208}$Pb+$^{208}$Pb @ 2.76 TeV  |  "
    r"$b_\mathrm{max}=b_{10\%}\approx 4.95$ fm  |  "
    f"$N_\\mathrm{{tot}}={N_TOTAL/1e6:.0f}$M",
    ha="center", va="top", fontsize=10, color=TEXT, fontweight="bold"
)

out_pdf = OUTDIR / "nc_convergence.pdf"
out_png = OUTDIR / "nc_convergence.png"
fig.savefig(out_pdf, dpi=180, bbox_inches="tight", facecolor=BG)
fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"  Saved: {out_pdf}")
print(f"  Saved: {out_png}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6 — SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  SUMMARY: HOW MANY EVENTS DO WE NEED?")
print("=" * 64)
print(f"""
Observable: nc_n{{4,ε}} = <ε^4>/<ε²>² − 2   in 0–1% centrality bin

Statistical error scales as:  σ ≈ A_n / √(N_bin)
  A_2 = {A2:.4f}
  A_3 = {A3:.4f}
  A_4 = {A4:.4f}

Required events in 0–1% bin (σ_target):
  σ < 0.10:  nc3 → {int((A3/0.10)**2):>8,}   nc4 → {int((A4/0.10)**2):>8,}
  σ < 0.05:  nc3 → {int((A3/0.05)**2):>8,}   nc4 → {int((A4/0.05)**2):>8,}
  σ < 0.02:  nc3 → {int((A3/0.02)**2):>8,}   nc4 → {int((A4/0.02)**2):>8,}
  σ < 0.01:  nc3 → {int((A3/0.01)**2):>8,}   nc4 → {int((A4/0.01)**2):>8,}

Total events needed (with b_max = b_10% = {B_MAX_10PCT:.3f} fm):
  × 10  (since 1% bin = 1/10 of all events when b_max=b_10%)
  σ < 0.05:  nc3 → {int((A3/0.05)**2 * 10):>8,}   nc4 → {int((A4/0.05)**2 * 10):>8,}
  σ < 0.02:  nc3 → {int((A3/0.02)**2 * 10):>8,}   nc4 → {int((A4/0.02)**2 * 10):>8,}
  σ < 0.01:  nc3 → {int((A3/0.01)**2 * 10):>8,}   nc4 → {int((A4/0.01)**2 * 10):>8,}

With 5 M events + b_max=b_10%:
  N in 0–1% bin = 500,000
  σ(nc3) ≈ {A3/np.sqrt(500000):.5f}   σ(nc4) ≈ {A4/np.sqrt(500000):.5f}
  → relative error ≈ {A3/np.sqrt(500000)/abs(nc3_true)*100:.2f}% for nc3,
                      {A4/np.sqrt(500000)/abs(nc4_true)*100:.2f}% for nc4  ✓
""")

print("=" * 64)
print(f"  All outputs in: {WORK_DIR.resolve()}")
print(f"    events_5M.npy")
print(f"    plots/nc_convergence.pdf")
print(f"    plots/nc_convergence.png")
print("=" * 64)
