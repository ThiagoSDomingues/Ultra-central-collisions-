#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  nc_convergence_colab.py  (v2 — three event counts)                        ║
# ║                                                                              ║
# ║  Computes  nc_n{4,ε} = <εₙ⁴>/<εₙ²>² − 2  for n=2,3,4                     ║
# ║  in ultra-central bins 0–10% (1%-wide intervals).                           ║
# ║                                                                              ║
# ║  Three separate runs:                                                        ║
# ║    N =  10 000  (~1 min on Colab)                                           ║
# ║    N = 100 000  (~10 min)                                                   ║
# ║    N = 1 000 000  (~100 min Colab free, ~30 min Colab Pro)                  ║
# ║                                                                              ║
# ║  b_max = b_{10%} ≈ 4.95 fm  → 100% efficiency in 0–10%                     ║
# ║  → N/10 events per 1%-wide bin                                              ║
# ║                                                                              ║
# ║  Fig 1: nc_2, nc_3, nc_4 vs centrality for all 3 runs (3×3 grid)           ║
# ║  Fig 2: convergence σ(N) + nc vs centrality all 3 runs overlaid             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import subprocess, sys, os, shutil, time, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
EVENT_COUNTS  = [10_000, 100_000, 1_000_000]
COUNT_COLORS  = ["#ffd54f", "#4fc3f7", "#ff4081"]
COUNT_MARKERS = ["^", "s", "o"]
COUNT_LS      = [":", "--", "-"]

SIGMA_INEL  = 770.0
SIGMA_NN    = 6.4
B_MAX_10PCT = float(np.sqrt(0.10 * SIGMA_INEL / np.pi))  # 4.9507 fm

TRENTO_P    = 0.063
TRENTO_K    = 1.0 / 0.97**2
TRENTO_NORM = 18.12
TRENTO_W    = 0.9
TRENTO_DMIN = 0.52**(1./3.)
TRENTO_SEED = 42

WORK_DIR    = Path("/content/nc_convergence")
TRENTO_REPO = Path("/content/trento")
OUTDIR      = WORK_DIR / "plots"

EVENTS_FILES = {n: WORK_DIR / f"events_{n//1000}k.npy" for n in EVENT_COUNTS}

N_REF           = 1_000_000
SUBSAMPLE_SIZES = [200, 500, 1_000, 2_000, 5_000, 10_000,
                   20_000, 50_000, 100_000, 200_000, 500_000]
N_BOOT = 500
SEED_BOOT = 42

N_TRENTO_COLS = 8
COL_MULT = 3; COL_E2 = 4; COL_E3 = 5; COL_E4 = 6

UC_EDGES   = np.arange(0, 11, 1)
UC_CENTRES = 0.5 * (UC_EDGES[:-1] + UC_EDGES[1:])
N_UC_BINS  = len(UC_CENTRES)

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
    for p in [Path("/usr/local/bin/trento"),
              TRENTO_REPO / "build" / "src" / "trento",
              TRENTO_REPO / "build" / "trento"]:
        if p.exists() and os.access(p, os.X_OK):
            return p
    r = subprocess.run(["which", "trento"], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return Path(r.stdout.strip())
    return None

def nc(eps):
    """nc_n = <ε⁴>/<ε²>² − 2  (user sign convention)"""
    m2 = np.mean(eps**2); m4 = np.mean(eps**4)
    return m4 / m2**2 - 2.0

def nc_err(eps, rng):
    val  = nc(eps); N = len(eps)
    boot = np.array([nc(rng.choice(eps, N, replace=True)) for _ in range(N_BOOT)])
    return val, float(boot.std())

def assign_centrality(arr):
    rank = np.argsort(np.argsort(-arr[:, COL_MULT]))
    return rank / len(arr) * 100.0

def parse_trento(stdout_text):
    rows = []
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        parts = line.split()
        if len(parts) < N_TRENTO_COLS: continue
        try:   rows.append([float(x) for x in parts[:N_TRENTO_COLS]])
        except ValueError: continue
    if not rows: raise RuntimeError("No events parsed from TRENTo")
    return np.array(rows, dtype=np.float64)

def generate_events(trento_bin, n_events, out_file):
    if out_file.exists() and out_file.stat().st_size > 0:
        arr = np.load(out_file)
        if len(arr) == n_events:
            print(f"  Cached: {out_file.name}  ({len(arr):,} events)")
            return arr
        print(f"  Wrong size ({len(arr):,} != {n_events:,}) — re-generating")
        out_file.unlink()
    cmd = [
        str(trento_bin), "Pb", "Pb", str(n_events),
        "-p", f"{TRENTO_P:.6f}", "-k", f"{TRENTO_K:.6f}",
        "-w", f"{TRENTO_W:.4f}", "-d", f"{TRENTO_DMIN:.6f}",
        "-n", f"{TRENTO_NORM:.6f}", "-x", f"{SIGMA_NN:.4f}",
        "--b-max", f"{B_MAX_10PCT:.6f}",
        "--random-seed", str(TRENTO_SEED),
    ]
    print(f"  Generating {n_events:,} events "
          f"(~{fmt(n_events/165.)}, b_max={B_MAX_10PCT:.4f} fm)")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"TRENTo failed:\n{result.stderr[-400:]}")
    arr = parse_trento(result.stdout)
    np.save(out_file, arr)
    print(f"  Done: {len(arr):,} events in {fmt(time.perf_counter()-t0)} "
          f"→ {out_file.name}")
    return arr

def compute_uc(arr, rng):
    cent = assign_centrality(arr)
    r = {k: np.full(N_UC_BINS, np.nan)
         for k in ["nc2","nc3","nc4","nc2e","nc3e","nc4e"]}
    r["n_bin"] = np.zeros(N_UC_BINS, dtype=int)
    for k, (lo, hi) in enumerate(zip(UC_EDGES[:-1], UC_EDGES[1:])):
        mask = (cent >= lo) & (cent < hi)
        r["n_bin"][k] = mask.sum()
        if mask.sum() < 20: continue
        r["nc2"][k], r["nc2e"][k] = nc_err(arr[mask, COL_E2], rng)
        r["nc3"][k], r["nc3e"][k] = nc_err(arr[mask, COL_E3], rng)
        r["nc4"][k], r["nc4e"][k] = nc_err(arr[mask, COL_E4], rng)
    return r

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 64)
print("  Stage 0 — Dependencies")
print("=" * 64)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "numpy", "matplotlib", "scipy"], check=True)
subprocess.run(["apt-get", "update", "-qq"], check=False)
subprocess.run(["apt-get", "install", "-y", "-qq",
                "build-essential", "cmake", "git", "libboost-all-dev"],
               check=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
print("  OK")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — BUILD TRENTO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 1 — TRENTo")
print("=" * 64)
TRENTO_BIN = find_trento()
if TRENTO_BIN:
    print(f"  Already built → {TRENTO_BIN}")
else:
    if TRENTO_REPO.exists(): shutil.rmtree(TRENTO_REPO)
    run(["git", "clone", "--depth=1",
         "https://github.com/Duke-QCD/trento.git", str(TRENTO_REPO)])
    bd = TRENTO_REPO / "build"; bd.mkdir(exist_ok=True)
    run(["cmake", str(TRENTO_REPO), "-DCMAKE_BUILD_TYPE=Release"], cwd=bd)
    import multiprocessing
    run(["make", f"-j{multiprocessing.cpu_count()}"], cwd=bd)
    subprocess.run(["make", "install"], cwd=str(bd), check=False)
    TRENTO_BIN = find_trento()
    if not TRENTO_BIN: raise RuntimeError("TRENTo not found after build")
v = subprocess.run([str(TRENTO_BIN), "--version"], capture_output=True, text=True)
print(f"  {(v.stdout+v.stderr).strip().splitlines()[0]}")
print(f"  b_max = {B_MAX_10PCT:.4f} fm  (= b_{{10%}}, all events in 0–10%)")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — GENERATE THREE EVENT SETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 2 — Generate 10k, 100k, 1M events")
print("=" * 64)
print("\n  Estimated wall time on Colab free tier (~165 evt/s):")
for N in EVENT_COUNTS:
    print(f"    {N:>10,}  →  {fmt(N/165.)}")
print()

all_arrs = {}
for N in EVENT_COUNTS:
    print(f"\n  ── {N:,} events ──")
    all_arrs[N] = generate_events(TRENTO_BIN, N, EVENTS_FILES[N])

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CUMULANT RATIOS IN UC BINS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 3 — Cumulant ratios in 1%-wide UC bins")
print("=" * 64)
rng_main = np.random.default_rng(SEED_BOOT)
profiles = {}
for N in EVENT_COUNTS:
    profiles[N] = compute_uc(all_arrs[N], rng_main)
    p = profiles[N]
    print(f"\n  N_tot={N:,}  (N_bin={N//10:,})")
    print(f"  {'bin':7s} {'N':>7s} "
          f"{'nc2':>10s}±{'':4s} {'nc3':>10s}±{'':4s} {'nc4':>10s}±{'':4s}")
    for k in range(N_UC_BINS):
        print(f"  {UC_EDGES[k]:.0f}-{UC_EDGES[k+1]:.0f}%  "
              f"{p['n_bin'][k]:>7,}  "
              f"{p['nc2'][k]:>+10.5f}±{p['nc2e'][k]:.4f}  "
              f"{p['nc3'][k]:>+10.5f}±{p['nc3e'][k]:.4f}  "
              f"{p['nc4'][k]:>+10.5f}±{p['nc4e'][k]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — CONVERGENCE STUDY  (subsampling 1M, 0–1% bin)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 4 — Convergence (subsampling 1M run, 0–1% bin)")
print("=" * 64)
arr_ref  = all_arrs[N_REF]
cent_ref = assign_centrality(arr_ref)
mask_01  = (cent_ref >= 0) & (cent_ref < 1)
e2_01 = arr_ref[mask_01, COL_E2]
e3_01 = arr_ref[mask_01, COL_E3]
e4_01 = arr_ref[mask_01, COL_E4]
N_01  = len(e3_01)
nc2_true = nc(e2_01)
nc3_true = nc(e3_01)
nc4_true = nc(e4_01)
print(f"\n  Reference (N_bin={N_01:,}, from 1M run):")
print(f"    nc2={nc2_true:+.6f}  nc3={nc3_true:+.6f}  nc4={nc4_true:+.6f}")

sub_sizes = [s for s in SUBSAMPLE_SIZES if s <= N_01]
sub_arr   = np.array(sub_sizes, dtype=float)
rng_conv  = np.random.default_rng(SEED_BOOT + 1)
conv = {obs: {"mean": [], "std": []} for obs in ["nc2","nc3","nc4"]}

print(f"\n  {'N_bin':>8s}  "
      f"{'nc2':>9s}±σ        {'nc3':>9s}±σ        {'nc4':>9s}±σ")
for ns in sub_sizes:
    idx = rng_conv.integers(0, N_01, size=(N_BOOT, ns))
    b2 = np.array([nc(e2_01[idx[b]]) for b in range(N_BOOT)])
    b3 = np.array([nc(e3_01[idx[b]]) for b in range(N_BOOT)])
    b4 = np.array([nc(e4_01[idx[b]]) for b in range(N_BOOT)])
    for obs, boot in [("nc2",b2),("nc3",b3),("nc4",b4)]:
        conv[obs]["mean"].append(boot.mean())
        conv[obs]["std"].append(boot.std())
    print(f"  {ns:>8,}  "
          f"{b2.mean():>+9.5f}±{b2.std():.5f}  "
          f"{b3.mean():>+9.5f}±{b3.std():.5f}  "
          f"{b4.mean():>+9.5f}±{b4.std():.5f}")

def fit_A(sizes, stds):
    """A = σ * sqrt(N) averaged over last 3 points"""
    A = np.array(stds[-3:]) * np.sqrt(np.array(sizes[-3:]))
    return float(A.mean())

A2 = fit_A(sub_sizes, conv["nc2"]["std"])
A3 = fit_A(sub_sizes, conv["nc3"]["std"])
A4 = fit_A(sub_sizes, conv["nc4"]["std"])
n_fit = np.logspace(np.log10(sub_sizes[0]),
                    np.log10(max(N_01*5, sub_sizes[-1]*5)), 300)

print(f"\n  σ = A/√N:  A2={A2:.3f}  A3={A3:.3f}  A4={A4:.3f}")
print(f"\n  Per-run precision (N_bin = N_tot/10):")
print(f"  {'N_tot':>10s}  {'N_bin':>8s}  {'σ(nc3)':>9s}  {'σ(nc4)':>9s}")
for N in EVENT_COUNTS:
    nb = N // 10
    print(f"  {N:>10,}  {nb:>8,}  {A3/np.sqrt(nb):>9.4f}  {A4/np.sqrt(nb):>9.4f}")

print(f"\n  Total events needed (b_max=b_{{10%}}) for σ < target:")
print(f"  {'σ':>8s}  {'nc3':>12s}  {'nc4':>12s}")
for sig in [0.10, 0.05, 0.02, 0.01]:
    print(f"  {sig:>8.2f}  {int((A3/sig)**2*10):>12,}  {int((A4/sig)**2*10):>12,}")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — FIGURE 1: nc vs centrality (3 obs × 3 event counts)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  Stage 5 — Plots")
print("=" * 64)

BG    = "#0d0f14"; PANEL = "#13161e"; GRID  = "#1e2230"
TEXT  = "#d8dce8"; TICK  = "#7f8c9a"; FIT   = "#b0b8c8"

def _style(ax, ylabel, title, xlim=None, xlog=False, ylog=False):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TICK, labelcolor=TICK, labelsize=8.5)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID); sp.set_linewidth(0.6)
    ax.grid(True, color=GRID, lw=0.4, alpha=0.6, ls="--")
    if not (xlog or ylog):
        ax.axhline(0, color=TICK, lw=1.0, ls="-", alpha=0.6, zorder=1)
    ax.set_ylabel(ylabel, fontsize=9.5, color=TEXT)
    ax.set_title(title, fontsize=9, color=TEXT, pad=4)
    if xlim: ax.set_xlim(*xlim)

NC_META = {
    "nc2": (r"$\mathrm{nc}_2 = \langle\varepsilon_2^4\rangle/\langle\varepsilon_2^2\rangle^2-2$",
            "#4fc3f7"),
    "nc3": (r"$\mathrm{nc}_3 = \langle\varepsilon_3^4\rangle/\langle\varepsilon_3^2\rangle^2-2$",
            "#ff4081"),
    "nc4": (r"$\mathrm{nc}_4 = \langle\varepsilon_4^4\rangle/\langle\varepsilon_4^2\rangle^2-2$",
            "#ffd54f"),
}

# Figure 1: 3 rows (obs) × 3 cols (N)
fig1 = plt.figure(figsize=(14, 10), facecolor=BG)
gs1  = gridspec.GridSpec(3, 3, figure=fig1,
                         hspace=0.48, wspace=0.30,
                         left=0.07, right=0.97, top=0.91, bottom=0.10)
for row, (obs_key, (ylabel, color)) in enumerate(NC_META.items()):
    err_key = obs_key + "e"
    for col, N in enumerate(EVENT_COUNTS):
        ax = fig1.add_subplot(gs1[row, col])
        n_bin = N // 10
        _style(ax, ylabel,
               f"$N_\\mathrm{{tot}}={N//1000}$k  ($N_\\mathrm{{bin}}={n_bin:,}$)",
               xlim=(-0.3, 10.3))
        ax.set_xlabel("Centrality (%)", fontsize=9, color=TEXT)
        ax.set_xticks(UC_CENTRES)
        ax.set_xticklabels([f"{int(c)}" for c in UC_CENTRES], fontsize=7)

        p  = profiles[N]
        v  = p[obs_key]; e = p[err_key]
        ok = np.isfinite(v) & np.isfinite(e)
        if ok.sum() >= 2:
            ax.fill_between(UC_CENTRES[ok], v[ok]-e[ok], v[ok]+e[ok],
                            color=color, alpha=0.22, zorder=3)
            ax.errorbar(UC_CENTRES[ok], v[ok], yerr=e[ok],
                        fmt="o-", ms=5, lw=1.8, color=color,
                        ecolor=color, elinewidth=1.3, capsize=3,
                        zorder=4)
            mean_err = np.nanmean(e[ok])
            ax.text(0.97, 0.05,
                    f"$\\bar{{\\sigma}}={mean_err:.3f}$",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color=color,
                    bbox=dict(facecolor=PANEL, edgecolor=GRID, alpha=0.7, pad=2))

fig1.text(0.5, 0.97,
          r"$\mathrm{nc}_n\{4,\varepsilon\}=\langle\varepsilon_n^4\rangle/"
          r"\langle\varepsilon_n^2\rangle^2-2$  vs centrality  |  "
          r"$^{208}$Pb+$^{208}$Pb @ 2.76 TeV  |  "
          r"$b_\mathrm{max}=b_{10\%}=4.95$ fm",
          ha="center", va="top", fontsize=9.5, color=TEXT, fontweight="bold")
out1 = OUTDIR / "nc_vs_cent.pdf"
fig1.savefig(out1, dpi=170, bbox_inches="tight", facecolor=BG)
fig1.savefig(out1.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print(f"  Saved: {out1}")

# Figure 2: convergence (2×2)
fig2 = plt.figure(figsize=(13, 9.5), facecolor=BG)
gs2  = gridspec.GridSpec(2, 2, figure=fig2,
                         hspace=0.42, wspace=0.32,
                         left=0.08, right=0.97, top=0.91, bottom=0.10)

# [0,0] σ(nc3) log-log
ax00 = fig2.add_subplot(gs2[0, 0])
_style(ax00, r"$\sigma(\mathrm{nc}_3)$  in 0–1% bin",
       r"Convergence $\mathrm{nc}_3$  (0–1% bin)", xlog=True, ylog=True)
ax00.set_xscale("log"); ax00.set_yscale("log")
ax00.set_xlabel(r"$N_\mathrm{bin}$  (events in 0–1%)", fontsize=9.5, color=TEXT)
ax00.loglog(sub_arr, conv["nc3"]["std"],
            "s-", ms=7, lw=1.8, color="#ff4081", zorder=4,
            label=r"$\sigma(\mathrm{nc}_3)$ (bootstrap)")
ax00.loglog(n_fit, A3/np.sqrt(n_fit),
            "--", lw=1.4, color=FIT, alpha=0.75,
            label=r"$A_3/\sqrt{N}$, $A_3=$"+f"{A3:.2f}")
for N, col, mk in zip(EVENT_COUNTS, COUNT_COLORS, COUNT_MARKERS):
    nb = N // 10
    ax00.scatter([nb], [A3/np.sqrt(nb)],
                 color=col, s=130, marker=mk, zorder=7,
                 label=f"$N_\\mathrm{{tot}}={N//1000}$k "
                       f"($\\sigma$≈{A3/np.sqrt(nb):.3f})")
for sig_t, clr in [(0.05,"#80cbc4"),(0.01,"#b39ddb")]:
    ax00.axhline(sig_t, color=clr, lw=0.9, ls=":", alpha=0.85)
    ax00.text(sub_arr[0]*0.9, sig_t*1.06,
              f"σ={sig_t}", color=clr, fontsize=7, ha="right")
ax00.legend(fontsize=7.5, framealpha=0.35, facecolor=PANEL,
            edgecolor=GRID, labelcolor=TEXT, loc="upper right")

# [0,1] σ(nc4) log-log
ax01 = fig2.add_subplot(gs2[0, 1])
_style(ax01, r"$\sigma(\mathrm{nc}_4)$  in 0–1% bin",
       r"Convergence $\mathrm{nc}_4$  (0–1% bin)", xlog=True, ylog=True)
ax01.set_xscale("log"); ax01.set_yscale("log")
ax01.set_xlabel(r"$N_\mathrm{bin}$  (events in 0–1%)", fontsize=9.5, color=TEXT)
ax01.loglog(sub_arr, conv["nc4"]["std"],
            "^-", ms=7, lw=1.8, color="#ffd54f", zorder=4,
            label=r"$\sigma(\mathrm{nc}_4)$ (bootstrap)")
ax01.loglog(n_fit, A4/np.sqrt(n_fit),
            "--", lw=1.4, color=FIT, alpha=0.75,
            label=r"$A_4/\sqrt{N}$, $A_4=$"+f"{A4:.2f}")
for N, col, mk in zip(EVENT_COUNTS, COUNT_COLORS, COUNT_MARKERS):
    nb = N // 10
    ax01.scatter([nb], [A4/np.sqrt(nb)],
                 color=col, s=130, marker=mk, zorder=7,
                 label=f"$N_\\mathrm{{tot}}={N//1000}$k "
                       f"($\\sigma$≈{A4/np.sqrt(nb):.3f})")
for sig_t, clr in [(0.05,"#80cbc4"),(0.01,"#b39ddb")]:
    ax01.axhline(sig_t, color=clr, lw=0.9, ls=":", alpha=0.85)
ax01.legend(fontsize=7.5, framealpha=0.35, facecolor=PANEL,
            edgecolor=GRID, labelcolor=TEXT, loc="upper right")

# [1,0] nc3 vs cent, all 3 N overlaid
ax10 = fig2.add_subplot(gs2[1, 0])
_style(ax10, r"$\mathrm{nc}_3\{4,\varepsilon\}$",
       r"$\mathrm{nc}_3$ vs centrality — all $N$", xlim=(-0.3,10.3))
ax10.set_xlabel("Centrality (%)", fontsize=9.5, color=TEXT)
ax10.set_xticks(UC_CENTRES)
ax10.set_xticklabels([f"{int(c)}" for c in UC_CENTRES], fontsize=7.5)
for N, col, mk, ls in zip(EVENT_COUNTS, COUNT_COLORS, COUNT_MARKERS, COUNT_LS):
    p = profiles[N]; v = p["nc3"]; e = p["nc3e"]
    ok = np.isfinite(v) & np.isfinite(e)
    ax10.fill_between(UC_CENTRES[ok], v[ok]-e[ok], v[ok]+e[ok],
                      color=col, alpha=0.18, zorder=2)
    ax10.plot(UC_CENTRES[ok], v[ok], color=col, ls=ls, lw=2.0,
              marker=mk, ms=5, zorder=4,
              label=f"$N_\\mathrm{{tot}}={N//1000}$k")
ax10.legend(fontsize=8.5, framealpha=0.35, facecolor=PANEL,
            edgecolor=GRID, labelcolor=TEXT)

# [1,1] nc4 vs cent, all 3 N overlaid
ax11 = fig2.add_subplot(gs2[1, 1])
_style(ax11, r"$\mathrm{nc}_4\{4,\varepsilon\}$",
       r"$\mathrm{nc}_4$ vs centrality — all $N$", xlim=(-0.3,10.3))
ax11.set_xlabel("Centrality (%)", fontsize=9.5, color=TEXT)
ax11.set_xticks(UC_CENTRES)
ax11.set_xticklabels([f"{int(c)}" for c in UC_CENTRES], fontsize=7.5)
for N, col, mk, ls in zip(EVENT_COUNTS, COUNT_COLORS, COUNT_MARKERS, COUNT_LS):
    p = profiles[N]; v = p["nc4"]; e = p["nc4e"]
    ok = np.isfinite(v) & np.isfinite(e)
    ax11.fill_between(UC_CENTRES[ok], v[ok]-e[ok], v[ok]+e[ok],
                      color=col, alpha=0.18, zorder=2)
    ax11.plot(UC_CENTRES[ok], v[ok], color=col, ls=ls, lw=2.0,
              marker=mk, ms=5, zorder=4,
              label=f"$N_\\mathrm{{tot}}={N//1000}$k")
ax11.legend(fontsize=8.5, framealpha=0.35, facecolor=PANEL,
            edgecolor=GRID, labelcolor=TEXT)

fig2.text(0.5, 0.97,
          r"Convergence study  |  "
          r"$^{208}$Pb+$^{208}$Pb @ 2.76 TeV  |  "
          r"$b_\mathrm{max}=b_{10\%}$  |  $N_\mathrm{tot}=10$k / 100k / 1M",
          ha="center", va="top", fontsize=9.5, color=TEXT, fontweight="bold")
out2 = OUTDIR / "nc_convergence.pdf"
fig2.savefig(out2, dpi=170, bbox_inches="tight", facecolor=BG)
fig2.savefig(out2.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"  Saved: {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("  SUMMARY")
print("=" * 64)
print(f"""
nc_n values at 0-1% centrality (from {N_01:,}-event reference):
  nc2 = {nc2_true:+.5f}
  nc3 = {nc3_true:+.5f}
  nc4 = {nc4_true:+.5f}

Statistical error  σ ≈ A/√N_bin:
  A2={A2:.3f}   A3={A3:.3f}   A4={A4:.3f}

Per-run uncertainty at 0-1%:
  N_tot= 10k:  N_bin=  1k  σ(nc3)={A3/np.sqrt(1e3):.3f}  σ(nc4)={A4/np.sqrt(1e3):.3f}
  N_tot=100k:  N_bin= 10k  σ(nc3)={A3/np.sqrt(1e4):.3f}  σ(nc4)={A4/np.sqrt(1e4):.3f}
  N_tot=  1M:  N_bin=100k  σ(nc3)={A3/np.sqrt(1e5):.4f}  σ(nc4)={A4/np.sqrt(1e5):.4f}

Total events for σ < target (b_max=b_{{10%}}):
  σ<0.05: nc3={int((A3/0.05)**2*10):,}  nc4={int((A4/0.05)**2*10):,}
  σ<0.02: nc3={int((A3/0.02)**2*10):,}  nc4={int((A4/0.02)**2*10):,}
  σ<0.01: nc3={int((A3/0.01)**2*10):,}  nc4={int((A4/0.01)**2*10):,}
""")
print("=" * 64)
print(f"  Outputs in {WORK_DIR}/plots/:")
for f in ["nc_vs_cent.pdf", "nc_vs_cent.png",
          "nc_convergence.pdf", "nc_convergence.png"]:
    p = OUTDIR / f
    if p.exists(): print(f"    {f}  ({p.stat().st_size/1e3:.0f} kB)")
print("=" * 64)
