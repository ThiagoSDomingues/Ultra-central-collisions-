#!/usr/bin/env python3
"""
pbpb_run_trento.py
==================
PART 3 of the 208Pb+208Pb ultracentral flow puzzle pipeline.

Reads the work directory produced by pbpb_isobar_gen.py and runs TRENTo
for every design point using the WS1.hdf / WS2.hdf nuclear configurations
already on disk. Nucleon width w is read from each design_meta.yaml, so
no external design matrix is needed at this stage.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRENTO FLAG REFERENCE  (verified against Duke-QCD/trento docs + release log)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Short  Long flag               Description
  ─────  ───────────────────     ──────────────────────────────────────────
  -p     --p-mean                entropy exponent  (reduced-thickness ansatz)
  -k     --fluctuation           Gamma-distribution shape for nucleon weights
  -w     --nucleon-width         Gaussian nucleon interaction width [fm]
  -d     --deposition-width      Gaussian entropy deposition width  [fm]
                                 ← DISTINCT from -w since TRENTo v1.3
                                 ← we set -d = -w (Bernhard 2019 convention)
  -n     --normalization         overall multiplicity scale
         --cross-section         σ_NN inelastic NN cross section [fm²]
         --random-seed           integer RNG seed
         --b-max                 maximum impact parameter [fm]

  IMPORTANT: -d (--deposition-width) was decoupled from -w in TRENTo v1.3.
  Using the wrong long flag name (e.g. --nucleon-min-dist for -d) would
  silently produce wrong results.  This script always uses the short flags
  -p/-k/-w/-d/-n to avoid any ambiguity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETERS PER DESIGN POINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WS1.hdf / WS2.hdf  → nuclear configs from modified Isobar-Sampler (R,a,β₃,β₄)
  w                  → read from design_NNNN/design_meta.yaml  (LHS-sampled)
  d                  → set equal to w  (Bernhard 2019 choice)

  Fixed (energy-dependent, Bayesian posterior medians):
    p    = 0.0              geometric-mean entropy ansatz
    k    = 1.6              Gamma fluctuation shape
    norm = 18.0 / 20.0     multiplicity scale  (2.76 / 5.02 TeV)
    σ_NN = 6.4  / 7.0  fm²  inelastic NN cross section

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRENTO STDOUT COLUMNS  (default mode, no --ncoll)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0  event_id
  1  b        impact parameter [fm]
  2  npart    participant count
  3  mult     total reduced thickness  (≈ dS/dη, used for centrality)
  4  e2       ε₂  eccentricity
  5  e3       ε₃
  6  e4       ε₄
  7  e5       ε₅

  No -o flag: stdout-only mode.  Eliminates N_events HDF5 disk writes
  (dominant I/O cost), then caches the full array as a single .npy file.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  <work_dir>/design_NNNN/trento_events.npy    float64 (N_events × 8)
  <work_dir>/trento_run_report.txt            timing + budget estimate

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Minimal (500k events / pt, 2.76 TeV)
  python pbpb_run_trento.py \\
      --trento   /path/to/trento \\
      --work-dir pbpb_scan

  # Full options
  python pbpb_run_trento.py \\
      --trento      /path/to/trento \\
      --work-dir    pbpb_scan \\
      --n-events    500000 \\
      --sqrts       2.76 \\
      --design-range 0 50

  # Force-rerun one failed point
  python pbpb_run_trento.py \\
      --trento   /path/to/trento \\
      --work-dir pbpb_scan \\
      --design-range 7 8 --force

PREREQUISITES
  trento   https://github.com/Duke-QCD/trento  (compile from source)
  numpy, pyyaml

REFERENCES
  Moreland, Bernhard, Bass — PRC 92, 011901 (2015) [arXiv:1412.4708]
  Bernhard et al.          — Nature Phys. 15, 1113 (2019) [arXiv:1901.07808]
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FIXED TRENTO PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

FIXED_TRENTO = {
    # sqrts_TeV : {p, k, norm, cross_section [fm²]}
    2.76: {"p": 0.0, "k": 1.6, "norm": 18.0, "cross_section": 6.4},
    5.02: {"p": 0.0, "k": 1.6, "norm": 20.0, "cross_section": 7.0},
}

# TRENTo stdout column indices (default output, no --ncoll flag)
N_COLS    = 8
COL_EVT   = 0   # event number
COL_B     = 1   # impact parameter [fm]
COL_NPART = 2   # participants
COL_MULT  = 3   # total reduced thickness  ← centrality proxy
COL_E2    = 4   # ε₂
COL_E3    = 5   # ε₃
COL_E4    = 6   # ε₄
COL_E5    = 7   # ε₅


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_dur(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _check_trento(trento_bin: Path) -> None:
    """Verify TRENTo binary exists and print its version string."""
    if not trento_bin.exists():
        log.error(f"TRENTo binary not found: {trento_bin}")
        sys.exit(1)
    r = subprocess.run([str(trento_bin), "--version"],
                       capture_output=True, text=True)
    lines = (r.stdout + r.stderr).strip().splitlines()
    ver   = lines[0] if lines else "(unknown)"
    log.info(f"  TRENTo binary  : {trento_bin}")
    log.info(f"  TRENTo version : {ver}")


def _read_meta(design_dir: Path) -> dict:
    """Load design_meta.yaml written by pbpb_isobar_gen.py."""
    p = design_dir / "design_meta.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"design_meta.yaml not found in {design_dir}. "
            "Run pbpb_isobar_gen.py (Stage 2) first."
        )
    with open(p) as fh:
        return yaml.safe_load(fh)


def _discover_dirs(work_dir: Path) -> list[Path]:
    """Return sorted design_NNNN/ dirs that have both WS1.hdf and WS2.hdf."""
    return sorted(
        [d for d in work_dir.glob("design_????")
         if (d / "WS1.hdf").exists() and (d / "WS2.hdf").exists()]
    )


def _idx_of(d: Path) -> int:
    return int(d.name.split("_")[1])


def _parse_stdout(stdout: str, idx: int) -> np.ndarray | None:
    """
    Parse TRENTo per-event stdout lines into a float64 array (N, 8).
    Lines starting with '#' (comments/headers) are skipped.
    Returns None if no events were parsed.
    """
    rows = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < N_COLS:
            continue
        try:
            rows.append([float(x) for x in parts[:N_COLS]])
        except ValueError:
            continue
    if not rows:
        log.warning(f"  [{idx:04d}] No parseable event lines in TRENTo stdout.")
        return None
    return np.array(rows, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE DESIGN-POINT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
        trento_bin: Path,
        design_dir: Path,
        idx:        int,
        n_events:   int,
        fixed:      dict,
        force:      bool,
) -> dict:
    """
    Run TRENTo for one design point.

    Returns dict:
        idx, success, elapsed, n_events, skipped, w
    """
    cache = design_dir / "trento_events.npy"

    # ── Skip if cached ────────────────────────────────────────────────────────
    if cache.exists() and not force:
        arr = np.load(cache)
        log.info(f"  [{idx:04d}] Cache exists ({len(arr):,} events) — skipping."
                 " (--force to overwrite)")
        return dict(idx=idx, success=True, elapsed=0.0,
                    n_events=len(arr), skipped=True, w=None)

    # ── Check HDF inputs ──────────────────────────────────────────────────────
    ws1 = design_dir / "WS1.hdf"
    ws2 = design_dir / "WS2.hdf"
    for f, lbl in [(ws1, "WS1.hdf"), (ws2, "WS2.hdf")]:
        if not f.exists():
            log.error(f"  [{idx:04d}] {lbl} not found: {f}")
            return dict(idx=idx, success=False, elapsed=0.0,
                        n_events=0, skipped=False, w=None)

    # ── Read w from design_meta.yaml ──────────────────────────────────────────
    try:
        meta = _read_meta(design_dir)
    except FileNotFoundError as e:
        log.error(f"  [{idx:04d}] {e}")
        return dict(idx=idx, success=False, elapsed=0.0,
                    n_events=0, skipped=False, w=None)

    w   = float(meta["trento_stage_params"]["nucleon_width_w_fm"])
    isp = meta["isobar_sampler_params"]

    log.info(
        f"  [{idx:04d}]  "
        f"R={isp['WS_radius_fm']:.4f}  "
        f"a={isp['WS_diffusiveness_fm']:.4f}  "
        f"β₃={isp['beta_3']:.5f}  "
        f"β₄={isp['beta_4']:.5f}  "
        f"w={w:.4f}"
    )

    # ── Build command ─────────────────────────────────────────────────────────
    #
    # Always use short flags (-p/-k/-w/-d/-n) to avoid long-flag name bugs.
    # Especially critical: -d is --deposition-width (NOT --nucleon-min-dist).
    # Both -w and -d are set to the same value w (Bernhard 2019 convention).
    #
    # No -o / --output flag → stdout-only; avoids per-event HDF5 file writes.
    #
    cmd = [
        str(trento_bin),
        str(ws1), str(ws2),         # projectile and target HDF configs
        str(n_events),              # number of events
        "-p", f"{fixed['p']:.6f}",         # entropy exponent
        "-k", f"{fixed['k']:.6f}",         # Gamma fluctuation shape
        "-w", f"{w:.6f}",                  # nucleon interaction width [fm]
        "-d", f"{w:.6f}",                  # deposition width = nucleon width
        "-n", f"{fixed['norm']:.6f}",      # multiplicity normalization
        "--cross-section", f"{fixed['cross_section']:.4f}",  # σ_NN [fm²]
        "--random-seed",   str(1000 + idx),                  # reproducible
        "--b-max",         "20.0",                           # generous b cut
    ]

    log.info(f"    $ {' '.join(cmd)}")

    # ── Execute ───────────────────────────────────────────────────────────────
    t0     = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        log.error(f"  [{idx:04d}] TRENTo exited with code {result.returncode}")
        for line in result.stderr.strip().splitlines()[-6:]:
            log.error(f"    stderr: {line}")
        return dict(idx=idx, success=False, elapsed=elapsed,
                    n_events=0, skipped=False, w=w)

    # ── Parse stdout ──────────────────────────────────────────────────────────
    arr = _parse_stdout(result.stdout, idx)
    if arr is None:
        log.error(f"  [{idx:04d}] stdout parsing failed — first 8 lines:")
        for line in result.stdout.splitlines()[:8]:
            log.error(f"    {line}")
        return dict(idx=idx, success=False, elapsed=elapsed,
                    n_events=0, skipped=False, w=w)

    # ── Cache to disk ─────────────────────────────────────────────────────────
    np.save(cache, arr)
    n_got = len(arr)

    log.info(
        f"    → {n_got:,} events saved  |  "
        f"{_fmt_dur(elapsed)}  |  "
        f"{n_got / elapsed:,.0f} evt/s"
    )
    return dict(idx=idx, success=True, elapsed=elapsed,
                n_events=n_got, skipped=False, w=w)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 LOOP
# ─────────────────────────────────────────────────────────────────────────────

def stage3(
        trento_bin: Path,
        work_dir:   Path,
        n_events:   int,
        sqrts:      float,
        idx_range:  tuple[int, int] | None,
        force:      bool,
) -> list[dict]:
    """
    Iterate over all design points and run TRENTo for each.
    Returns list of per-design-point result dicts.
    """
    if sqrts not in FIXED_TRENTO:
        log.error(f"sqrts={sqrts} not supported. Choose 2.76 or 5.02.")
        sys.exit(1)
    fixed = FIXED_TRENTO[sqrts]

    dirs = _discover_dirs(work_dir)
    if not dirs:
        log.error(
            f"No design_NNNN/ dirs with WS1.hdf+WS2.hdf found in {work_dir}. "
            "Run pbpb_isobar_gen.py (Stage 2) first."
        )
        sys.exit(1)

    if idx_range is not None:
        start, end = idx_range
        dirs = [d for d in dirs if start <= _idx_of(d) < end]
        if not dirs:
            log.error(f"No dirs in [{start},{end}). Check --design-range.")
            sys.exit(1)

    n_total = len(dirs)

    print()
    print("─" * 72)
    log.info(f"[Stage 3] TRENTo — {n_total} design pts  |  "
             f"{n_events:,} events each  |  √s = {sqrts} TeV")
    log.info(f"  Flags: -p {fixed['p']}  -k {fixed['k']}  "
             f"-n {fixed['norm']}  --cross-section {fixed['cross_section']} fm²")
    log.info(f"  -w = -d = LHS nucleon width per design pt")
    log.info(f"  --random-seed = 1000 + idx  |  --b-max 20.0 fm")
    log.info(f"  Stdout-only mode (no -o → no per-event HDF5 writes)")
    print("─" * 72)
    print()

    records      = []
    t_loop_start = time.perf_counter()

    for i, ddir in enumerate(dirs):
        idx = _idx_of(ddir)
        rec = run_one(trento_bin, ddir, idx, n_events, fixed, force)
        records.append(rec)

        # Live ETA from actual (non-cached) runs
        real = [r for r in records if not r["skipped"] and r["success"]]
        if real:
            avg_t  = sum(r["elapsed"] for r in real) / len(real)
            n_left = n_total - (i + 1)
            wall   = time.perf_counter() - t_loop_start
            log.info(
                f"    ── {i+1}/{n_total} done  |  "
                f"avg {_fmt_dur(avg_t)}/pt  |  "
                f"ETA {_fmt_dur(avg_t * n_left)}  |  "
                f"wall {_fmt_dur(wall)}"
            )
        print()

    return records


# ─────────────────────────────────────────────────────────────────────────────
# TIMING + BUDGET REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
        records:    list[dict],
        work_dir:   Path,
        n_events:   int,
        sqrts:      float,
        trento_bin: Path,
        t_wall:     float,
) -> None:
    """
    Print the full timing and budget estimate, then save to
    <work_dir>/trento_run_report.txt.
    """
    fixed   = FIXED_TRENTO[sqrts]
    ran     = [r for r in records if not r["skipped"] and r["success"]]
    skipped = [r for r in records if r["skipped"]]
    failed  = [r for r in records if not r["success"]]

    t_all  = sum(r["elapsed"] for r in ran)
    t_avg  = t_all / len(ran) if ran else 0.0
    t_min  = min((r["elapsed"] for r in ran), default=0.0)
    t_max  = max((r["elapsed"] for r in ran), default=0.0)
    t_std  = (
        (sum((r["elapsed"] - t_avg) ** 2 for r in ran) / len(ran)) ** 0.5
        if len(ran) > 1 else 0.0
    )
    rate   = n_events / t_avg if t_avg > 0 else 0.0

    # Extrapolations to full 100-pt design
    t_100   = t_avg * 100
    t_100h  = t_100 / 3600

    SEP = "=" * 72
    BOX = "─" * 56

    lines = [
        "",
        SEP,
        "  TRENTo Stage 3 — Running Time & Budget Report",
        f"  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        SEP,
        "",
        "  RUN CONFIGURATION",
        f"    TRENTo binary       : {trento_bin}",
        f"    Work directory      : {work_dir}",
        f"    √s_NN               : {sqrts} TeV",
        f"    Events / pt         : {n_events:,}",
        f"    -p  (p-mean)        : {fixed['p']}",
        f"    -k  (fluctuation)   : {fixed['k']}",
        f"    -n  (normalization) : {fixed['norm']}",
        f"    σ_NN                : {fixed['cross_section']} fm²",
        f"    -d = -w             : deposition width = nucleon width",
        f"    --random-seed       : 1000 + design_index",
        f"    --b-max             : 20.0 fm",
        "",
        "  COMPLETION SUMMARY",
        f"    Total design pts    : {len(records)}",
        f"    Ran  (new)          : {len(ran)}",
        f"    Skipped  (cached)   : {len(skipped)}",
        f"    Failed              : {len(failed)}",
    ]

    if failed:
        lines += ["", "  FAILED DESIGN POINTS:"]
        lines += [f"    design_{r['idx']:04d}/" for r in failed]

    if ran:
        lines += [
            "",
            f"  ┌{BOX}┐",
            "  │         RUNNING TIME SUMMARY                        │",
            f"  ├{BOX}┤",
            f"  │  Min / pt           {_fmt_dur(t_min):>14}                  │",
            f"  │  Max / pt           {_fmt_dur(t_max):>14}                  │",
            f"  │  Mean / pt          {_fmt_dur(t_avg):>14}                  │",
            f"  │  Std dev            {_fmt_dur(t_std):>14}                  │",
            f"  │  Total (ran pts)    {_fmt_dur(t_all):>14}                  │",
            f"  │  Throughput         {rate:>12,.0f} evt/s                │",
            f"  ├{BOX}┤",
            "  │         BUDGET ESTIMATE  (full N=100 design)        │",
            f"  ├{BOX}┤",
            f"  │  Sequential total   {_fmt_dur(t_100):>14}  ({t_100h:.1f} h)      │",
            f"  │  Cluster 100 nodes  {_fmt_dur(t_avg):>14}  (1 pt / node)   │",
            f"  │  Cluster  50 nodes  {_fmt_dur(t_avg*2):>14}  (2 pts / node)  │",
            f"  │  Cluster  10 nodes  {_fmt_dur(t_avg*10):>14}  (10 pts / node) │",
            f"  └{BOX}┘",
            "",
            "  NOTE: TRENTo is single-threaded. Each design point is fully",
            "  independent → embarrassingly parallel. On a cluster, the wall",
            f"  time with 100 nodes collapses to ≈ {_fmt_dur(t_avg)} per node.",
        ]

    # ── Per-design-point table ────────────────────────────────────────────────
    lines += [
        "",
        "  PER-DESIGN-POINT TABLE",
        f"  {'idx':>5}  {'status':>9}  {'time':>9}  {'events':>9}  "
        f"{'evt/s':>9}  {'β₃':>8}  {'β₄':>8}  {'w [fm]':>7}",
        "  " + "─" * 70,
    ]

    for r in records:
        ddir = work_dir / f"design_{r['idx']:04d}"
        try:
            meta = _read_meta(ddir)
            isp  = meta["isobar_sampler_params"]
            b3   = f"{isp['beta_3']:.5f}"
            b4   = f"{isp['beta_4']:.5f}"
            w_s  = (f"{r['w']:.4f}" if r["w"] is not None
                    else f"{meta['trento_stage_params']['nucleon_width_w_fm']:.4f}")
        except Exception:
            b3 = b4 = w_s = "?"

        if r["skipped"]:
            status = "skipped"
            t_s    = "—"
            evts   = f"{r['n_events']:,}"
            rate_s = "—"
        elif r["success"]:
            status = "OK"
            t_s    = _fmt_dur(r["elapsed"])
            evts   = f"{r['n_events']:,}"
            rate_s = (f"{r['n_events']/r['elapsed']:,.0f}"
                      if r["elapsed"] > 0 else "—")
        else:
            status = "FAILED"
            t_s = evts = rate_s = "—"

        lines.append(
            f"  {r['idx']:>5}  {status:>9}  {t_s:>9}  {evts:>9}  "
            f"{rate_s:>9}  {b3:>8}  {b4:>8}  {w_s:>7}"
        )

    lines += [
        "",
        f"  Total wall time (this run)  :  {_fmt_dur(t_wall)}",
        SEP,
        "",
    ]

    report = "\n".join(lines)
    print(report)

    rpt = work_dir / "trento_run_report.txt"
    rpt.write_text(report)
    log.info(f"Report saved  →  {rpt}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="208Pb+208Pb ultracentral scan — Stage 3: TRENTo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--trento", required=True, metavar="PATH",
        help="Path to the compiled TRENTo binary.",
    )
    p.add_argument(
        "--work-dir", required=True, metavar="PATH",
        help="Work directory from pbpb_isobar_gen.py (contains design_NNNN/).",
    )
    p.add_argument(
        "--n-events", type=int, default=500_000, metavar="N",
        help="Events per design point (default: 500000). "
             "Gives ~5000 UC events in the 0–1%% centrality bin.",
    )
    p.add_argument(
        "--sqrts", type=float, default=2.76, choices=[2.76, 5.02],
        metavar="TEV",
        help="Collision energy in TeV — sets norm and σ_NN (default: 2.76).",
    )
    p.add_argument(
        "--design-range", nargs=2, type=int, default=None,
        metavar=("START", "END"),
        help="Only process design indices [START, END). Default: all.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing trento_events.npy cache files.",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    trento_bin = Path(args.trento).expanduser().resolve()
    work_dir   = Path(args.work_dir).expanduser().resolve()

    if not work_dir.exists():
        log.error(f"Work directory not found: {work_dir}")
        sys.exit(1)

    print()
    print("=" * 72)
    print("  208Pb+208Pb Ultracentral Scan — Stage 3: TRENTo")
    print("=" * 72)
    print(f"  TRENTo binary  :  {trento_bin}")
    print(f"  Work directory :  {work_dir}")
    print(f"  Events / pt    :  {args.n_events:,}")
    print(f"  √s_NN          :  {args.sqrts} TeV")
    if args.design_range:
        print(f"  Design range   :  [{args.design_range[0]}, {args.design_range[1]})")
    print()

    _check_trento(trento_bin)
    print()

    t0      = time.perf_counter()
    records = stage3(
        trento_bin = trento_bin,
        work_dir   = work_dir,
        n_events   = args.n_events,
        sqrts      = args.sqrts,
        idx_range  = tuple(args.design_range) if args.design_range else None,
        force      = args.force,
    )
    t_wall = time.perf_counter() - t0

    print_report(
        records    = records,
        work_dir   = work_dir,
        n_events   = args.n_events,
        sqrts      = args.sqrts,
        trento_bin = trento_bin,
        t_wall     = t_wall,
    )

    if any(not r["success"] for r in records):
        log.warning("Some design points failed. See trento_run_report.txt.")
        sys.exit(1)


if __name__ == "__main__":
    main()
