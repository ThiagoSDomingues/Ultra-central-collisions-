#!/usr/bin/env python3
"""
setup_isobar_sampler.py
=======================
Stages 1+2 for the 208Pb+208Pb ultracentral flow puzzle pipeline:

  Stage 1 – Create the nucleon‑seed bank (once, shared by all design points)
  Stage 2 – Build nuclear configuration HDF5 files (WS1.hdf, WS2.hdf)
            for each design point from an LHS design matrix.

This script is the improved, all‑in‑one version that replaces the basic
setup_isobar_sampler.py and includes all features from pbpb_isobar_gen.py.

USAGE (command line)
--------------------
  python setup_isobar_sampler.py \\
      --isobar-dir /path/to/Isobar-Sampler \\
      --design     lhs_design_matrix.npz \\
      --work-dir   pbpb_scan \\
      --n-configs  10000 \\
      --n-parallel -1 \\
      [--skip-seeds] \\
      [--design-range 0 50]

USAGE (import in Jupyter)
-------------------------
  from setup_isobar_sampler import setup_isobar_sampler
  
  seeds_hdf, design, report = setup_isobar_sampler(
      isobar_dir="/path/to/Isobar-Sampler",
      design_npz="lhs_design_matrix.npz",
      work_dir="pbpb_scan",
      n_configs=10000,
      n_parallel=-1,
      skip_seeds=False,
      design_range=(0, None)   # all points
  )
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import yaml

# ----------------------------------------------------------------------
# Constants & fixed nuclear parameters
# ----------------------------------------------------------------------
A_PB = 208
FIXED_GAMMA = 0.0          # radians
FIXED_CORR_LENGTH = 0.4    # fm
FIXED_CORR_STRENGTH = -1.0

EXPECTED_PARAM_NAMES = ["WS_R", "WS_A", "beta2", "beta3", "beta4", "w"]

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _run(cmd: list, label: str, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a command, stream output, exit on error."""
    cmd_str = " ".join(str(c) for c in cmd)
    log.info(f"  $ {cmd_str}")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
    )
    if result.returncode != 0:
        log.error(f"{label} failed (exit {result.returncode})")
        sys.exit(1)
    return result


def _load_design(npz_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load design matrix from pbpb_lhs_design.py output."""
    if not npz_path.exists():
        log.error(f"Design matrix not found: {npz_path}")
        sys.exit(1)
    data = np.load(npz_path, allow_pickle=True)
    design = data["design"]                     # (N, 5)
    names = list(data["param_names"].astype(str))
    if names != EXPECTED_PARAM_NAMES:
        log.error(f"Unexpected param names: {names}\nExpected: {EXPECTED_PARAM_NAMES}")
        sys.exit(1)
    log.info(f"Loaded {design.shape[0]} design points from {npz_path}")
    return design, names


def _fmt(val: float, decimals: int = 6) -> float:
    return round(float(val), decimals)


# ----------------------------------------------------------------------
# Stage 1: Seed bank (identical to pbpb_isobar_gen.py)
# ----------------------------------------------------------------------
def write_seeds_conf(seeds_dir: Path, n_configs: int, n_parallel: int) -> Tuple[Path, Path]:
    """Write seeds-conf.yaml and return (conf_path, seeds_hdf_path)."""
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seeds_hdf = seeds_dir / "nucleon-seeds.hdf"
    cfg = {
        "isobar_seeds": {
            "description": f"{A_PB}Pb seed bank, {n_configs} configs.",
            "number_nucleons": {"description": "A", "value": A_PB},
            "number_configs": {"description": "N_configs", "value": n_configs},
            "output_file": {"description": "HDF5 output", "filename": str(seeds_hdf.resolve())},
            "number_of_parallel_processes": {"description": "-1=all CPUs", "value": n_parallel},
        }
    }
    conf_path = seeds_dir / "seeds-conf.yaml"
    with open(conf_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return conf_path, seeds_hdf


def stage1_seeds(isobar_dir: Path, work_dir: Path,
                 n_configs: int, n_parallel: int,
                 skip_if_exists: bool) -> Path:
    """Create the nucleon-seed bank (shared by all design points)."""
    seeds_dir = work_dir / "seeds"
    seeds_hdf = seeds_dir / "nucleon-seeds.hdf"
    if skip_if_exists and seeds_hdf.exists() and seeds_hdf.stat().st_size > 0:
        log.info(f"[Stage 1] Seed bank already exists: {seeds_hdf} "
                 f"({seeds_hdf.stat().st_size/1e6:.1f} MB)")
        return seeds_hdf

    log.info("[Stage 1] Creating nucleon-seed bank...")
    log.info(f"  n_configs = {n_configs:,}, n_parallel = {n_parallel}")
    conf_path, seeds_hdf = write_seeds_conf(seeds_dir, n_configs, n_parallel)
    make_seeds = isobar_dir / "exec" / "make_seeds.py"
    if not make_seeds.exists():
        log.error(f"make_seeds.py not found in {isobar_dir}/exec/")
        sys.exit(1)

    t0 = time.perf_counter()
    _run(["python3", str(make_seeds), str(conf_path)], label="make_seeds.py", cwd=isobar_dir)
    elapsed = time.perf_counter() - t0

    if not seeds_hdf.exists() or seeds_hdf.stat().st_size == 0:
        log.error(f"Seed bank not created correctly: {seeds_hdf}")
        sys.exit(1)

    log.info(f"  Done in {elapsed:.1f} s | {seeds_hdf.name} ({seeds_hdf.stat().st_size/1e6:.1f} MB)")
    return seeds_hdf


# ----------------------------------------------------------------------
# Stage 2: Per-design-point isobar configuration generation
# ----------------------------------------------------------------------
def _isobar_block(name: str, R: float, a: float, beta2: float, beta3: float, beta4: float) -> Dict[str, Any]:
    """Create an isobar dictionary for the modified Isobar Sampler (supports beta4)."""
    return {
        "isobar_name": name,
        "WS_radius": {"description": "R [fm]", "value": _fmt(R)},
        "WS_diffusiveness": {"description": "a [fm]", "value": _fmt(a)},
        "beta_2": {"description": "quadrupole β₂", "value": _fmt(beta2)},
        "gamma": {"description": "γ [rad]", "value": _fmt(FIXED_GAMMA)},
        "beta_3": {"description": "octupole β₃", "value": _fmt(beta3)},
        "correlation_length": {"description": "C_l [fm]", "value": _fmt(FIXED_CORR_LENGTH)},
        "correlation_strength": {"description": "C_s", "value": _fmt(FIXED_CORR_STRENGTH)},
        "beta_4": {"description": "hexadecapole β₄", "value": _fmt(beta4)},
    }


def write_isobar_conf(design_dir: Path, seeds_hdf: Path,
                      n_configs: int, n_parallel: int,
                      R: float, a: float, beta2: float, beta3: float, beta4: float) -> Path:
    """Write isobar-conf.yaml for build_isobars.py."""
    cfg = {
        "isobar_samples": {
            "description": "Isobar sampler configuration",
            "number_configs": {"description": "N", "value": n_configs},
            "number_nucleons": {"description": "A", "value": A_PB},
            "seeds_file": {"description": "seed bank", "filename": str(seeds_hdf.resolve())},
            "output_path": {"description": "output dir", "dirname": str(design_dir.resolve())},
            "number_of_parallel_processes": {"description": "-1=all", "value": n_parallel},
        },
        "isobar_properties": {
            "description": "Projectile and target (identical)",
            "isobar1": _isobar_block("WS1", R, a, beta2, beta3, beta4),
            "isobar2": _isobar_block("WS2", R, a, beta2, beta3, beta4),
        },
    }
    conf_path = design_dir / "isobar-conf.yaml"
    with open(conf_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return conf_path


def write_design_meta(design_dir: Path, idx: int,
                      R: float, a: float, beta2: float, beta3: float, beta4: float, w: float) -> Path:
    """Write design_meta.yaml with all parameters for the TRENTo stage."""
    meta = {
        "design_index": idx,
        "nucleus": "208Pb",
        "A": A_PB,
        "isobar_sampler_params": {
            "WS_radius_fm": _fmt(R),
            "WS_diffusiveness_fm": _fmt(a),
            "beta_2": _fmt(beta2),
            "gamma_rad": _fmt(FIXED_GAMMA),
            "beta_3": _fmt(beta3),
            "beta_4": _fmt(beta4),
            "correlation_length_fm": _fmt(FIXED_CORR_LENGTH),
            "correlation_strength": _fmt(FIXED_CORR_STRENGTH),
        },
        "trento_stage_params": {
            "nucleon_width_w_fm": _fmt(w),
            "note": "w is passed to TRENTo via -w flag, not used by Isobar Sampler.",
        },
        "hdf_files": {
            "WS1": str((design_dir / "WS1.hdf").resolve()),
            "WS2": str((design_dir / "WS2.hdf").resolve()),
        },
    }
    meta_path = design_dir / "design_meta.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False, sort_keys=False)
    return meta_path


def stage2_isobars(isobar_dir: Path, work_dir: Path, seeds_hdf: Path,
                   design: np.ndarray, n_configs: int, n_parallel: int,
                   idx_range: Tuple[int, int]) -> Tuple[List[int], List[int], List[int], float]:
    """
    Build isobar configurations for design points in [idx_start, idx_end).

    Returns: (completed_indices, skipped_indices, failed_indices, total_elapsed)
    """
    build_isobars = isobar_dir / "exec" / "build_isobars.py"
    if not build_isobars.exists():
        log.error(f"build_isobars.py not found: {build_isobars}")
        sys.exit(1)

    idx_start, idx_end = idx_range
    n_total = idx_end - idx_start
    completed = []
    skipped = []
    failed = []

    log.info(f"[Stage 2] Building isobar configurations for {n_total} design points...")
    log.info(f"  n_configs = {n_configs:,}, n_parallel = {n_parallel}")
    log.info(f"  Seeds: {seeds_hdf}")

    t_total_start = time.perf_counter()

    for i in range(idx_start, idx_end):
        row = design[i]
        R, a, beta2, beta3, beta4, w = row

        design_dir = work_dir / f"design_{i:04d}"
        design_dir.mkdir(parents=True, exist_ok=True)

        ws1 = design_dir / "WS1.hdf"
        ws2 = design_dir / "WS2.hdf"

        # Skip if both HDF5 files already exist and are non‑empty
        if ws1.exists() and ws2.exists() and ws1.stat().st_size > 0 and ws2.stat().st_size > 0:
            log.info(f"  [{i:04d}] WS1.hdf and WS2.hdf already exist – skipping.")
            skipped.append(i)
            continue

        # Write configuration files
        isobar_conf = write_isobar_conf(design_dir, seeds_hdf, n_configs, n_parallel, R, a, beta2, beta3, beta4)
        meta_path = write_design_meta(design_dir, i, R, a, beta2, beta3, beta4, w)

        log.info(f"  [{i:04d}] R={R:.4f} a={a:.4f} β₂={beta2:.5f} β₃={beta3:.5f} β₄={beta4:.5f} w={w:.4f}")
        log.info(f"    isobar-conf -> {isobar_conf}")
        log.info(f"    design_meta -> {meta_path}")

        t0 = time.perf_counter()
        try:
            _run(["python3", str(build_isobars), str(isobar_conf)],
                 label=f"build_isobars [{i:04d}]", cwd=isobar_dir)
        except SystemExit:
            log.error(f"  [{i:04d}] build_isobars.py FAILED.")
            failed.append(i)
            continue
        elapsed = time.perf_counter() - t0

        # Verify outputs
        if ws1.exists() and ws2.exists() and ws1.stat().st_size > 0 and ws2.stat().st_size > 0:
            s1, s2 = ws1.stat().st_size / 1e6, ws2.stat().st_size / 1e6
            log.info(f"    WS1.hdf {s1:.1f} MB | WS2.hdf {s2:.1f} MB | {elapsed:.1f} s")
            completed.append(i)
        else:
            log.error(f"  [{i:04d}] Output HDF files missing or empty.")
            failed.append(i)

        print()  # blank line for readability

    elapsed_total = time.perf_counter() - t_total_start
    return completed, skipped, failed, elapsed_total


# ----------------------------------------------------------------------
# Report generation
# ----------------------------------------------------------------------
def write_report(work_dir: Path, design: np.ndarray,
                 completed: List[int], skipped: List[int], failed: List[int],
                 elapsed: float, seeds_hdf: Path, isobar_dir: Path) -> None:
    """Write a plain‑text completion report."""
    n_pts = design.shape[0]
    lines = [
        "=" * 72,
        "  Isobar Sampler Completion Report",
        f"  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        f"  Isobar-Sampler dir : {isobar_dir.resolve()}",
        f"  Work directory     : {work_dir.resolve()}",
        f"  Seed bank          : {seeds_hdf.resolve()}",
        f"  Total design pts   : {n_pts}",
        f"  Completed          : {len(completed)}",
        f"  Skipped (exist)    : {len(skipped)}",
        f"  Failed             : {len(failed)}",
        f"  Wall time          : {elapsed:.1f} s",
        "",
    ]
    if failed:
        lines += ["  FAILED design points:", *[f"    design_{i:04d}/" for i in failed], ""]
    lines += [
        "  Per-design-point summary:",
        f"  {'idx':>5}  {'R [fm]':>8}  {'a [fm]':>8}  "
        f"{'beta2':>8}  {'beta3':>8}"  
        f"{'beta4':>8}  {'w [fm]':>8}  {'status':>10}",
        "  " + "-" * 65,
    ]
    for i, row in enumerate(design):
        if i in completed:
            status = "OK"
        elif i in skipped:
            status = "skipped"
        elif i in failed:
            status = "FAILED"
        else:
            status = "not run"
        lines.append(
            f"  {i:>5}  {row[0]:>8.5f}  {row[1]:>8.5f}  "
            f"{row[2]:>8.5f}  {row[3]:>8.5f}  {row[4]:>8.5f}  {status:>10}"
        )
    lines += [
        "",
        "  Next step: run TRENTo using the WS1.hdf/WS2.hdf files and design_meta.yaml",
        "  from each design_XXXX/ folder.",
        "=" * 72,
    ]
    out_path = work_dir / "isobar_gen_report.txt"
    out_path.write_text("\n".join(lines))
    log.info(f"Report saved to {out_path}")


# ----------------------------------------------------------------------
# Main orchestrator (importable and CLI)
# ----------------------------------------------------------------------
def setup_isobar_sampler(
    isobar_dir: Path,
    design_npz: Path,
    work_dir: Path,
    n_configs: int = 10000,
    n_parallel: int = -1,
    skip_seeds: bool = False,
    design_range: Optional[Tuple[int, Optional[int]]] = None,
) -> Tuple[Path, np.ndarray, Dict[str, Any]]:
    """
    Run the full isobar sampler pipeline (stages 1+2).

    Parameters
    ----------
    isobar_dir : Path
        Root directory of the Isobar-Sampler repository.
    design_npz : Path
        Path to lhs_design_matrix.npz from pbpb_lhs_design.py.
    work_dir : Path
        Working directory for all outputs.
    n_configs : int
        Number of nucleon configurations per nucleus.
    n_parallel : int
        Number of parallel processes (-1 = all CPUs).
    skip_seeds : bool
        If True and seed bank already exists, reuse it.
    design_range : tuple (start, end) or None
        Range of design indices to process. If None, process all.

    Returns
    -------
    seeds_hdf : Path
        Path to the nucleon seed bank.
    design : np.ndarray
        The full design matrix (all points).
    report : dict
        Summary information: completed, skipped, failed indices.
    """
    # Convert to absolute paths
    isobar_dir = Path(isobar_dir).expanduser().resolve()
    design_npz = Path(design_npz).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 68)
    log.info("  Isobar Sampler Pipeline (Stages 1+2)")
    log.info(f"  Isobar dir: {isobar_dir}")
    log.info(f"  Design file: {design_npz}")
    log.info(f"  Work dir: {work_dir}")
    log.info(f"  n_configs: {n_configs:,}, n_parallel: {n_parallel}")
    log.info("=" * 68)

    # Load design matrix
    design, param_names = _load_design(design_npz)
    n_pts = design.shape[0]

    # Determine design point range
    if design_range is None:
        idx_start, idx_end = 0, n_pts
    else:
        idx_start, idx_end = design_range
        if idx_end is None:
            idx_end = n_pts
        if not (0 <= idx_start < idx_end <= n_pts):
            raise ValueError(f"Invalid design range {idx_start}:{idx_end} for {n_pts} points")

    # Stage 1: seeds
    seeds_hdf = stage1_seeds(isobar_dir, work_dir, n_configs, n_parallel, skip_seeds)

    # Stage 2: per‑point isobars
    completed, skipped, failed, elapsed = stage2_isobars(
        isobar_dir, work_dir, seeds_hdf, design,
        n_configs, n_parallel, (idx_start, idx_end)
    )

    # Report
    write_report(work_dir, design, completed, skipped, failed, elapsed, seeds_hdf, isobar_dir)

    report = {
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "elapsed": elapsed,
        "seeds_hdf": seeds_hdf,
        "work_dir": work_dir,
    }
    return seeds_hdf, design, report


# ----------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isobar Sampler pipeline (seeds + per‑point configs)")
    p.add_argument("--isobar-dir", required=True, help="Isobar-Sampler repository root")
    p.add_argument("--design", required=True, help="LHS design matrix .npz file")
    p.add_argument("--work-dir", default="pbpb_scan", help="Working directory")
    p.add_argument("--n-configs", type=int, default=10000, help="Number of configurations per nucleus")
    p.add_argument("--n-parallel", type=int, default=-1, help="Parallel processes (-1 = all cores)")
    p.add_argument("--skip-seeds", action="store_true", help="Reuse existing seed bank")
    p.add_argument("--design-range", nargs=2, type=int, default=None,
                   metavar=("START", "END"), help="Indices of design points to process")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    design_range = None
    if args.design_range is not None:
        design_range = (args.design_range[0], args.design_range[1])

    seeds_hdf, design, report = setup_isobar_sampler(
        isobar_dir=args.isobar_dir,
        design_npz=args.design,
        work_dir=args.work_dir,
        n_configs=args.n_configs,
        n_parallel=args.n_parallel,
        skip_seeds=args.skip_seeds,
        design_range=design_range,
    )
    if report["failed"]:
        log.warning(f"{len(report['failed'])} design point(s) failed. Check report.")
        sys.exit(1)
    log.info("All done.")


if __name__ == "__main__":
    main()
