#!/usr/bin/env python3
"""
pbpb_isobar_gen.py
==================
PART 2 of the 208Pb+208Pb ultracentral flow puzzle pipeline.

Reads the LHS design matrix produced by pbpb_lhs_design.py and runs the
Isobar-Sampler in two stages:

  Stage 1 — Create the nucleon-seed bank (runs ONCE, shared by all design pts)
  Stage 2 — Build nuclear configurations (runs once per design point)

Each design point produces two HDF files:
  WS1.hdf  — projectile 208Pb configurations
  WS2.hdf  — target    208Pb configurations (identical nuclear params)

These files are directly consumed by TRENTo in Part 3.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT: beta_4 and the Isobar Sampler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The Isobar Sampler (mluzum/Isobar-Sampler, arXiv:2302.14026) implements
deformation up to octupole order. The supported nuclear shape parameters are:

  WS_radius, WS_diffusiveness, beta_2, gamma, beta_3,
  correlation_length, correlation_strength

beta_4 (hexadecapole) is NOT a parameter of the Isobar Sampler.
The hexadecapole term beta_4 from the LHS design is stored in the per-design
metadata YAML for bookkeeping and will be needed at the TRENTo stage if a
modified TRENTo that accepts beta_4 is available. It is NOT passed to
build_isobars.py — doing so would cause a YAML parsing error.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECTORY LAYOUT produced by this script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  <WORK_DIR>/
    seeds/
      seeds-conf.yaml          ← isobar-sampler Stage-1 config
      nucleon-seeds.hdf        ← seed bank (created by make_seeds.py)
    design_0000/
      isobar-conf.yaml         ← isobar-sampler Stage-2 config
      design_meta.yaml         ← ALL LHS parameters including beta_4
      WS1.hdf                  ← projectile configs (from build_isobars.py)
      WS2.hdf                  ← target configs    (from build_isobars.py)
    design_0001/
      ...
    isobar_gen_report.txt      ← plain-text completion report

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Minimal — uses all defaults
  python pbpb_isobar_gen.py \\
      --isobar-dir /path/to/Isobar-Sampler \\
      --design     pbpb_lhs_output/lhs_design_matrix.npz

  # Full options
  python pbpb_isobar_gen.py \\
      --isobar-dir /path/to/Isobar-Sampler \\
      --design     pbpb_lhs_output/lhs_design_matrix.npz \\
      --work-dir   pbpb_scan_276GeV \\
      --n-configs  10000 \\
      --n-parallel -1 \\
      --skip-seeds         ← skip Stage 1 if seeds already exist
      --design-range 0 50  ← only run design points 0..49

PREREQUISITES
  pip install pyyaml numpy h5py
  git clone https://github.com/mluzum/Isobar-Sampler
  pip install -r Isobar-Sampler/requirements.txt

CITATION
  Luzum et al., Eur. Phys. J. A 59, 110 (2023) [arXiv:2302.14026]
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FIXED NUCLEAR DEFAULTS  (not sampled by the LHS)
# ─────────────────────────────────────────────────────────────────────────────

# These values are written into every isobar-conf.yaml.
# beta_2 = 0 because 208Pb is doubly magic (near-spherical ground state).
# gamma  = 0 because beta_2 = 0, so the triaxiality angle is irrelevant.
# correlation_length / correlation_strength: hard-core NN repulsion defaults
# from Luzum et al. arXiv:2302.14026 / arXiv:2312.10129.
FIXED_BETA2              = 0.0
FIXED_GAMMA              = 0.0    # radians
FIXED_CORR_LENGTH        = 0.4    # fm
FIXED_CORR_STRENGTH      = -1.0   # full Pauli exclusion

# 208Pb mass number
A_PB = 208

# Expected LHS column names (must match pbpb_lhs_design.py output)
EXPECTED_PARAM_NAMES = ["R", "a", "beta3", "beta4", "w"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list, label: str, cwd: Path = None) -> subprocess.CompletedProcess:
    """
    Run a subprocess command.  Streams stdout/stderr live to the terminal
    so the user sees progress from make_seeds.py / build_isobars.py.
    Exits with a clear error message on non-zero return code.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    log.info(f"  $ {cmd_str}")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        # Stream live output — do NOT capture, user must see progress
    )
    if result.returncode != 0:
        log.error(f"{label} failed with exit code {result.returncode}.")
        log.error(f"  Command: {cmd_str}")
        sys.exit(1)
    return result


def _load_design(npz_path: Path) -> tuple[np.ndarray, list[str]]:
    """
    Load design matrix from the Stage-0 .npz archive.
    Returns (design array (N,5), list of param names).
    """
    if not npz_path.exists():
        log.error(f"Design matrix not found: {npz_path}")
        log.error("Run pbpb_lhs_design.py first to generate the LHS.")
        sys.exit(1)

    data   = np.load(npz_path, allow_pickle=True)
    design = data["design"]                        # shape (N, 5)
    names  = list(data["param_names"].astype(str)) # ['R','a','beta3','beta4','w']

    # Validate column names — fail loudly if layout has changed
    if names != EXPECTED_PARAM_NAMES:
        log.error(
            f"Unexpected param names in design matrix: {names}\n"
            f"Expected: {EXPECTED_PARAM_NAMES}\n"
            "Check that pbpb_lhs_design.py and pbpb_isobar_gen.py are in sync."
        )
        sys.exit(1)

    log.info(f"Loaded design matrix: {design.shape[0]} points × "
             f"{design.shape[1]} params  from  {npz_path}")
    return design, names


def _fmt(val: float, decimals: int = 6) -> float:
    """Round to fixed decimal places for clean YAML output."""
    return round(float(val), decimals)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — NUCLEON SEED BANK
# ─────────────────────────────────────────────────────────────────────────────

def write_seeds_conf(seeds_dir: Path, n_configs: int, n_parallel: int) -> Path:
    """
    Write seeds-conf.yaml following the exact schema of
    Isobar-Sampler/examples/seeds-conf.yaml.
    """
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seeds_hdf = seeds_dir / "nucleon-seeds.hdf"

    cfg = {
        "isobar_seeds": {
            "description": (
                "Configurations for making list of seeds for nucleon positions. "
                "208Pb, 208 nucleons per configuration."
            ),
            "number_nucleons": {
                "description": "Mass number A of the nuclei.",
                "value": A_PB,
            },
            "number_configs": {
                "description": "How many sets of nucleon positions to sample?",
                "value": n_configs,
            },
            "output_file": {
                "description": "Path where to save list of seeds for nucleon positions.",
                "filename": str(seeds_hdf.resolve()),
            },
            "number_of_parallel_processes": {
                "description": (
                    "Number of processes to compute in parallel. "
                    "A value of -1 automatically selects all available CPUs."
                ),
                "value": n_parallel,
            },
        }
    }

    conf_path = seeds_dir / "seeds-conf.yaml"
    with open(conf_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    return conf_path, seeds_hdf


def stage1_seeds(isobar_dir: Path, work_dir: Path,
                 n_configs: int, n_parallel: int,
                 skip_if_exists: bool) -> Path:
    """
    Stage 1: create the nucleon-seed bank.

    The seed bank is created ONCE and reused by every design point in Stage 2.
    The key property is that all nuclear configurations across different design
    points are generated from the same underlying seeds, so that differences
    between design points reflect genuine nuclear-structure sensitivity and not
    statistical noise.
    """
    seeds_dir = work_dir / "seeds"
    seeds_hdf = seeds_dir / "nucleon-seeds.hdf"

    if skip_if_exists and seeds_hdf.exists():
        log.info(f"[Stage 1] Seed bank already exists — skipping.")
        log.info(f"  {seeds_hdf}")
        return seeds_hdf

    log.info("[Stage 1] Creating nucleon-seed bank for 208Pb ...")
    log.info(f"  n_configs  = {n_configs:,}")
    log.info(f"  n_parallel = {n_parallel}  (-1 = all CPUs)")

    conf_path, seeds_hdf = write_seeds_conf(seeds_dir, n_configs, n_parallel)
    log.info(f"  Seeds config  ->  {conf_path}")

    make_seeds = isobar_dir / "exec" / "make_seeds.py"
    if not make_seeds.exists():
        log.error(f"make_seeds.py not found: {make_seeds}")
        log.error("Check --isobar-dir points to the root of the Isobar-Sampler repo.")
        sys.exit(1)

    t0 = time.perf_counter()
    _run(["python3", str(make_seeds), str(conf_path)],
         label="make_seeds.py",
         cwd=isobar_dir)
    elapsed = time.perf_counter() - t0

    if not seeds_hdf.exists():
        log.error(
            f"Expected seed file not found after make_seeds.py:\n  {seeds_hdf}\n"
            "Possible causes:\n"
            "  • The output_file path in seeds-conf.yaml was overridden by the script\n"
            "  • make_seeds.py wrote the file relative to its own CWD\n"
            "Check the output above for the actual path used."
        )
        sys.exit(1)

    size_mb = seeds_hdf.stat().st_size / 1e6
    log.info(f"[Stage 1] Done in {elapsed:.1f} s  "
             f"|  {seeds_hdf.name}  ({size_mb:.1f} MB)")
    return seeds_hdf


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ISOBAR CONFIGURATIONS (per design point)
# ─────────────────────────────────────────────────────────────────────────────

def _isobar_block(name: str, R: float, a: float,
                  beta3: float, beta4: float) -> dict:
    """
    Build one isobar entry matching the schema of the modified
    Isobar-Sampler (ThiagoSDomingues/Isobar-Sampler), which adds beta_4
    support on top of the upstream mluzum/Isobar-Sampler.

    Supported fields:
      WS_radius, WS_diffusiveness, beta_2, gamma, beta_3, beta_4,
      correlation_length, correlation_strength
    """
    return {
        "isobar_name": name,
        "WS_radius": {
            "description": "Woods-Saxon radius parameter R [fm].",
            "value": _fmt(R),
        },
        "WS_diffusiveness": {
            "description": "Woods-Saxon diffusiveness parameter a [fm].",
            "value": _fmt(a),
        },
        "beta_2": {
            "description": (
                "Quadrupolar deformation beta_2. "
                "Fixed to 0: 208Pb is doubly magic (Z=82, N=126), "
                "near-spherical ground state."
            ),
            "value": _fmt(FIXED_BETA2),
        },
        "gamma": {
            "description": (
                "Quadrupolar deformation angle gamma [radians]. "
                "Fixed to 0 (axially symmetric; irrelevant when beta_2=0)."
            ),
            "value": _fmt(FIXED_GAMMA),
        },
        "beta_3": {
            "description": (
                "Octupolar deformation beta_3. "
                "KEY parameter for the v2-to-v3 ultracentral puzzle "
                "(Carzon et al. PRC 102, 054905, 2020)."
            ),
            "value": _fmt(beta3),
        },
        "correlation_length": {
            "description": (
                "Radius of the step-function short-range correlation "
                "function C(r) [fm]. Hard-core NN scale."
            ),
            "value": _fmt(FIXED_CORR_LENGTH),
        },
        "correlation_strength": {
            "description": (
                "Depth of correlation function for r < correlation_length. "
                "Must be >= -1. Value -1 = full Pauli exclusion."
            ),
            "value": _fmt(FIXED_CORR_STRENGTH),
        },
        "beta_4": {
            "description": (
                "Hexadecapolar deformation beta_4 of isobar. "
                "Supported by the modified Isobar-Sampler "
                "(ThiagoSDomingues/Isobar-Sampler)."
            ),
            "value": _fmt(beta4),
        },
    }


def write_isobar_conf(design_dir: Path, seeds_hdf: Path,
                      n_configs: int, n_parallel: int,
                      R: float, a: float, beta3: float, beta4: float) -> Path:
    """
    Write isobar-conf.yaml for build_isobars.py, following the schema of
    the modified Isobar-Sampler (ThiagoSDomingues/Isobar-Sampler) which
    adds beta_4 support.

    Both isobars (WS1 = projectile, WS2 = target) receive identical
    nuclear parameters — this is correct for symmetric Pb+Pb collisions.

    The output_path dirname is set to design_dir so that WS1.hdf and
    WS2.hdf land directly inside that design point's folder.
    """
    cfg = {
        "isobar_samples": {
            "description": "Options for the isobar nucleon-position samples.",
            "number_configs": {
                "description": "Number of configurations to be sampled.",
                "value": n_configs,
            },
            "number_nucleons": {
                "description": "Mass number A of the nuclei.",
                "value": A_PB,
            },
            "seeds_file": {
                "description": "Input file with list of seeds for nucleon positions.",
                "filename": str(seeds_hdf.resolve()),
            },
            "output_path": {
                "description": (
                    "Output directory where WS1.hdf and WS2.hdf will be saved."
                ),
                "dirname": str(design_dir.resolve()),
            },
            "number_of_parallel_processes": {
                "description": (
                    "Number of processes to compute in parallel. "
                    "-1 = use all available CPUs."
                ),
                "value": n_parallel,
            },
        },
        "isobar_properties": {
            "description": (
                "Nuclear properties of both 208Pb nuclei. "
                "WS1 = projectile, WS2 = target. "
                "Identical parameters for symmetric Pb+Pb. "
                "Results saved to WS1.hdf and WS2.hdf in output_path."
            ),
            "isobar1": _isobar_block("WS1", R, a, beta3, beta4),
            "isobar2": _isobar_block("WS2", R, a, beta3, beta4),
        },
    }

    conf_path = design_dir / "isobar-conf.yaml"
    with open(conf_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    return conf_path


def write_design_meta(design_dir: Path, idx: int,
                      R: float, a: float, beta3: float,
                      beta4: float, w: float) -> Path:
    """
    Write design_meta.yaml — stores ALL 5 LHS parameters for this design
    point. beta_4 is now passed to the modified Isobar Sampler directly.
    w (nucleon width) is the only parameter deferred to the TRENTo stage.
    This file is consumed by the TRENTo stage (Part 3).
    """
    meta = {
        "design_index": int(idx),
        "nucleus": "208Pb",
        "A": A_PB,

        # All parameters passed to the modified Isobar Sampler
        "isobar_sampler_params": {
            "WS_radius_fm":           _fmt(R),
            "WS_diffusiveness_fm":    _fmt(a),
            "beta_2":                 _fmt(FIXED_BETA2),
            "gamma_rad":              _fmt(FIXED_GAMMA),
            "beta_3":                 _fmt(beta3),
            "beta_4":                 _fmt(beta4),
            "correlation_length_fm":  _fmt(FIXED_CORR_LENGTH),
            "correlation_strength":   _fmt(FIXED_CORR_STRENGTH),
        },

        # Parameters deferred to the TRENTo stage
        "trento_stage_params": {
            "nucleon_width_w_fm": _fmt(w),
            "note_w": (
                "w is passed directly to TRENTo via the -w flag. "
                "It is not a nuclear structure parameter of the Isobar Sampler."
            ),
        },

        # HDF files produced by the Isobar Sampler
        "hdf_files": {
            "WS1": str((design_dir / "WS1.hdf").resolve()),
            "WS2": str((design_dir / "WS2.hdf").resolve()),
        },
    }

    meta_path = design_dir / "design_meta.yaml"
    with open(meta_path, "w") as fh:
        yaml.dump(meta, fh, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)
    return meta_path


def stage2_isobars(isobar_dir: Path, work_dir: Path, seeds_hdf: Path,
                   design: np.ndarray, n_configs: int, n_parallel: int,
                   idx_range: tuple[int, int]) -> list[int]:
    """
    Stage 2: build nuclear configurations for each design point.

    For each design point i in [idx_start, idx_end):
      1. Create work_dir/design_NNNN/
      2. Write isobar-conf.yaml  (passed to build_isobars.py)
      3. Write design_meta.yaml  (bookkeeping including beta_4, w)
      4. Run build_isobars.py    (produces WS1.hdf and WS2.hdf)
      5. Verify WS1.hdf and WS2.hdf exist and have non-zero size

    Returns list of successfully completed design-point indices.
    """
    build_isobars = isobar_dir / "exec" / "build_isobars.py"
    if not build_isobars.exists():
        log.error(f"build_isobars.py not found: {build_isobars}")
        sys.exit(1)

    idx_start, idx_end = idx_range
    n_total   = idx_end - idx_start
    completed = []
    skipped   = []
    failed    = []

    log.info(f"[Stage 2] Building isobar configurations ...")
    log.info(f"  Design points : {idx_start} – {idx_end - 1}  "
             f"({n_total} total)")
    log.info(f"  n_configs     : {n_configs:,} per design point")
    log.info(f"  n_parallel    : {n_parallel}  (-1 = all CPUs)")
    log.info(f"  Seeds file    : {seeds_hdf}")
    print()

    t_total_start = time.perf_counter()

    for i in range(idx_start, idx_end):
        row   = design[i]
        R     = float(row[0])   # WS radius [fm]
        a     = float(row[1])   # WS diffuseness [fm]
        beta3 = float(row[2])   # octupole  — passed to isobar sampler
        beta4 = float(row[3])   # hexadecapole — NOT passed to isobar sampler
        w     = float(row[4])   # nucleon width — for TRENTo stage only

        design_dir = work_dir / f"design_{i:04d}"
        design_dir.mkdir(parents=True, exist_ok=True)

        ws1 = design_dir / "WS1.hdf"
        ws2 = design_dir / "WS2.hdf"

        # ── Skip if already done ─────────────────────────────────────────────
        if ws1.exists() and ws2.exists():
            s1, s2 = ws1.stat().st_size, ws2.stat().st_size
            if s1 > 0 and s2 > 0:
                log.info(f"  [{i:04d}] WS1.hdf and WS2.hdf exist "
                         f"({s1/1e6:.1f} MB / {s2/1e6:.1f} MB) — skipping.")
                skipped.append(i)
                continue
            else:
                log.warning(f"  [{i:04d}] Existing HDF files have zero size "
                            "— regenerating.")

        # ── Write configs ─────────────────────────────────────────────────────
        isobar_conf = write_isobar_conf(
            design_dir, seeds_hdf, n_configs, n_parallel, R, a, beta3, beta4)
        meta_path = write_design_meta(
            design_dir, i, R, a, beta3, beta4, w)

        # ── Log this design point ─────────────────────────────────────────────
        log.info(
            f"  [{i:04d}/{idx_end-1:04d}]  "
            f"R={R:.4f}  a={a:.4f}  "
            f"β₃={beta3:.5f}  β₄={beta4:.5f}  w={w:.4f}"
        )
        log.info(f"    isobar-conf  ->  {isobar_conf}")
        log.info(f"    design_meta  ->  {meta_path}")

        # ── Run build_isobars.py ──────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            _run(
                ["python3", str(build_isobars), str(isobar_conf)],
                label=f"build_isobars [{i:04d}]",
                cwd=isobar_dir,
            )
        except SystemExit:
            log.error(f"  [{i:04d}] build_isobars.py FAILED — continuing to "
                      "next design point.")
            failed.append(i)
            continue
        elapsed = time.perf_counter() - t0

        # ── Verify outputs ───────────────────────────────────────────────────
        missing = []
        for hdf, label in [(ws1, "WS1.hdf"), (ws2, "WS2.hdf")]:
            if not hdf.exists():
                missing.append(label)
            elif hdf.stat().st_size == 0:
                missing.append(f"{label} (zero bytes)")

        if missing:
            log.error(
                f"  [{i:04d}] Expected output file(s) not found or empty:\n"
                + "".join(f"    • {m}\n" for m in missing)
                + f"  Output path specified in isobar-conf.yaml:\n"
                + f"    {design_dir.resolve()}\n"
                + "  Possible causes:\n"
                + "  • build_isobars.py wrote files relative to its own CWD\n"
                + "  • The dirname key in output_path was interpreted differently\n"
                + "  Inspect the output above for the actual path used.\n"
                + "  You can re-run this point after fixing the path issue."
            )
            failed.append(i)
        else:
            s1 = ws1.stat().st_size / 1e6
            s2 = ws2.stat().st_size / 1e6
            log.info(f"    WS1.hdf  {s1:.1f} MB  |  WS2.hdf  {s2:.1f} MB  "
                     f"|  {elapsed:.1f} s")
            completed.append(i)

        print()  # blank line between design points for readability

    elapsed_total = time.perf_counter() - t_total_start
    return completed, skipped, failed, elapsed_total


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_report(work_dir: Path, design: np.ndarray,
                 completed: list, skipped: list, failed: list,
                 elapsed: float, seeds_hdf: Path,
                 isobar_dir: Path) -> None:
    """Write a plain-text completion report."""
    n = design.shape[0]
    lines = [
        "=" * 72,
        "  208Pb+208Pb Isobar Generation Report",
        f"  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 72,
        "",
        f"  Isobar-Sampler dir : {isobar_dir.resolve()}",
        f"  Work directory     : {work_dir.resolve()}",
        f"  Seed bank          : {seeds_hdf.resolve()}",
        f"  Total design pts   : {n}",
        f"  Completed          : {len(completed)}",
        f"  Skipped (exist)    : {len(skipped)}",
        f"  Failed             : {len(failed)}",
        f"  Wall time          : {elapsed:.1f} s",
        "",
    ]

    if failed:
        lines += [
            "  FAILED design points:",
            *[f"    design_{i:04d}/" for i in failed],
            "",
        ]

    lines += [
        "  Per-design-point summary:",
        f"  {'idx':>5}  {'R [fm]':>8}  {'a [fm]':>8}  "
        f"{'beta3':>8}  {'beta4':>8}  {'w [fm]':>8}  {'status':>10}",
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
        "  Next step: run  pbpb_trento_run.py  to run TRENTo on all design",
        "  points.  Pass --work-dir pointing to this directory.",
        "=" * 72,
    ]

    text     = "\n".join(lines)
    out_path = work_dir / "isobar_gen_report.txt"
    out_path.write_text(text)
    print(text)
    log.info(f"Report  ->  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pb+Pb ultracentral scan — Stage 1 (seeds) + Stage 2 (isobars)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument(
        "--isobar-dir", required=True, metavar="PATH",
        help="Root directory of the Isobar-Sampler repository "
             "(must contain exec/make_seeds.py and exec/build_isobars.py).",
    )
    p.add_argument(
        "--design", required=True, metavar="NPZ",
        help="Path to lhs_design_matrix.npz produced by pbpb_lhs_design.py.",
    )

    # Optional paths
    p.add_argument(
        "--work-dir", default="pbpb_scan", metavar="PATH",
        help="Working directory for all outputs (default: pbpb_scan/).",
    )

    # Sampler settings
    p.add_argument(
        "--n-configs", type=int, default=10000, metavar="N",
        help="Number of nucleon-position configurations per nucleus "
             "(seed bank size, default: 10000).",
    )
    p.add_argument(
        "--n-parallel", type=int, default=-1, metavar="N",
        help="Parallel processes for the isobar sampler. "
             "-1 = use all available CPUs (default: -1).",
    )

    # Stage control
    p.add_argument(
        "--skip-seeds", action="store_true",
        help="Skip Stage 1 if nucleon-seeds.hdf already exists.",
    )
    p.add_argument(
        "--skip-isobars", action="store_true",
        help="Skip Stage 2 entirely (only run Stage 1).",
    )

    # Design-point range
    p.add_argument(
        "--design-range", nargs=2, type=int, default=None,
        metavar=("START", "END"),
        help="Only run design points in [START, END). "
             "Default: run all design points.",
    )

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    isobar_dir = Path(args.isobar_dir).expanduser().resolve()
    work_dir   = Path(args.work_dir).expanduser().resolve()
    design_npz = Path(args.design).expanduser().resolve()

    work_dir.mkdir(parents=True, exist_ok=True)

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  208Pb+208Pb Ultracentral Scan — Isobar Generation  (Parts 1–2)")
    print("=" * 72)
    print(f"  Isobar-Sampler  :  {isobar_dir}")
    print(f"  Design matrix   :  {design_npz}")
    print(f"  Work directory  :  {work_dir}")
    print(f"  n_configs       :  {args.n_configs:,}")
    print(f"  n_parallel      :  {args.n_parallel}  (-1 = all CPUs)")
    print()

    # ── Load design ───────────────────────────────────────────────────────────
    design, param_names = _load_design(design_npz)
    n_pts               = design.shape[0]

    # Determine design-point index range
    if args.design_range is not None:
        idx_start, idx_end = args.design_range
        if not (0 <= idx_start < idx_end <= n_pts):
            log.error(f"--design-range {idx_start} {idx_end} is out of bounds "
                      f"for design with {n_pts} points.")
            sys.exit(1)
    else:
        idx_start, idx_end = 0, n_pts

    print(f"  Design points to process: {idx_start} – {idx_end - 1} "
          f"({idx_end - idx_start} points)\n")

    # ── Stage 1: Seed bank ────────────────────────────────────────────────────
    seeds_hdf = stage1_seeds(
        isobar_dir=isobar_dir,
        work_dir=work_dir,
        n_configs=args.n_configs,
        n_parallel=args.n_parallel,
        skip_if_exists=args.skip_seeds,
    )
    print()

    # ── Stage 2: Isobar configurations ───────────────────────────────────────
    if args.skip_isobars:
        log.info("[Stage 2] Skipped (--skip-isobars flag set).")
        return

    completed, skipped, failed, elapsed = stage2_isobars(
        isobar_dir=isobar_dir,
        work_dir=work_dir,
        seeds_hdf=seeds_hdf,
        design=design,
        n_configs=args.n_configs,
        n_parallel=args.n_parallel,
        idx_range=(idx_start, idx_end),
    )

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(
        work_dir=work_dir,
        design=design,
        completed=completed,
        skipped=skipped,
        failed=failed,
        elapsed=elapsed,
        seeds_hdf=seeds_hdf,
        isobar_dir=isobar_dir,
    )

    # Exit with error code if any design points failed
    if failed:
        log.warning(f"{len(failed)} design point(s) failed. "
                    "See report above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
