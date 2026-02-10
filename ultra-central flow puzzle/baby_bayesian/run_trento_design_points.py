"""
This script generates realistic nucleon positions for nuclei with arbitrary deformation parameters.
Author: OptimusThi
"""

import os
import subprocess
from pathlib import Path

# -------------------------------
# USER CONFIGURATION
# -------------------------------
TRENT0_EXEC = "./trento"
NUCLEAR_DIR = "/home/thiagodomingues/"
OUTPUT_DIR = "./trento_events"

N_EVENTS = 100_000
SEED = 1

# Trento parameters (example)
TRENTO_PARAMS = {
    "--reduced-thickness": "0.063",
    "--fluctuation": "1.05",
    "--nucleon-width": "1.12",
    "--cross-section": "4.23",
    "--normalization": "5.73",
    "--b-min": "0",
    "--b-max": "20"
}

# -------------------------------
# SETUP
# -------------------------------
Path(OUTPUT_DIR).mkdir(exist_ok=True)

hdf_files = sorted([
    f for f in os.listdir(NUCLEAR_DIR)
    if f.endswith(".hdf")
])

# -------------------------------
# RUN TRENTO
# -------------------------------
for i, hdf in enumerate(hdf_files, start=1):
    nucleus = os.path.join(NUCLEAR_DIR, hdf)
    output = os.path.join(OUTPUT_DIR, f"design_{i}.txt")

    cmd = [
        TRENT0_EXEC,
        nucleus,
        nucleus,
        str(N_EVENTS),
        "--random-seed", str(SEED),
        "--quiet"
    ]

    for k, v in TRENTO_PARAMS.items():
        cmd.extend([k, v])

    print(f"Running design point {i}")
    with open(output, "w") as f:
        subprocess.run(cmd, stdout=f)

print("All TRENTo runs completed.")
