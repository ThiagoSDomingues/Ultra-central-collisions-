import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("trento_events")
CENTRALITY_BINS = [(0, 1), (0, 0.5), (0, 0.1)]  # %

def load_events(fname):
    return np.loadtxt(fname)

def select_central(events, cmin, cmax):
    # multiplicity column = 3
    mult = events[:, 3]
    order = np.argsort(mult)[::-1]
    events = events[order]

    n = len(events)
    i_min = int(cmin / 100 * n)
    i_max = int(cmax / 100 * n)
    return events[i_min:i_max]

def cumulants(eps):
    eps2 = eps**2
    eps4 = eps**4

    e2_2 = np.sqrt(np.mean(eps2))
    e2_4 = (2*np.mean(eps2)**2 - np.mean(eps4))**0.25

    return e2_2, e2_4

# -------------------------------
# MAIN LOOP
# -------------------------------
for fname in sorted(DATA_DIR.glob("design_*.txt")):
    events = load_events(fname)

    eps2 = events[:, 4]
    eps3 = events[:, 5]

    for cmin, cmax in CENTRALITY_BINS:
        central = select_central(events, cmin, cmax)

        e2_2, e2_4 = cumulants(central[:, 4])
        e3_2, _    = cumulants(central[:, 5])

        print(
            fname.name,
            f"{cmin}-{cmax}%",
            f"ε2{{2}}={e2_2:.4f}",
            f"ε2{{4}}={e2_4:.4f}",
            f"ε3{{2}}={e3_2:.4f}"
        )
