
"""
Script to Generate a TRENTo design (LHS)

Author: OptimusThi
"""

import numpy as np
from pyDOE import lhs

# Number of design points
N = 100

# Parameter ranges
ranges = {
    "p": (-1.0, 1.0),
    "k": (0.5, 2.0),
    "w": (0.4, 0.8),
    "beta2": (0.0, 0.2),
    "beta4": (-0.05, 0.05),
}

param_names = list(ranges.keys())
lhs_samples = lhs(len(ranges), samples=N)

design = np.zeros_like(lhs_samples)

for i, p in enumerate(param_names):
    lo, hi = ranges[p]
    design[:, i] = lo + (hi - lo) * lhs_samples[:, i]

np.savetxt("trento_design.txt", design,
           header=" ".join(param_names))
