"""
Script to End-to-end example (one design point) of calculate weigthed observables.
Author: OptimusThi
"""

import numpy as np
from reweighting import compute_weights
from eccentricities import epsilon_cumulants

data = np.loadtxt("trento_events/design_1.txt")

mult = data[:, 3]
eps2 = data[:, 4]
eps3 = data[:, 5]

# Define ultra-central target
weights = compute_weights(
    mult,
    target="gaussian",
    mean=np.percentile(mult, 99.8),
    sigma=0.01 * np.mean(mult)
)

e2_2, e2_4 = epsilon_cumulants(eps2, weights)
e3_2, _    = epsilon_cumulants(eps3, weights)

print("Reweighted ultra-central:")
print(f"ε2{{2}} = {e2_2:.4f}")
print(f"ε2{{4}} = {e2_4:.4f}")
print(f"ε3{{2}} = {e3_2:.4f}")
