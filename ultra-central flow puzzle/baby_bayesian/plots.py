"""
Script to plot eccentricities cumulants. 
"""

import matplotlib.pyplot as plt
from cumulants import * 

plt.hist(eps2[uc_idx], bins=60, density=True, alpha=0.6, label=r"$\varepsilon_2$")
plt.hist(eps3[uc_idx], bins=60, density=True, alpha=0.6, label=r"$\varepsilon_3$")
plt.xlabel(r"$\varepsilon_n$")
plt.ylabel("Probability density")
plt.legend()
plt.tight_layout()
plt.show()

# Cumulant comparison plot
labels = [r"$\varepsilon_2\{2\}$", r"$\varepsilon_2\{4\}$", r"$\varepsilon_3\{2\}$"]
values = [obs["eps2{2}"], obs["eps2{4}"], obs["eps3{2}"]]

plt.bar(labels, values)
plt.ylabel("Value")
plt.title("Ultra-central eccentricity cumulants")
plt.tight_layout()
plt.show()

# Ratio diagnostic
plt.bar(
    [r"$\varepsilon_2\{2\}/\varepsilon_3\{2\}$",
     r"$\varepsilon_2\{4\}/\varepsilon_2\{2\}$"],
    [obs["eps2{2}/eps3{2}"], obs["eps2{4}/eps2{2}"]]
)
plt.ylabel("Ratio")
plt.title("Ultra-central flow puzzle diagnostics")
plt.tight_layout()
plt.show()
