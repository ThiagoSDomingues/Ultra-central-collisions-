"""
Script to create plots for the ultra-central flow puzzle project
Author: OptimusThi
"""
import matplotlib.pyplot as plt
 
plt.hist(central[:, 4], bins=50, density=True)
plt.xlabel(r"$\varepsilon_2$")
plt.ylabel("Probability density")
plt.title("Ultra-central (0–0.1%)")
plt.show()
