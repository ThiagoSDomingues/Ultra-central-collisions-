#!shebang 

import numpy as np
import h5py

# Ultracentral centrality bins
UCC_cent_bins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) # 0-10% in intervals of 1%


# input: Trento events eccentricities
# epsilon_n 

rms_n = np.sqrt(epsilon_n**2)

# geometry observables

# 2-particle correlation
epsilon_n_2 = np.sqrt(np.average(epsilon_n**2)) # rms of Trento's output

# 4-particle correlation 
epsilon_n_4 = (2 * rms_n)**(1/4) - np.average(epsilon_n**4)


# ratios: fluctuations of geometry eccentricities
r_4_2 = epsilon_n_4 / epsilon_n_2

# need to add jackknife sampling algorithm to add calculation statistical uncertainties