#!shebang 
"""
Making important plots for ultracentral collisions project.

"""
from compute_eccentricities import *
from matplotlib import pyplot as plt
from pbpb_run_trento import centrality bins


# plot 2-particle eccentricities individually. Centrality bins range from 0-90%. n = 2 and 3 overlaid 
epsilon_n_2 vs centrality 
plt.plot(centrality_bins, epsilon_2_2, )
plt.plot(centrality_bins, epsilon_3_2, )


# plot eccentricities against parameters 
plt.plot()
plt.plot()
