#!shebang 
"""
Making important plots for ultracentral collisions project.

"""
from compute_eccentricities import * 
from matplotlib import pyplot as plt
from pbpb_run_trento import centrality bins
from matplotlib import pyplot as plt

plt.rcParams.update({
    "figure.facecolor": 
    "axes.facecolor":
    "axes.edgecolor":
    "axes.labelcolor":
    "xtick.color":
    "ytick.color":
    "xtick.labelsize": 8,
    ""

})

# Load it from LHS code
#PARAM_LABELS_TEX = 
PARAM_NAMES=list(PRIOR.keys())

# Load files generated from compute_eccentricities calculations
# "eccentricities.npz"

# ============================
# ECCENTRICITIES VS CENTRALITY
# ============================

def plot_vs_centrality(all_profiles, out_dir):
    """
    Left - ultracentral 0-10%, 1% bins, all design points as faint lines;    
    Right - RMS over 0-90%, wide bins, all design points as faint lines.   
    """
    obs_keys = list(OBS_META.keys())
    n_obs = len(obs_keys)
    
    fig, axes = plt.subplots(n_obs, 2,
                             figsize=(13, n_obs * 2.6),
                             squeeze=False)
    fig.patch.set_facecolor(BG)
    fig.suptitle(r"${208}$Pb+${208}$Pb $\sqrt{s_{NN}}=2.76$ TeV" | {n_design_pts} | 1M events|, fontsize=11, color=TEXT, y=0.95, fontweight='bold')

plt.figure()

# plot 2-particle eccentricities individually (RMS ecccentricities). Centrality bins range from 0-90%. n = 2,3, and 4 overlaid 
plt.plot(centrality_bins, eps22, label=r'$\epsilon_2{2}$')
plt.plot(centrality_bins, eps32, label=r'$\epsilon_3{2}$')
plt.plot(centrality_bins, eps42, label=r'$\epsilon_4{2}$')

plt.xlabel("Centrality (%)")
plt.ylabel("Eccentricity")
plt.legend()

plt.savefig(WORKDIR/"ecc_vs_centrality.pdf")

plt.figure()

# Eccentricities ratios
plt.plot(centrality_bins, ratio32, label=r'$\epsilon_3{2}/\epsilon_2{2}$')
plt.plot(centrality_bins, ratio24, label=r'$\epsilon_2{2}/\epsilon_2{4}$')
plt.plot(centrality_bins, ratio34, label=r'$\epsilon_3{2}/\epsilon_3{4}$')

plt.xlabel("Centrality (%)")
plt.legend()

plt.savefig(WORKDIR/"ratios_vs_centrality.pdf")

# ==================================
# ECCENTRICITIES VS MODEL PARAMETERS
# ==================================

#ratio

ratio = np.mean(ratio32) # avarage over events?

for i,name in enumerate(PARAM_NAMES):
    
    plt.figure()
    
    plt.scatter(design[:,i], np.repeat(ratio, len(design)))
    
    plt.xlabel(name)
    plt.ylabel(r'$\epsilon_3{2}/\epsilon_2{2}$')
    
    plt.savefig(WORKDIR/f"ratio_vs_{name}.pdf")
