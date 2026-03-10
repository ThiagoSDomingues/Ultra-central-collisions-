#!/usr/bin/env python3

"""
Stage 5: Generate plots for Pb+Pb ultracentral eccentricity scan.

Produces:

1. epsilon_n vs centrality
2. epsilon2/epsilon3 vs centrality
3. epsilon_n vs nuclear parameters
4. ratio vs parameters

Error bars are estimated using Jackknife resampling over centrality bins.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

WORK_DIR = Path("pbpb_scan")

DESIGN_FILE = WORK_DIR / "lhs_design_matrix.npz"

CENTRALITY_BINS = [
    (0,0.2),
    (0.2,0.5),
    (0.5,1.0),
    (1.0,2.0),
    (2.0,3.0)
]

FREE_NAMES = ["R","a","beta3","beta4","w"]

# ============================================================
# JACKKNIFE
# ============================================================

def jackknife(samples, func):

    N = len(samples)

    jk_vals = []

    for i in range(N):

        subsample = np.delete(samples, i)

        jk_vals.append(func(subsample))

    jk_vals = np.array(jk_vals)

    mean = np.mean(jk_vals)

    err = np.sqrt((N-1)/N * np.sum((jk_vals-mean)**2))

    return mean, err


# ============================================================
# CENTRALITY SELECTION
# ============================================================

def select_centrality(events, cent_min, cent_max):

    mult = events[:,3]

    percentile_min = np.percentile(mult, 100-cent_max)
    percentile_max = np.percentile(mult, 100-cent_min)

    mask = (mult >= percentile_min) & (mult < percentile_max)

    return events[mask]


# ============================================================
# ECCENTRICITY RMS
# ============================================================

def eps_rms(eps):

    return np.sqrt(np.mean(eps**2))


# ============================================================
# CENTRALITY PLOTS
# ============================================================

def plot_eccentricities_vs_centrality(events):

    eps2_vals=[]
    eps3_vals=[]
    eps4_vals=[]
    eps5_vals=[]

    err2=[]
    err3=[]
    err4=[]
    err5=[]

    centers=[]

    for cmin,cmax in CENTRALITY_BINS:

        subset = select_centrality(events,cmin,cmax)

        if len(subset)<20:
            continue

        centers.append(0.5*(cmin+cmax))

        eps2 = subset[:,4]
        eps3 = subset[:,5]
        eps4 = subset[:,6]
        eps5 = subset[:,7]

        m,e = jackknife(eps2, eps_rms)
        eps2_vals.append(m)
        err2.append(e)

        m,e = jackknife(eps3, eps_rms)
        eps3_vals.append(m)
        err3.append(e)

        m,e = jackknife(eps4, eps_rms)
        eps4_vals.append(m)
        err4.append(e)

        m,e = jackknife(eps5, eps_rms)
        eps5_vals.append(m)
        err5.append(e)

    plt.figure()

    plt.errorbar(centers,eps2_vals,yerr=err2,marker="o",label="ε2")
    plt.errorbar(centers,eps3_vals,yerr=err3,marker="o",label="ε3")
    plt.errorbar(centers,eps4_vals,yerr=err4,marker="o",label="ε4")
    plt.errorbar(centers,eps5_vals,yerr=err5,marker="o",label="ε5")

    plt.xlabel("Centrality (%)")
    plt.ylabel("√<ε_n²>")
    plt.legend()
    plt.title("Eccentricities vs centrality")

    plt.savefig(WORK_DIR/"eccentricity_vs_centrality.png",dpi=200)
    plt.close()


# ============================================================
# RATIO PLOT
# ============================================================

def plot_ratio_vs_centrality(events):

    ratios=[]
    errors=[]
    centers=[]

    for cmin,cmax in CENTRALITY_BINS:

        subset = select_centrality(events,cmin,cmax)

        if len(subset)<20:
            continue

        centers.append(0.5*(cmin+cmax))

        eps2=subset[:,4]
        eps3=subset[:,5]

        def ratio(x):

            eps2 = x[:,0]
            eps3 = x[:,1]

            return eps_rms(eps2)/eps_rms(eps3)

        pair=np.column_stack((eps2,eps3))

        m,e = jackknife(pair,ratio)

        ratios.append(m)
        errors.append(e)

    plt.figure()

    plt.errorbar(centers,ratios,yerr=errors,marker="o")

    plt.xlabel("Centrality (%)")
    plt.ylabel("ε2 / ε3")

    plt.title("Ultracentral flow puzzle diagnostic")

    plt.savefig(WORK_DIR/"ratio23_vs_centrality.png",dpi=200)
    plt.close()


# ============================================================
# PARAMETER SCANS
# ============================================================

def plot_vs_parameters(design,results):

    eps2=results[:,1]
    eps3=results[:,2]

    ratio = eps2/eps3

    for i,name in enumerate(FREE_NAMES):

        x=design[:,i]

        plt.figure()

        plt.scatter(x,ratio)

        plt.xlabel(name)
        plt.ylabel("ε2 / ε3")

        plt.title(f"Ultracentral sensitivity to {name}")

        plt.savefig(WORK_DIR/f"ratio_vs_{name}.png",dpi=200)

        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    design = np.load(DESIGN_FILE)["design"]

    all_events=[]

    for d in sorted(WORK_DIR.glob("design_*")):

        f=d/"trento_events.npy"

        if f.exists():

            ev=np.load(f)

            all_events.append(ev)

    events=np.concatenate(all_events)

    plot_eccentricities_vs_centrality(events)

    plot_ratio_vs_centrality(events)

    results=np.load(WORK_DIR/"eccentricity_observables.npz")["data"]

    plot_vs_parameters(design,results)

    print("Plots saved in:",WORK_DIR)


if __name__=="__main__":
    main()
