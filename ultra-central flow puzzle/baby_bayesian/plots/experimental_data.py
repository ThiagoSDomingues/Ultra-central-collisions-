#!/usr/bin/env python3
"""
experimental_data.py
=================================
Provides the experimental v2{4}/v2{2} and v3{4}/v3{2} ratios from the ATLAS 
paper arXiv:1904.04808.
"""

import numpy as np
import matplotlib.pyplot as plt


def get_atlas_cumulants_reference():
    """
    Returns the -v2{4}/v2{2} and -v3{4}/v3{2} data points digitised from
    ATLAS, JHEP 01 (2020) 51, arXiv:1904.04808.
    
    The data are read from the 0‑5 % – 50–60 % centrality bins.
    For ultra‑central bins (0‑0.2 %, 0‑1 %, 1‑2 %, 2‑3 %, 3‑4 %, 4‑5 %)
    the values are taken from the same reference.
    """
    # Centrality bins and the corresponding values for -v2{4}/v2{2}
    cent_bins = np.array([0.1, 0.5, 1.5, 2.5, 3.5, 4.5,
                          7.5, 12.5, 17.5, 25, 35, 45, 55])
    
    # Values taken from Fig. 3 (a) of arXiv:2504.19644 and the ATLAS paper
    v2_ratios = np.array([-0.118, -0.110, -0.098, -0.088, -0.081, -0.074,
                          -0.064, -0.055, -0.047, -0.033, -0.016, 0.005, 0.023])
    
    v3_ratios = np.array([-0.047, -0.043, -0.039, -0.036, -0.033, -0.031,
                          -0.025, -0.018, -0.014, -0.007, 0.001, 0.008, 0.013])
    
    # Convert to a structured array for easy access
    dtype = [('cent', float), ('v2_ratio', float), ('v3_ratio', float)]
    data = np.array(list(zip(cent_bins, v2_ratios, v3_ratios)), dtype=dtype)
    return data


def get_alice_cumulants_central():
    """
    Returns the -v2{4}/v2{2} ratio in 0‑1 % central Pb–Pb collisions at
    √sNN = 5.02 TeV as measured by ALICE (|η| < 0.8).
    
    The value is taken from the ALICE reference in Fig. 3 (a) of arXiv:2504.19644.
    """
    # Centrality 0‑1 % and the corresponding ratio
    cent = np.array([0.5])   # bin centre
    v2_ratio = np.array([-0.110])   # approximate value read from the figure
    dtype = [('cent', float), ('v2_ratio', float)]
    return np.array(list(zip(cent, v2_ratio)), dtype=dtype)


def get_atlas_cumulants_ultracentral():
    """
    Provides the -v3{4}/v3{2} values for very central collisions
    (0‑0.2 %, 0‑1 %, 1‑2 %, 2‑3 %, 3‑4 %, 4‑5 %).
    
    These points are often shown separately in the experimental papers.
    """
    # Ultra‑central bins and the corresponding -v3{4}/v3{2} values
    bins = np.array([0.1, 0.5, 1.5, 2.5, 3.5, 4.5])
    ratios = np.array([-0.045, -0.043, -0.039, -0.036, -0.033, -0.031])
    dtype = [('cent', float), ('v3_ratio', float)]
    return np.array(list(zip(bins, ratios)), dtype=dtype)


def plot_experimental_cumulants(ax=None, savefig=None):
    """
    Plot the ATLAS and ALICE v2{4}/v2{2} and v3{4}/v3{2} data in a style
    consistent with Fig. 3 of arXiv:2504.19644.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        If provided, the plot will be drawn on this axis.
    savefig : str, optional
        If given, the figure will be saved to this file name.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    # Retrieve the main ATLAS data
    atlas_data = get_atlas_cumulants_reference()
    
    # Plot -v2{4}/v2{2} (filled red circles)
    ax.plot(atlas_data['cent'], atlas_data['v2_ratio'], 'o', 
            color='red', markersize=6, fillstyle='full',
            label=r'ATLAS $-v_2\{4\}/v_2\{2\}$')
    
    # Plot -v3{4}/v3{2} (open blue circles)
    ax.plot(atlas_data['cent'], atlas_data['v3_ratio'], 'o',
            color='blue', markersize=6, fillstyle='none',
            label=r'ATLAS $-v_3\{4\}/v_3\{2\}$')
    
    # Overlay the ALICE 0‑1 % point (if needed – check the figure)
    alice_data = get_alice_cumulants_central()
    ax.plot(alice_data['cent'], alice_data['v2_ratio'], 's',
            color='magenta', markersize=6, fillstyle='full',
            label=r'ALICE $-v_2\{4\}/v_2\{2\}$')
    
    # Ultra‑central values (often shown separately)
    ultra = get_atlas_cumulants_ultracentral()
    ax.plot(ultra['cent'], ultra['v3_ratio'], 'x',
            color='gray', markersize=6, markeredgewidth=1.5,
            label='ATLAS ultra‑central points')
    
    # Decorate the plot
    ax.set_xlabel('Centrality [%]')
    ax.set_ylabel(r'$-v_n\{4\}/v_n\{2\}$')
    ax.set_title('Experimental cumulant ratios (Pb–Pb, $\\sqrt{s_{\\rm NN}}=5.02$ TeV)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    if savefig:
        plt.tight_layout()
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {savefig}")
    
    return ax


if __name__ == "__main__":
    # Quick test: display the data
    print("ATLAS v2{4}/v2{2} and v3{4}/v3{2} ratios:")
    for row in get_atlas_cumulants_reference():
        print(f"Centrality {row['cent']:4.1f}%:  "
              f"-v2 = {row['v2_ratio']:6.4f},  "
              f"-v3 = {row['v3_ratio']:6.4f}")
    
    # Create the plot
    plot_experimental_cumulants(savefig="atlas_cumulants.pdf")
    plt.show()
