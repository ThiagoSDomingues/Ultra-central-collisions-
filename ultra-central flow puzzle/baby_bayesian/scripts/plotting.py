import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

# -----------------------------------------------------------------------------
# Design points plots
# -----------------------------------------------------------------------------

# ── plot setup ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman","DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4, "ytick.minor.size": 2.5,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.framealpha": 1.0, "legend.edgecolor": "0.70",
    "legend.fancybox": False, "legend.fontsize": 7,
})

COL_NC2      = "#C0392B"
COL_NC3      = "#111111"
COL_NC4      = "#1A7A1A"
COL_ALICE_276= "#E67E22"
COL_BAND     = "#2471A3"    # design point envelope
COL_MEDIAN   = "#154360"
EXP_KW = dict(ls="none", capsize=3, capthick=0.9, elinewidth=0.9, markersize=4.5)

def add_label(ax, txt):
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=11, fontweight="bold")

def style_ax(ax, ylabel, xlim, ylim=None):
    ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=1)
    ax.set_xlabel("Centrality (%)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", labelsize=9)

def plot_design_envelope(ax, cents, nc_matrix):
    cents = np.asarray(cents)
    if nc_matrix.shape[1] != len(cents):
        raise ValueError(
            f"Mismatch: matrix has {nc_matrix.shape[1]} bins "
            f"but cents has {len(cents)}"
        )
    for row in nc_matrix:
        fin = np.isfinite(row)
        ax.plot(cents[fin], row[fin], color=COL_BAND, lw=0.5, alpha=0.25, zorder=2)
        
    p00 = np.nanpercentile(nc_matrix, 0,  axis=0)
    p05 = np.nanpercentile(nc_matrix, 5,  axis=0)
    p50 = np.nanpercentile(nc_matrix, 50, axis=0)
    p95 = np.nanpercentile(nc_matrix, 95, axis=0)
    p100 = np.nanpercentile(nc_matrix, 100, axis=0)
    
    fin = np.isfinite(p50)
    ax.fill_between(cents[fin], p00[fin], p100[fin],
                    color=COL_BAND, alpha=0.20, zorder=3)
    ax.plot(cents[fin], p50[fin], color=COL_MEDIAN, lw=1.8, ls="-",
            zorder=4, label="TRENTo median (LHS)")

fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                         gridspec_kw={"hspace": 0.42, "wspace": 0.32})
PANEL = [["(a)","(b)","(c)"], ["(d)","(e)","(f)"]]

# ── Row 0: wide range (0–60%) ─────────────────────────────────────────
# nc2 wide
ax = axes[0,0]
fin = np.isfinite(alice_R_502)
plot_design_envelope(ax, alice_c_502[fin], -m2)
ax.errorbar(alice_c_502[fin], alice_R_502[fin], yerr=alice_eR_502[fin],
            fmt="o", color=COL_NC2, mfc=COL_NC2,
            label=r"ALICE 5.02 TeV  $-c_2\{4\}/c_2\{2\}^2$", **EXP_KW)
fin2 = np.isfinite(alice_R_276)
ax.errorbar(alice_c_276[fin2], alice_R_276[fin2], yerr=alice_eR_276[fin2],
            fmt="o", color=COL_ALICE_276, mfc="none",
            label=r"ALICE 2.76 TeV", **EXP_KW)
style_ax(ax, r"$-\mathrm{nc}_2$", xlim=(-1,61))
add_label(ax, PANEL[0][0])
ax.legend(loc="lower right", fontsize=6.5)

# nc3 wide
ax = axes[0,1]
fin = np.isfinite(neg_atl3)
plot_design_envelope(ax, atl3_c[fin], -m3)
if np.any(fin):
    ax.errorbar(atl3_c[fin], neg_atl3[fin], yerr=err_atl3[fin],
                fmt="o", color=COL_NC3, mfc="none",
                label=r"ATLAS $-nc_3\{4\}$", **EXP_KW)
style_ax(ax, r"$-\mathrm{nc}_3$", xlim=(-1,55))
add_label(ax, PANEL[0][1])
ax.legend(loc="lower right", fontsize=6.5)

# nc4 wide
ax = axes[0,2]
plot_design_envelope(ax, atl4_c, -m4)
if len(neg_atl4) > 0:
    fin = np.isfinite(neg_atl4)
    ax.errorbar(atl4_c[fin], neg_atl4[fin], yerr=err_atl4[fin],
                fmt="s", color=COL_NC4, mfc=COL_NC4,
                label=r"ATLAS $-nc_4\{4\}$", **EXP_KW)
style_ax(ax, r"$-\mathrm{nc}_4$", xlim=(-1,55))
add_label(ax, PANEL[0][2])
ax.legend(loc="lower right", fontsize=6.5)

# ── Row 1: zoom 0–10% ────────────────────────────────────────────────
# nc2 zoom
ax = axes[1,0]
plot_design_envelope(ax, zoom_cents_nc2, -zm2)

mask_a = alice_c_502 <= 10
ax.errorbar(alice_c_502[mask_a], alice_R_502[mask_a], yerr=alice_eR_502[mask_a],
            fmt="o", color=COL_NC2, mfc=COL_NC2, label=r"ALICE 5.02 TeV", **EXP_KW)

mask_a2 = alice_c_276 <= 10
ax.errorbar(alice_c_276[mask_a2], alice_R_276[mask_a2], yerr=alice_eR_276[mask_a2],
            fmt="o", color=COL_ALICE_276, mfc="none", label=r"ALICE 2.76 TeV", **EXP_KW)

style_ax(ax, r"$-\mathrm{nc}_2$", xlim=(-0.3,10.3))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_title("0–10% zoom", fontsize=8.5, color="#444", pad=2)
add_label(ax, PANEL[1][0])
ax.legend(loc="lower right", fontsize=6.5)

# nc3 zoom
ax = axes[1,1]
plot_design_envelope(ax, zoom_cents_nc3, -zm3)

if len(neg_atl3) > 0:
    mask_3 = atl3_c <= 10
    ax.errorbar(atl3_c[mask_3], neg_atl3[mask_3], yerr=err_atl3[mask_3],
                fmt="o", color=COL_NC3, mfc="none", label=r"ATLAS $-nc_3\{4\}$", **EXP_KW)

style_ax(ax, r"$-\mathrm{nc}_3$", xlim=(-0.3,10.3))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_title("0–10% zoom", fontsize=8.5, color="#444", pad=2)
add_label(ax, PANEL[1][1])
ax.legend(loc="lower right", fontsize=6.5)

# nc4 zoom
ax = axes[1,2]
plot_design_envelope(ax, zoom_cents_nc4, -zm4)

if len(neg_atl4) > 0:
    mask_4 = atl4_c <= 10
    ax.errorbar(atl4_c[mask_4], neg_atl4[mask_4], yerr=err_atl4[mask_4],
                fmt="s", color=COL_NC4, mfc=COL_NC4, label=r"ATLAS $-nc_4\{4\}$", **EXP_KW)
else:
    ax.text(0.5, 0.5, "no nc4 data", transform=ax.transAxes,
            ha="center", va="center", fontsize=10, color="gray")

style_ax(ax, r"$-\mathrm{nc}_4$", xlim=(-0.3,10.3))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_title("0–10% zoom", fontsize=8.5, color="#444", pad=2)
add_label(ax, PANEL[1][2])
ax.legend(loc="lower right", fontsize=6.5)

# ── shared legend patch ───────────────────────────────────────────────
band_patch  = Patch(facecolor=COL_BAND, alpha=0.35, label="TRENTo LHS envelope Full")
thin_line   = Line2D([0],[0], color=COL_BAND, lw=0.8, alpha=0.5, label="Individual design points")
median_line = Line2D([0],[0], color=COL_MEDIAN, lw=1.8, label="TRENTo LHS median")

fig.legend(handles=[thin_line, band_patch, median_line],
           loc="upper center", ncol=3, fontsize=8.5,
           bbox_to_anchor=(0.5, 1.02), framealpha=1.0,
           edgecolor="0.70", fancybox=False)

fig.suptitle(
    r"$-\mathrm{nc}_n \equiv -(\langle\varepsilon_n^4\rangle/\langle\varepsilon_n^2\rangle^2 - 2)$"
    f"  —  Pb+Pb @ 5.02 TeV  |  TRENTo {VERSION}  |  {len(X)} LHS design points",
    fontsize=10.5, y=1.05
)

out_file = f"plots/nc_lhs_overlay_{VERSION}.pdf"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
print(f"Saved: {out_file}")
plt.show()
plt.close(fig)


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4,
    "ytick.minor.size": 2.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "0.70",
    "legend.fancybox": False,
    "legend.fontsize": 8,
})

COL_BAND   = "#2471A3"
COL_MEDIAN = "#154360"
COL_EXP    = "#C0392B"

fig, ax = plt.subplots(figsize=(8, 6))

# Plot TRENTo envelope (individual thin lines)
for row in ratios_zoom:
    fin = np.isfinite(row)
    ax.plot(centers_zoom, row[fin], color=COL_BAND, lw=0.5, alpha=0.25, zorder=2)

# Percentile bands
p00 = np.nanpercentile(ratios_zoom, 0, axis=0)
p05 = np.nanpercentile(ratios_zoom, 5, axis=0)
p50 = np.nanpercentile(ratios_zoom, 50, axis=0)
p95 = np.nanpercentile(ratios_zoom, 95, axis=0)
p100 = np.nanpercentile(ratios_zoom, 100, axis=0)

fin = np.isfinite(p50)
ax.fill_between(centers_zoom, p00[fin], p100[fin], color=COL_BAND, alpha=0.20, zorder=3)
ax.plot(centers_zoom[fin], p50[fin], color=COL_MEDIAN, lw=2.0, ls="-", zorder=4,
        label=r"TRENTo median (LHS)")

# Experimental data (ALICE)
ax.errorbar(centers_exp, ratio, yerr=ratio_err,
            fmt='o', color=COL_EXP, mfc=COL_EXP, capsize=3,
            label=r"ALICE $v_{22}/v_{32}$ (stat+syst)", zorder=5)

# Decoration
ax.set_xlabel("Centrality (%)", fontsize=12)
ax.set_ylabel(r"$\varepsilon_2\{2\}/\varepsilon_3\{2\}$", fontsize=12)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0.8, 2.2)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Legend
band_patch = Patch(facecolor=COL_BAND, alpha=0.35, label="LHS envelope Full")
thin_line = Line2D([0], [0], color=COL_BAND, lw=0.8, alpha=0.5, label="Individual design points")
median_line = Line2D([0], [0], color=COL_MEDIAN, lw=2.0, label="TRENTo median")
ax.legend(handles=[thin_line, band_patch, median_line, ax.get_legend_handles_labels()[0][-1]],
          loc="best", fontsize=9)

ax.set_title(r"$\varepsilon_2\{2\}/\varepsilon_3\{2\}$ in ultracentral Pb+Pb @ 2.76 TeV" +
             f"\nTRENTo {VERSION}  |  {ratios_zoom.shape[0]} LHS design points", fontsize=10)

plt.tight_layout()
plt.savefig("plots/e2e3_ratio_alice_data.pdf", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------------------
# PCA Diagnostic plots
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("PCA Diagnostics (Full Experimental Bins)", fontsize=12, fontweight='bold')

# (a) Cumulative variance
ax = axes[0, 0]
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components = np.arange(1, len(cumsum)+1)
ax.plot(n_components, cumsum, 'o-', color='blue', linewidth=2, markersize=4)
ax.axhline(0.95, linestyle='--', color='red', alpha=0.7, label='95% Threshold')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('(a) Cumulative variance')
ax.grid(alpha=0.3)
ax.legend()

# (b) Scree plot
ax = axes[0, 1]
show_bins = min(15, len(n_components))
ax.bar(n_components[:show_bins], pca.explained_variance_ratio_[:show_bins], color='steelblue', alpha=0.7)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title(f'(b) Scree plot (First {show_bins} PCs)')
ax.grid(alpha=0.3, axis='y')

# (c) Design points in PC1‑PC2 space
ax = axes[0, 2]
ax.scatter(Z[:, 0], Z[:, 1], c='steelblue', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('(c) Design points in PC1‑PC2 space')
ax.grid(alpha=0.3)
ax.axhline(0, linestyle='--', color='gray', alpha=0.5)
ax.axvline(0, linestyle='--', color='gray', alpha=0.5)

# (d) Reconstruction errors
ax = axes[1, 0]
Y_reconstructed = pca.inverse_transform(Z)
reconstruction_errors = np.mean((Y_scaled - Y_reconstructed)**2, axis=1)
ax.hist(reconstruction_errors, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Mean squared error (scaled space)')
ax.set_ylabel('Number of design points')
ax.set_title('(d) Reconstruction error profile')
ax.grid(alpha=0.3, axis='y')

# (e) Individual PC variance (bar)
ax = axes[1, 1]
ax.bar(n_components[:Z.shape[1]], pca.explained_variance_ratio_[:Z.shape[1]], color='coral', alpha=0.7)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('(e) Individual PC variance')
ax.grid(alpha=0.3, axis='y')

# (f) Cumulative variance in percent
ax = axes[1, 2]
ax.plot(n_components[:Z.shape[1]], cumsum[:Z.shape[1]]*100, 'o-', color='green', linewidth=2, markersize=4)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance (%)')
ax.set_title('(f) Cumulative variance (%)')
ax.grid(alpha=0.3)
ax.axhline(75, linestyle='--', color='red', alpha=0.7)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 4. Loadings heatmap (first 6 PCs)
# -----------------------------------------------------------------------------
n_pcs_show = min(6, Z.shape[1])
components = pca.components_[:n_pcs_show]

# Create labels for the heatmap: "harm\ncenter"
labels = [f"{m['harm']}\n{m['center']:.1f}" for m in feature_metadata]

fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.25), 5))
im = ax.imshow(components, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_yticks(np.arange(n_pcs_show))
ax.set_yticklabels([f'PC{i+1}' for i in range(n_pcs_show)])
ax.set_xlabel('Original observables (centrality bins)')
ax.set_ylabel('Principal Components')
ax.set_title(f'PCA Loadings Map (First {n_pcs_show} PCs)')
plt.colorbar(im, ax=ax, label='Loading weight coefficient')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 5. PC weights vs centrality (separate for nc2, nc3, nc4)
# -----------------------------------------------------------------------------
# Group indices by harmonic
indices_by_harm = {
    'nc2': [i for i, m in enumerate(feature_metadata) if m['harm'] == 'nc2'],
    'nc3': [i for i, m in enumerate(feature_metadata) if m['harm'] == 'nc3'],
    'nc4': [i for i, m in enumerate(feature_metadata) if m['harm'] == 'nc4'],
}
centres_by_harm = {
    'nc2': np.array([feature_metadata[i]['center'] for i in indices_by_harm['nc2']]),
    'nc3': np.array([feature_metadata[i]['center'] for i in indices_by_harm['nc3']]),
    'nc4': np.array([feature_metadata[i]['center'] for i in indices_by_harm['nc4']]),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False)
harmonics = ['nc2', 'nc3', 'nc4']
titles = ['nc₂', 'nc₃', 'nc₄']
colors = {'nc2': '#C0392B', 'nc3': '#111111', 'nc4': '#1A7A1A'}

for i, harm in enumerate(harmonics):
    ax = axes[i]
    idx = indices_by_harm[harm]
    cents = centres_by_harm[harm]
    # Weights for PC1 and PC2
    w_pc1 = pca.components_[0, idx]
    w_pc2 = pca.components_[1, idx]
    # Sort by centrality (already sorted but ensure)
    sort_order = np.argsort(cents)
    cents_sorted = cents[sort_order]
    w_pc1_sorted = w_pc1[sort_order]
    w_pc2_sorted = w_pc2[sort_order]
    ax.plot(cents_sorted, w_pc1_sorted, 'o-', color=colors[harm], label='PC1', linewidth=1.5, markersize=3)
    ax.plot(cents_sorted, w_pc2_sorted, 's--', color='gray', label='PC2', linewidth=1.5, markersize=3, alpha=0.7)
    ax.axhline(0, linestyle='--', color='black', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Centrality (%)')
    ax.set_ylabel('Loading weight')
    ax.set_title(f'{titles[i]} PCA weights')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')
plt.tight_layout()
plt.show()

# =============================================================================
# Prior predictive (90% credible interval) using design-point bin centres
# =============================================================================

# ----------------------------------------------------------------------
# 1. Generate prior samples and predict observables
# ----------------------------------------------------------------------
n_prior = 5000
np.random.seed(123)
prior_samples = np.random.uniform(param_lo, param_hi, (n_prior, len(param_lo)))
Y_prior, _ = emulator_predict(prior_samples)   # shape (n_prior, n_outputs)

# Determine number of bins per harmonic from the actual bin centres
n_bins_nc2 = len(cents_nc2)
n_bins_nc3 = len(cents_nc3)
n_bins_nc4 = len(cents_nc4)
total_bins = n_bins_nc2 + n_bins_nc3 + n_bins_nc4

# Check if emulator output matches expected total bins
if Y_prior.shape[1] != total_bins:
    print(f"Warning: Emulator output has {Y_prior.shape[1]} columns, but expected {total_bins}")
    print("Assuming emulator was trained on zoom observables only. Adjusting bin centres.")
    # Assume the emulator was trained on Y_zoom (only bins within 0-10%)
    # Then we need to use the zoom bin centres
    try:
        cents_nc2 = zoom_cents_nc2
        cents_nc3 = zoom_cents_nc3
        cents_nc4 = zoom_cents_nc4
        n_bins_nc2 = len(cents_nc2)
        n_bins_nc3 = len(cents_nc3)
        n_bins_nc4 = len(cents_nc4)
        total_bins = n_bins_nc2 + n_bins_nc3 + n_bins_nc4
        if Y_prior.shape[1] != total_bins:
            raise ValueError("Still mismatch. Please check emulator training data.")
    except NameError:
        raise ValueError("zoom_cents_nc2 not defined. Train emulator on full Y or define zoom bin centres.")

# Unpack predictions (convert to -nc for plotting)
Y_prior_nc2 = Y_prior[:, 0:n_bins_nc2]
Y_prior_nc3 = Y_prior[:, n_bins_nc2:n_bins_nc2+n_bins_nc3]
Y_prior_nc4 = Y_prior[:, n_bins_nc2+n_bins_nc3:]

print(f"Prior prediction: {n_prior} samples, nc2: {n_bins_nc2} bins, nc3: {n_bins_nc3} bins, nc4: {n_bins_nc4} bins")

# Compute median and 5‑95% bands (90% credible interval)
def prior_band(Y):
    med = np.median(Y, axis=0)
    lo0 = np.percentile(Y, 0, axis=0)
    hi100 = np.percentile(Y, 100, axis=0)
    return med, lo0, hi100

med2, lo2, hi2 = prior_band(Y_prior_nc2)
med3, lo3, hi3 = prior_band(Y_prior_nc3)
med4, lo4, hi4 = prior_band(Y_prior_nc4)

# ----------------------------------------------------------------------
# 2. Match experimental data to the exact bin centres (no interpolation)
# ----------------------------------------------------------------------
def match_experimental(exp_cent, exp_val, exp_err, target_centers, tol=1.5):
    matched_val = np.full(len(target_centers), np.nan)
    matched_err = np.full(len(target_centers), np.nan)
    for i, tc in enumerate(target_centers):
        idx = np.argmin(np.abs(exp_cent - tc))
        if np.abs(exp_cent[idx] - tc) <= tol:
            matched_val[i] = exp_val[idx]
            matched_err[i] = exp_err[idx]
    return matched_val, matched_err

exp_nc2, exp_nc2_err = match_experimental(alice_c_502, alice_R_502, alice_eR_502, cents_nc2)
exp_nc3, exp_nc3_err = match_experimental(atl3_c, neg_atl3, err_atl3, cents_nc3)
exp_nc4, exp_nc4_err = match_experimental(atl4_c, neg_atl4, err_atl4, cents_nc4)

# ----------------------------------------------------------------------
# 3. Build the 2×3 figure (same layout as design point plot)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman","DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 4, "ytick.minor.size": 2.5,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.framealpha": 1.0, "legend.edgecolor": "0.70",
    "legend.fancybox": False, "legend.fontsize": 7,
})

COL_BAND   = "#2471A3"
COL_MEDIAN = "#154360"
COL_EXP    = "#C0392B"
EXP_KW = dict(ls="none", capsize=3, capthick=0.9, elinewidth=0.9, markersize=4.5)

def plot_band(ax, cents, med, lo, hi, label="Full Prior"):
    fin = np.isfinite(med)
    ax.fill_between(cents[fin], lo[fin], hi[fin], color=COL_BAND, alpha=0.25, zorder=2, label=label)
    ax.plot(cents[fin], med[fin], color=COL_MEDIAN, lw=1.8, ls="-", zorder=3, label="Prior median")

def style_ax(ax, ylabel, xlim, ylim=None):
    ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=1)
    ax.set_xlabel("Centrality (%)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which="both", labelsize=9)

def add_label(ax, txt):
    ax.text(0.04, 0.96, txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=11, fontweight="bold")

fig, axes = plt.subplots(2, 3, figsize=(14, 9),
                         gridspec_kw={"hspace": 0.42, "wspace": 0.32})
PANEL = [["(a)","(b)","(c)"], ["(d)","(e)","(f)"]]

# ----------------------------------------------------------------------
# Row 0: full range (all bins)
# ----------------------------------------------------------------------
# nc2 full
ax = axes[0,0]
plot_band(ax, cents_nc2, med2, lo2, hi2)
valid = np.isfinite(exp_nc2)
ax.errorbar(cents_nc2[valid], exp_nc2[valid], yerr=exp_nc2_err[valid],
            fmt="o", color=COL_EXP, mfc=COL_EXP, label=r"ALICE 5.02 TeV", **EXP_KW)
style_ax(ax, r"$-\mathrm{nc}_2$", xlim=(-1, max(cents_nc2)+5))
add_label(ax, PANEL[0][0])
ax.legend(loc="lower right", fontsize=6.5)

# nc3 full
ax = axes[0,1]
plot_band(ax, cents_nc3, med3, lo3, hi3)
valid = np.isfinite(exp_nc3)
if np.any(valid):
    ax.errorbar(cents_nc3[valid], exp_nc3[valid], yerr=exp_nc3_err[valid],
                fmt="o", color=COL_EXP, mfc=COL_EXP, label=r"ATLAS $-nc_3\{4\}$", **EXP_KW)
style_ax(ax, r"$-\mathrm{nc}_3$", xlim=(-1, max(cents_nc3)+5))
add_label(ax, PANEL[0][1])
ax.legend(loc="lower right", fontsize=6.5)

# nc4 full
ax = axes[0,2]
plot_band(ax, cents_nc4, med4, lo4, hi4)
valid = np.isfinite(exp_nc4)
if np.any(valid):
    ax.errorbar(cents_nc4[valid], exp_nc4[valid], yerr=exp_nc4_err[valid],
                fmt="s", color=COL_EXP, mfc=COL_EXP, label=r"ATLAS $-nc_4\{4\}$", **EXP_KW)
else:
    ax.text(0.5, 0.5, "no nc4 data", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="gray")
style_ax(ax, r"$-\mathrm{nc}_4$", xlim=(-1, max(cents_nc4)+5))
add_label(ax, PANEL[0][2])
ax.legend(loc="lower right", fontsize=6.5)

# ----------------------------------------------------------------------
# Row 1: zoom (only bins with centrality ≤10)
# ----------------------------------------------------------------------
zoom_mask_nc2 = cents_nc2 <= 10
zoom_mask_nc3 = cents_nc3 <= 10
zoom_mask_nc4 = cents_nc4 <= 10

# nc2 zoom
ax = axes[1,0]
plot_band(ax, cents_nc2[zoom_mask_nc2], med2[zoom_mask_nc2], lo2[zoom_mask_nc2], hi2[zoom_mask_nc2])
valid_z = np.isfinite(exp_nc2[zoom_mask_nc2])
if np.any(valid_z):
    ax.errorbar(cents_nc2[zoom_mask_nc2][valid_z], exp_nc2[zoom_mask_nc2][valid_z],
                yerr=exp_nc2_err[zoom_mask_nc2][valid_z],
                fmt="o", color=COL_EXP, mfc=COL_EXP, label=r"ALICE 5.02 TeV", **EXP_KW)
style_ax(ax, r"$-\mathrm{nc}_2$", xlim=(-0.3, 10.3))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_title("0–10% zoom", fontsize=8.5, color="#444", pad=2)
add_label(ax, PANEL[1][0])
ax.legend(loc="lower right", fontsize=6.5)

# nc3 zoom
ax = axes[1,1]
plot_band(ax, cents_nc3[zoom_mask_nc3], med3[zoom_mask_nc3], lo3[zoom_mask_nc3], hi3[zoom_mask_nc3])
valid_z = np.isfinite(exp_nc3[zoom_mask_nc3])
if np.any(valid_z):
    ax.errorbar(cents_nc3[zoom_mask_nc3][valid_z], exp_nc3[zoom_mask_nc3][valid_z],
                yerr=exp_nc3_err[zoom_mask_nc3][valid_z],
                fmt="o", color=COL_EXP, mfc=COL_EXP, label=r"ATLAS $-nc_3\{4\}$", **EXP_KW)
style_ax(ax, r"$-\mathrm{nc}_3$", xlim=(-0.3, 10.3))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_title("0–10% zoom", fontsize=8.5, color="#444", pad=2)
add_label(ax, PANEL[1][1])
ax.legend(loc="lower right", fontsize=6.5)

# nc4 zoom
ax = axes[1,2]
plot_band(ax, cents_nc4[zoom_mask_nc4], med4[zoom_mask_nc4], lo4[zoom_mask_nc4], hi4[zoom_mask_nc4])
valid_z = np.isfinite(exp_nc4[zoom_mask_nc4])
if np.any(valid_z):
    ax.errorbar(cents_nc4[zoom_mask_nc4][valid_z], exp_nc4[zoom_mask_nc4][valid_z],
                yerr=exp_nc4_err[zoom_mask_nc4][valid_z],
                fmt="s", color=COL_EXP, mfc=COL_EXP, label=r"ATLAS $-nc_4\{4\}$", **EXP_KW)
else:
    ax.text(0.5, 0.5, "no nc4 data", transform=ax.transAxes, ha="center", va="center", fontsize=10, color="gray")
style_ax(ax, r"$-\mathrm{nc}_4$", xlim=(-0.3, 10.3))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_title("0–10% zoom", fontsize=8.5, color="#444", pad=2)
add_label(ax, PANEL[1][2])
ax.legend(loc="lower right", fontsize=6.5)

# Shared legend for the prior band and median
band_patch  = Patch(facecolor=COL_BAND, alpha=0.35, label="Full Prior")
median_line = Line2D([0],[0], color=COL_MEDIAN, lw=1.8, label="Prior median")
fig.legend(handles=[band_patch, median_line],
           loc="upper center", ncol=2, fontsize=8.5,
           bbox_to_anchor=(0.5, 1.02), framealpha=1.0,
           edgecolor="0.70", fancybox=False)

fig.suptitle(
    r"$-\mathrm{nc}_n \equiv -(\langle\varepsilon_n^4\rangle/\langle\varepsilon_n^2\rangle^2 - 2)$"
    f"  —  Pb+Pb @ 5.02 TeV  |  Prior predictive ({n_prior} samples, uniform prior)",
    fontsize=10.5, y=1.05
)

#plt.savefig("prior_predictive_2x3.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# Emulator's validation: LOO-CV
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Plot: predicted vs true PC scores
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, min(4, n_pc_global), figsize=(5*min(4, n_pc_global), 4))
if n_pc_global == 1:
    axes = [axes]
for i, ax in enumerate(axes.flatten()[:min(4, n_pc_global)]):
    # Scatter plot
    ax.scatter(Z_true_all[:, i], Z_pred_all[:, i], alpha=0.7, edgecolors='k', s=50)
    # Perfect prediction line
    lims = [np.min([Z_true_all[:, i], Z_pred_all[:, i]]),
            np.max([Z_true_all[:, i], Z_pred_all[:, i]])]
    ax.plot(lims, lims, 'k--', lw=1.5, label='Ideal')
    # Compute correlation
    corr = np.corrcoef(Z_true_all[:, i], Z_pred_all[:, i])[0, 1]
    ax.set_xlabel(f'True PC{i+1}')
    ax.set_ylabel(f'Predicted PC{i+1}')
    ax.set_title(f'PC{i+1}  (r = {corr:.3f})')
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig("loocv_pc_predictions.pdf", dpi=150, bbox_inches="tight")
plt.show()

# ----------------------------------------------------------------------
# Also keep the original error bar plots (optional)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# per-fold error bar chart
ax = axes[0]
ax.bar(range(n_folds), all_errs, color="steelblue", alpha=0.7, width=0.8)
ax.axhline(np.mean(all_errs), color="red", lw=1.5, ls="--",
           label=f"mean = {np.mean(all_errs):.4f}")
ax.axhline(threshold, color="orange", lw=1.2, ls=":",
           label=f"threshold = {threshold:.0%}")
ax.set_xlabel("Design point (left-out fold)")
ax.set_ylabel("Mean relative error")
ax.set_title("LOO-CV error per design point")
ax.legend(fontsize=9)

# per-observable comparison
ax = axes[1]
labels = ["nc2", "nc3", "nc4"]
means = [np.mean(nc2_errs), np.mean(nc3_errs), np.mean(nc4_errs)]
stds = [np.std(nc2_errs), np.std(nc3_errs), np.std(nc4_errs)]
x = np.arange(3)
bars = ax.bar(x, means, yerr=stds, color=["#C0392B","#111111","#1A7A1A"],
              alpha=0.75, capsize=6, width=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Mean relative error")
ax.set_title("LOO-CV error by observable")
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{m:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
#plt.savefig("loocv_diagnostics.pdf", dpi=150, bbox_inches="tight")
plt.show()
print("Diagnostic plots saved: loocv_pc_predictions.pdf, loocv_diagnostics.pdf")

# ----------------------------------------------------------------------
# Posterior distribution
# ----------------------------------------------------------------------

# Corner plot
fig = corner.corner(samples, labels=param_names, truths=theta_true, show_titles=True)
#plt.savefig("closure_test_corner.pdf", dpi=150)
plt.show()
