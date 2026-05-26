from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator
from sklearn.metrics import r2_score
import textwrap

np.random.seed(123)

outdir = Path("/mnt/data/loocv_emulator_demo")
outdir.mkdir(exist_ok=True)

# --------------------------------------------------
# Synthetic model
# --------------------------------------------------

bounds = np.array([
    [0.05, 0.30],   # eta/s
    [0.0, 2.0],     # bulk normalization
    [0.135, 0.165]  # switching temperature
])

param_names = [r'$\eta/s$', r'$\zeta/s$ norm', r'$T_{\mathrm{sw}}$']
obs_names = [r'$dN_{\mathrm{ch}}/d\eta$', r'$\langle p_T \rangle$', r'$v_2$']

def true_model(theta):
    x1, x2, x3 = theta[...,0], theta[...,1], theta[...,2]

    y1 = 1100 + 120*np.sin(5*x1) - 60*x2 + 250*(x3-0.15)
    y2 = 0.45 + 0.08*x1 + 0.03*x2 - 0.8*(x3-0.15)**2
    y3 = 0.055 - 0.05*x1 + 0.012*x2 + 0.10*(x3-0.15)

    return np.stack([y1, y2, y3], axis=-1)

# --------------------------------------------------
# Generate 50 design points
# --------------------------------------------------

n_design = 50

sampler = qmc.LatinHypercube(d=3, seed=123)
u = sampler.random(n_design)

design = bounds[:,0] + u*(bounds[:,1]-bounds[:,0])
observables = true_model(design)

# --------------------------------------------------
# Leave-One-Out Cross Validation
# --------------------------------------------------

predictions = np.zeros_like(observables)

for i in range(n_design):

    mask = np.ones(n_design, dtype=bool)
    mask[i] = False

    train_x = design[mask]
    train_y = observables[mask]

    emulator = RBFInterpolator(
        train_x,
        train_y,
        kernel='thin_plate_spline'
    )

    pred = emulator(design[i].reshape(1,-1))[0]
    predictions[i] = pred

# --------------------------------------------------
# Errors
# --------------------------------------------------

abs_error = np.abs(predictions - observables)
rel_error = 100 * abs_error / np.abs(observables)

median_rel = np.median(rel_error, axis=0)
max_rel = np.max(rel_error, axis=0)
r2_vals = [
    r2_score(observables[:,i], predictions[:,i])
    for i in range(3)
]

summary_df = pd.DataFrame({
    "Observable": [r"$dN/d\eta$", r"$\langle p_T \rangle$", r"$v_2$"],
    "Median relative error (%)": median_rel,
    "Maximum relative error (%)": max_rel,
    "R2 score": r2_vals
})

summary_df.to_csv(outdir / "loocv_summary.csv", index=False)

# --------------------------------------------------
# Plot 1 — parity plots
# --------------------------------------------------

for j, obs_name in enumerate(obs_names):

    fig, ax = plt.subplots(figsize=(5,5))

    ax.scatter(
        observables[:,j],
        predictions[:,j],
        s=40
    )

    minv = min(observables[:,j].min(), predictions[:,j].min())
    maxv = max(observables[:,j].max(), predictions[:,j].max())

    ax.plot([minv,maxv],[minv,maxv], linestyle='--')

    ax.set_xlabel(f"True {obs_name}")
    ax.set_ylabel(f"Predicted {obs_name}")

    med = median_rel[j]

    ax.set_title(
        f"LOOCV parity plot\nMedian relative error = {med:.2f}%"
    )

    fig.tight_layout()

    fig.savefig(
        outdir / f"parity_plot_obs_{j}.pdf",
        dpi=220
    )

    plt.close(fig)

# --------------------------------------------------
# Plot 2 — relative error distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7,4))

bins = np.linspace(0, np.max(rel_error)*1.1, 20)

for j, obs_name in enumerate(obs_names):
    ax.hist(
        rel_error[:,j],
        bins=bins,
        alpha=0.6,
        label=obs_name
    )

ax.set_xlabel("Relative error (%)")
ax.set_ylabel("Counts")
ax.set_title("LOOCV relative error distributions")
ax.legend()

fig.tight_layout()
fig.savefig(outdir / "error_histograms.pdf", dpi=220)
plt.close(fig)

# --------------------------------------------------
# Plot 3 — identify problematic design points
# --------------------------------------------------

worst_error = np.max(rel_error, axis=1)

fig, ax = plt.subplots(figsize=(6,5))

sc = ax.scatter(
    design[:,0],
    design[:,1],
    c=worst_error,
    s=70
)

ax.set_xlabel(param_names[0])
ax.set_ylabel(param_names[1])
ax.set_title("Worst LOOCV error per design point")

cbar = plt.colorbar(sc)
cbar.set_label("Max relative error (%)")

fig.tight_layout()
fig.savefig(outdir / "problematic_design_points.pdf", dpi=220)
plt.close(fig)

# --------------------------------------------------
# Plot 4 — emulator uncertainty summary
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6,4))

x = np.arange(len(obs_names))

ax.bar(
    x,
    median_rel
)

ax.axhline(
    5.0,
    linestyle='--',
    label='5% target'
)

ax.set_xticks(x)
ax.set_xticklabels(obs_names)

ax.set_ylabel("Median relative error (%)")
ax.set_title("Median LOOCV emulator errors")
ax.legend()

fig.tight_layout()
fig.savefig(outdir / "median_error_summary.pdf", dpi=220)
plt.close(fig)

# --------------------------------------------------
# Save mock data
# --------------------------------------------------

design_df = pd.DataFrame(
    design,
    columns=["eta_over_s", "bulk_norm", "Tsw"]
)

true_df = pd.DataFrame(
    observables,
    columns=["dNch_deta_true", "mean_pT_true", "v2_true"]
)

pred_df = pd.DataFrame(
    predictions,
    columns=["dNch_deta_pred", "mean_pT_pred", "v2_pred"]
)

err_df = pd.DataFrame(
    rel_error,
    columns=["dNch_relerr_pct", "mean_pT_relerr_pct", "v2_relerr_pct"]
)

full_df = pd.concat(
    [design_df, true_df, pred_df, err_df],
    axis=1
)

full_df.to_csv(outdir / "loocv_mock_data.csv", index=False)

# --------------------------------------------------
# Tutorial markdown
# --------------------------------------------------

tutorial = f"""
# Emulator Validation with Leave-One-Out Cross Validation (LOOCV)

This example demonstrates how to validate a Gaussian Process emulator using Leave-One-Out Cross Validation.

---

# Goal

The objective is to quantify emulator predictive accuracy.

A Bayesian analysis is only trustworthy if the emulator reproduces the underlying simulation accurately.

---

# LOOCV Procedure

Suppose we have:

N_design = {n_design}

design points.

For each point i:

1. Remove point i from the training set
2. Train the emulator on the remaining N-1 points
3. Predict observables at point i
4. Compare prediction to the true simulation value

This produces a predictive error estimate for every design point.

---

# Mathematical Definition

The relative error for observable y is:

relative_error = |y_pred - y_true| / |y_true|

Usually expressed as a percentage.

---

# Interpretation

Typical criteria:

- median relative error < 5%:
  emulator is accurate enough

- localized large errors:
  suggest poor coverage of parameter space

- boundary points often fail first:
  indicates insufficient design density near edges

---

# Important Diagnostic Plots

The most important plots are:

1. Parity plots:
   predicted vs true values

2. Error histograms:
   distribution of emulator errors

3. Parameter-space error maps:
   identify problematic regions

4. Median error summary:
   determines whether emulator quality is acceptable

---

# Recommended Real-World Thresholds

Typical targets for heavy-ion Bayesian analyses:

| Observable Type | Good Emulator Error |
|---|---|
| Multiplicities | < 2% |
| Mean pT | < 3% |
| Flow harmonics | < 5% |
| Correlations | < 10% |

The acceptable threshold depends on experimental uncertainty.

The emulator uncertainty should ideally be smaller than experimental uncertainties.

---

# Important Physical Interpretation

If emulator errors are too large:

- posterior distributions become artificially broad
- likelihood surfaces become distorted
- Bayesian inference becomes unreliable

Increasing the number of design points usually improves emulator accuracy.

---

# Files Included

- loocv_mock_data.csv
- loocv_summary.csv
- parity plots
- error histograms
- parameter-space error maps

"""

tutorial_path = outdir / "loocv_tutorial.md"
tutorial_path.write_text(textwrap.dedent(tutorial))

print("Generated files:")
for f in sorted(outdir.iterdir()):
    print("-", f.name)
