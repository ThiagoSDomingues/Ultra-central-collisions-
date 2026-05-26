from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
import textwrap

np.random.seed(42)

outdir = Path("closure_test_demo")
outdir.mkdir(exist_ok=True)

# -----------------------------
# Synthetic "heavy-ion" model
# -----------------------------
# Parameters:
# theta1 = eta/s
# theta2 = bulk normalization
# theta3 = switching temperature

param_names = [r'$\eta/s$', r'$\zeta/s$ norm', r'$T_{\mathrm{sw}}$']
bounds = np.array([
    [0.05, 0.30],
    [0.0, 2.0],
    [0.135, 0.165]
])

def true_model(theta):
    """
    Synthetic observable map:
    y = [dNch/deta, <pT>, v2]
    """
    x1, x2, x3 = theta[...,0], theta[...,1], theta[...,2]

    y1 = 1100 + 120*np.sin(5*x1) - 60*x2 + 250*(x3-0.15)
    y2 = 0.45 + 0.08*x1 + 0.03*x2 - 0.8*(x3-0.15)**2
    y3 = 0.055 - 0.05*x1 + 0.012*x2 + 0.10*(x3-0.15)

    return np.stack([y1, y2, y3], axis=-1)

# -----------------------------
# Generate LHS design
# -----------------------------
n_design = 120
sampler = qmc.LatinHypercube(d=3, seed=42)
u = sampler.random(n_design)

design = bounds[:,0] + u*(bounds[:,1]-bounds[:,0])
observables = true_model(design)

# -----------------------------
# Emulator using RBF interpolation
# -----------------------------
emulator = RBFInterpolator(design, observables, kernel='thin_plate_spline')

# -----------------------------
# Choose "true" parameter
# -----------------------------
theta_true = np.array([0.16, 1.0, 0.150])
y_true = true_model(theta_true.reshape(1,-1))[0]

# -----------------------------
# Statistical uncertainty model
# sigma ~ 1/sqrt(Nevents)
# -----------------------------
base_sigma = np.array([30.0, 0.010, 0.008])

event_counts = [5_000, 20_000, 100_000]

results = []

# -----------------------------
# Posterior via Gaussian likelihood
# -----------------------------
grid_n = 45

axes = [
    np.linspace(bounds[i,0], bounds[i,1], grid_n)
    for i in range(3)
]

mesh = np.meshgrid(*axes, indexing='ij')
grid = np.stack([m.flatten() for m in mesh], axis=-1)

emu_pred = emulator(grid)

for Nev in event_counts:

    sigma = base_sigma / np.sqrt(Nev/5000)

    y_mock = y_true + np.random.normal(scale=sigma)

    chi2 = np.sum(((emu_pred - y_mock)/sigma)**2, axis=1)
    logp = -0.5*chi2

    p = np.exp(logp - np.max(logp))
    p /= np.sum(p)

    mean = np.sum(grid * p[:,None], axis=0)
    std = np.sqrt(np.sum((grid-mean)**2 * p[:,None], axis=0))

    contains = np.abs(theta_true - mean) < std

    results.append({
        "Nev": Nev,
        "sigma_obs_1": sigma[0],
        "sigma_obs_2": sigma[1],
        "sigma_obs_3": sigma[2],
        "mean_eta_over_s": mean[0],
        "std_eta_over_s": std[0],
        "mean_bulk": mean[1],
        "std_bulk": std[1],
        "mean_Tsw": mean[2],
        "std_Tsw": std[2],
        "coverage_all_1sigma": bool(np.all(contains))
    })

    # --------------------------------------------------
    # Posterior slices / marginal distributions
    # --------------------------------------------------
    fig, axs = plt.subplots(1,3, figsize=(15,4))

    for i in range(3):
        marg = np.sum(
            p.reshape(grid_n,grid_n,grid_n),
            axis=tuple(j for j in range(3) if j != i)
        )

        axs[i].plot(axes[i], marg)
        axs[i].axvline(theta_true[i], linestyle='--')
        axs[i].set_xlabel(param_names[i])
        axs[i].set_ylabel("Posterior density")
        axs[i].set_title(f"{param_names[i]}")

    fig.suptitle(f"Posterior marginals — N_events = {Nev:,}")
    fig.tight_layout()

    fig.savefig(outdir / f"posterior_marginals_{Nev}.pdf", dpi=220)
    plt.close(fig)

# -----------------------------
# Plot 1: LHS design
# -----------------------------
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111)

sc = ax.scatter(design[:,0], design[:,1], c=observables[:,2], s=35)

ax.scatter(
    theta_true[0],
    theta_true[1],
    marker='*',
    s=250
)

ax.set_xlabel(param_names[0])
ax.set_ylabel(param_names[1])
ax.set_title("Latin hypercube design points")

cbar = plt.colorbar(sc)
cbar.set_label(r'$v_2$')

fig.tight_layout()
fig.savefig(outdir / "lhs_design.pdf", dpi=220)
plt.close(fig)

# -----------------------------
# Plot 2: observable uncertainties vs events
# -----------------------------
fig, ax = plt.subplots(figsize=(6,4))

sigmas = np.array([
    base_sigma / np.sqrt(N/5000)
    for N in event_counts
])

for i, label in enumerate([r'$dN/d\eta$', r'$\langle p_T \rangle$', r'$v_2$']):
    ax.plot(event_counts, sigmas[:,i], marker='o', label=label)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Events per design point")
ax.set_ylabel("Statistical uncertainty")
ax.set_title("Statistical uncertainty decreases as $1/\\sqrt{N}$")
ax.legend()

fig.tight_layout()
fig.savefig(outdir / "uncertainty_scaling.pdf", dpi=220)
plt.close(fig)

# -----------------------------
# Plot 3: posterior width scaling
# -----------------------------
df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(6,4))

ax.plot(df["Nev"], df["std_eta_over_s"], marker='o', label=r'$\eta/s$')
ax.plot(df["Nev"], df["std_bulk"], marker='o', label=r'$\zeta/s$ norm')
ax.plot(df["Nev"], df["std_Tsw"], marker='o', label=r'$T_{\mathrm{sw}}$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel("Events per design point")
ax.set_ylabel("Posterior width (1σ)")
ax.set_title("Posterior narrowing with more events")
ax.legend()

fig.tight_layout()
fig.savefig(outdir / "posterior_widths.pdf", dpi=220)
plt.close(fig)

# -----------------------------
# Save mock data
# -----------------------------
design_df = pd.DataFrame(
    design,
    columns=["eta_over_s", "bulk_norm", "Tsw"]
)

obs_df = pd.DataFrame(
    observables,
    columns=["dNch_deta", "mean_pT", "v2"]
)

design_full = pd.concat([design_df, obs_df], axis=1)
design_full.to_csv(outdir / "mock_design_points.csv", index=False)

df.to_csv(outdir / "closure_results.csv", index=False)

# -----------------------------
# Build markdown tutorial
# -----------------------------
tutorial = f"""
# Closure Test Tutorial for Bayesian Calibration

This example demonstrates how to perform a closure test to determine how many events per design point are needed in a Bayesian analysis.

The idea is:

1. Generate a Latin hypercube design (LHS)
2. Train an emulator
3. Pick a known “true” parameter point
4. Generate mock experimental data
5. Run Bayesian inference
6. Check whether:
   - the posterior recovers the true parameter
   - the posterior width is sufficiently constrained

---

# 1. Synthetic model

We use 3 parameters:

- eta/s
- bulk viscosity normalization
- switching temperature

and 3 observables:

- charged multiplicity
- mean pT
- v2

---

# 2. Mock design points

The file:

mock_design_points.csv

contains the synthetic training data.

The emulator is trained using an RBF interpolator.

---

# 3. Closure test procedure

For each number of events:

N_events = {event_counts}

we assume statistical uncertainties scale like:

sigma ~ 1/sqrt(N_events)

We then:

- generate mock data around the true point
- compute likelihoods
- build the posterior
- measure posterior widths

---

# 4. Interpretation

If the posterior:

- contains the true parameter
- is significantly narrower than the prior

then the event statistics are sufficient.

Typically:

- too few events -> broad posterior
- enough events -> constrained posterior
- excessive events -> diminishing returns

---

# 5. Recommended workflow for your real analysis

For each candidate event count:

1. Run a small pilot simulation
2. Estimate observable covariance matrix
3. Perform emulator training
4. Run closure tests
5. Measure posterior widths
6. Plot width vs N_events
7. Choose the smallest N_events where widths saturate

This gives the optimal computational cost.

"""

tutorial_path = outdir / "closure_test_tutorial.md"
tutorial_path.write_text(textwrap.dedent(tutorial))

print("Generated files:")
for f in sorted(outdir.iterdir()):
    print("-", f.name)
