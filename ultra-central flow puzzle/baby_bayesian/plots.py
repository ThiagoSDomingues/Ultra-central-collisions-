"""
Script to plot eccentricities cumulants. 
"""

import matplotlib.pyplot as plt
from cumulants import * 
from analyze_ultracentral_trento import * 

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

# Multiplicity distribution + ultra-central cut
plt.figure()
plt.hist(mult, bins=100, histtype="step", label="All events")
plt.axvline(mult_sorted[n_uc], color="red", linestyle="--",
            label=f"Ultra-central {UC_PERCENTILE}% cut")
plt.xlabel("Multiplicity (entropy)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "multiplicity_cut.pdf")
plt.close()

# Impact parameter distribution (sanity check)
plt.figure()
plt.hist(uc_b, bins=40, density=True)
plt.xlabel("Impact parameter b [fm]")
plt.ylabel("Probability density")
plt.title("Ultra-central impact parameter distribution")
plt.tight_layout()
plt.savefig(OUTDIR / "impact_parameter_uc.pdf")
plt.close()

# ε₂ vs ε₃ scatter (the ultra-central puzzle)
plt.figure()
plt.scatter(uc_eps3, uc_eps2, s=5, alpha=0.4)
plt.xlabel(r"$\varepsilon_3$")
plt.ylabel(r"$\varepsilon_2$")
plt.title("Ultra-central eccentricities")
plt.tight_layout()
plt.savefig(OUTDIR / "eps2_vs_eps3_uc.pdf")
plt.close()

# Distributions of ε₂ and ε₃
plt.figure()
plt.hist(uc_eps2, bins=50, density=True, alpha=0.6, label=r"$\varepsilon_2$")
plt.hist(uc_eps3, bins=50, density=True, alpha=0.6, label=r"$\varepsilon_3$")
plt.xlabel(r"$\varepsilon_n$")
plt.ylabel("Probability density")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "eps_distributions_uc.pdf")
plt.close()

# Event-by-event profiles (visual intuition): plot a grid of ultra-central events.
def plot_event_grid(events, nrows=3, ncols=4, fname="event_grid.pdf"):
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8))
    axes = axes.flatten()

    for ax, ev in zip(axes, events):
        profile = np.array(ev)
        ax.imshow(profile, origin="lower", cmap="inferno")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTDIR / fname)
    plt.close()

plot_event_grid(uc_events[:12], fname="ultracentral_profiles.pdf")

# Ultra-central eccentricity cumulants
def eps2_cumulants(eps):
    e2_2 = np.sqrt(np.mean(eps**2))
    e2_4 = (2*np.mean(eps**2)**2 - np.mean(eps**4))**0.25
    return e2_2, e2_4

e22, e24 = eps2_cumulants(uc_eps2)
e32 = np.sqrt(np.mean(uc_eps3**2))

print("Ultra-central observables:")
print(f"ε2{{2}} = {e22:.4f}")
print(f"ε2{{4}} = {e24:.4f}")
print(f"ε3{{2}} = {e32:.4f}")
print(f"ε2{{2}}/ε3{{2}} = {e22/e32:.4f}")
print(f"ε2{{4}}/ε2{{2}} = {e24/e22:.4f}")
