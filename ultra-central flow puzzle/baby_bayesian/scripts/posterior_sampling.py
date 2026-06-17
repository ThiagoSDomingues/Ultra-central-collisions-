import emcee
import corner

# True parameter (median of prior)
theta_true = (param_lo + param_hi) / 2

Y_true, std_true = emulator_predict(theta_true.reshape(1,-1))

print("At true parameter:")
print(f"  Y[0:3] = {Y_true[0][:3]}")
print(f"  std[0:3] = {std_true[0][:3]}")

# Increase synthetic noise to a realistic level (e.g., 10% of experimental errors)
noise_std = 0.02 * np.ones_like(Y_true[0])   # absolute error of 0.02
np.random.seed(42)
Y_obs = Y_true[0] + np.random.normal(0, noise_std)

# Log‑likelihood and prior
def log_prior(theta):
    if np.all((theta >= param_lo) & (theta <= param_hi)):
        return 0.0
    return -np.inf

# In log_likelihood, add the emulator's own uncertainty (if available)
def log_likelihood(theta):
    Y_pred, Y_std_pred = emulator_predict(theta.reshape(1,-1))
    total_std = np.sqrt(noise_std**2 + Y_std_pred[0]**2 + 1e-8)
    return -0.5 * np.sum(((Y_pred[0] - Y_obs) / total_std)**2 + np.log(2*np.pi*total_std**2))

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# MCMC
ndim = len(theta_true)
nwalkers = 32
nsteps = 5000
np.random.seed(42)
p0 = theta_true + 1e-4 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(p0, nsteps, progress=True)
samples = sampler.get_chain(discard=500, flat=True)

med = np.percentile(samples, 50, axis=0)
lo68 = np.percentile(samples, 16, axis=0)
hi68 = np.percentile(samples, 84, axis=0)

print("Closure test results (true = median prior):")
for i, name in enumerate(param_names):
    print(f"{name:5} : true={theta_true[i]:.4f}, median={med[i]:.4f}, 68% CI=[{lo68[i]:.4f}, {hi68[i]:.4f}]")
print(f"All within 68%? {np.all((theta_true >= lo68) & (theta_true <= hi68))}")
