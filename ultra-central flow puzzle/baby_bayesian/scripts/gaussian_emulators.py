from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from pathlib import Path
import numpy as np

X_train     = X.copy()
#Y_train     = Y.copy()
Y_train     = Y_zoom.copy() # only most central bins

print(f"X_train shape : {X_train.shape}")
print(f"Y_train shape : {Y_train.shape}")
print(f"Parameters    : {param_names}")

# ── train one GP per principal component ─────────────────────────────
gps = []
for i in range(Z.shape[1]):
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              RBF(length_scale=np.ones(X_train.shape[1]),
                  length_scale_bounds=(1e-2, 1e2)) +
              WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=5, random_state=42)
    gp.fit(X_train, Z[:, i])
    gps.append(gp)
    print(f"  GP {i+1}/{Z.shape[1]} trained — log-marginal-likelihood: "
          f"{gp.log_marginal_likelihood_value_:.2f}")

# ── prediction function ───────────────────────────────────────────────
def emulator_predict(X_new):
    X_new  = np.asarray(X_new).reshape(-1, X_train.shape[1])
    Z_mean = np.column_stack([gp.predict(X_new) for gp in gps])
    Z_std  = np.column_stack([gp.predict(X_new, return_std=True)[1] for gp in gps])
    Y_mean = y_scaler.inverse_transform(pca.inverse_transform(Z_mean))
    Y_std  = np.sqrt(np.sum((Z_std[:, :, None] *
                             pca.components_[None, :, :])**2, axis=1))
    Y_std  = Y_std * y_scaler.scale_
    return Y_mean, Y_std

print("Emulator ready.")
