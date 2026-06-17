# =============================================================================
# LOOCV with PC score prediction (using global PCA)
# =============================================================================

from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from joblib import Parallel, delayed
import numpy as np
import warnings

# ----------------------------------------------------------------------
# 1. Pre‑compute global PCA model (on all training data) for later projection
# ----------------------------------------------------------------------
imputer_global = SimpleImputer(strategy="median")
Y_imp_global = imputer_global.fit_transform(Y_train)
scaler_global = StandardScaler()
Y_scaled_global = scaler_global.fit_transform(Y_imp_global)
pca_global = PCA(n_components=0.9, random_state=42)
Z_global = pca_global.fit_transform(Y_scaled_global)
n_pc_global = Z_global.shape[1]
print(f"Global PCA: {Y_train.shape[1]} → {n_pc_global} components")

# ----------------------------------------------------------------------
# 2. Helper function for a single LOOCV fold (returns predictions in PC space)
# ----------------------------------------------------------------------
def make_kernel(n_dims):
    return (ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(length_scale=np.ones(n_dims), length_scale_bounds=(1e-2, 1e2)) +
            WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2)))

def loo_one_fold(train_idx, test_idx):
    """Retrain emulator on n-1 points, predict PC scores for left-out point."""
    X_tr = X_train[train_idx]
    X_te = X_train[test_idx]
    Y_tr = Y_train[train_idx]

    # fresh preprocessing per fold (to avoid leakage)
    _imp = SimpleImputer(strategy="median")
    _scaler = StandardScaler()
    _pca = PCA(n_components=0.9, random_state=42)

    Y_tr_imp = _imp.fit_transform(Y_tr)
    Y_tr_scaled = _scaler.fit_transform(Y_tr_imp)
    Z_tr = _pca.fit_transform(Y_tr_scaled)
    n_comp = Z_tr.shape[1]

    # Train GP on each PC of the fold's PCA space
    gps_loo = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_comp):
            gp = GaussianProcessRegressor(kernel=make_kernel(X_tr.shape[1]),
                                          normalize_y=True,
                                          n_restarts_optimizer=10,
                                          random_state=42)
            gp.fit(X_tr, Z_tr[:, i])
            gps_loo.append(gp)

    # Predict PC scores in the fold's PCA space
    Z_pred_fold = np.array([gp.predict(X_te.reshape(1, -1))[0] for gp in gps_loo])
    # Transform back to observable space and then to global PC space
    Y_pred = _scaler.inverse_transform(_pca.inverse_transform(Z_pred_fold.reshape(1, -1)))[0]
    # Project predicted observables into global PCA space
    Y_pred_scaled = scaler_global.transform(Y_pred.reshape(1, -1))
    Z_pred_global = pca_global.transform(Y_pred_scaled)[0]

    # True global PC scores for the left-out point
    Y_true = Y_train[test_idx[0]]
    Y_true_scaled = scaler_global.transform(Y_true.reshape(1, -1))
    Z_true_global = pca_global.transform(Y_true_scaled)[0]

    # Relative error in observable space (as before)
    rel_err = np.abs(Y_pred - Y_true) / (np.abs(Y_true) + 1e-6)
    n_bins = len(Y_true) // 3
    block_errs = {
        "nc2": np.nanmean(rel_err[:n_bins]),
        "nc3": np.nanmean(rel_err[n_bins:2*n_bins]),
        "nc4": np.nanmean(rel_err[2*n_bins:]),
        "all": np.nanmean(rel_err),
    }
    # Also return the PC predictions
    return block_errs, Z_pred_global, Z_true_global

# ----------------------------------------------------------------------
# 3. Run LOOCV in parallel
# ----------------------------------------------------------------------
loo = LeaveOneOut()
splits = list(loo.split(X_train))
n_folds = len(splits)

print(f"Running {n_folds}-fold LOO-CV in parallel (n_jobs={min(n_folds, 50)})...")
results = Parallel(n_jobs=min(n_folds, 60), verbose=5)(
    delayed(loo_one_fold)(tr, te) for tr, te in splits
)

# Unpack results
fold_errs = [r[0] for r in results]
Z_pred_all = np.array([r[1] for r in results])
Z_true_all = np.array([r[2] for r in results])

all_errs = np.array([e["all"] for e in fold_errs])
nc2_errs = np.array([e["nc2"] for e in fold_errs])
nc3_errs = np.array([e["nc3"] for e in fold_errs])
nc4_errs = np.array([e["nc4"] for e in fold_errs])

# ----------------------------------------------------------------------
# 4. Print summary
# ----------------------------------------------------------------------
print(f"\nLOO-CV results ({n_folds} folds)")
print(f"{'Observable':<12} {'Mean rel. err':>14}  {'Std':>8}  {'Max':>8}")
print("-" * 46)
for name, errs in [("overall", all_errs), ("nc2", nc2_errs),
                    ("nc3", nc3_errs), ("nc4", nc4_errs)]:
    print(f"{name:<12} {np.mean(errs):>14.4f}  {np.std(errs):>8.4f}  {np.max(errs):>8.4f}")

threshold = 0.20
bad = [(i, all_errs[i]) for i in range(n_folds) if all_errs[i] > threshold]
if bad:
    print(f"\nDesign points with LOO error > {threshold*100:.0f}%:")
    for idx, err in sorted(bad, key=lambda x: -x[1]):
        print(f"  design {idx:02d}: {err:.4f}")
else:
    print(f"\nAll design points within {threshold*100:.0f}% — emulator looks good.")
