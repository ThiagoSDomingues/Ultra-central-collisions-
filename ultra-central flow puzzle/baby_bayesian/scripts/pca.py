# =============================================================================
# PCA Analysis on the full observable matrix Y
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# 1. Build feature metadata for the columns of Y
#    The order of columns in Y is: all nc2 bins, then all nc3 bins, then all nc4 bins.
#    The bin centres are stored in cents_nc2, cents_nc3, cents_nc4.
# -----------------------------------------------------------------------------
feature_metadata = []

# nc2 bins
for cent in cents_nc2:
    feature_metadata.append({'harm': 'nc2', 'center': cent, 'type': 'exp_bin'})
# nc3 bins
for cent in cents_nc3:
    feature_metadata.append({'harm': 'nc3', 'center': cent, 'type': 'exp_bin'})
# nc4 bins
for cent in cents_nc4:
    feature_metadata.append({'harm': 'nc4', 'center': cent, 'type': 'exp_bin'})

print(f"Total number of observables: {len(feature_metadata)}")
print(f"nc2 bins: {len(cents_nc2)}, nc3 bins: {len(cents_nc3)}, nc4 bins: {len(cents_nc4)}")

# -----------------------------------------------------------------------------
# 2. Standardize and perform PCA (keep 90% variance, or adjust as needed)
# -----------------------------------------------------------------------------
y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y_zoom) # Y_zoom to use only most central bins

pca = PCA(n_components=0.90, svd_solver="full", random_state=42)
Z = pca.fit_transform(Y_scaled)

print(f"\nPCA Compression Diagnostics:")
print(f"  Original Space Dimensionality = {Y_zoom.shape[1]}")
print(f"  Retained Component Space Size = {Z.shape[1]}")
print(f"  Total Retained Variance       = {100 * pca.explained_variance_ratio_.sum():.2f}%\n")

# -----------------------------------------------------------------------------
# 6. Summary statistics
# -----------------------------------------------------------------------------
print("\n" + "="*60)
print("PCA SUMMARY")
print("="*60)
print(f"Total observables:                 {Y.shape[1]}")
print(f"Principal Components retained:     {Z.shape[1]}")
print(f"Cumulative explained variance:     {100*pca.explained_variance_ratio_.sum():.2f}%")
print(f"\nVariance per component:")
for i, ev in enumerate(pca.explained_variance_ratio_[:8]):
    print(f"  PC{i+1}: {100*ev:.2f}% (cumulative: {100*np.sum(pca.explained_variance_ratio_[:i+1]):.2f}%)")
if Z.shape[1] > 8:
    print(f"  ... and {Z.shape[1]-8} more components.")
print("="*60)
