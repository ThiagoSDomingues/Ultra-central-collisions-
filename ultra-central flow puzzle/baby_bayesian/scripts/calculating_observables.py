import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

# ── configuration ─────────────────────────────────────────────────────
VERSION  = "sync"   # "std" or "sync"
BASE_DIR = Path("/home/thiagos/UCC_project/Trento_events")
LHS_DIR  = BASE_DIR / "my_lhs_design_210"

if VERSION == "std":
    CACHE_DIR = BASE_DIR / "trento_events_cache"
else:
    CACHE_DIR = BASE_DIR / "trento_sync_cache/750k_new"

# ── load design matrix ───────────────────────────────
design_data = np.load(LHS_DIR / "lhs_design_matrix.npz")
X = design_data["design"]
param_lo = design_data["param_lo"]
param_hi = design_data["param_hi"]
param_names = design_data["param_names"]

# ── Latex labels ───────────────────────────────
param_labels = {
    "WS_R": r"$W\!S_R$",
    "WS_A": r"$W\!S_A$",
    "beta3": r"$\beta_3$",
    "beta4": r"$\beta_4$",
    "w": r"$w$",
}

# ── load experimental data ─────────────────────────────────────────────────
HEPDATA_DIR = Path(".")

def read_hepdata_alice(path, val_col):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            rows.append(line.strip().split(","))
    hdr = rows[0]; data = rows[1:]
    ci = hdr.index("Centrality"); vi = hdr.index(val_col)
    sp = hdr.index("stat +");      yp = hdr.index("sys +")
    cent  = np.array([float(r[ci]) for r in data])
    val   = np.array([float(r[vi]) for r in data])
    stat  = np.array([abs(float(r[sp])) for r in data])
    syst  = np.array([abs(float(r[yp])) for r in data])
    return cent, val, stat, syst

def read_hepdata_atlas(path, val_col):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            rows.append(line.strip().split(","))
    hdr = rows[0]; data = rows[1:]
    ci = hdr.index("Centrality"); vi = hdr.index(val_col)
    sp = hdr.index("stat. uncertainty +")
    yp = hdr.index("syst. uncertainty +"); ym = hdr.index("syst. uncertainty -")
    cent = np.array([float(r[ci]) for r in data])
    val  = np.array([float(r[vi]) for r in data])
    stat = np.array([abs(float(r[sp])) for r in data])
    syst = np.array([max(abs(float(r[yp])), abs(float(r[ym]))) for r in data])
    return cent, val, stat, syst

def tot_err(s, y): return np.sqrt(s**2 + y**2)

try:
    c, c22, s22, y22 = read_hepdata_alice("HEPData-ins1666817-v1-Table_201.csv", "c22")
    _, c24, s24, y24 = read_hepdata_alice("HEPData-ins1666817-v1-Table_202.csv", "c24")
    alice_c_502  = c
    alice_R_502  = -c24 / c22**2
    alice_eR_502 = np.abs(alice_R_502) * np.sqrt((tot_err(s24,y24)/np.abs(c24))**2 + (2*tot_err(s22,y22)/c22)**2)
    
    c, c22, s22, y22 = read_hepdata_alice("HEPData-ins1666817-v1-Table_205.csv", "c22")
    _, c24, s24, y24 = read_hepdata_alice("HEPData-ins1666817-v1-Table_206.csv", "c24")
    alice_c_276  = c
    alice_R_276  = -c24 / c22**2
    alice_eR_276 = np.abs(alice_R_276) * np.sqrt((tot_err(s24,y24)/np.abs(c24))**2 + (2*tot_err(s22,y22)/c22)**2)
    
    atl2_c, atl2_v, atl2_s, atl2_y = read_hepdata_atlas("HEPData-ins1728935-v1-Figure4_Panel1_pt0.csv", "nc_2{4}")
    atl3_c, atl3_v, atl3_s, atl3_y = read_hepdata_atlas("HEPData-ins1728935-v1-Figure4_Panel2_pt0.csv", "nc_3{4}")
    atl4_c, atl4_v, atl4_s, atl4_y = read_hepdata_atlas("HEPData-ins1728935-v1-Figure4_Panel3_pt0.csv", "nc_4{4}")
    neg_atl2 = -atl2_v;  err_atl2 = tot_err(atl2_s, atl2_y)
    neg_atl3 = -atl3_v;  err_atl3 = tot_err(atl3_s, atl3_y)
    neg_atl4 = -atl4_v;  err_atl4 = tot_err(atl4_s, atl4_y)
    print("Experimental data loaded from HEPData CSV files.")
except FileNotFoundError:
    print("CSV files not found — using fallback placeholder data.")
    alice_c_502  = np.array([0.5,1.5,2.5,3.5,4.5,7.5,12.5,17.5,25,35,45,55])
    alice_R_502  = np.array([-0.118,-0.110,-0.098,-0.088,-0.081,-0.074,-0.064,-0.055,-0.047,-0.033,-0.016, 0.005])
    alice_eR_502 = np.abs(alice_R_502)*0.03
    alice_c_276  = alice_c_502.copy()
    alice_R_276  = alice_R_502*0.95
    alice_eR_276 = alice_eR_502.copy()
    atl2_c = atl3_c = atl4_c = np.array([2.5,7.5,12.5,17.5,22.5,27.5,32.5,37.5,42.5,47.5])
    neg_atl2 = np.array([-0.115,-0.072,-0.060,-0.052,-0.043,-0.034,-0.024,-0.014,-0.004, 0.005])
    neg_atl3 = np.array([-0.048,-0.035,-0.028,-0.022,-0.016,-0.010,-0.004, 0.001, 0.003, 0.005])
    neg_atl4 = np.zeros(10)
    err_atl2 = err_atl3 = err_atl4 = np.full(10, 0.005)

# ── helper function for edge generation ───────────────────────────────
def centers_to_edges(c):
    c = np.asarray(c, dtype=float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    edges = np.empty(len(c) + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0]  = c[0]  - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges

BINNING_EXP = {
    2: centers_to_edges(alice_c_502), 
    3: centers_to_edges(atl3_c),
    4: centers_to_edges(atl4_c),
}

# ── recompute zoom profiles per design point ──────────────────────────
def assign_centrality(mult):
    rank = np.argsort(np.argsort(-mult))
    return rank / len(mult) * 100.0

def jackknife_nc(eps):
    N = len(eps)
    if N < 10:
        return np.nan, np.nan
    e2, e4 = eps**2, eps**4
    m2, m4 = e2.mean(), e4.mean()
    nc_val  = m4 / m2**2 - 2.0
    m2_loo  = (N*m2 - e2) / (N-1)
    m4_loo  = (N*m4 - e4) / (N-1)
    nc_loo  = m4_loo / m2_loo**2 - 2.0
    jk_var  = (N-1)/N * np.sum((nc_loo - nc_loo.mean())**2)
    return nc_val, np.sqrt(jk_var)

def compute_nc_profile(mult, eps_dict, bin_edges_dict):
    cent = assign_centrality(mult)
    prof = {}
    for n, eps in eps_dict.items():
        edges = bin_edges_dict[n]
        mids = []
        vals = []
        errs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (cent >= lo) & (cent < hi)
            if mask.sum() < 10:
                mids.append(0.5*(lo+hi))
                vals.append(np.nan)
                errs.append(np.nan)
                continue
            v, e = jackknife_nc(eps[mask])
            mids.append(0.5*(lo+hi))
            vals.append(v)
            errs.append(e)
        prof[n] = {
            "cent": np.array(mids),
            "val": np.array(vals),
            "err": np.array(errs)
        }
    return prof

if VERSION == "std":
    COL_MULT, COL_E2, COL_E3, COL_E4 = 3, 4, 5, 6
else: # sync version – 13 columns
    # indices for real and imaginary parts
    COL_MULT = 3
    COL_E2_RE = 5
    COL_E2_IM = 6
    COL_E3_RE = 7
    COL_E3_IM = 8
    COL_E4_RE = 9
    COL_E4_IM = 10
    # dummy variables for backward compatibility
    COL_E2 = COL_E3 = COL_E4 = None

# calculating the input matrix (observables).
def load_results(cache_file):
    idx = int(cache_file.stem.split("_")[-1])
    arr = np.load(cache_file)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    mult = arr[:, COL_MULT]
    if VERSION == "std":
        e2 = arr[:, COL_E2]
        e3 = arr[:, COL_E3]
        e4 = arr[:, COL_E4]
    else:  # sync: compute |ε| = sqrt(re² + im²)
        e2 = np.sqrt(arr[:, COL_E2_RE]**2 + arr[:, COL_E2_IM]**2)
        e3 = np.sqrt(arr[:, COL_E3_RE]**2 + arr[:, COL_E3_IM]**2)
        e4 = np.sqrt(arr[:, COL_E4_RE]**2 + arr[:, COL_E4_IM]**2)
    eps_dict = {2: e2, 3: e3, 4: e4}            
    return idx, compute_nc_profile(mult, eps_dict, BINNING_EXP)

cache_files = sorted(CACHE_DIR.glob("events_design_*.npy"))
n_jobs = 60 # increase to run faster
results = Parallel(n_jobs, verbose=0)(
    delayed(load_results)(f) for f in cache_files
)
results.sort(key=lambda x: x[0])

cents_nc2 = results[0][1][2]["cent"]
cents_nc3 = results[0][1][3]["cent"]
cents_nc4 = results[0][1][4]["cent"]

def input_matrix(harm):
    return np.asarray([
        prof[harm]["val"]
        for _, prof in results
    ])

m2 = input_matrix(2)
m3 = input_matrix(3)
m4 = input_matrix(4)

# Create boolean masks for bins between 0 and 10%
mask_nc2 = (cents_nc2 >= 0) & (cents_nc2 <= 10)
mask_nc3 = (cents_nc3 >= 0) & (cents_nc3 <= 10)
mask_nc4 = (cents_nc4 >= 0) & (cents_nc4 <= 10)

# Slice the 1D coordinate arrays to get the 0-10% bin centers
zoom_cents_nc2 = cents_nc2[mask_nc2]
zoom_cents_nc3 = cents_nc3[mask_nc3]
zoom_cents_nc4 = cents_nc4[mask_nc4]

# 2. Slice the columns of your matrices using the masks
zm2 = m2[:, mask_nc2]
zm3 = m3[:, mask_nc3]
zm4 = m4[:, mask_nc4]

### in the future, better to save these input matrices or Y

# concatenate and flatten the input matrix
Y = np.concatenate([-m2, -m3, -m4], axis=1)

# concatenate and flatten the input zoom matrix (0-10% events)
Y_zoom = np.concatenate([-zm2, -zm3, -zm4], axis=1)

print(f"Loaded {len(X)} design points, Y shape: {Y.shape}, Y_zoom shape: {Y_zoom.shape}")


# =============================================================================
# ε₂{2}/ε₃{2} ratio: TRENTo LHS envelope vs. ALICE v22/v32 data (0–10%)
# =============================================================================

from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd

# ----------------------------------------------------------------------
# 1. Load ALICE experimental data (v22/v32)
# ----------------------------------------------------------------------

# For safety, we recompute using the same code you provided:
t7 = pd.read_csv("HEPData-ins1666817-v1-Table_7.csv", comment="#").iloc[:10].copy()
t9 = pd.read_csv("HEPData-ins1666817-v1-Table_9.csv", comment="#").copy()

cent_exp = t7["Centrality"].to_numpy() # experimental centers 
v22 = t7["v22Deta1"].to_numpy()
v22_stat = t7["stat +"].abs().to_numpy()
v22_syst = t7["sys +"].abs().to_numpy()

cent_v32 = t9["Centrality"].to_numpy()
v32 = t9["v32Deta1"].to_numpy()
v32_stat = t9["stat +"].abs().to_numpy()
v32_syst = t9["sys +"].abs().to_numpy()

# Interpolate v32 to the same centrality bins as v22
v32_i = np.interp(cent_exp, cent_v32, v32)
v32_stat_i = np.interp(cent_exp, cent_v32, v32_stat)
v32_syst_i = np.interp(cent_exp, cent_v32, v32_syst)

# Compute ratio v22/v32 and propagate errors (stat and syst separately)
ratio = v22 / v32_i
ratio_stat = ratio * np.sqrt((v22_stat / v22)**2 + (v32_stat_i / v32_i)**2)
ratio_syst = ratio * np.sqrt((v22_syst / v22)**2 + (v32_syst_i / v32_i)**2)

# Combine stat and syst in quadrature for total error
ratio_err = np.sqrt(ratio_stat**2 + ratio_syst**2)

# The centrality bin centres are the midpoints of the bins:
centers_exp = cent_exp   # Table_7 gives the bin centres (e.g., 0.5, 1.5, ... , 9.5)

# ----------------------------------------------------------------------
# 2. TRENTo results for the same bins
# ----------------------------------------------------------------------
# =============================================================================
# Compute ε₂{2}/ε₃{2} ratios for ultracentral bins (0‑10%, 1% bins)
# from cached event files, and save to disk.
# =============================================================================

VERSION = "sync"   # or "sync"
BASE_DIR = Path("/home/thiagos/UCC_project/Trento_events")
if VERSION == "std":
    COL_MULT, COL_E2, COL_E3, COL_E4 = 3, 4, 5, 6
else: # sync version – 13 columns
    # indices for real and imaginary parts
    COL_MULT = 3
    COL_E2_RE = 5
    COL_E2_IM = 6
    COL_E3_RE = 7
    COL_E3_IM = 8
    COL_E4_RE = 9
    COL_E4_IM = 10
    # dummy variables for backward compatibility
    COL_E2 = COL_E3 = COL_E4 = None
 
centers_zoom = cent_exp # centers
BINS_ZOOM = centers_to_edges(cent_exp) # edges

# ----------------------------------------------------------------------
# Helper functions (jackknife ratio)
# ----------------------------------------------------------------------
def assign_centrality(mult):
    rank = np.argsort(np.argsort(-mult))
    return rank / len(mult) * 100.0

def jackknife_ratio(eps2, eps3):
    N = len(eps2)
    if N < 10:
        return np.nan, np.nan
    m2_2 = np.mean(eps2**2)
    m2_3 = np.mean(eps3**2)
    ratio_val = np.sqrt(m2_2 / m2_3) if m2_3 > 0 else np.nan
    # leave-one-out jackknife
    ratio_loo = []
    for i in range(N):
        m2_2_loo = (N * m2_2 - eps2[i]**2) / (N - 1)
        m2_3_loo = (N * m2_3 - eps3[i]**2) / (N - 1)
        r_loo = np.sqrt(m2_2_loo / m2_3_loo) if m2_3_loo > 0 else np.nan
        if np.isfinite(r_loo):
            ratio_loo.append(r_loo)
    if len(ratio_loo) < 2:
        return ratio_val, np.nan
    ratio_loo = np.array(ratio_loo)
    jk_var = (N - 1) / N * np.sum((ratio_loo - ratio_loo.mean())**2)
    return ratio_val, np.sqrt(jk_var)

def compute_ratio_profile(mult, e2, e3, bin_edges):
    cent = assign_centrality(mult)
    vals = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (cent >= lo) & (cent < hi)
        if mask.sum() < 10:
            vals.append(np.nan)
            continue
        v, _ = jackknife_ratio(e2[mask], e3[mask])
        vals.append(v)
    return np.array(vals)

# ----------------------------------------------------------------------
# Process all design points in parallel
# ----------------------------------------------------------------------
cache_files = sorted(CACHE_DIR.glob("events_design_*.npy"))
print(f"Found {len(cache_files)} design point cache files")

def process_ratio(cache_file):
    arr = np.load(cache_file)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    mult = arr[:, COL_MULT]
    if VERSION == "std":
        e2 = arr[:, COL_E2]
        e3 = arr[:, COL_E3]
    else:  # sync: compute |ε| = sqrt(re² + im²)
        e2 = np.sqrt(arr[:, COL_E2_RE]**2 + arr[:, COL_E2_IM]**2)
        e3 = np.sqrt(arr[:, COL_E3_RE]**2 + arr[:, COL_E3_IM]**2)    
    return compute_ratio_profile(mult, e2, e3, BINS_ZOOM)

print("Computing ε₂/ε₃ ratios for ultracentral bins...")
n_jobs=10
results = Parallel(n_jobs, verbose=5)(
    delayed(process_ratio)(f) for f in cache_files
)

ratios_zoom = np.array(results)   # shape (n_designs, 10)
print(f"Ratios array shape: {ratios_zoom.shape}")

# Save for later use
#np.save("ratios_zoom.npy", ratios_zoom)
#np.save("centers_zoom.npy", centers_zoom)
print("Saved ratios_zoom.npy and centers_zoom.npy")

# ----------------------------------------------------------------------
# Sanity check: print median ratio in first bin
median_first_bin = np.nanmedian(ratios_zoom[:, 0])
print(f"Median ε₂/ε₃ in 0‑1% bin: {median_first_bin:.3f}")
