from pathlib import Path


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
