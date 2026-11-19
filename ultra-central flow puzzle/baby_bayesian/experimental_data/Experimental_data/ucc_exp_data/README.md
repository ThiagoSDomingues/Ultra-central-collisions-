# UCC experimental data extracted from figures

Extracted 2026-11-19 for the ultracentral flow puzzle campaign. Every CSV has `#` comment lines on top (figure ID, system, kinematics, caveats); read with `pandas.read_csv(f, comment='#')`. `plot_all.py` remakes all six figures (`replots.pdf`).

## Files

| File | Content | How obtained |
|---|---|---|
| `alice_prel_5p36_v32_over_v22_ucc.csv` | ALICE prel. v3{2}/v2{2}, bins 0–0.005, 0–0.02, 0–0.2, 0–1, 0–2.5, 2.5–5 % (ALI-PREL-621021) + Trajectum points | digitized |
| `alice_prel_5p36_v34_over_v32.csv` | ALICE prel. v3{4}/v3{2}, 1 % FT0C bins in 0–5 % (ALI-PREL-621026) | digitized |
| `alice_prel_5p36_v32_over_v22_0to5.csv` | ALICE prel. v3{2}/v2{2}, Run 3 5.36 TeV and Run 2 5.02 TeV, 1 % bins (ALI-PREL-598067) | digitized |
| `alice_prel_5p36_vn_norm_vs_Nch.csv` | ALICE prel. v2{2}, v3{2} normalized to 0–5 % vs Nch/Nch(0–5 %) (ALI-PREL-598097) | digitized |
| `*_trajectum_curve.csv` | Trajectum (PRC 106 (2022) 044903) dotted curves from the ALICE figures | digitized |
| `cms_2p76_vn2_ucc.csv` | CMS vn{2}, n = 2, 3, 4, bins 0–0.02, 0–0.2, 0–1, 0–2.5, 2.5–5 % (JHEP 02 (2014) 088) with band | exact, vector PDF |
| `cms_2p76_v32_over_v22_ucc.csv` | CMS v3{2}/v2{2} with band | exact, vector PDF |
| `alice_5p02_vn2_1pct.csv` | ALICE vn{2}, n = 2, 3, 4, 1 % bins in 0–5 % (JHEP 07 (2018) 103) with band | exact, vector PDF |
| `models_from_group_figures.csv` | Trajectum 1, Trajectum 2, JETSCAPE Grad, Duke points from the group figures | exact, vector PDF |

## Accuracy

The group figures are vector PDFs, so those values are the plotted numbers to floating-point precision (axis calibration residual < 3×10⁻⁵ in log10 centrality, 0 in value).

The ALICE preliminary figures are 720 ppi raster images. Points were found by exact colour, axes calibrated with a linear fit to the major ticks. One pixel is 0.0001–0.0003 in value. Check: the Run 2 points digitized from ALI-PREL-598067 agree with the exact Run 2 values from the vector figure to **3×10⁻⁵** (max over 5 points). Error bars and boxes are read the same way; expect about ±0.001.

## Caveats

- **CMS v3 is drawn ×0.8** in the vn{2} figure. The CSV gives the unscaled value (`plot_scale` column records it). CMS v4 is not scaled.
- **CMS and ALICE 5.02 TeV bands:** the figures show one grey band per series and no separate stat bars. Check the papers / HEPData for whether it is syst only or total before using it in a likelihood.
- **ALICE preliminary stat errors in v3{2}/v2{2} (UCC bins) are hidden by the marker.** The CSV gives `stat_upper_limit` = marker radius (0.008) instead of a value.
- `stat`, `syst` = larger of the two measured sides. A few sides are covered by overlapping markers or curves; raw `*_up` / `*_dn` are kept.
- Run 2 points in ALI-PREL-598067 are drawn shifted by +0.1 % for visibility; `centrality_bin_FT0C_pct` gives the true bin.
- ALICE preliminary numbers must be replaced by HEPData when the paper appears; cite as ALICE Preliminary with the ALI-PREL ID.

## Reproduce the extraction

`scripts/extract_alice.py` (raster) and `scripts/extract_cms.py` + `scripts/vec.py` (vector), `scripts/build_package.py` writes the CSVs.
