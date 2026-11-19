"""Assemble digitized/extracted data into CSV files (long format, one file per figure)."""
import json, math, os, csv
import numpy as np

OUT = '/home/claude/ucc_exp_data'
os.makedirs(OUT, exist_ok=True)
A = json.load(open('/home/claude/alice_figs/work/alice_raw.json'))
V = json.load(open('/home/claude/cms_figs/work/vector_raw.json'))

LOGBINS = {0.0025: '0-0.005', 0.01: '0-0.02', 0.1: '0-0.2', 0.5: '0-1', 1.25: '0-2.5', 3.75: '2.5-5'}
def logbin(x):
    k = min(LOGBINS, key=lambda b: abs(math.log10(b) - math.log10(x)))
    return k, LOGBINS[k]
def onepct(x):
    lo = int(math.floor(x + 1e-6)) if abs(x - round(x - 0.5) - 0.5) < 0.2 else int(math.floor(x))
    lo = int(math.floor(x - 0.0001)) if x % 1 > 0.05 else int(x) - 1
    return f'{lo}-{lo + 1}', lo + 0.5

def r(v, n=5):
    return '' if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), n)

def sym(a, b):
    vals = [v for v in (a, b) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return max(vals) if vals else None

def write(name, header, rows, comments):
    with open(f'{OUT}/{name}', 'w', newline='') as f:
        for c in comments: f.write('# ' + c + '\n')
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

H_ALICE = ['series', 'x', 'value', 'stat', 'stat_up', 'stat_dn', 'stat_upper_limit', 'syst', 'syst_up', 'syst_dn']
def alice_rows(series, pts, xfun=lambda x: (x,)):
    out = []
    for p in pts:
        st = sym(p['stat_up'], p['stat_dn'])
        lim = p['stat_hidden_below'] if st is None else None
        out.append([series, *xfun(p['x']), r(p['y']), r(st), r(p['stat_up']), r(p['stat_dn']), r(lim),
                    r(sym(p['syst_up'], p['syst_dn'])), r(p['syst_up']), r(p['syst_dn'])])
    return out

COMMON = ['DIGITIZED from ALICE Preliminary figure (raster, 720 ppi). Not official numbers: replace with HEPData when released.',
          'stat/syst = symmetric best estimate (max of the two measured sides; one side can be hidden by overlapping objects).',
          'stat_upper_limit: stat bar fully hidden inside the marker -> stat error < this value.',
          'Typical digitization uncertainty on central values: +-0.0005 or better (validated against exact ALICE Run 2 values, see README).']

# ---- Fig A
rows = alice_rows('v2{2}/v2{2}(0-5%)', A['A_v2']) + alice_rows('v3{2}/v3{2}(0-5%)', A['A_v3'])
write('alice_prel_5p36_vn_norm_vs_Nch.csv', ['series', 'Nch_over_Nch0-5', 'value', 'stat', 'stat_up', 'stat_dn', 'stat_upper_limit', 'syst', 'syst_up', 'syst_dn'], rows,
      ['ALI-PREL-598097 | Pb-Pb 5.36 TeV | 0.2<pT<3 GeV/c, |eta|<0.8, |Delta eta|>1 | vn{2}/vn{2}(0-5%) vs Nch/Nch(0-5%)'] + COMMON)
rows = [['Trajectum v2{2}', r(x), r(y)] for x, y in A['A_traj_v2']] + [['Trajectum v3{2}', r(x), r(y)] for x, y in A['A_traj_v3']]
write('alice_prel_5p36_vn_norm_vs_Nch_trajectum_curve.csv', ['series', 'Nch_over_Nch0-5', 'value'], rows,
      ['ALI-PREL-598097 | Trajectum PRC 106 (2022) 044903 dotted curves, digitized (averaged in 40-px x bins)'])

# ---- Fig B
rows = []
for series, key in [('ALICE', 'B_alice'), ('Trajectum', 'B_traj')]:
    for p in A[key]:
        xb, lab = logbin(p['x'])
        rr = alice_rows(series, [p])[0]
        rows.append([rr[0], lab, xb] + rr[2:])
write('alice_prel_5p36_v32_over_v22_ucc.csv', ['series', 'centrality_bin_pct', 'x_plot', 'value', 'stat', 'stat_up', 'stat_dn', 'stat_upper_limit', 'syst', 'syst_up', 'syst_dn'], rows,
      ['ALI-PREL-621021 | Pb-Pb 5.36 TeV | 0.2<pT<3 GeV/c, |eta|<0.8, |Delta eta|>1 | v3{2}/v2{2} in ultra-central bins',
       'x_plot is the bin midpoint used on the log axis. ALICE syst = shaded band at the point position. Trajectum error = its stat bar.'] + COMMON)

# ---- Fig C
rows = []
for p in A['C_alice']:
    lab, mid = f"{int(p['x'] - 0.5 + 0.01)}-{int(p['x'] - 0.5 + 0.01) + 1}", round(p['x'] * 2) / 2
    rr = alice_rows('ALICE', [p])[0]
    rows.append([rr[0], lab, mid] + rr[2:])
write('alice_prel_5p36_v34_over_v32.csv', ['series', 'centrality_bin_FT0C_pct', 'x_plot', 'value', 'stat', 'stat_up', 'stat_dn', 'stat_upper_limit', 'syst', 'syst_up', 'syst_dn'], rows,
      ['ALI-PREL-621026 | Pb-Pb 5.36 TeV | 0.2<pT<3 GeV/c, |eta|<0.8, |Delta eta|>1 | v3{4}/v3{2}',
       f"Bessel-Gaussian limit line read at {A['C_bessel_gaussian']:.4f} (expected sqrt(4/pi-1)=0.5227)"] + COMMON)

# ---- Fig D
rows = []
for series, key, shift in [('ALICE Run 3 (5.36 TeV)', 'D_run3', 0.0), ('ALICE Run 2 (5.02 TeV), JHEP 07 (2018) 103', 'D_run2', 0.1)]:
    for p in A[key]:
        mid = round((p['x'] - shift) * 2) / 2
        lab = f'{int(mid - 0.5)}-{int(mid + 0.5)}'
        rr = alice_rows(series, [p])[0]
        rows.append([rr[0], lab, round(p['x'], 3)] + rr[2:])
write('alice_prel_5p36_v32_over_v22_0to5.csv', ['series', 'centrality_bin_FT0C_pct', 'x_plot', 'value', 'stat', 'stat_up', 'stat_dn', 'stat_upper_limit', 'syst', 'syst_up', 'syst_dn'], rows,
      ['ALI-PREL-598067 | Pb-Pb 5.36 TeV Run 3 vs Run 2 5.02 TeV | 0.2<pT<3 GeV/c, |eta|<0.8, |Delta eta|>1 | v3{2}/v2{2}',
       'Run 2 points are drawn shifted by +0.1 in x for visibility; x_plot keeps the drawn position. Exact Run 2 values: alice_5p02_vn2_1pct.csv.'] + COMMON)
write('alice_prel_5p36_v32_over_v22_0to5_trajectum_curve.csv', ['series', 'centrality_pct', 'value'],
      [['Trajectum', r(x), r(y)] for x, y in A['D_traj']], ['ALI-PREL-598067 | Trajectum PRC 106 (2022) 044903 dotted curve, digitized'])

# ---- vector figures: experimental data
def exp_points(fig):
    return [m for m in V[fig]['markers'] if m['series'] == 'EXP']
def bands(fig, name='EXP band'):
    return [b for b in V[fig]['bands'] if b['series'] == name]

def harmonics_at(points):
    """split markers at the same x into v2 > v3 > v4 by value"""
    byx = {}
    for m in points: byx.setdefault(round(m['x'], 3), []).append(m)
    out = []
    for x, ms in sorted(byx.items()):
        ms = sorted(ms, key=lambda m: -m['y'])
        for n, m in zip((2, 3, 4), ms): out.append((n, m))
    return out

def band_at(blist, x):
    for b in blist:
        for bx, lo, hi in b['rows']:
            if abs(math.log10(bx) - math.log10(x)) < 0.01 if x > 0 else abs(bx - x) < 0.01:
                return lo, hi
    return None

# CMS vn{2} 2.76 TeV
pts = harmonics_at(exp_points('vn276'))
bl = sorted(bands('vn276'), key=lambda b: -b['mean'])  # v2, v3, v4 bands
rows = []
for n, m in pts:
    xb, lab = logbin(m['x'])
    lo, hi = band_at([bl[n - 2]], m['x'])
    scale = 0.8 if n == 3 else 1.0
    v = m['y'] / scale; blo = lo / scale; bhi = hi / scale
    rows.append([f'v{n}{{2}}', lab, xb, r(v, 6), r(blo, 6), r(bhi, 6), r((bhi - blo) / 2, 6), r(m['y'], 6), scale])
write('cms_2p76_vn2_ucc.csv', ['observable', 'centrality_bin_pct', 'x_plot', 'value', 'band_lo', 'band_hi', 'band_halfwidth', 'value_as_plotted', 'plot_scale'], rows,
      ['CMS, JHEP 02 (2014) 088 (arXiv:1312.1845) | Pb-Pb 2.76 TeV | 0.3<pT<3 GeV, |eta|<2.4, |Delta eta|>2 | vn{2}, n=2,3,4',
       'EXACT values read from the vector PDF of the group figure (arXiv:2203.17011 style). v3 was drawn x0.8: value/band here are UNSCALED.',
       'band = grey uncertainty band drawn with the data (check paper for whether it is syst or stat+syst).'])
# CMS ratio
rows = []
bl = bands('ratio276')
for m in sorted(exp_points('ratio276'), key=lambda m: m['x']):
    xb, lab = logbin(m['x']); lo, hi = band_at(bl, m['x'])
    rows.append(['v3{2}/v2{2}', lab, xb, r(m['y'], 6), r(lo, 6), r(hi, 6), r((hi - lo) / 2, 6)])
write('cms_2p76_v32_over_v22_ucc.csv', ['observable', 'centrality_bin_pct', 'x_plot', 'value', 'band_lo', 'band_hi', 'band_halfwidth'], rows,
      ['CMS 2.76 TeV v3{2}/v2{2} as drawn in the group figure (ratio of the CMS vn{2}); EXACT from vector PDF.',
       'Consistency check: equals v3/v2 from cms_2p76_vn2_ucc.csv to <1e-4.'])
# ALICE 5.02 vn{2} 1% bins
pts = harmonics_at(exp_points('vn502'))
bl = sorted(bands('vn502'), key=lambda b: -b['mean'])
rows = []
for n, m in pts:
    lo, hi = band_at([bl[n - 2]], m['x'])
    lab = f"{int(m['x'] - 0.5 + 0.01)}-{int(m['x'] - 0.5 + 0.01) + 1}"
    rows.append([f'v{n}{{2}}', lab, round(m['x'], 3), r(m['y'], 6), r(lo, 6), r(hi, 6), r((hi - lo) / 2, 6)])
write('alice_5p02_vn2_1pct.csv', ['observable', 'centrality_bin_pct', 'x_plot', 'value', 'band_lo', 'band_hi', 'band_halfwidth'], rows,
      ['ALICE, JHEP 07 (2018) 103 (arXiv:1804.02944) | Pb-Pb 5.02 TeV | 0.2<pT<3 GeV, |eta|<0.8, |Delta eta|>1 | vn{2} in 1% bins (0-5%)',
       'EXACT values read from the vector PDF of the group figure. Prefer HEPData for the official stat/syst split.'])

# ---- models from the vector figures (bonus)
rows = []
for fig, obs in [('ratio276', 'v3{2}/v2{2} 2.76 TeV'), ('vn276', 'vn{2} 2.76 TeV'), ('vn502', 'vn{2} 5.02 TeV')]:
    ms = [m for m in V[fig]['markers'] if m['series'] != 'EXP']
    groups = {}
    for m in ms: groups.setdefault(m['series'], []).append(m)
    for s, g in groups.items():
        if fig == 'ratio276':
            hh = [(None, m) for m in g]
        else:
            hh = harmonics_at(g)
        for n, m in hh:
            o = obs if n is None else obs.replace('vn', f'v{n}')
            scale = 0.8 if (fig == 'vn276' and n == 3) else 1.0
            lab = logbin(m['x'])[1] if fig != 'vn502' else f"{int(round(m['x'] - 0.5))}-{int(round(m['x'] - 0.5)) + 1}"
            err = sym(m['err_up'], m['err_dn'])
            rows.append([s, o, lab, round(m['x'], 4), r(m['y'] / scale, 6), r(err / scale if err else None, 6), scale])
    for b in V[fig]['bands']:
        if b['series'].startswith('Trajectum 2'):
            pass
write('models_from_group_figures.csv', ['model', 'observable', 'centrality_bin_pct', 'x_plot', 'value', 'error', 'plot_scale_removed'], rows,
      ['Model points (Trajectum 1 MAP, Trajectum 2, JETSCAPE Grad, Duke) from the group vector figures; v3 at 2.76 TeV unscaled (/0.8).',
       'error = model stat bar (blank if hidden by the marker). Trajectum 2 posterior bands are in the vector_raw.json file.'])
print(sorted(os.listdir(OUT)))
