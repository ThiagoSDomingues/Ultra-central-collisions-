"""Recreate the figures from the CSV files in this folder.

usage:  python plot_all.py            -> writes replots.pdf and replots.png next to this script
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
def rd(name):
    return pd.read_csv(os.path.join(HERE, name), comment='#')

RED, GREY, BLUE = '#cc0000', '#666666', '#1f3a93'

def stat_err(df):
    """stat bar; where hidden by the marker use 0 (the limit is in stat_upper_limit)"""
    return df['stat'].fillna(0).to_numpy()

def boxes(ax, x, y, s, w, color, logx=False):
    for xi, yi, si in zip(x, y, s):
        if np.isnan(si): continue
        x0, x1 = (xi / 10 ** w, xi * 10 ** w) if logx else (xi - w, xi + w)
        ax.add_patch(Rectangle((x0, yi - si), x1 - x0, 2 * si, color=color, alpha=0.35, lw=0))

fig, axs = plt.subplots(2, 3, figsize=(16, 10))

# (a) ALICE prelim: vn/vn(0-5%) vs Nch/Nch(0-5%)
ax = axs[0, 0]
d = rd('alice_prel_5p36_vn_norm_vs_Nch.csv'); c = rd('alice_prel_5p36_vn_norm_vs_Nch_trajectum_curve.csv')
for s, col, mfc in [('v2{2}/v2{2}(0-5%)', RED, RED), ('v3{2}/v3{2}(0-5%)', GREY, 'white')]:
    q = d[d.series == s]
    boxes(ax, q['Nch_over_Nch0-5'], q['value'], q['syst'], 0.006, col)
    ax.errorbar(q['Nch_over_Nch0-5'], q['value'], yerr=stat_err(q), fmt='o', color=col, mfc=mfc, ms=8, label=s.split('/')[0] + ' ALICE prel.')
for s, col in [('Trajectum v2{2}', RED), ('Trajectum v3{2}', GREY)]:
    q = c[c.series == s].sort_values('Nch_over_Nch0-5')
    ax.plot(q['Nch_over_Nch0-5'], q['value'], ':', color=col, lw=2, alpha=0.6, label=s)
ax.set(xlim=(0.8, 1.18), ylim=(0.6, 1.5), xlabel=r'$N_{ch}/N_{ch}^{0-5\%}$', ylabel=r'$v_n/v_n^{0-5\%}$',
       title='ALICE prel. Pb-Pb 5.36 TeV (ALI-PREL-598097)')
ax.legend(fontsize=8)

# (b) v3{2}/v2{2} in UCC bins: ALICE prelim 5.36 + CMS 2.76
ax = axs[0, 1]
d = rd('alice_prel_5p36_v32_over_v22_ucc.csv'); cm = rd('cms_2p76_v32_over_v22_ucc.csv')
q = d[d.series == 'ALICE']
ax.fill_between(q['x_plot'], q['value'] - q['syst'], q['value'] + q['syst'], color=RED, alpha=0.2, lw=0)
ax.plot(q['x_plot'], q['value'], 'o', color=RED, ms=8, label='ALICE prel. 5.36 TeV')
q = d[d.series == 'Trajectum']
ax.errorbar(q['x_plot'], q['value'], yerr=stat_err(q), fmt='o-', color=GREY, ms=6, label='Trajectum (ALICE fig.)')
ax.fill_between(cm['x_plot'], cm['band_lo'], cm['band_hi'], color='k', alpha=0.12, lw=0)
ax.plot(cm['x_plot'], cm['value'], 's', color='k', ms=7, label='CMS 2.76 TeV')
ax.set(xscale='log', xlim=(1e-3, 6), ylim=(0.55, 1.2), xlabel='Centrality (%)', ylabel=r'$v_3\{2\}/v_2\{2\}$',
       title='Ultra-central $v_3/v_2$')
ax.legend(fontsize=8)

# (c) v3{4}/v3{2}
ax = axs[0, 2]
d = rd('alice_prel_5p36_v34_over_v32.csv')
boxes(ax, d['x_plot'], d['value'], d['syst'], 0.1, RED)
ax.errorbar(d['x_plot'], d['value'], yerr=stat_err(d), fmt='o', color=RED, ms=8, capsize=2, label='ALICE prel.')
ax.axhline(np.sqrt(4 / np.pi - 1), ls='--', color='k', label='Bessel-Gaussian limit')
ax.set(xlim=(0, 5), ylim=(0.15, 1.2), xlabel='Centrality FT0C (%)', ylabel=r'$v_3\{4\}/v_3\{2\}$',
       title='ALICE prel. 5.36 TeV (ALI-PREL-621026)')
ax.legend(fontsize=8)

# (d) v3{2}/v2{2} 0-5%: Run 3 vs Run 2
ax = axs[1, 0]
d = rd('alice_prel_5p36_v32_over_v22_0to5.csv'); c = rd('alice_prel_5p36_v32_over_v22_0to5_trajectum_curve.csv')
q = d[d.series.str.startswith('ALICE Run 3')]
boxes(ax, q['x_plot'], q['value'], q['syst'], 0.06, RED)
ax.errorbar(q['x_plot'], q['value'], yerr=stat_err(q), fmt='o', color=RED, ms=8, label='Run 3 5.36 TeV (prel.)')
q = d[d.series.str.startswith('ALICE Run 2')]
ax.errorbar(q['x_plot'], q['value'], yerr=stat_err(q), fmt='o', color=GREY, mfc='white', ms=8, label='Run 2 5.02 TeV (digitized)')
a5 = rd('alice_5p02_vn2_1pct.csv').pivot(index='x_plot', columns='observable', values='value')
ax.plot(a5.index + 0.1, a5['v3{2}'] / a5['v2{2}'], 'x', color='k', ms=7, label='Run 2 exact (vector fig.)')
ax.plot(c['centrality_pct'], c['value'], ':', color=GREY, lw=2, alpha=0.6, label='Trajectum')
ax.set(xlim=(0, 5), ylim=(0.55, 0.95), xlabel='Centrality FT0C (%)', ylabel=r'$v_3\{2\}/v_2\{2\}$',
       title='ALICE prel. (ALI-PREL-598067)')
ax.legend(fontsize=8)

# (e) CMS vn{2} 2.76 TeV
ax = axs[1, 1]
d = rd('cms_2p76_vn2_ucc.csv')
for (o, col) in zip(['v2{2}', 'v3{2}', 'v4{2}'], [RED, BLUE, 'k']):
    q = d[d.observable == o]
    ax.fill_between(q['x_plot'], q['band_lo'], q['band_hi'], color=col, alpha=0.15, lw=0)
    ax.plot(q['x_plot'], q['value'], 'o', color=col, label=f'CMS {o} (unscaled)')
ax.set(xscale='log', xlim=(1e-3, 6), ylim=(0, 0.04), xlabel='Centrality (%)', ylabel=r'$v_n\{2\}$',
       title='CMS Pb-Pb 2.76 TeV, JHEP 02 (2014) 088')
ax.legend(fontsize=8)

# (f) ALICE vn{2} 5.02 TeV 1% bins
ax = axs[1, 2]
d = rd('alice_5p02_vn2_1pct.csv')
for (o, col) in zip(['v2{2}', 'v3{2}', 'v4{2}'], [RED, BLUE, 'k']):
    q = d[d.observable == o]
    ax.fill_between(q['x_plot'], q['band_lo'], q['band_hi'], color=col, alpha=0.15, lw=0)
    ax.plot(q['x_plot'], q['value'], 'o', color=col, label=f'ALICE {o}')
ax.set(xlim=(0, 5), ylim=(0, 0.05), xlabel='Centrality (%)', ylabel=r'$v_n\{2\}$',
       title='ALICE Pb-Pb 5.02 TeV, JHEP 07 (2018) 103')
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(HERE, 'replots.pdf')); fig.savefig(os.path.join(HERE, 'replots.png'), dpi=110)
print('wrote replots.pdf / replots.png')
