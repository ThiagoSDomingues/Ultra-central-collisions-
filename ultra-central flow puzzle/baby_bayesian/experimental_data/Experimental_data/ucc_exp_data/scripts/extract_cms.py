"""Exact extraction of data points from the three ROOT vector PDFs (group figures, arXiv:2203.17011 style)."""
import sys, json, math
import numpy as np
import pymupdf as fitz
sys.path.insert(0, '/home/claude/cms_figs')
from vec import calib, shapes

SRC = '/home/claude/cms_figs/src/'
COL = {
    (0.4, 0.4, 0.4): 'EXP',            # CMS (2.76) or ALICE (5.02) markers
    (0.0, 0.0, 0.6): 'Trajectum 1',
    (0.6, 0.6, 0.0): 'Trajectum 2',
    (0.0, 0.4, 0.2): 'JETSCAPE Grad',
    (0.8, 0.0, 0.0): 'Duke',
}

def rc(c):
    return tuple(round(v, 1) for v in c) if c else None

def lin(pairs):
    p = np.array([a for a, b in pairs]); v = np.array([b for a, b in pairs])
    k, b = np.polyfit(p, v, 1)
    return (lambda z: k * z + b), float(np.abs(k * p + b - v).max()), k

def process(fn, xlog, xpairs, ypairs, legend):
    d = fitz.open(SRC + fn); ds = d[0].get_drawings()
    if xlog:
        fx_, rx, kx = lin([(a, math.log10(b)) for a, b in xpairs]); fx = lambda z: 10 ** fx_(z)
    else:
        fx, rx, kx = lin(xpairs)
    fy, ry, ky = lin(ypairs)
    print(fn[:8], 'calib max residual (axis units):', rx, ry)
    S = shapes(ds)
    # page-x -> value (fy), page-y -> centrality (fx)
    def P(px, py): return fx(py), fy(px)
    def inleg(x, y): return legend[0] <= x <= legend[1] and legend[2] <= y <= legend[3]
    markers = []
    for s in S:
        if s['type'] == 'f' and (s['op'] in (None, 1.0)) and rc(s['fill']) in COL:
            r = s['rect']; w = r[2] - r[0]; h = r[3] - r[1]
            if max(w, h) > 12: continue
            cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            # triangles: ROOT places the marker centre at the centroid of the bounding box (marker size box)
            x, y = P(cx, cy)
            if inleg(x, y) or (not xlog and x < 0.4): continue
            markers.append(dict(series=COL[rc(s['fill'])], px=cx, py=cy, half=w / 2, x=x, y=y, items=s['items']))
    # error bar segments: single-line strokes in series colour, constant page-y
    segs = [(rc(s['color']), s['pts']) for s in S if s['type'] == 's' and s['items'] == 'l' and rc(s['color']) in COL]
    for m in markers:
        col = [k for k, v in COL.items() if v == m['series']][0]
        xs = []
        for c, pts in segs:
            if c != col: continue
            (ax, ay), (bx, by) = pts
            if abs(ay - m['py']) < 0.35 and abs(by - m['py']) < 0.35 and min(abs(ax - m['px']), abs(bx - m['px'])) <= m['half'] + 0.6:
                xs += [ax, bx]
        if xs:
            lo_px, hi_px = min(xs), max(xs)          # page-x grows as value decreases
            m['err_up'] = fy(lo_px) - m['y'] if fy(lo_px) > m['y'] else None
            m['err_dn'] = m['y'] - fy(hi_px) if fy(hi_px) < m['y'] else None
        else:
            m['err_up'] = m['err_dn'] = None
        m['marker_half_height'] = abs(ky) * m['half']
    # bands: translucent polygons
    bands = []
    for s in S:
        if s['type'] == 'f' and s['op'] is not None and s['op'] < 0.5:
            name = 'EXP band' if rc(s['fill']) == (0.2, 0.2, 0.2) else COL.get(rc(s['fill']), str(rc(s['fill']))) + ' band'
            pts = [P(px, py) for px, py in s['pts']]
            # unique vertices keyed by centrality
            byx = {}
            for x, y in pts:
                key = round(x, 6)
                byx.setdefault(key, set()).add(round(y, 7))
            rows = sorted((x, min(v), max(v)) for x, v in byx.items())
            bands.append(dict(series=name, rows=rows, mean=float(np.mean([r[1] + r[2] for r in rows]) / 2)))
    return markers, bands

out = {}
LOGX = [(481.96, 1e-3), (358.92, 1e-2), (235.87, 1e-1), (112.83, 1.0)]
# ratio v3{2}/v2{2} at 2.76 TeV
out['ratio276'] = process('96bfe8e3-ratio_v32_over_v22-int-PbPb2p76_w_EtaGap_paper_no_shadow_w_CMS_pt-reference.pdf', True, LOGX,
                          [(11.83 + i * 52.6483, 1.2 - 0.1 * i) for i in range(7)], legend=(1.2e-3, 1.1e-2, 0.57, 0.73))
# vn{2} at 2.76 TeV (v3 scaled x0.8)
out['vn276'] = process('cc06cc3c-vn2-int-PbPb2p76_with_eta_gap_paper_no_shadow_labels_back_w_CMS_pt-reference.pdf', True, LOGX,
                       [(11.83 + i * 42.77625, 0.04 - 0.005 * i) for i in range(9)], legend=(1.1e-3, 1.2e-2, 0.026, 0.04))
# vn{2} at 5.02 TeV
out['vn502'] = process('60ba6a83-vn2-int-PbPb5p02TeV_with_eta_gap_paper_no_shadow.pdf', False,
                       [(493.3 - i * 95.2441, float(i)) for i in range(6)],
                       [(11.83 + i * 34.22093, 0.05 - 0.005 * i) for i in range(11)], legend=(0.05, 1.0, 0.0375, 0.05))

ser = {}
for fig, (mk, bands) in out.items():
    ser[fig] = dict(markers=[{k: v for k, v in m.items() if k not in ('px', 'py', 'items', 'half')} for m in mk], bands=bands)
    print('=====', fig)
    for m in sorted(mk, key=lambda m: (m['series'], m['x'], -m['y'])):
        print(f"{m['series']:14s} x={m['x']:.5g} y={m['y']:.5f} up={m['err_up']} dn={m['err_dn']} mh={m['marker_half_height']:.5f}")
    for b in bands:
        print(b['series'], round(b['mean'], 5), [(round(x, 5), round(lo, 5), round(hi, 5)) for x, lo, hi in b['rows']])
json.dump(ser, open('/home/claude/cms_figs/work/vector_raw.json', 'w'), indent=1)
