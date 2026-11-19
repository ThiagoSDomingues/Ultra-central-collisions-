"""Digitize the four ALICE Preliminary figures (raster images embedded at 720 ppi).

Method: the embedded image is extracted at full resolution, flipped to display
orientation, axes are calibrated with a linear fit to the major tick marks, and
markers / stat bars / syst boxes are found by exact ROOT colours.
"""
import sys, json, numpy as np
from PIL import Image
from scipy import ndimage as ndi
Image.MAX_IMAGE_PIXELS = None
W = sys.argv[1] if len(sys.argv) > 1 else '.'

def load(fn):
    return np.flipud(np.asarray(Image.open(f'{W}/{fn}').convert('RGB'))).astype(np.int16)

def near(a, c, tol=6):
    return (np.abs(a - np.array(c)).max(axis=2) <= tol)

def cal(points, log=False):
    p = np.array([q[0] for q in points], float); v = np.array([q[1] for q in points], float)
    if log: v = np.log10(v)
    k, b = np.polyfit(p, v, 1)
    res = (k * p + b) - v
    f = (lambda px: 10 ** (k * px + b)) if log else (lambda px: k * px + b)
    return f, k, float(np.abs(res).max())

def markers(mask, D, exclude, ring=False, opening=0):
    m = mask
    if opening:
        st = ndi.generate_binary_structure(2, 1)
        m = ndi.binary_opening(mask, structure=st, iterations=opening)
    lab, n = ndi.label(m)
    out = []
    for i, s in enumerate(ndi.find_objects(lab)):
        x0, x1, y0, y1 = s[1].start, s[1].stop, s[0].start, s[0].stop
        if (x1 - x0) < 0.85 * D or (x1 - x0) > 1.15 * D: continue
        if any(x0 < ex[1] and x1 > ex[0] and y0 < ex[3] and y1 > ex[2] for ex in exclude): continue
        sub = (lab[s] == i + 1)
        ext = []
        for r in range(sub.shape[0]):
            xs = np.where(sub[r])[0]
            ext.append(xs[-1] - xs[0] + 1 if len(xs) else 0)
        ext = np.array(ext)
        rows = np.where(ext >= 0.92 * (x1 - x0))[0]
        yc = y0 + rows.mean()
        xc = (x0 + x1 - 1) / 2
        out.append(dict(xc=xc, yc=yc, D=x1 - x0))
    return sorted(out, key=lambda d: d['xc'])

def run_through(pred, x, y_start, step, maxgap=0, limit=3000):
    """walk from y_start in direction step while pred holds (gaps up to maxgap). return last y that satisfied."""
    last = None; gap = 0; y = y_start
    for _ in range(limit):
        if y < 0 or y >= pred.shape[0]: break
        if pred[y, x]:
            last = y; gap = 0
        else:
            gap += 1
            if gap > maxgap: break
        y += step
    return last

def errors(mk, mpred, bpred, bgap=0, cols=(0,)):
    xc = int(round(mk['xc'])); yc = mk['yc']; R = mk['D'] / 2
    # stat: contiguous marker-colour run at centre column containing the marker edge
    top_edge = int(round(yc - R + 3)); bot_edge = int(round(yc + R - 3))
    st_top = run_through(mpred, xc, top_edge, -1)
    st_bot = run_through(mpred, xc, bot_edge, +1)
    st_up = yc - st_top if st_top is not None else np.nan
    st_dn = st_bot - yc if st_bot is not None else np.nan
    hid_up = st_up <= R + 1.5; hid_dn = st_dn <= R + 1.5
    # syst: continue outward from end of stat run with box predicate (allow gaps)
    sy_up = sy_dn = np.nan
    if bpred is not None:
        vals_u, vals_d = [], []
        for dx in cols:
            x = xc + dx
            yt = (st_top if st_top is not None else top_edge) - 1
            # skip small white gap
            t = run_through(bpred, x, yt, -1, maxgap=bgap)
            b = run_through(bpred, x, (st_bot if st_bot is not None else bot_edge) + 1, +1, maxgap=bgap)
            if t is not None: vals_u.append(yc - t)
            if b is not None: vals_d.append(b - yc)
        if vals_u: sy_up = float(np.max(vals_u))
        if vals_d: sy_dn = float(np.max(vals_d))
    return dict(st_up=st_up, st_dn=st_dn, hid_up=bool(hid_up), hid_dn=bool(hid_dn), sy_up=sy_up, sy_dn=sy_dn, R=R)

def curve(mask, xcal, ycal, exclude_rects, xstep=40):
    ys, xs = np.where(mask)
    keep = np.ones(len(xs), bool)
    for ex in exclude_rects:
        keep &= ~((xs >= ex[0]) & (xs < ex[1]) & (ys >= ex[2]) & (ys < ex[3]))
    xs, ys = xs[keep], ys[keep]
    out = []
    for b0 in range(xs.min(), xs.max() + 1, xstep):
        sel = (xs >= b0) & (xs < b0 + xstep)
        if sel.sum() < 20: continue
        out.append((float(xcal(xs[sel].mean())), float(ycal(ys[sel].mean()))))
    return out

RES = {}

def finish(name, mk, er, xcal, ycal, kx, ky):
    rows = []
    for m, e in zip(mk, er):
        y = ycal(m['yc'])
        dyp = abs(ky)
        row = dict(x=float(xcal(m['xc'])), y=float(y),
                   stat_up=None if e['hid_up'] else float(e['st_up'] * dyp),
                   stat_dn=None if e['hid_dn'] else float(e['st_dn'] * dyp),
                   stat_hidden_below=float(e['R'] * dyp) if (e['hid_up'] or e['hid_dn']) else None,
                   syst_up=None if np.isnan(e['sy_up']) else float(e['sy_up'] * dyp),
                   syst_dn=None if np.isnan(e['sy_dn']) else float(e['sy_dn'] * dyp))
        rows.append(row)
    RES[name] = rows

RED = (204, 0, 0); GRAY = (102, 102, 102)

# ---------------- Fig A: vn/vn(0-5%) vs Nch/Nch(0-5%)  (ALI-PREL-598097)
a = load('full_0676a40c-normvntrajactumsys_new-000.png')
xcal, kx, ex = cal([(1321.5, .85), (1963, .9), (2604.5, .95), (3246, 1.0), (3887.5, 1.05), (4529, 1.10), (5170.5, 1.15)])
ycal, ky, ey = cal([(634, 1.4), (1681, 1.2), (2728, 1.0), (3775, 0.8)])
print('A calib resid', ex, ey, 'per px', kx, ky)
mred = near(a, RED); mgray = near(a, GRAY)
leg = [(3300, 3700, 1350, 2050)]
mk = markers(mred, 141, leg); er = [errors(m, mred, near(a, (245, 204, 204)), bgap=60, cols=(0, -15, 15)) for m in mk]
finish('A_v2', mk, er, xcal, ycal, kx, ky)
mk = markers(mgray, 160, leg, ring=True); er = [errors(m, mgray, near(a, (224, 224, 224)), bgap=60, cols=(0, -15, 15)) for m in mk]
finish('A_v3', mk, er, xcal, ycal, kx, ky)
cx = [(684, 1450, 3900, 4818)]
RES['A_traj_v2'] = curve(near(a, (240, 178, 178), 4), xcal, ycal, cx)
RES['A_traj_v3'] = curve(near(a, (209, 209, 209), 3), xcal, ycal, cx)

# ---------------- Fig B: v3{2}/v2{2} vs centrality, log x (ALI-PREL-621021)
a = load('full_0ebac9ac-ratiouccplot_v32v22-000.png')
xcal, kx, ex = cal([(2368, 1e-2), (3361.5, 1e-1), (4354.5, 1.0)], log=True)
ycal, ky, ey = cal([(642, 1.1), (1197, 1.0), (1752.5, .9), (2308, .8), (2863, .7), (3418, .6)])
print('B calib resid', ex, ey, 'per px', kx, ky)
mred = near(a, RED); mgray = near(a, GRAY)
leg = [(3000, 5046, 800, 1200)]
mk = markers(mred, 88, leg); er = [errors(m, mred, near(a, (237, 166, 166), 8), bgap=20, cols=(0, -20, 20)) for m in mk]
finish('B_alice', mk, er, xcal, ycal, kx, ky)
mk = markers(mgray, 88, leg, opening=12); er = [errors(m, mgray, None) for m in mk]
finish('B_traj', mk, er, xcal, ycal, kx, ky)

# ---------------- Fig C: v3{4}/v3{2} vs centrality (ALI-PREL-621026)
a = load('full_9d19d7df-ratio05plotv34v32-000.png')
xcal, kx, ex = cal([(2153, 1), (2844.5, 2), (3536, 3), (4227.5, 4)])
ycal, ky, ey = cal([(852, 1.0), (1491.5, .8), (2131, .6), (2770.5, .4), (3410, .2)])
print('C calib resid', ex, ey, 'per px', kx, ky)
mred = near(a, RED)
leg = [(3000, 4916, 900, 1150)]
mk = markers(mred, 88, leg); er = [errors(m, mred, near(a, (245, 204, 204), 8), bgap=60, cols=(0, -20, 20)) for m in mk]
finish('C_alice', mk, er, xcal, ycal, kx, ky)
# Bessel-Gaussian dashed line: black dashes inside frame around row 2387
blk = a.max(2) < 70
rows = [r for r in range(2300, 2480) if blk[r, 1600:4800].mean() > 0.3]
RES['C_bessel_gaussian'] = float(ycal(np.mean(rows)))

# ---------------- Fig D: v3{2}/v2{2} vs centrality, Run 3 vs Run 2 (ALI-PREL-598067)
a = load('full_ba5e5673-v32overv22_vs_centrality_new-000.png')
xcal, kx, ex = cal([(1655, 1), (2630, 2), (3605, 3), (4580.5, 4)])
ycal, ky, ey = cal([(769.5, .9), (1878, .8), (2986.5, .7), (4095, .6)])
print('D calib resid', ex, ey, 'per px', kx, ky)
mred = near(a, RED); mgray = near(a, GRAY)
leg = [(900, 1700, 3550, 4300)]
mk = markers(mred, 140, leg); er = [errors(m, mred, mred, bgap=32, cols=tuple(range(-45, 46, 5))) for m in mk]
finish('D_run3', mk, er, xcal, ycal, kx, ky)
mk = markers(mgray, 159, leg, ring=True); er = [errors(m, mgray, None) for m in mk]
finish('D_run2', mk, er, xcal, ycal, kx, ky)
RES['D_traj'] = curve(near(a, (204, 204, 204), 3), xcal, ycal, [(684, 1700, 3550, 4756)])

json.dump(RES, open(f'{W}/alice_raw.json', 'w'), indent=1)
for k, v in RES.items():
    print('==', k)
    if isinstance(v, list) and v and isinstance(v[0], dict):
        for r in v: print({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()})
    elif isinstance(v, list):
        print(len(v), 'curve points', [tuple(round(t, 4) for t in p) for p in v[::max(1, len(v)//8)]])
    else:
        print(v)
