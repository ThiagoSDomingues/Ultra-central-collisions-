"""Extract exact data from ROOT vector PDFs (rotated landscape pages).

Page coords: page-x  <-> plot y-axis (linear), page-y <-> plot x-axis.
Calibration from the major tick segments of the frame.
"""
import sys, json, math, collections
import numpy as np
import pymupdf as fitz


def segs(ds, color=(0.0, 0.0, 0.0)):
    out = []
    for g in ds:
        if g['type'] == 's' and g.get('color') == color:
            for it in g['items']:
                if it[0] == 'l':
                    out.append((it[1].x, it[1].y, it[2].x, it[2].y))
    return out


def calib(ds):
    S = segs(ds)
    # frame
    frame = [g['rect'] for g in ds if g['type'] == 's' and g.get('color') == (0.0, 0.0, 0.0) and len(g['items']) == 4][0]
    x0, y0, x1, y1 = frame.x0, frame.y0, frame.x1, frame.y1
    # major ticks on the edge x=x0 (value axis): horizontal segments starting at x0 of length ~10
    vmaj = sorted({round(s[1], 2) for s in S if abs(s[1] - s[3]) < 0.01 and min(s[0], s[2]) - x0 < 0.05 and 9 < abs(s[2] - s[0]) < 11})
    # major ticks on the centrality axis edge y=y1: vertical segments ending on y1 of length ~14
    hmaj = sorted({round(s[0], 2) for s in S if abs(s[0] - s[2]) < 0.01 and abs(max(s[1], s[3]) - y1) < 0.05 and 13 < abs(s[3] - s[1]) < 15})
    return dict(frame=(x0, y0, x1, y1), vmaj=vmaj, hmaj=hmaj)


def shapes(ds):
    """group drawings by (type, fill, color) with vertex lists"""
    out = []
    for g in ds:
        pts = []
        for it in g['items']:
            if it[0] == 'l':
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            elif it[0] == 'c':
                pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
            elif it[0] == 're':
                r = it[1]; pts += [(r.x0, r.y0), (r.x1, r.y1)]
        out.append(dict(type=g['type'], fill=g.get('fill'), color=g.get('color'), op=g.get('fill_opacity'),
                        items=''.join(i[0] for i in g['items']), rect=tuple(g['rect']), pts=pts, width=g.get('width')))
    return out


if __name__ == '__main__':
    d = fitz.open(sys.argv[1]); p = d[0]
    ds = p.get_drawings()
    c = calib(ds)
    print(json.dumps(c))
    cnt = collections.Counter((s['type'], s['fill'], s['color'], s['op'], s['items'][:6]) for s in shapes(ds))
    for k, v in cnt.most_common(): print(v, k)
