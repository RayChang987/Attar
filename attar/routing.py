"""The 'routing' fact and global-routing congestion from it: OpenROAD's GCell grid and edge capacities, RUDY demand.

attach(ir_dir, lefs) distills what OpenROAD's GlobalRouter blocks capacity with into routing.safetensors:
  rm_master, rm_layer, rm_rect   every master's pin and OBS rectangles on the routing layers (dbu, master frame, N)
  sn_layer, sn_rect              the special nets' wires and vias on the routing layers (dbu, absolute; from the
                                 design's DEF frame, vias from its VIAS section and the LEF)
and records the routing layer range and the layer adjustment (manifest 'routing': ORFS's MIN / MAX_ROUTING_LAYER and
ROUTING_LAYER_ADJUSTMENT where the design comes from ORFS).

capacity(d) rebuilds OpenROAD's grid and capacities from the IR (GlobalRouter: initGrid, computeGCellCapacity,
computeObstructionsAdjustments, applyObstructionAdjustment + initBlockedIntervals, the global layer adjustment; the
rules gsta's FastRoute port reproduces): gcells of 15 x the median M2-M4 track pitch, each edge's capacity the tracks
of its layer inside the gcell strip, minus what instance pins / OBS (at the current placement) and special nets block,
times (1 - adjustment). overflow(d) puts RUDY demand on that grid, per direction, and reports the share of gcell edges
whose demand exceeds their capacity (GROV).
"""
import gzip
import json
import math
import os
import re

import numpy as np
import torch

from . import st

SCHEMA = 'attar-routing/0.1'
# OpenROAD orientation as (a, b, c, d): x' = a x + b y, y' = c x + d y (attar.deffile.ORIENTS order)
_ORIENT = {'N': (1, 0, 0, 1), 'S': (-1, 0, 0, -1), 'W': (0, -1, 1, 0), 'E': (0, 1, -1, 0), 'FN': (-1, 0, 0, 1),
           'FS': (1, 0, 0, -1), 'FE': (0, 1, 1, 0), 'FW': (0, -1, -1, 0)}


# -- distillation -----------------------------------------------------------------------------------------------------
def _def_tokens(text):
    return re.findall(r'[()+;]|[^\s()+;]+', text)


def _special_nets(frame_text, lid, tech_vias, dbu_lef):
    """Special-net wires and vias of a DEF as (level index, rect) on the routing layers (gsta collect_obstructions)."""
    out = []
    # DEF vias: fixed RECT vias and generated VIARULE vias (dbVia create_via_boxes)
    dvias = {}
    i, j = frame_text.find('\nVIAS'), frame_text.find('\nEND VIAS')
    if i >= 0 and j > i:
        tok = _def_tokens(frame_text[frame_text.index('\n', i + 1):j])
        k = 0
        while k < len(tok):
            if tok[k] != '-':
                k += 1; continue
            name = tok[k + 1]; k += 2
            boxes, gen = [], False
            p = {}
            while k < len(tok) and tok[k] != ';':
                w = tok[k]
                if w == 'RECT':
                    lay = tok[k + 1]
                    nums = [x for x in tok[k + 2:k + 10] if x not in '()']
                    if lay in lid:
                        a, b, c, d = (int(float(v)) for v in nums[:4])
                        boxes.append((lid[lay], (min(a, c), min(b, d), max(a, c), max(b, d))))
                    k += 2; continue
                if w == 'VIARULE':
                    gen = True
                elif w in ('CUTSIZE', 'CUTSPACING', 'ROWCOL', 'ORIGIN'):
                    p[w] = (int(tok[k + 1]), int(tok[k + 2]))
                elif w in ('ENCLOSURE', 'OFFSET'):
                    p[w] = tuple(int(tok[k + n]) for n in range(1, 5))
                elif w == 'LAYERS':
                    p[w] = (tok[k + 1], tok[k + 2], tok[k + 3])
                k += 1
            if gen and 'ROWCOL' in p and 'LAYERS' in p:
                cw, ch = p.get('CUTSIZE', (0, 0)); csx, csy = p.get('CUTSPACING', (0, 0))
                rows, cols = p['ROWCOL']
                ebx, eby, etx, ety = p.get('ENCLOSURE', (0, 0, 0, 0)); ox, oy = p.get('ORIGIN', (0, 0))
                bofx, bofy, tofx, tofy = p.get('OFFSET', (0, 0, 0, 0))
                maxx = maxy = y = 0
                for _ in range(rows):
                    x = 0
                    for _ in range(cols):
                        maxx, maxy = x + cw, y + ch
                        x = maxx + csx
                    y = maxy + csy
                dx, dy = maxx // 2, maxy // 2
                mnx, mny, mxx, mxy = -dx, -dy, maxx - dx, maxy - dy
                bot, _cut, top = p['LAYERS']
                if top in lid:
                    boxes.append((lid[top], (mnx - etx + ox + tofx, mny - ety + oy + tofy, mxx + etx + ox + tofx, mxy + ety + oy + tofy)))
                if bot in lid:
                    boxes.append((lid[bot], (mnx - ebx + ox + bofx, mny - eby + oy + bofy, mxx + ebx + ox + bofx, mxy + eby + oy + bofy)))
            dvias[name] = boxes
            k += 1
    vias = {n: [(lid[l], tuple(int(round(v * dbu_lef)) for v in r)) for l, r in b if l in lid] for n, b in tech_vias.items()}
    vias.update(dvias)
    i, j = frame_text.find('\nSPECIALNETS'), frame_text.find('\nEND SPECIALNETS')
    if i < 0 or j < i:
        return out
    tok = _def_tokens(frame_text[frame_text.index('\n', i + 1):j])
    k = 0
    while k < len(tok):
        if tok[k] not in ('ROUTED', 'NEW', 'FIXED', 'COVER', 'SHIELD') or k + 2 >= len(tok):
            k += 1; continue
        lay = tok[k + 1]
        try:
            width = int(tok[k + 2])
        except ValueError:
            k += 1; continue
        lv = lid.get(lay)
        k += 3
        prev = None
        while k < len(tok):
            s = tok[k]
            if s in ('NEW', ';', '-') or (s == '+' and k + 1 < len(tok) and tok[k + 1] not in ('SHAPE', 'MASK', 'STYLE')):
                break
            if s == '(':
                x = prev[0] if tok[k + 1] == '*' else int(tok[k + 1])
                y = prev[1] if tok[k + 2] == '*' else int(tok[k + 2])
                m = k + 3
                if tok[m] != ')':
                    m += 1                                   # extension: not part of OpenROAD's blockage
                if prev is not None and lv is not None and width > 0:
                    dw = width >> 1
                    if x == prev[0]:
                        out.append((lv, (x - dw, min(prev[1], y), x + dw, max(prev[1], y))))
                    elif y == prev[1]:
                        out.append((lv, (min(prev[0], x), y - dw, max(prev[0], x), y + dw)))
                prev = (x, y)
                k = m + 1
                continue
            if s not in ('+', 'SHAPE', 'MASK', 'STYLE', ')') and prev is not None and tok[k - 1] == ')':
                for lvv, r in vias.get(s, []):
                    out.append((lvv, (r[0] + prev[0], r[1] + prev[1], r[2] + prev[0], r[3] + prev[1])))
            k += 1
    return out


def _layer_rules(lef_paths, layers, tech, dbu):
    """Per IR routing layer (dbu): the largest of its plain SPACING and SPACINGTABLE values (dbTechLayer::getSpacing at
    the largest width and length; LEF58 property rules are not read there), and the width across its direction of its
    default via's box on it (dbBlock::getDefaultVias: the first tech via with the layer as its bottom layer)."""
    ext = {}
    for path in lef_paths:
        cur, table, prop = None, False, False
        for line in open(path):
            t = line.split('#')[0].split()
            if not t:
                continue
            if prop:                                   # a PROPERTY's quoted text: not a spacing rule of getSpacing
                prop = not line.rstrip().endswith(';') or line.count('"') % 2 == 1 and t[0] != '"'
                if '";' in line or line.strip().endswith('" ;'):
                    prop = False
                continue
            if t[0] == 'LAYER' and len(t) == 2:
                cur, table = t[1], False
            elif cur is None:
                continue
            elif t[0] == 'END' and t[1:2] == [cur]:
                cur = None
            elif t[0] == 'PROPERTY':
                prop = not line.rstrip().endswith(';')
            elif t[0] == 'SPACINGTABLE':
                table = True
            elif t[0] == 'SPACING' and len(t) > 1:
                try:
                    ext[cur] = max(ext.get(cur, 0.0), float(t[1]))
                except ValueError:
                    pass
            elif table and t[0] == 'WIDTH':
                vals = [float(v) for v in t[2:] if v != ';' and v.replace('.', '', 1).isdigit()]
                if vals:
                    ext[cur] = max(ext.get(cur, 0.0), max(vals))
                table = not line.rstrip().endswith(';')
    names = [l['name'] for l in tech['layers']]
    horiz = {l['name']: l.get('direction') == 'HORIZONTAL' for l in tech['layers']}
    via_w = {}
    for boxes in tech.get('vias', {}).values():
        ls = [l for l, _ in boxes if l in names]
        if not ls:
            continue
        bot = min(ls, key=names.index)
        if bot in via_w:
            continue
        r = next(r for l, r in boxes if l == bot)
        via_w[bot] = (r[2] - r[0]) if horiz[bot] else (r[3] - r[1])
    q = lambda v: int(round(v * dbu))  # noqa: E731
    return [q(ext.get(n, 0.0)) for n in layers], [q(via_w.get(n, 0.0)) for n in layers]


def attach(ir_dir, lef_paths, layers=None, adjustment=None):
    """Distill the 'routing' fact into the IR at ir_dir from its LEF files (see the module doc). layers (lowest,
    highest routing layer name) and adjustment default to the manifest's 'routing' entry (ORFS designs have one)."""
    from .lef import read_lef
    with gzip.open(os.path.join(ir_dir, 'names.json.gz'), 'rt') as f:
        names = json.load(f)
    mf = os.path.join(ir_dir, 'manifest.json')
    m = json.load(open(mf))
    dbu = m['dbu']
    macros, tech = read_lef(list(lef_paths))
    lid = {n: i for i, n in enumerate(names['layers'])}
    q = lambda v: int(round(v * dbu))  # noqa: E731
    rm_master, rm_layer, rm_rect, rm_obs = [], [], [], []
    for mi, mn in enumerate(names['masters']):
        mac = macros.get(mn)
        if mac is None:
            continue
        for lay, r, kind in mac.shapes:
            if lay in lid:
                rm_master.append(mi); rm_layer.append(lid[lay]); rm_rect.append([q(v) for v in r])
                rm_obs.append(kind == 'OBS')
    # per routing layer: the macro obstruction extension (findLayerExtensions: the largest spacing rule) and the
    # width of its default via on it (findTransitionLayers: fat vias make a transition layer)
    ext, via_w = _layer_rules(lef_paths, names['layers'], tech, dbu)
    frame = os.path.join(ir_dir, 'frame.def.gz')
    sn = []
    if os.path.exists(frame):
        with gzip.open(frame, 'rt') as f:
            sn = _special_nets(f.read(), lid, tech.get('vias', {}), dbu)
    # special nets: identical rectangles block nothing more (edges zeroed once, intervals merged), keep one
    sn_a = np.asarray([[s[0], *s[1]] for s in sn], np.int64).reshape(-1, 5)
    if len(sn_a):
        sn_a = sn_a[np.sort(np.unique(sn_a, axis=0, return_index=True)[1])]
    tens = {'rm_master': np.asarray(rm_master, np.int32), 'rm_layer': np.asarray(rm_layer, np.int16),
            'rm_rect': np.asarray(rm_rect, np.int32).reshape(-1, 4), 'rm_obs': np.asarray(rm_obs, np.uint8),
            'layer_ext': np.asarray(ext, np.int32), 'layer_via_w': np.asarray(via_w, np.int32),
            'sn_layer': sn_a[:, 0].astype(np.int16), 'sn_rect': sn_a[:, 1:].astype(np.int32)}
    st.save(os.path.join(ir_dir, 'routing.safetensors'), tens, {'schema': SCHEMA})
    r = dict(m.get('routing') or {})
    if layers:
        r['layers'] = list(layers)
    if adjustment is not None:
        r['adjustment'] = float(adjustment)
    r.setdefault('adjustment', 0.0)
    r.setdefault('source', 'default')
    m['routing'] = r
    m.setdefault('facts', {})['routing'] = 'lef+def'
    json.dump(m, open(mf, 'w'), indent=1)
    return {'master_shapes': len(rm_master), 'special_net_rects': len(sn_a), 'routing': r}


# -- OpenROAD's grid and capacities -------------------------------------------------------------------------------------
def _layer_tracks(a, layer, horizontal):
    """(init, pitch, count) of a layer's preferred-direction tracks, as dbTrackGrid averages several patterns."""
    want = 1 if horizontal else 0                  # tr_axis: 0 'X' (vertical lines), 1 'Y'
    sel = np.nonzero((a['tr_layer'] == layer) & (a['tr_axis'] == want))[0]
    if not len(sel):
        return 0, 0, 0
    if len(sel) == 1:
        k = sel[0]
        return int(a['tr_start'][k]), int(a['tr_step'][k]), int(a['tr_num'][k])
    c = np.unique(np.concatenate([int(a['tr_start'][k]) + np.arange(int(a['tr_num'][k])) * int(a['tr_step'][k])
                                  for k in sel]))
    return int(c[0]), int(math.ceil(float(c[-1] - c[0]) / float(len(c)))), len(c)


def grid(d):
    """OpenROAD's GCell grid: tile = 15 x the median of the (even-rounded) M2..M4 track pitches, floor(die / tile) gcells."""
    a = d.raw
    tp, _ = st.load(os.path.join(d.path, 'tech_profile.safetensors'))
    horiz = tp['layer_dir'] == 0
    x0, y0, x1, y1 = (int(v) for v in a['die_box'][0])
    sp = []
    for lv in (1, 2, 3):                                     # M2, M3, M4 (layer indices)
        p = _layer_tracks(a, lv, bool(horiz[lv]))[1]
        sp.append(p if p % 2 == 0 else p - 1)
    tile = sorted(sp)[1] * 15
    return {'tile': tile, 'nx': max(1, (x1 - x0) // tile), 'ny': max(1, (y1 - y0) // tile),
            'die': (x0, y0, x1, y1), 'horizontal': horiz}


def capacity(d, adjustment=None, layers=None):
    """Per layer, the capacity of every gcell edge as OpenROAD computes it: (g, h3 [L][ny][nx-1], v3 [L][ny-1][nx])
    (layer L = names['layers'][L]; outside the routing range 0). adjustment / layers override the manifest's."""
    a, names = d.raw, d.names
    r = d.manifest.get('routing') or {}
    lay = layers or r.get('layers') or [names['layers'][1], names['layers'][-1]]
    adj = float(r.get('adjustment', 0.0) if adjustment is None else adjustment)
    lid = {n: i for i, n in enumerate(names['layers'])}
    lo, hi = lid[lay[0]], lid[lay[1]]
    g = grid(d)
    tile, nx, ny = g['tile'], g['nx'], g['ny']
    x0, y0, x1, y1 = g['die']
    horiz = g['horizontal']
    L = len(names['layers'])
    # gcell rectangles (GlobalRouter::getGCellRect): the last row / column reaches the die edge
    def bounds(n, o, e):
        c = np.arange(n) * tile + tile // 2 + o
        mn, mx = c - tile // 2, c + tile // 2
        mx = np.where((e - mx) // tile < 1, e, mx)
        return mn, mx
    gx0, gx1 = bounds(nx, x0, x1)
    gy0, gy1 = bounds(ny, y0, y1)
    h3 = np.zeros((L, ny, max(nx - 1, 0)), np.int64); v3 = np.zeros((L, max(ny - 1, 0), nx), np.int64)
    pitch = np.ones(L, np.int64)
    for l in range(L):
        init, p, cnt = _layer_tracks(a, l, bool(horiz[l]))
        pitch[l] = max(p, 1)
        if not (lo <= l <= hi) or cnt == 0 or p == 0:
            continue
        mn, mx = (gy0, gy1) if horiz[l] else (gx0, gx1)
        first = np.where(mn <= init, 0, (mn - init + p - 1) // p)
        last = np.where(mx < init, -1, (mx - init - 1) // p)
        first = np.maximum(first, 0); last = np.minimum(cnt - 1, last)
        cap = np.where(first > last, 0, last - first + 1)
        if horiz[l]:
            h3[l] = cap[:, None]
        else:
            v3[l] = cap[None, :]
    # obstructions (computeObstructionsAdjustments): instances' pin / OBS shapes at the current placement (macro OBS
    # extended), special nets; edges they cover go to 0 now, their partially blocked intervals are kept; then the
    # transition layers are halved over macros; then the intervals cost their tracks (initBlockedIntervals)
    t, _ = st.load(os.path.join(d.path, 'routing.safetensors'))
    rects, macro_obs = _obstructions(d, t, lo, hi, horiz)
    hint, vint = _apply_obstructions(rects, h3, v3, g, horiz, pitch)
    _transition_layers(t, macro_obs, h3, v3, g, horiz, pitch, hi)
    _blocked_intervals(hint, vint, h3, v3, pitch)
    if adj != 0.0:
        for l in range(lo, hi + 1):
            for c in (h3[l], v3[l]):
                nc = np.floor(c.astype(np.float32) * np.float32(1 - adj)).astype(np.int64)
                if adj != 1:
                    nc = np.where(c > 0, np.maximum(nc, 1), nc)
                c[...] = nc
    g.update({'layers': (lo, hi), 'adjustment': adj, 'pitch': pitch})
    return g, h3, v3


def _xform(A, B, C, D, r):
    xs = (A * r[0] + B * r[1], A * r[2] + B * r[3]); ys = (C * r[0] + D * r[1], C * r[2] + D * r[3])
    return [min(xs), min(ys), max(xs), max(ys)]


def _intersect(rs1, rs2):
    out = []
    for a in rs1:
        for b in rs2:
            r = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
            if r[0] < r[2] and r[1] < r[3]:
                out.append(r)
    return out


def _obstructions(d, t, lo, hi, horiz):
    """GlobalRouter::findInstancesObstructions and findNetsObstructions at the current placement:
    ([(layer, x0, y0, x1, y1, is_macro)], {layer: [macro OBS rects]}) in dbu. A macro's (CLASS BLOCK) OBS on a layer
    also blocks a layer between two obstructed ones where both overlap, and the routing range's end layer next to an
    obstructed one (extendObstructions); macro OBS grows across the layer's direction by the layer's spacing extension.
    Every other shape (pins, standard-cell OBS, special nets) blocks as it is."""
    from .deffile import ORIENTS
    from .distill import M_BLOCK
    out, macro_obs = [], {}
    keep = (t['rm_layer'] >= lo) & (t['rm_layer'] <= hi)
    rm_m, rm_l, rm_r = t['rm_master'][keep], t['rm_layer'][keep], t['rm_rect'][keep]
    rm_o = t['rm_obs'][keep] if 'rm_obs' in t else np.zeros(len(rm_m), np.uint8)
    ext = t['layer_ext'] if 'layer_ext' in t else np.zeros(len(horiz), np.int64)
    if len(rm_m):
        a = d.raw
        cm = a['c_master'].astype(np.int64)
        has = np.zeros(len(a['m_size']), bool); has[rm_m] = True
        cells = np.nonzero((cm >= 0) & has[np.maximum(cm, 0)])[0]
        if len(cells):
            pos = np.rint(d.pos[torch.as_tensor(cells, device=d.pos.device)].double().cpu().numpy() * d.dbu).astype(np.int64)
            ori = d.orient.cpu().numpy()[cells]
            msz = a['m_size'].astype(np.int64); mcls = a['m_class']
            o = np.argsort(rm_m, kind='stable'); rs = np.searchsorted(rm_m[o], np.arange(len(msz) + 1))
            for ci, c in enumerate(cells):
                mi = cm[c]
                A, B, C, D = _ORIENT[ORIENTS[int(ori[ci])]]
                box = _xform(A, B, C, D, (0, 0, msz[mi][0], msz[mi][1]))
                ox, oy = pos[ci, 0] - box[0], pos[ci, 1] - box[1]
                shapes = [(int(rm_l[k]), bool(rm_o[k]), _xform(A, B, C, D, [int(v) for v in rm_r[k]]))
                          for k in o[rs[mi]:rs[mi + 1]]]
                shapes = [(l, ob, [r[0] + ox, r[1] + oy, r[2] + ox, r[3] + oy]) for l, ob, r in shapes]
                if mcls[mi] == M_BLOCK:
                    per = {}
                    for l, ob, r in shapes:
                        if ob:
                            per.setdefault(l, []).append(r)
                    if per:
                        bottom, top = min(per), max(per)
                        if bottom - 1 == lo:
                            bottom -= 1
                        if top + 1 == hi:
                            top += 1
                        for l in range(bottom, top + 1):         # extendObstructions / layerIsBlocked
                            extra = []
                            if l == hi and (l - 1) in per:
                                extra = list(per[l - 1])
                            if l == lo and (l + 1) in per:
                                extra = list(per[l + 1])
                            extra += _intersect(per.get(l + 1, []), per.get(l - 1, []))
                            if extra:
                                per.setdefault(l, []).extend(extra)
                        for l, rs_ in per.items():
                            e = int(ext[l])
                            for r in rs_:
                                r = [r[0], r[1] - e, r[2], r[3] + e] if horiz[l] else [r[0] - e, r[1], r[2] + e, r[3]]
                                macro_obs.setdefault(l, []).append(r)
                                out.append((l, *r, True))
                    shapes = [(l, ob, r) for l, ob, r in shapes if not ob]
                for l, ob, r in shapes:
                    out.append((l, *r, False))
    keep = (t['sn_layer'] >= lo) & (t['sn_layer'] <= hi)
    for l, r in zip(t['sn_layer'][keep], t['sn_rect'][keep]):
        out.append((int(l), *(int(v) for v in r), False))
    return out, macro_obs


def _tiles(rx0, ry0, rx1, ry1, g):
    """Grid::getBlockedTiles: the first and last gcell (centers) of a rectangle, and their gcell boxes."""
    tile, nx, ny = g['tile'], g['nx'], g['ny']
    X0, Y0, X1, Y1 = g['die']

    def on_grid(x, y):
        gx, gy = (x - X0) // tile, (y - Y0) // tile
        gx -= gx >= nx; gy -= gy >= ny
        return gx * tile + tile // 2 + X0, gy * tile + tile // 2 + Y0
    lx, ly = on_grid(rx0, ry0); ux, uy = on_grid(rx1, ry1)
    ft = (lx - tile // 2, ly - tile // 2, lx + tile // 2, ly + tile // 2)
    lt = [ux - tile // 2, uy - tile // 2, ux + tile // 2, uy + tile // 2]
    if (X1 - lt[2]) // tile < 1:
        lt[2] = X1
    if (Y1 - lt[3]) // tile < 1:
        lt[3] = Y1
    return (lx - X0) // tile, (ly - Y0) // tile, (ux - X0) // tile, (uy - Y0) // tile, ft, lt


def _apply_obstructions(rects, h3, v3, g, horiz, pitch):
    """GlobalRouter::applyObstructionAdjustment for every rectangle: edges it covers on its inner gcell rows (columns)
    go to 0; on its first and last row (column) the blocked interval is kept per edge (for initBlockedIntervals). For a
    macro's obstruction an interval that would leave one track grows by a track (Grid::computeTileReduceInterval)."""
    nx, ny = g['nx'], g['ny']
    X0, Y0, X1, Y1 = g['die']
    hint, vint = {}, {}
    for l, rx0, ry0, rx1, ry1, macro in rects:
        rx0, ry0, rx1, ry1 = max(rx0, X0), max(ry0, Y0), min(rx1, X1), min(ry1, Y1)
        if rx0 > rx1 or ry0 > ry1:
            continue
        fx, fy, lastx, lasty, ft, lt = _tiles(rx0, ry0, rx1, ry1, g)
        vertical = not horiz[l]

        def interval(tl, first, cap):
            if vertical:
                s, e = (rx0, rx1) if rx0 >= tl[0] and rx1 <= tl[2] else ((rx0, tl[2]) if first else (tl[0], rx1))
            else:
                s, e = (ry0, ry1) if ry0 >= tl[1] and ry1 <= tl[3] else ((ry0, tl[3]) if first else (tl[1], ry1))
            if macro and cap - math.ceil(abs(e - s) / pitch[l]) == 1:
                e += pitch[l]
            return s, e
        if not vertical:
            cap_f = h3[l, fy, min(fx, nx - 2)] if nx > 1 else 0
            cap_l = h3[l, lasty, min(lastx, nx - 2)] if nx > 1 else 0
            fi, li = interval(ft, True, cap_f), interval(lt, False, cap_l)
            if fx == lastx and lastx + 1 < nx:
                lastx += 1
            for x in range(fx, lastx):
                for y in range(fy, lasty + 1):
                    if y == fy:
                        hint.setdefault((x, y, l), []).append(fi)
                    elif y == lasty:
                        hint.setdefault((x, y, l), []).append(li)
                    else:
                        h3[l, y, x] = 0
        else:
            cap_f = v3[l, min(fy, ny - 2), fx] if ny > 1 else 0
            cap_l = v3[l, min(lasty, ny - 2), lastx] if ny > 1 else 0
            fi, li = interval(ft, True, cap_f), interval(lt, False, cap_l)
            if fy == lasty and lasty + 1 < ny:
                lasty += 1
            for x in range(fx, lastx + 1):
                for y in range(fy, lasty):
                    if x == fx:
                        vint.setdefault((x, y, l), []).append(fi)
                    elif x == lastx:
                        vint.setdefault((x, y, l), []).append(li)
                    else:
                        v3[l, y, x] = 0
    return hint, vint


def _transition_layers(t, macro_obs, h3, v3, g, horiz, pitch, hi):
    """GlobalRouter::findTransitionLayers + adjustTransitionLayers: a layer whose default via is wider than 0.8 of its
    track pitch has its edges halved (floor, at least 1) over the gcells a macro obstructs on the layer below."""
    via_w = t['layer_via_w'] if 'layer_via_w' in t else None
    if via_w is None:
        return
    for l in range(1, hi + 1):
        if not via_w[l] or via_w[l] / float(pitch[l]) <= 0.8 or not macro_obs.get(l - 1):
            continue
        tiles = set()
        for r in macro_obs[l - 1]:
            fx, fy, lastx, lasty, _ft, _lt = _tiles(*r, g)
            if fx == lastx and fy == lasty:
                continue
            for y in range(fy, lasty):
                for x in range(fx, lastx):
                    tiles.add((x, y))
        for x, y in tiles:
            if horiz[l]:
                if x < h3.shape[2]:
                    c = h3[l, y, x]; h3[l, y, x] = max(int(c * 0.5), 1) if c > 0 else c
            elif y < v3.shape[1]:
                c = v3[l, y, x]; v3[l, y, x] = max(int(c * 0.5), 1) if c > 0 else c


def _blocked_intervals(hint, vint, h3, v3, pitch):
    """FastRouteCore::initBlockedIntervals: an edge's blocked intervals, merged, cost ceil(length / pitch) tracks."""
    def blocked(v):
        v.sort()
        total, lo_, hi_ = 0, None, None
        for s, e in v:
            if e <= s:
                continue
            if hi_ is None:
                lo_, hi_ = s, e
            elif s <= hi_:
                hi_ = max(hi_, e)
            else:
                total += hi_ - lo_; lo_, hi_ = s, e
        if hi_ is not None:
            total += hi_ - lo_
        return total
    for (x, y, l), v in hint.items():
        if h3[l, y, x] > 0:
            h3[l, y, x] = max(h3[l, y, x] - int(math.ceil(blocked(v) / pitch[l])), 0)
    for (x, y, l), v in vint.items():
        if v3[l, y, x] > 0:
            v3[l, y, x] = max(v3[l, y, x] - int(math.ceil(blocked(v) / pitch[l])), 0)


# -- RUDY demand and overflow ------------------------------------------------------------------------------------------
def rudy(d, g, pos=None):
    """RUDY per gcell and direction on grid g, in tracks: (H [ny][nx], V [ny][nx]). A net spreads its horizontal wire
    (bbox width) evenly over its bbox: horizontal demand density 1 / bbox height, vertical 1 / bbox width (bbox at
    least one gcell each way); times the gcell area over the gcell length = wires crossing the gcell."""
    from . import metrics as M
    from .kernels import Grid
    from .evaluator import Evaluator
    ev = Evaluator(d)
    ev._prepare()
    s = 1.0 / d.dbu
    t = g['tile'] * s
    x0, y0 = g['die'][0] * s, g['die'][1] * s
    grid_ = Grid.over_box([x0, y0, x0 + g['nx'] * t, y0 + g['ny'] * t], g['nx'], g['ny'], d.pos.device)
    p = ev._pins(pos)
    lo, hi = M.net_bbox(p, ev.net, ev.E)
    ok = ev.deg >= 2
    lo, hi, w = lo[ok], hi[ok], ev.wmask[ok]
    c = (lo + hi) / 2
    sz = torch.maximum(hi - lo, torch.tensor([t, t], device=p.device))
    lo, hi = c - sz / 2, c + sz / 2
    mh = grid_.accumulate(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], w / sz[:, 1]) / grid_.bin_area * t
    mv = grid_.accumulate(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], w / sz[:, 0]) / grid_.bin_area * t
    return mh.double().cpu().numpy().T, mv.double().cpu().numpy().T


def edge_overflow(dem_h, dem_v, cap_h, cap_v):
    """GROV from 2D maps: gcell-edge demand (H edges: the mean of the two gcells' H demand; V likewise) against the
    edge capacity summed over the layers of that direction. Shares of edges whose demand exceeds capacity, in %."""
    eh = (dem_h[:, :-1] + dem_h[:, 1:]) / 2 if dem_h.shape == (cap_h.shape[0], cap_h.shape[1] + 1) else dem_h
    ev = (dem_v[:-1, :] + dem_v[1:, :]) / 2 if dem_v.shape == (cap_v.shape[0] + 1, cap_v.shape[1]) else dem_v
    oh, ov = eh > cap_h, ev > cap_v
    n_h, n_v = oh.size, ov.size
    uh = eh / np.maximum(cap_h, 1e-9); uv = ev / np.maximum(cap_v, 1e-9)
    return {'grov_%': 100.0 * (oh.sum() + ov.sum()) / max(n_h + n_v, 1),
            'grov_h_%': 100.0 * oh.sum() / max(n_h, 1), 'grov_v_%': 100.0 * ov.sum() / max(n_v, 1),
            'overflow_edges': int(oh.sum() + ov.sum()), 'edges': int(n_h + n_v),
            'total_overflow': float(np.maximum(eh - cap_h, 0).sum() + np.maximum(ev - cap_v, 0).sum()),
            'max_util': float(max(uh.max(initial=0), uv.max(initial=0)))}


def overflow(d, pos=None, adjustment=None, layers=None, maps=False):
    """Global-routing overflow (GROV) estimated with RUDY on OpenROAD's grid and capacities: the share of gcell edges
    (horizontal and vertical) whose RUDY demand exceeds their capacity, in %, with per-direction values, the total
    overflow (tracks) and the worst edge utilization. maps=True adds the demand and capacity maps."""
    g, h3, v3 = capacity(d, adjustment, layers)
    dh, dv = rudy(d, g, pos)
    cap_h, cap_v = h3.sum(0).astype(np.float64), v3.sum(0).astype(np.float64)
    r = edge_overflow(dh, dv, cap_h, cap_v)
    r.update({'gcells': (g['nx'], g['ny']), 'tile_um': g['tile'] / d.dbu, 'adjustment': g['adjustment'],
              'layers': [d.names['layers'][g['layers'][0]], d.names['layers'][g['layers'][1]]]})
    if maps:
        r.update({'demand_h': dh, 'demand_v': dv, 'cap_h': cap_h, 'cap_v': cap_v, 'cap3_h': h3, 'cap3_v': v3})
    return r
