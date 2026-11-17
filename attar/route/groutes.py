"""Global routes of a design (the solution state global routing produces; Timer(parasitics='global_routing') reads it).

A GlobalRoutes holds, as OpenROAD's GlobalRouter keeps them (grt::GRoute, grt::PinGridLocation):
  segments  per net, 3D segments between gcell centers: (x0, y0, l0, x1, y1, l1), dbu and routing level (1 = the
            first routing layer, names['layers'][0]); a via has x0 == x1, y0 == y1, l0 != l1
  pins      per pin of a routed net, where the route meets it: pt (the pin's position, GlobalRouter's
            makeItermPins / makeBtermPins: the lower-left corner of its last routing-layer shape), grid (its gcell's
            center, Pin::getOnGridPosition) and conn_layer (Pin::getConnectionLayer); -1 where unknown
Routes come from Attar's router (attar.route) or are read from OpenROAD's write_global_route_segments output
(read_segments) with its pins (pins_from_shapes + read_pin_locations): an explicit input, never a filled-in fact.
"""
import os

import numpy as np


def _norm(name):
    return name.replace('\\', '')


class GlobalRoutes:
    """seg_net (S,), seg (S, 6) int64; pin_pt (P, 2), pin_grid (P, 2), pin_layer (P,) int64 per pin of the design
    (-1: unknown); min_layer: the lowest routing level (GlobalRouter::getMinRoutingLayer); source: where they came
    from."""

    def __init__(self, d, seg_net, seg, pin_pt=None, pin_grid=None, pin_layer=None, min_layer=1, source=''):
        P = len(d.raw['p_cell'])
        self.E = len(d.raw['n_ptr']) - 1
        self.seg_net = np.asarray(seg_net, np.int64)
        self.seg = np.asarray(seg, np.int64).reshape(-1, 6)
        self.pin_pt = np.full((P, 2), -1, np.int64) if pin_pt is None else np.asarray(pin_pt, np.int64)
        self.pin_grid = np.full((P, 2), -1, np.int64) if pin_grid is None else np.asarray(pin_grid, np.int64)
        self.pin_layer = np.full(P, -1, np.int64) if pin_layer is None else np.asarray(pin_layer, np.int64)
        self.min_layer = int(min_layer)
        self.source = source
        self.dbu = d.dbu

    def routed(self):
        """Per net: has a non-empty route."""
        r = np.zeros(self.E, bool)
        r[self.seg_net] = True
        return r

    def wirelength(self, tile, per_net=False):
        """Wire length as grt reports it (GlobalRouter::computeNetWirelength, 'Total wirelength'): every segment of
        non-zero length counts its length plus one gcell (tile, dbu); vias count nothing. per_net: um per net, else
        the total in whole um (grt divides the dbu sum by the DEF units)."""
        s = self.seg
        L = np.abs(s[:, 3] - s[:, 0]) + np.abs(s[:, 4] - s[:, 1])
        L = np.where(L > 0, L + int(tile), 0)
        per = np.bincount(self.seg_net, L, minlength=self.E)
        if per_net:
            return per / self.dbu
        return int(per.sum()) // int(round(self.dbu))

    def vias(self):
        """Number of vias (one per routing level crossed)."""
        s = self.seg
        return int(np.abs(s[:, 5] - s[:, 2]).sum())


def pin_names(d):
    """Per pin of the design, its OpenROAD name: 'instance/pin' for a cell pin, the port name for an IO pin."""
    a, n = d.raw, d.names
    mp = [p for pins in n['master_pins'] for p in pins]
    cells = n['cells']
    out = []
    for c, m in zip(a['p_cell'].tolist(), a['p_mpin'].tolist()):
        out.append(_norm(cells[c]) if m < 0 else '%s/%s' % (_norm(cells[c]), mp[m]))
    return out


def read_segments(d, path, min_layer=None, source=None):
    """OpenROAD's write_global_route_segments output -> GlobalRoutes (segments only; pins: set them with
    pins_from_shapes / read_pin_locations). Layer names are the IR's routing layers."""
    lid = {nm: i + 1 for i, nm in enumerate(d.names['layers'])}
    nid = {_norm(nm): i for i, nm in enumerate(d.names['nets'])}
    seg_net, seg = [], []
    cur = -1
    missing = 0
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0] in ('(', ')'):
                continue
            if len(t) == 1:
                cur = nid.get(_norm(t[0]), -1)
                missing += cur < 0
            elif len(t) == 6 and cur >= 0:
                seg_net.append(cur)
                seg.append((int(t[0]), int(t[1]), lid[t[2]], int(t[3]), int(t[4]), lid[t[5]]))
    if missing:
        raise ValueError('%d routed nets of %s are not in the design' % (missing, os.path.basename(path)))
    if min_layer is None:
        r = d.manifest.get('routing') or {}
        min_layer = lid[r['layers'][0]] if r.get('layers') else 1
    return GlobalRoutes(d, seg_net, seg, min_layer=min_layer,
                        source=source or 'write_global_route_segments %s' % os.path.basename(path))


def read_pin_locations(d, path):
    """OpenROAD's grt::write_pin_locations output -> (pin_grid (P, 2), found (P,) bool): each pin's on-grid position."""
    pid = {nm: i for i, nm in enumerate(pin_names(d))}
    P = len(pid)
    grid = np.full((P, 2), -1, np.int64)
    with open(path) as f:
        for line in f:
            t = line.split()
            if len(t) == 3 and not t[1].lstrip('-').isdigit() or len(t) != 3:
                continue
            p = pid.get(_norm(t[0]))
            if p is not None:
                grid[p] = (int(t[1]), int(t[2]))
    return grid, grid[:, 0] >= 0


# -- grt's pin model (GlobalRouter::makeItermPins / makeBtermPins, Pin, findOnGridPositions without access points) ----
def read_shapes(path):
    """A shape dump (tools/gr_truth.py dump_shapes): ({master: {'type', 'w', 'h', 'block', 'pad', 'pins': {pin: [(level,
    x0, y0, x1, y1), ...]}}}, {port: [(level, x0, y0, x1, y1), ...]}, die). Only routing-layer shapes, in odb order."""
    masters, ports, die = {}, {}, None
    cur = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == 'M':
                cur = masters[t[1]] = {'type': t[2], 'w': int(t[3]), 'h': int(t[4]), 'block': t[5] == '1',
                                       'pad': t[6] == '1', 'pins': {}}
            elif t[0] == 'S' and t[3] == 'ROUTING':
                cur['pins'].setdefault(t[1], []).append((int(t[4]), int(t[5]), int(t[6]), int(t[7]), int(t[8])))
            elif t[0] == 'P' and t[3] == 'ROUTING':
                ports.setdefault(t[1], []).append((int(t[4]), int(t[5]), int(t[6]), int(t[7]), int(t[8])))
            elif t[0] == 'D':
                die = tuple(int(v) for v in t[1:5])
    return masters, ports, die


_ORI = {'N': (1, 0, 0, 1), 'S': (-1, 0, 0, -1), 'W': (0, -1, 1, 0), 'E': (0, 1, -1, 0), 'FN': (-1, 0, 0, 1),
        'FS': (1, 0, 0, -1), 'FW': (0, -1, -1, 0), 'FE': (0, 1, 1, 0)}


def _edge(boxes, bounds):
    """Pin::determineEdge's vote: the bounds' edge most of the pin's boxes are nearest to (n, s, e, w on ties)."""
    cnt = {'n': 0, 's': 0, 'e': 0, 'w': 0}
    for _, x0, y0, x1, y1 in boxes:
        dn, ds, de, dw = bounds[3] - y1, y0 - bounds[1], bounds[2] - x1, x0 - bounds[0]
        m = min(dn, ds, dw, de)
        cnt['n' if dn == m else 's' if ds == m else 'e' if de == m else 'w'] += 1
    best = max(cnt.values())
    return next(k for k in 'nsew' if cnt[k] == best)


def pins_from_shapes(d, routes, shapes, max_layer, tile, tracks=None, horizontal=None):
    """Fill routes.pin_pt / pin_grid / pin_layer from pin shapes (read_shapes) at the design's placement, as
    GlobalRouter does without access points: pt = the lower-left of the pin's last routing shape; connection layer =
    its highest layer up to max_layer (a macro / pad / IO pin: the highest one in the direction across its nearest
    instance / die edge); on-grid position = the most common gcell center of its connection-layer shapes' middles (a
    long macro / IO shape: the point near the edge), moved to a single track it overlaps (pinOverlapsWithSingleTrack).
    tracks: {level: (location, pitch)} of each layer's preferred-direction tracks; horizontal: per level (1-based)."""
    from ..io.deffile import ORIENTS
    masters, ports, die = shapes
    a, n = d.raw, d.names
    dbu = d.dbu
    x0d, y0d = die[0], die[1]
    nxg, nyg = (die[2] - die[0]) // tile, (die[3] - die[1]) // tile

    def on_grid(x, y):
        gx = (x - x0d) // tile; gy = (y - y0d) // tile
        gx = gx - 1 if gx >= nxg else gx; gy = gy - 1 if gy >= nyg else gy
        return gx * tile + tile // 2 + x0d, gy * tile + tile // 2 + y0d

    mnames = n['masters']
    mp = n['master_pins']
    ptr = a['m_pin_ptr']
    pos = np.rint(d.pos.detach().double().cpu().numpy() * dbu).astype(np.int64)
    ori = d.orient.cpu().numpy()
    cm = a['c_master'].astype(np.int64)
    cells = n['cells']
    P = len(a['p_cell'])
    pt = np.full((P, 2), -1, np.int64); grid = np.full((P, 2), -1, np.int64); layer = np.full(P, -1, np.int64)
    for p, (c, m) in enumerate(zip(a['p_cell'].tolist(), a['p_mpin'].tolist())):
        if m < 0:                                                   # an IO pin: its port's shapes (die frame)
            bx = ports.get(_norm(cells[c]))
            if not bx:
                continue
            # boxes outside the die are clipped to it (makeBtermPins)
            bx = [(l, max(x0, die[0]), max(y0, die[1]), min(x1, die[2]), min(y1, die[3])) for l, x0, y0, x1, y1 in bx]
            bounds, edge_pin = die, True
            layers = sorted({b[0] for b in bx})
        else:
            mi = int(cm[c])
            mn = mnames[mi]
            info = masters.get(mn)
            if info is None:
                continue
            pin = mp[mi][m - int(ptr[mi])]
            raw = info['pins'].get(pin)
            if not raw:
                continue
            A, B, C, D = _ORI[ORIENTS[int(ori[c])]]
            w, h = info['w'], info['h']
            ox = pos[c, 0] - min(0, A * w + B * h); oy = pos[c, 1] - min(0, C * w + D * h)
            bx = []
            for l, x0, y0, x1, y1 in raw:
                xa, xb = A * x0 + B * y0, A * x1 + B * y1
                ya, yb = C * x0 + D * y0, C * x1 + D * y1
                bx.append((l, min(xa, xb) + ox, min(ya, yb) + oy, max(xa, xb) + ox, max(ya, yb) + oy))
            edge_pin = info['block'] or info['pad']
            bounds = (pos[c, 0], pos[c, 1], pos[c, 0] + (w if A != 0 else h), pos[c, 1] + (h if A != 0 else w))
            layers = sorted({b[0] for b in bx if b[0] <= max_layer})
        if not layers:
            continue
        pt[p] = bx[-1][1:3]
        edge = None
        conn = layers[-1]
        if edge_pin:
            edge = _edge(bx, bounds)
            want_h = edge in 'ew'
            cand = [l for l in layers if horizontal is not None and bool(horizontal[l]) == want_h]
            if cand:
                conn = max(cand)
        boxes = [b for b in bx if b[0] == conn]
        cands = []
        last = None
        for _, x0, y0, x1, y1 in boxes:
            mx, my = (x0 + x1) // 2, (y0 + y1) // 2                 # getRectMiddle
            if edge is not None and max(x1 - x0, y1 - y0) >= tile:
                ex, ey = mx, my
                if edge == 'n':
                    ey = y1
                elif edge == 's':
                    ey = y0
                elif edge == 'e':
                    ex = x1
                else:
                    ex = x0
                last = on_grid(ex, ey)
            else:
                last = on_grid(mx, my)
            cands.append(last)
        best, votes = None, -1
        for c_ in cands:                                            # computePinPositionOnGrid: first most common
            k = cands.count(c_)
            if k > votes:
                best, votes = c_, k
        if best is None:
            best = tuple(pt[p])
        if tracks is not None and conn in tracks and last is not None:
            loc, pitch = tracks[conn]
            hz = bool(horizontal[conn])
            ux0 = min(b[1] for b in boxes); uy0 = min(b[2] for b in boxes)
            ux1 = max(b[3] for b in boxes); uy1 = max(b[4] for b in boxes)
            lo, hi = (uy0, uy1) if hz else (ux0, ux1)
            if pitch > 0 and float(hi - lo) / pitch <= 3:
                t1 = int(np.floor(np.float32(hi - loc) / np.float32(pitch))) * pitch + loc
                t2 = int(np.floor(np.float32(hi - loc) / np.float32(pitch) - 1)) * pitch + loc
                tp = None
                if not (lo <= t1 <= hi and lo <= t2 <= hi):
                    if lo < t1 < hi:
                        tp = (last[0], t1) if hz else (t1, last[1])
                    elif lo < t2 < hi:
                        tp = (last[0], t2) if hz else (t2, last[1])
                if tp is not None:
                    g = on_grid(*tp)
                    if g != best and ((hz and g[1] != best[1]) or (not hz and g[0] != best[0])):
                        best = g
        grid[p] = best
        layer[p] = conn
    routes.pin_pt, routes.pin_grid, routes.pin_layer = pt, grid, layer
    return routes
