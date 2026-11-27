"""Pin access against the PDN (the Evaluator's 'pin_access' metric): can the router reach each signal pin from the
layer above without a spacing violation to the power grid?

Every signal pin is reached by a via up to the next routing layer (Lu = the pin layer + 1). An access point is where
a preferred-direction track of Lu crosses the pin; there the via lands a square of the layer's width on the pin layer
and on Lu. An access point is blocked when either landing is closer than that layer's minimum spacing to PG metal of
the same layer (wires, vias, fill; Euclidean, corner to corner too; facts 'pdn' and 'tracks', widths and spacings
from the tech profile). A pin is blocked when all of its access points are. A pin that no track of Lu crosses has one
access point at its centre (an off-track access).

Measured on 988 routed Innovus runs of i3dPDK_NS3K (pa_pdn): OpenROAD dpl's rule (attar.rules.pdn.conflicts) flags
almost none of the failing pins, since a single-access M1 pin can sit exactly one M1 spacing from a PG via (legal)
while its M2 landing is 0.040 um from the via's M2 pad (M2 spacing 0.048): a signal-to-PG M2 DRV after routing.

Nothing here depends on the PDK beyond the IR: layer directions, widths, spacings, tracks and the PG shapes. Pins are
tested by the bounding box of their shapes (the IR keeps no more); the via landings are squares of the layer width
(the smallest a via enclosure can be), so the rule errs towards calling a pin reachable.
"""
import os

import numpy as np
import torch

from . import pdn as PDN


def _tech(d):
    from ..io import st
    tp, _ = st.load(os.path.join(d.path, 'tech_profile.safetensors'))
    return tp


def pg_shapes(d, tp=None):
    """(layer, x0, y0, x1, y1, g) float64 DBU of every PG shape grown by g, its layer's spacing - 0.25 DBU (shapes are on
    the DBU grid, landing centres on the half grid): what a landing closer than the spacing overlaps. Cached on the
    design."""
    cached = getattr(d, '_pa_pg', None)
    if cached is not None:
        return cached
    a = d.raw
    if 'pdn' not in d.facts or 'pg_rect' not in a or len(a['pg_rect']) == 0:
        res = np.zeros((0, 6))
    else:
        tp = _tech(d) if tp is None else tp
        sp = tp['layer_spacing'].astype(np.float64)
        L = a['pg_layer'].astype(np.int64)
        R = a['pg_rect'].astype(np.float64)
        key = np.unique(np.concatenate([L[:, None], R], 1), axis=0)      # via stacks repeat their metal
        g = np.maximum(sp[key[:, 0].astype(np.int64)] - 0.25, 0)[:, None]
        res = np.concatenate([key[:, :1], key[:, 1:3] - g, key[:, 3:] + g, g], 1)
    d._pa_pg = res
    return res


def access_points(d, pb, layer, tp):
    """Access points of placed pin boxes pb (P, 4, um) on layers `layer` (P,):
    -> (owner pin index, x, y) in um, one or more per pin; the number of pins accessed off-track."""
    dbu = float(d.dbu)
    dev = pb.device
    nL = len(d.names['layers'])
    ldir = tp['layer_dir'].astype(np.int64)
    lu = (layer + 1).clamp(max=nL - 1)
    cx = (pb[:, 0] + pb[:, 2]) / 2; cy = (pb[:, 1] + pb[:, 3]) / 2
    count = torch.zeros(len(pb), dtype=torch.long, device=dev)
    O, X, Y = [], [], []
    eps = 0.25 / dbu
    for L in torch.unique(lu).tolist():
        horiz = ldir[L] == 0                                 # tracks of a horizontal layer cross the pin at y
        sel = torch.nonzero(lu == L).flatten()
        lo = pb[sel, 1] if horiz else pb[sel, 0]
        hi = pb[sel, 3] if horiz else pb[sel, 2]
        for s0, sp, num in track_patterns(d, tp, L, 1 if horiz else 0):
            k0 = torch.ceil((lo - s0 - eps) / sp).clamp(min=0)
            k1 = torch.floor((hi - s0 + eps) / sp).clamp(max=num - 1)
            n = (k1 - k0 + 1).clamp(min=0).long()
            own = torch.repeat_interleave(sel, n)
            rank = torch.arange(len(own), device=dev) - torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
            t = s0 + (torch.repeat_interleave(k0, n) + rank) * sp
            O.append(own)
            X.append(cx[own] if horiz else t); Y.append(t if horiz else cy[own])
            count.index_add_(0, sel, n)
    off = torch.nonzero(count == 0).flatten()                # no track crosses the pin: its centre
    O.append(off); X.append(cx[off]); Y.append(cy[off])
    return torch.cat(O), torch.cat(X), torch.cat(Y), len(off)


def track_patterns(d, tp, L, axis):
    """The track patterns of layer L along axis (0: x positions, 1: y): [(start um, step um, count)]. Every DEF TRACKS
    statement of that layer and axis (a PDK may need several: ASAP7 M2); without any, the LEF offset and pitch."""
    a = d.raw
    dbu = float(d.dbu)
    out = []
    if 'tr_layer' in a and len(a['tr_layer']):
        m = (a['tr_layer'].astype(np.int64) == L) & (a['tr_axis'].astype(np.int64) == axis) & (a['tr_step'] > 0)
        out = sorted({(int(s), int(p), int(n)) for s, p, n in zip(a['tr_start'][m], a['tr_step'][m], a['tr_num'][m])})
    if not out and tp['layer_pitch'][L] > 0:
        out = [(int(tp['layer_offset'][L]), int(tp['layer_pitch'][L]), 1 << 40)]
    return [(s / dbu, p / dbu, n) for s, p, n in out]


class _Spacing(PDN.Index):
    """PG shapes binned as attar.rules.pdn.Index; hits(boxes, layers): is a box closer than the spacing to a shape of
    its layer, Euclidean between the rectangles (LEF minimum spacing: corner to corner too, not only along the
    edges)? The bins hold the shapes grown by the spacing; each candidate pair is then measured."""

    def __init__(self, d, dev, pg):
        super().__init__(d, dev, blockages=pg[:, :5])
        self.g = torch.as_tensor(pg[:, 5], device=dev, dtype=torch.float64) / d.dbu if self.n else None

    def hits(self, pb, pl):
        dev = pb.device
        hit = torch.zeros(len(pb), dtype=torch.bool, device=dev)
        if self.n == 0 or len(pb) == 0:
            return hit
        bx0 = self.gx(pb[:, 0]); by0 = self.gy(pb[:, 1])
        bx1 = self.gx(pb[:, 2]).clamp(max=self.nx - 1)
        by1 = self.gy(pb[:, 3])
        nbx = (bx1 - bx0 + 1).clamp(min=0); nby = (by1 - by0 + 1).clamp(min=0)
        for sl in PDN._chunks(nbx * nby, self.BUDGET):
            q, k = PDN._expand(nbx[sl] * nby[sl])
            q = q + sl.start
            key = self.key(pl[q], bx0[q] + k % nbx[q], by0[q] + k // nbx[q])
            lo = torch.searchsorted(self.keys, key); cnt = torch.searchsorted(self.keys, key, right=True) - lo
            m = cnt > 0
            q, lo, cnt = q[m], lo[m], cnt[m]
            for s2 in PDN._chunks(cnt, self.BUDGET):
                j, off = PDN._expand(cnt[s2])
                rep = q[s2][j]
                ids = self.ids[lo[s2][j] + off]
                r, g = self.R[ids], self.g[ids]
                dx = torch.maximum(torch.maximum(r[:, 0] + g - pb[rep, 2], pb[rep, 0] - (r[:, 2] - g)), torch.zeros_like(g))
                dy = torch.maximum(torch.maximum(r[:, 1] + g - pb[rep, 3], pb[rep, 1] - (r[:, 3] - g)), torch.zeros_like(g))
                close = dx * dx + dy * dy < g * g
                hit[rep[close]] = True
        return hit


def blocked_pins(d, pos=None, orient=None):
    """Per pin of the design (bool): a signal pin whose every access point is blocked by PG metal (module doc).
    Also returns counts: {'signal_pins', 'access_points', 'blocked_access_points', 'off_track_pins'}."""
    pos = d.pos if pos is None else pos
    orient = d.orient if orient is None else orient
    dev = pos.device
    nP = len(d.p_cell)
    out = torch.zeros(nP, dtype=torch.bool, device=dev)
    rel, layer = PDN.pin_shapes(d, dev)
    pins = torch.nonzero(layer >= 0).flatten()
    stats = {'signal_pins': len(pins), 'access_points': 0, 'blocked_access_points': 0, 'off_track_pins': 0}
    pg = pg_shapes(d)
    if len(pins) == 0 or len(pg) == 0:
        return out, stats
    tp = _tech(d)
    c = d.p_cell[pins]
    pb = PDN.place_boxes(rel[pins], d.size[c, 0].double(), d.size[c, 1].double(), orient[c], pos[c, 0].double(),
                         pos[c, 1].double())
    lp = layer[pins]
    nL = len(d.names['layers'])
    owner, x, y, n_off = access_points(d, pb, lp, tp)
    hw = torch.as_tensor(tp['layer_width'].astype(np.float64) / 2 / float(d.dbu), device=dev)
    index = _Spacing(d, dev, pg)
    blocked = torch.zeros(len(owner), dtype=torch.bool, device=dev)
    for L in (lp[owner], (lp[owner] + 1).clamp(max=nL - 1)):     # the landing on the pin layer and on the one above
        h = hw[L]
        blocked |= index.hits(torch.stack([x - h, y - h, x + h, y + h], 1), L)
    npts = torch.bincount(owner, minlength=len(pins))
    nblk = torch.bincount(owner[blocked], minlength=len(pins))
    out[pins[(nblk == npts) & (npts > 0)]] = True
    stats.update({'access_points': len(owner), 'blocked_access_points': int(blocked.sum()), 'off_track_pins': n_off})
    return out, stats
