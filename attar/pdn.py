"""The PDN as a placement rule (fact 'pdn'), as OpenROAD's detailed placer checks it (dpl PlacementDRC::
checkBlockedLayers, Grid::initBlockedLayers): on the PDN_LEVELS lowest routing layers,

  pin blockages     via and DRC-fill metal of the PG nets that no PG wire of the same layer covers, grown by the
                    layer's spacing - 1: no signal pin shape on that layer may overlap one;
  stripe blockages  vertical PG wires on the 2nd and 3rd routing layer: no cell with a signal pin on that layer may
                    overlap one.

Pins are tested by the bounding box of their shapes (the IR keeps no more), which can only be stricter than OpenROAD.
"""
import numpy as np
import torch

from .deffile import PG_FILL, PG_VIA, PG_VWIRE
from .distill import TERMINAL


def _subtract_many(parts, owner, wires, w_owner):
    """Rects `parts` (P, 4) minus, for every (part's owner == w_owner) pair, that wire: vectorized rect - rect, each
    piece up to 4 rects. Returns the pieces and their owners."""
    if len(wires) == 0 or len(parts) == 0:
        return parts, owner
    # pair every part with every wire of its owner (owners have few wires)
    order = np.argsort(w_owner, kind='stable')
    wo, wr = w_owner[order], wires[order]
    lo = np.searchsorted(wo, owner, 'left'); hi = np.searchsorted(wo, owner, 'right')
    for k in range(int((hi - lo).max(initial=0))):         # the k-th wire of every owner, one round each
        has = (hi - lo) > k
        if not has.any():
            break
        keep_p, keep_o = [parts[~has]], [owner[~has]]
        r = parts[has]; o = owner[has]; w = wr[lo[has] + k]
        miss = (w[:, 0] >= r[:, 2]) | (w[:, 2] <= r[:, 0]) | (w[:, 1] >= r[:, 3]) | (w[:, 3] <= r[:, 1])
        keep_p.append(r[miss]); keep_o.append(o[miss])
        r, o, w = r[~miss], o[~miss], w[~miss]
        y0, y1 = np.maximum(r[:, 1], w[:, 1]), np.minimum(r[:, 3], w[:, 3])
        cand = [(np.stack([r[:, 0], r[:, 1], r[:, 2], w[:, 1]], 1), r[:, 1] < w[:, 1]),      # below the wire
                (np.stack([r[:, 0], w[:, 3], r[:, 2], r[:, 3]], 1), w[:, 3] < r[:, 3]),      # above
                (np.stack([r[:, 0], y0, w[:, 0], y1], 1), r[:, 0] < w[:, 0]),                # left
                (np.stack([w[:, 2], y0, r[:, 2], y1], 1), w[:, 2] < r[:, 2])]                # right
        for piece, m in cand:
            keep_p.append(piece[m]); keep_o.append(o[m])
        parts, owner = np.concatenate(keep_p), np.concatenate(keep_o)
        lo = np.searchsorted(wo, owner, 'left'); hi = np.searchsorted(wo, owner, 'right')
    return parts, owner


def pin_blockages(d):
    """(layer, x0, y0, x1, y1) int64 array in DBU of what signal pins must keep clear of; empty without 'pdn'.
    Computed once per design (cached on it)."""
    cached = getattr(d, '_pdn_blockages', None)
    if cached is not None:
        return cached
    a = d.raw
    if 'pdn' not in d.facts or 'pg_rect' not in a or len(a['pg_rect']) == 0:
        return np.zeros((0, 5), np.int64)
    from . import st
    import os
    tp, _ = st.load(os.path.join(d.path, 'tech_profile.safetensors'))
    spacing = tp['layer_spacing'] if 'layer_spacing' in tp else np.zeros(len(d.names['layers']), np.int32)
    layer, rect, kind = a['pg_layer'].astype(np.int64), a['pg_rect'].astype(np.int64), a['pg_kind']
    out = []
    for L in np.unique(layer):
        on = layer == L
        wires = rect[on & (kind != PG_VIA) & (kind != PG_FILL)]
        metal = np.unique(rect[on & ((kind == PG_VIA) | (kind == PG_FILL))], axis=0)   # via stacks repeat the metal
        b = max(int(spacing[L]) - 1, 0)
        if len(metal):
            # the wires each piece of metal may overlap: horizontal ones by y, vertical ones by x (both thin)
            hz = wires[(wires[:, 2] - wires[:, 0]) >= (wires[:, 3] - wires[:, 1])]
            vt = wires[(wires[:, 2] - wires[:, 0]) < (wires[:, 3] - wires[:, 1])]
            pr, pw = [], []
            for ws, ax in ((hz, 1), (vt, 0)):
                if not len(ws):
                    continue
                ws = ws[np.argsort(ws[:, ax])]
                span = int((ws[:, ax + 2] - ws[:, ax]).max())
                lo = np.searchsorted(ws[:, ax], metal[:, ax] - span, 'left')
                hi = np.searchsorted(ws[:, ax], metal[:, ax + 2], 'right')
                n = hi - lo
                ri = np.repeat(np.arange(len(metal)), n)
                wi = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
                m = metal[ri]; w = ws[wi]
                ov = (w[:, 0] < m[:, 2]) & (m[:, 0] < w[:, 2]) & (w[:, 1] < m[:, 3]) & (m[:, 1] < w[:, 3])
                pr.append(ri[ov]); pw.append(w[ov])
            ri = np.concatenate(pr) if pr else np.zeros(0, np.int64)
            ww = np.concatenate(pw) if pw else np.zeros((0, 4), np.int64)
            parts, owner = _subtract_many(metal, np.arange(len(metal)), ww, ri)
            parts = parts[(parts[:, 2] > parts[:, 0]) & (parts[:, 3] > parts[:, 1])]
            out.append(np.concatenate([np.full((len(parts), 1), L), parts + np.array([-b, -b, b, b])], 1))
        if L in (1, 2):                                              # vertical stripes on the 2nd and 3rd layer
            vw = rect[on & (kind == PG_VWIRE)]
            out.append(np.concatenate([np.full((len(vw), 1), L), vw], 1))
    res = np.concatenate(out).astype(np.int64).reshape(-1, 5) if out else np.zeros((0, 5), np.int64)
    d._pdn_blockages = res
    return res

class Index:
    """Pin blockages binned for overlap queries on the GPU: hits(boxes, layers) -> bool per box."""

    def __init__(self, d, dev, blockages=None):
        bl = pin_blockages(d) if blockages is None else blockages
        self.n = len(bl)
        if self.n == 0:
            return
        B = torch.as_tensor(bl, device=dev).double()
        self.R = B[:, 1:] / d.dbu
        L = B[:, 0].long(); R = self.R
        self.s = max(1.0, float((R[:, 2:] - R[:, :2]).max()))          # a rect spans at most 2 bins per axis
        self.ox, self.oy = float(R[:, 0].min()), float(R[:, 1].min())
        bx0, by0, bx1, by1 = self.gx(R[:, 0]), self.gy(R[:, 1]), self.gx(R[:, 2]), self.gy(R[:, 3])
        self.nx = int(bx1.max().item()) + 3
        keys, ids = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                m = (bx0 + dx <= bx1) & (by0 + dy <= by1)
                keys.append(self.key(L[m], bx0[m] + dx, by0[m] + dy)); ids.append(torch.nonzero(m).flatten())
        keys = torch.cat(keys); ids = torch.cat(ids)
        order = torch.argsort(keys)
        self.keys, self.ids = keys[order], ids[order]

    def gx(self, v):
        return ((v - self.ox) / self.s).floor().long().clamp(min=-1) + 1

    def gy(self, v):
        return ((v - self.oy) / self.s).floor().long().clamp(min=-1) + 1

    def key(self, layer, bx, by):
        return layer * (1 << 40) + by * self.nx + bx.clamp(max=self.nx - 1)

    def hits(self, pb, pl):
        """pb (P, 4) boxes in microns, pl (P,) layers: does each box overlap a blockage of its layer?"""
        dev = pb.device
        hit = torch.zeros(len(pb), dtype=torch.bool, device=dev)
        if self.n == 0 or len(pb) == 0:
            return hit
        for cx, cy in ((0, 1), (2, 1), (0, 3), (2, 3)):                  # the bins of the box's corners
            q = self.key(pl, self.gx(pb[:, cx]), self.gy(pb[:, cy]))
            lo = torch.searchsorted(self.keys, q); hi = torch.searchsorted(self.keys, q, right=True)
            cnt = hi - lo
            idx = torch.nonzero(cnt > 0).flatten()
            if len(idx) == 0:
                continue
            c = cnt[idx]
            rep = torch.repeat_interleave(idx, c)
            off = torch.arange(len(rep), device=dev) - torch.repeat_interleave(torch.cumsum(c, 0) - c, c)
            r = self.R[self.ids[lo[rep] + off]]
            ov = (pb[rep, 0] < r[:, 2]) & (r[:, 0] < pb[rep, 2]) & (pb[rep, 1] < r[:, 3]) & (r[:, 1] < pb[rep, 3])
            hit[rep[ov]] = True
        return hit


def pin_shapes(d, dev):
    """Per pin: master-relative bbox (P, 4) microns, layer (-1: not a cell signal pin)."""
    a = d.raw
    mp = torch.as_tensor(a['p_mpin'].astype(np.int64), device=dev)
    box = torch.as_tensor(a['m_pin_box'].astype(np.float64), device=dev) / d.dbu
    lay = torch.as_tensor(a['m_pin_layer'].astype(np.int64), device=dev)
    use = torch.as_tensor(a['m_pin_use'].astype(np.int64), device=dev)
    k = mp.clamp(min=0)
    layer = torch.where((mp >= 0) & (use[k] == 0), lay[k], torch.full_like(k, -1))
    term = (d.flags[d.p_cell.to(dev)] & TERMINAL) != 0
    return box[k], torch.where(term, torch.full_like(layer, -1), layer)


def place_boxes(rel, w, h, o, x, y):
    """Master-relative boxes placed at lower-left (x, y) with orientation o (N, S, FN, FS; others as N)."""
    x0, y0, x1, y1 = rel[:, 0], rel[:, 1], rel[:, 2], rel[:, 3]
    fx = (o == 1) | (o == 4)                        # S, FN: mirrored in x
    fy = (o == 1) | (o == 5)                        # S, FS: mirrored in y
    X0 = torch.where(fx, w - x1, x0); X1 = torch.where(fx, w - x0, x1)
    Y0 = torch.where(fy, h - y1, y0); Y1 = torch.where(fy, h - y0, y1)
    return torch.stack([x + X0, y + Y0, x + X1, y + Y1], 1)


def conflicts(d, pos=None, orient=None, index=None):
    """Bool per cell: a signal pin overlaps a PDN pin blockage of its layer."""
    pos = d.pos if pos is None else pos
    orient = d.orient if orient is None else orient
    dev = pos.device
    res = torch.zeros(len(pos), dtype=torch.bool, device=dev)
    index = Index(d, dev) if index is None else index
    if index.n == 0:
        return res
    rel, layer = pin_shapes(d, dev)
    pins = torch.nonzero(layer >= 0).flatten()
    c = d.p_cell[pins]
    pb = place_boxes(rel[pins], d.size[c, 0].double(), d.size[c, 1].double(), orient[c], pos[c, 0].double(), pos[c, 1].double())
    hit = index.hits(pb, layer[pins])
    res[c[hit]] = True
    return res
