"""Wire parasitics from global routes: OpenROAD's estimate_parasitics -global_routing (est::MakeWireParasitics),
reduced as OpenSTA reduces a parasitic network for its delay calculation (ReduceParasitics: reduceToPiElmore).

Network per routed net (MakeWireParasitics::estimateParasitics):
  - a node per distinct (x, y, routing level) of the route's segment ends (a planar segment below the lowest routing
    layer makes none: it is skipped);
  - a planar segment: a resistor r x L and its capacitance c x L split in half to its ends (L between the segment's
    route points: a gcell center holding pins of the net stands for the median of those pins, findPinCenters /
    routePoint; R at least 1e-3 ohm); a via: the resistance of the cut layer it crosses, no capacitance;
  - each pin: a resistor from the pin to its gcell's node on the layer above its connection layer (+ that cut's
    resistance) if the route has one there, else on its connection layer, of the wire from the pin's position to
    the route point (that layer's r and c, c split in half; R at least 1e-3 ohm).
r, c per layer: set_layer_rc (timing.json 'rc' 'layer'); a via: set_layer_rc -via ('rc' 'via'), else the cut layer's
LEF resistance (via_res, an explicit argument when timing.json has none).
Reduction from the driver pin (reducePiDfs / reduceElmoreDfs): admittance moments y1, y2, y3 -> the pi model
(c2 near, rpi, c1 far), each load's Elmore delay = sum of r x downstream capacitance; pin capacitances per rise / fall
and max / min. A net whose network has a loop (two segments meeting twice) is reduced over a spanning tree of it
(breadth first, which is OpenSTA's depth-first tree only where no loop exists): such nets are counted.
"""
import re

import numpy as np

FLOOR = 1e-6          # est's resistance floor, 1e-3 ohm, in kohm


def _via_table(rc, n_levels, via_res=None):
    """Per routing level l (1-based), the resistance (kohm) of the cut between l and l + 1."""
    out = np.zeros(n_levels + 2)
    num = lambda s: int(re.sub(r'\D', '', s)) if re.search(r'\d', s) else None  # noqa: E731
    for name, r in (rc.get('via') or {}).items():
        k = num(name)
        if k is not None and k < len(out):
            out[k] = r
    if via_res is not None:
        for k, r in (via_res.items() if isinstance(via_res, dict) else enumerate(via_res, 1)):
            if k < len(out) and out[k] == 0:
                out[k] = r
    return out


def build(d, routes, rc, via_res=None):
    """The parasitic networks of every routed net. Returns a dict of numpy arrays:
    node_net (N,), node_cap (N,) wire capacitance (fF) at each node; res_a, res_b (R,) node ids, res_r (R,) kohm;
    pin_node (P,) the node of each pin (-1: its net is not routed, or the pin has no route to it)."""
    names = d.names['layers']
    L = len(names)
    lay_r = np.zeros(L + 2); lay_c = np.zeros(L + 2)
    for i, nm in enumerate(names, 1):
        if nm in (rc.get('layer') or {}):
            lay_r[i], lay_c[i] = rc['layer'][nm]
    vt = _via_table(rc, L, via_res)
    dbu = float(d.dbu)
    s, sn = routes.seg, routes.seg_net
    a = d.raw
    p_net = np.repeat(np.arange(len(a['n_ptr']) - 1), np.diff(a['n_ptr']))
    P = len(p_net)
    routed = routes.routed()
    # route points: per (net, gcell center) holding pins of the net, the median of their positions
    pp = routes.pin_pt; pg = routes.pin_grid
    has = routed[p_net] & (pg[:, 0] >= 0) & (pp[:, 0] >= 0)
    hp = np.nonzero(has)[0]
    key_dt = np.dtype([('n', np.int64), ('x', np.int64), ('y', np.int64)])
    gkey = np.empty(len(hp), key_dt); gkey['n'] = p_net[hp]; gkey['x'] = pg[hp, 0]; gkey['y'] = pg[hp, 1]
    ug, ginv = np.unique(gkey, return_inverse=True)
    med = np.zeros((len(ug), 2), np.int64)
    order = np.lexsort((pp[hp, 0], ginv)); gs = ginv[order]
    cnt = np.bincount(ginv, minlength=len(ug)); start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    med[:, 0] = pp[hp, 0][order][start + cnt // 2]                  # nth_element at size / 2
    order = np.lexsort((pp[hp, 1], ginv))
    med[:, 1] = pp[hp, 1][order][start + cnt // 2]
    del gs

    def route_point(net, x, y):
        k = np.empty(len(net), key_dt); k['n'] = net; k['x'] = x; k['y'] = y
        i = np.searchsorted(ug, k)
        ok = (i < len(ug))
        i2 = np.minimum(i, max(len(ug) - 1, 0))
        ok &= (ug[i2] == k) if len(ug) else np.zeros(len(k), bool)
        X = np.where(ok, med[i2, 0] if len(ug) else x, x); Y = np.where(ok, med[i2, 1] if len(ug) else y, y)
        return X, Y

    # nodes: segment ends on valid layers (or any via end)
    via = (s[:, 0] == s[:, 3]) & (s[:, 1] == s[:, 4]) & (s[:, 2] != s[:, 5])
    planar = s[:, 2] == s[:, 5]
    v1 = (s[:, 2] >= routes.min_layer) | via
    v2 = (s[:, 5] >= routes.min_layer) | via
    keep = v1 & v2
    nk_dt = np.dtype([('n', np.int64), ('x', np.int64), ('y', np.int64), ('l', np.int64)])
    ends = np.empty(2 * int(keep.sum()), nk_dt)
    ks = np.nonzero(keep)[0]
    ends['n'] = np.concatenate([sn[ks], sn[ks]])
    ends['x'] = np.concatenate([s[ks, 0], s[ks, 3]]); ends['y'] = np.concatenate([s[ks, 1], s[ks, 4]])
    ends['l'] = np.concatenate([s[ks, 2], s[ks, 5]])
    nodes, ninv = np.unique(ends, return_inverse=True)
    K = len(ks)
    na, nb = ninv[:K], ninv[K:]
    node_cap = np.zeros(len(nodes))
    # planar segments
    is_p = planar[ks]
    X0, Y0 = route_point(sn[ks], s[ks, 0], s[ks, 1]); X1, Y1 = route_point(sn[ks], s[ks, 3], s[ks, 4])
    length = (np.abs(X1 - X0) + np.abs(Y1 - Y0)) / dbu
    lay = s[ks, 2]
    r_seg = np.where(is_p, np.maximum(lay_r[lay] * length, FLOOR), 0.0)
    c_seg = np.where(is_p, lay_c[lay] * length, 0.0)
    lo = np.minimum(s[ks, 2], s[ks, 5])
    r_seg = np.where(via[ks], vt[lo], r_seg)
    np.add.at(node_cap, na, c_seg / 2); np.add.at(node_cap, nb, c_seg / 2)
    # pins: to their gcell's node on conn_layer + 1 (with the cut below it), else on conn_layer
    pin_node = np.full(P, -1, np.int64)
    pr = np.nonzero(has & (routes.pin_layer >= 0))[0]
    cl = routes.pin_layer[pr]

    def find(net, x, y, l):
        k = np.empty(len(net), nk_dt); k['n'] = net; k['x'] = x; k['y'] = y; k['l'] = l
        i = np.searchsorted(nodes, k)
        i2 = np.minimum(i, max(len(nodes) - 1, 0))
        ok = (i < len(nodes)) & (nodes[i2] == k) if len(nodes) else np.zeros(len(k), bool)
        return np.where(ok, i2, -1)

    up = find(p_net[pr], pg[pr, 0], pg[pr, 1], cl + 1)
    at = find(p_net[pr], pg[pr, 0], pg[pr, 1], cl)
    use_up = up >= 0
    gnode = np.where(use_up, up, at)
    layer = np.where(use_up, cl + 1, cl)
    vr = np.where(use_up, vt[np.clip(cl, 0, len(vt) - 1)], 0.0)
    RX, RY = route_point(p_net[pr], pg[pr, 0], pg[pr, 1])
    plen = (np.abs(pp[pr, 0] - RX) + np.abs(pp[pr, 1] - RY)) / dbu
    pres = np.maximum(lay_r[layer] * plen + vr, FLOOR)
    pcap = lay_c[layer] * plen
    okp = gnode >= 0
    pr, gnode, pres, pcap = pr[okp], gnode[okp], pres[okp], pcap[okp]
    N0 = len(nodes)
    pnode = N0 + np.arange(len(pr))                                    # a node per pin
    pin_node[pr] = pnode
    node_net = np.concatenate([nodes['n'], p_net[pr]])
    node_cap = np.concatenate([node_cap, pcap / 2])
    np.add.at(node_cap, gnode, pcap / 2)
    res_a = np.concatenate([na, pnode]); res_b = np.concatenate([nb, gnode]); res_r = np.concatenate([r_seg, pres])
    return {'node_net': node_net, 'node_cap': node_cap, 'res_a': res_a, 'res_b': res_b, 'res_r': res_r,
            'pin_node': pin_node, 'pins_missing': int((has & (routes.pin_layer >= 0)).sum() - len(pr)),
            'wire_cap': np.bincount(node_net, node_cap, minlength=len(a['n_ptr']) - 1)}


def reduce(net, drv_pin, pin_cap):
    """Reduce the networks of build() from each net's driver pin. pin_cap (P, V) per pin and variant.
    Returns elmore (P, V) (0 where no network), pi (E, V, 3) as (near c2, rpi, far c1) where a network exists
    (NaN elsewhere), loops: the number of nets with a resistor loop."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order
    pin_node = net['pin_node']
    N = len(net['node_cap'])
    E = len(net['wire_cap'])
    P, V = pin_cap.shape
    elm = np.zeros((P, V)); pi = np.full((E, V, 3), np.nan)
    if N == 0:
        return elm, pi, 0
    dn = np.full(E, -1, np.int64)
    ok = drv_pin >= 0
    dn[ok] = pin_node[drv_pin[ok]]
    roots = np.nonzero(dn >= 0)[0]
    # one graph: a super root (node N) tied to every driver node, the resistors between nodes
    ra, rb, rr = net['res_a'], net['res_b'], net['res_r']
    loops = 0
    nn = np.bincount(net['node_net'], minlength=E)
    nr = np.bincount(net['node_net'][ra], minlength=E)
    loops = int(((nr > nn - 1) & (nn > 0)).sum())
    G = coo_matrix((np.ones(len(ra) + len(roots)), (np.concatenate([ra, np.full(len(roots), N)]),
                                                    np.concatenate([rb, dn[roots]]))), shape=(N + 1, N + 1)).tocsr()
    G = G + G.T
    order, pred = breadth_first_order(G, N, directed=False, return_predecessors=True)
    order = order[1:].astype(np.int64)                  # nodes reachable from a driver, parents first
    par = pred[order].astype(np.int64)                  # (scipy returns int32: the edge keys below need int64)
    # each tree edge's resistance: the resistor between a node and its parent (the first such if several)
    key = np.minimum(ra, rb) * (N + 1) + np.maximum(ra, rb)
    o = np.argsort(key, kind='stable'); ks = key[o]
    q = np.minimum(order, par) * (N + 1) + np.maximum(order, par)
    i = np.searchsorted(ks, q)
    root_edge = par == N
    r_edge = np.where(root_edge, 0.0, rr[o[np.minimum(i, len(ks) - 1)]])
    # downstream capacitance and moments, children before parents
    cap = np.repeat(net['node_cap'][:, None], V, 1)
    pn = pin_node >= 0
    np.add.at(cap, pin_node[pn], pin_cap[pn])
    y1 = np.zeros((N + 1, V)); y2 = np.zeros((N + 1, V)); y3 = np.zeros((N + 1, V))
    y1[:N] = cap
    pos = np.full(N + 1, -1, np.int64); pos[order] = np.arange(len(order))
    pp = np.where(root_edge, -1, pos[np.where(root_edge, 0, par)])      # parent's index in order (-1: a driver)
    depth_o = np.zeros(len(order), np.int64)
    anc = pp.copy()
    while True:                                          # depth = number of ancestors below the super root
        live = anc >= 0
        if not live.any():
            break
        depth_o += live
        anc = np.where(live, pp[np.maximum(anc, 0)], -1)
    for dpt in range(int(depth_o.max(initial=0)), -1, -1):
        sel = np.nonzero(depth_o == dpt)[0]
        c = order[sel]; p_ = par[sel]; r = r_edge[sel][:, None]
        a1, a2, a3 = y1[c], y2[c], y3[c]
        np.add.at(y1, p_, a1)
        np.add.at(y2, p_, a2 - r * a1 * a1)
        np.add.at(y3, p_, a3 - 2 * r * a1 * a2 + r * r * a1 ** 3)
    # Elmore, parents first
    el = np.zeros((N + 1, V))
    for dpt in range(1, int(depth_o.max(initial=0)) + 1):
        sel = np.nonzero(depth_o == dpt)[0]
        c = order[sel]
        el[c] = el[par[sel]] + r_edge[sel][:, None] * y1[c]
    elm[pn] = el[pin_node[pn]]
    # the pi model at each driver node
    Y1, Y2, Y3 = y1[dn[roots]], y2[dn[roots]], y3[dn[roots]]
    capl = (Y2 == 0) & (Y3 == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        c1 = np.where(capl, 0.0, Y2 * Y2 / np.where(capl, 1.0, Y3))
        rpi = np.where(capl, 0.0, -Y3 * Y3 / np.where(capl, 1.0, Y2 ** 3))
    pi[roots] = np.stack([Y1 - c1, rpi, c1], -1)
    return elm, pi, loops
