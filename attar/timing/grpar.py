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


def _packer(*cols):
    """A function packing integer columns into one int64 key whose order is the columns' lexicographic order (as
    np.unique / searchsorted on a structured array, which sort record by record: 20x slower). cols: every value each
    column will take; a column is packed as its value's rank among them (coordinates in DBU would overflow int64)."""
    uniq = []
    for c in cols:
        c = np.concatenate([np.asarray(x, np.int64).ravel() for x in c]) if isinstance(c, (list, tuple)) else c
        uniq.append(np.unique(c))
    sizes = [max(len(u), 1) for u in uniq]
    if float(np.prod([float(x) for x in sizes])) >= 2.0 ** 62:
        raise OverflowError('grpar: key columns too wide for int64 (%s)' % (sizes,))

    def pack(*vals):
        k = np.zeros(np.broadcast(*vals).shape, np.int64)
        for v, u, m in zip(vals, uniq, sizes):
            k = k * m + np.searchsorted(u, np.asarray(v, np.int64))
        return k
    return pack, uniq


def build(d, routes, rc, via_res=None, nets=None):
    """The parasitic networks of every routed net. Returns a dict of numpy arrays:
    node_net (N,), node_cap (N,) wire capacitance (fF) at each node; res_a, res_b (R,) node ids, res_r (R,) kohm;
    pin_node (P,) the node of each pin (-1: its net is not routed, or the pin has no route to it).
    nets: a bool mask (E,) to build only those nets' networks (the others get no node, wire_cap 0); a net's network,
    and so its reduction, is bit for bit the one it has in the full build (its nodes keep their relative order)."""
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
    if nets is not None:
        nets = np.asarray(nets, bool)
        routed = routed & nets
        sel = nets[sn]
        s, sn = s[sel], sn[sel]
    # route points: per (net, gcell center) holding pins of the net, the median of their positions
    pp = routes.pin_pt; pg = routes.pin_grid
    has = routed[p_net] & (pg[:, 0] >= 0) & (pp[:, 0] >= 0)
    hp = np.nonzero(has)[0]
    # nodes: segment ends on valid layers (or any via end)
    via = (s[:, 0] == s[:, 3]) & (s[:, 1] == s[:, 4]) & (s[:, 2] != s[:, 5])
    planar = s[:, 2] == s[:, 5]
    keep = ((s[:, 2] >= routes.min_layer) | via) & ((s[:, 5] >= routes.min_layer) | via)
    ks = np.nonzero(keep)[0]
    pr = np.nonzero(has & (routes.pin_layer >= 0))[0]            # pins that connect to their net's route
    # keys (net, x, y[, layer]) packed into one int64, ordered as the records (net first): sized by every value
    # they will hold
    gpack, _ = _packer((p_net[hp], sn), (pg[hp, 0], s[:, 0], s[:, 3]), (pg[hp, 1], s[:, 1], s[:, 4]))
    npack, nuq = _packer((sn[ks], p_net[pr]), (s[ks, 0], s[ks, 3], pg[pr, 0]), (s[ks, 1], s[ks, 4], pg[pr, 1]),
                         (s[ks, 2], s[ks, 5], routes.pin_layer[pr] + 1))
    ug, ginv = np.unique(gpack(p_net[hp], pg[hp, 0], pg[hp, 1]), return_inverse=True)
    med = np.zeros((len(ug), 2), np.int64)
    order = np.lexsort((pp[hp, 0], ginv)); gs = ginv[order]
    cnt = np.bincount(ginv, minlength=len(ug)); start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    med[:, 0] = pp[hp, 0][order][start + cnt // 2]                  # nth_element at size / 2
    order = np.lexsort((pp[hp, 1], ginv))
    med[:, 1] = pp[hp, 1][order][start + cnt // 2]
    del gs

    def route_point(net, x, y):
        k = gpack(net, x, y)
        i = np.searchsorted(ug, k)
        ok = (i < len(ug))
        i2 = np.minimum(i, max(len(ug) - 1, 0))
        ok &= (ug[i2] == k) if len(ug) else np.zeros(len(k), bool)
        X = np.where(ok, med[i2, 0] if len(ug) else x, x); Y = np.where(ok, med[i2, 1] if len(ug) else y, y)
        return X, Y

    ends = npack(np.concatenate([sn[ks], sn[ks]]), np.concatenate([s[ks, 0], s[ks, 3]]),
                 np.concatenate([s[ks, 1], s[ks, 4]]), np.concatenate([s[ks, 2], s[ks, 5]]))
    nodes, ninv = np.unique(ends, return_inverse=True)
    nodes_net = nuq[0][nodes // (max(len(nuq[1]), 1) * max(len(nuq[2]), 1) * max(len(nuq[3]), 1))]
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
    cl = routes.pin_layer[pr]

    def find(net, x, y, l):
        k = npack(net, x, y, l)
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
    node_net = np.concatenate([nodes_net, p_net[pr]])
    node_cap = np.concatenate([node_cap, pcap / 2])
    np.add.at(node_cap, gnode, pcap / 2)
    res_a = np.concatenate([na, pnode]); res_b = np.concatenate([nb, gnode]); res_r = np.concatenate([r_seg, pres])
    return {'node_net': node_net, 'node_cap': node_cap, 'res_a': res_a, 'res_b': res_b, 'res_r': res_r,
            'pin_node': pin_node, 'pins_missing': int((has & (routes.pin_layer >= 0)).sum() - len(pr)),
            'wire_cap': np.bincount(node_net, node_cap, minlength=len(a['n_ptr']) - 1)}


def _mix(x):
    """splitmix64 of uint64 values (wrapping arithmetic)."""
    with np.errstate(over='ignore'):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _row_hash(cols):
    h = np.zeros(len(cols[0]), np.uint64)
    with np.errstate(over='ignore'):
        for c in cols:
            h = _mix(h ^ np.asarray(c).astype(np.int64).astype(np.uint64))
    return h


def net_signatures(d, routes):
    """Per net (E,) a 64-bit hash of what its network depends on: its segments in their order and its pins' positions,
    gcells and layers. Two route sets give a net the same network (and the same reduction at the same pin
    capacitances) where the signatures agree, barring a hash collision."""
    a = d.raw
    E = len(a['n_ptr']) - 1
    s, sn = routes.seg, routes.seg_net.astype(np.int64)
    o = np.argsort(sn, kind='stable')
    sno = sn[o]
    first = np.searchsorted(sno, sno)
    rank = np.arange(len(sno)) - first                          # position of the segment within its net
    hs = _row_hash([s[o, k] for k in range(s.shape[1])] + [rank])
    seg_h = np.zeros(E, np.uint64)
    if len(sno):
        starts = np.nonzero(np.r_[True, sno[1:] != sno[:-1]])[0]
        with np.errstate(over='ignore'):
            seg_h[sno[starts]] = np.add.reduceat(hs, starts)
    p_net = np.repeat(np.arange(E), np.diff(a['n_ptr']))
    hp = _row_hash([np.arange(len(p_net)), routes.pin_pt[:, 0], routes.pin_pt[:, 1], routes.pin_grid[:, 0],
                    routes.pin_grid[:, 1], routes.pin_layer])
    pin_h = np.zeros(E, np.uint64)
    if len(p_net):
        starts = np.asarray(a['n_ptr'][:-1], np.int64)
        nz = np.diff(a['n_ptr']) > 0
        with np.errstate(over='ignore'):
            pin_h[nz] = np.add.reduceat(hp, starts[nz])
    cnt = np.bincount(sn, minlength=E).astype(np.uint64)
    return _mix(seg_h ^ _mix(pin_h ^ _mix(cnt)))


def loop_nets(net, E):
    """Per net (E,): its network has a resistor loop (more resistors than nodes less one)."""
    nn = np.bincount(net['node_net'], minlength=E)
    nr = np.bincount(net['node_net'][net['res_a']], minlength=E)
    return (nr > nn - 1) & (nn > 0)


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
    # breadth-first: the nodes of one depth are a contiguous run of `order`, and pp never decreases along it, so the
    # depth runs are found by searching pp (a tree routed through many gcells is thousands of nodes deep: a pass over
    # all nodes per depth took a minute on a 300k-cell design)
    if len(pp) > 1 and bool((pp[1:] < pp[:-1]).any()):
        raise AssertionError('grpar.reduce: the traversal is not breadth-first')
    bounds = [0, int(np.searchsorted(pp, 0, 'left'))]          # depth 0: the driver nodes (pp == -1)
    while bounds[-1] < len(order):
        bounds.append(int(np.searchsorted(pp, bounds[-1], 'left')))
    runs = list(zip(bounds[:-1], bounds[1:]))
    for a, b in reversed(runs):                                 # children before parents
        c = order[a:b]; p_ = par[a:b]; r = r_edge[a:b][:, None]
        a1, a2, a3 = y1[c], y2[c], y3[c]
        np.add.at(y1, p_, a1)
        np.add.at(y2, p_, a2 - r * a1 * a1)
        np.add.at(y3, p_, a3 - 2 * r * a1 * a2 + r * r * a1 ** 3)
    # Elmore, parents first
    el = np.zeros((N + 1, V))
    for a, b in runs[1:]:
        c = order[a:b]
        el[c] = el[par[a:b]] + r_edge[a:b][:, None] * y1[c]
    elm[pn] = el[pin_node[pn]]
    # the pi model at each driver node
    Y1, Y2, Y3 = y1[dn[roots]], y2[dn[roots]], y3[dn[roots]]
    capl = (Y2 == 0) & (Y3 == 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        c1 = np.where(capl, 0.0, Y2 * Y2 / np.where(capl, 1.0, Y3))
        rpi = np.where(capl, 0.0, -Y3 * Y3 / np.where(capl, 1.0, Y2 ** 3))
    pi[roots] = np.stack([Y1 - c1, rpi, c1], -1)
    return elm, pi, loops
