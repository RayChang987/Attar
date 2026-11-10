"""Bookshelf reader (ISPD 2005/2006, ICCAD 2004 style): .aux -> .nodes .nets .wts .pl .scl -> Attar IR.

Facts provided: netlist, cell_geometry, pin_offsets, floorplan.die, floorplan.io, floorplan.macros, rows. No cell library, no
slow loop: Bookshelf designs have no technology, so Attar is their end point (tier C).

Conventions: Bookshelf pin offsets are relative to the node center and may be half units, so coordinates are stored
in half units (dbu = 2 per Bookshelf unit) when that is exact, else in the smallest of 1/10, 1/100, 1/1000 unit that
is (or 1/1000, rounded: e.g. IBM-HB+ soft blocks are sqrt(area) squares; the rounding error goes to the manifest). .pl locations are lower-left corners. Movable nodes all at (0, 0) in the
.pl mean unplaced. Each node is its own master (Bookshelf has no cell library); its pins are zero-size boxes.
The die is the bounding box of the rows and the fixed nodes (Bookshelf has no die outline; the rows define the
placement region), recorded as provenance 'bookshelf(rows+fixed)'.
"""
import os
import time

import numpy as np

from .deffile import ORIENT_ID
from .distill import FIXED, MACRO, PLACED, TERMINAL, PHYSICAL, M_CORE, M_BLOCK, SCHEMA, save_arrays, save_names_manifest, source_ref, \
    D_IN, D_OUT, D_INOUT, D_UNKNOWN, IO_DIR, pin_dir_counts

DBU = 2  # half Bookshelf units (the default; see _pick_dbu)
DBUS = (2, 10, 100, 1000)


def _lines(path):
    with open(path, errors='replace') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line and not line.startswith(('UCLA', 'UCSC')):
                yield line


def read_aux(aux):
    d = os.path.dirname(os.path.abspath(aux))
    files = {}
    for line in _lines(aux):
        if ':' in line:
            for fn in line.split(':', 1)[1].split():
                files[os.path.splitext(fn)[1][1:]] = os.path.join(d, fn)
    return files


def read_nodes(path):
    names, w, h, term = [], [], [], []
    for line in _lines(path):
        if line.startswith(('NumNodes', 'NumTerminals')):
            continue
        t = line.split()
        names.append(t[0]); w.append(float(t[1])); h.append(float(t[2]))
        term.append(2 if len(t) > 3 and t[3] == 'terminal_NI' else 1 if len(t) > 3 and t[3] == 'terminal' else 0)
    return names, np.array(w), np.array(h), np.array(term, np.int8)


def read_pl(path, nid):
    """-> x, y (lower-left), orient ids, fixed flags (arrays over node ids); nodes missing from the file stay NaN."""
    n = len(nid)
    x = np.full(n, np.nan); y = np.full(n, np.nan); o = np.zeros(n, np.uint8); fx = np.zeros(n, bool)
    for line in _lines(path):
        t = line.split()
        i = nid.get(t[0])
        if i is None or len(t) < 3:
            continue
        x[i] = float(t[1]); y[i] = float(t[2])
        if ':' in t:
            k = t.index(':')
            if k + 1 < len(t):
                o[i] = ORIENT_ID.get(t[k + 1], 0)
        fx[i] = any(s.startswith('/FIXED') for s in t[3:])
    return x, y, o, fx


def read_nets(path, nid):
    """-> net names, net pin pointer (CSR), pin node ids, pin offsets (from the node center), pin directions."""
    names, ptr, node, ox, oy, dr = [], [0], [], [], [], []
    for line in _lines(path):
        if line.startswith('NetDegree'):
            t = [x for x in line.replace('NetDegree:', 'NetDegree :').split() if x != ':']
            names.append(t[2] if len(t) > 2 else 'net%d' % len(names))
            ptr.append(ptr[-1])
            continue
        if line.startswith(('NumNets', 'NumPins')):
            continue
        t = [x for x in line.split() if x != ':']           # names may contain ':' (Calypto: t1[6:6]_reg)
        if len(t) > 1 and t[1].endswith(':') and t[1][:-1] in ('I', 'O', 'B'):
            t[1] = t[1][:-1]
        node.append(nid[t[0]])
        dr.append(t[1] if len(t) > 1 and t[1] in ('I', 'O', 'B') else 'B')
        k = 2 if len(t) > 1 and t[1] in ('I', 'O', 'B') else 1
        ox.append(float(t[k]) if len(t) > k else 0.0); oy.append(float(t[k + 1]) if len(t) > k + 1 else 0.0)
        ptr[-1] += 1
    return names, np.array(ptr, np.int64), np.array(node, np.int64), np.array(ox), np.array(oy), dr


def read_wts(path, net_id):
    wt = np.ones(len(net_id), np.float32)
    if path and os.path.exists(path):
        for line in _lines(path):
            t = line.split()
            if len(t) >= 2 and t[0] in net_id:
                wt[net_id[t[0]]] = float(t[1])
    return wt


def read_scl(path):
    rows, cur = [], None
    for line in _lines(path):
        t = line.replace(':', ' ').split()
        if t[0] == 'CoreRow':
            cur = {}
        elif t[0] == 'End' and cur is not None:
            rows.append(cur); cur = None
        elif cur is not None:
            for k in range(0, len(t) - 1, 2):
                key = t[k].lower()                  # Numsites / NumSites, Sitewidth / SiteWidth
                try:
                    cur[key] = float(t[k + 1])
                except ValueError:                  # Siteorient : N, Sitesymmetry : Y
                    cur[key] = t[k + 1]
    return rows


def q(v, dbu=DBU, strict=True):
    """Bookshelf units -> integer DBU; with strict, refuses anything finer than 1/dbu unit."""
    a = np.asarray(v, np.float64) * dbu
    r = np.round(a)
    if strict and np.abs(a - r).max(initial=0) > 1e-6:
        raise ValueError('coordinates finer than 1/%d unit' % dbu)
    return r.astype(np.int64)


def _pick_dbu(*arrays):
    """The smallest DBU in DBUS that stores every value exactly, else the largest (rounded). -> (dbu, max error)."""
    a = np.concatenate([np.asarray(x, np.float64).ravel() for x in arrays])
    a = a[np.isfinite(a)]
    for d in DBUS:
        err = np.abs(a * d - np.round(a * d)).max(initial=0) / d
        if err * d <= 1e-6:
            return d, 0.0
    return DBUS[-1], float(err)


def read_blocks(path):
    """Floorplanning .blocks: -> {name: ('soft', area, min_aspect, max_aspect) | ('hard', w, h) | ('terminal',)}.
    Hard rectilinear blocks with more than 4 vertices become their bounding box ('hard_rectilinear')."""
    out = {}
    for line in _lines(path):
        if line.startswith(('Num', 'UCSC')):
            continue
        t = line.replace(',', ' ').replace('(', ' ').replace(')', ' ').split()
        if len(t) < 2:
            continue
        kind = t[1].lower()
        if kind == 'softrectangular':
            out[t[0]] = ('soft', float(t[2]), float(t[3]), float(t[4]))
        elif kind == 'hardrectilinear':
            n = int(t[2]); v = np.array([float(x) for x in t[3:3 + 2 * n]]).reshape(n, 2)
            lo, hi = v.min(0), v.max(0)
            out[t[0]] = ('hard' if n == 4 else 'hard_rectilinear', hi[0] - lo[0], hi[1] - lo[1])
        elif kind == 'terminal':
            out[t[0]] = ('terminal',)
    return out


def distill_bookshelf(aux, out_dir, snapshots=None, extra=None):
    """Distill a Bookshelf design. snapshots: {name: .pl path} kept as named reference placements (e.g. a placer's
    result). Returns the manifest."""
    t0 = time.time()
    files = read_aux(aux)
    names, w, h, term = read_nodes(files['nodes'])
    nid = {n: i for i, n in enumerate(names)}
    N = len(names)
    x, y, o, fixed_pl = read_pl(files['pl'], nid)
    net_names, ptr, pnode, pox, poy, pdir = read_nets(files['nets'], nid)
    wts = read_wts(files.get('wts'), {n: i for i, n in enumerate(net_names)})
    rows = read_scl(files['scl'])
    row_h = np.array([r.get('height', 0) for r in rows])
    hmin = row_h.min() if len(row_h) else 0

    fixed = (term > 0) | fixed_pl
    # IO pins: zero-size terminals, and terminals entirely outside the rows (pads, e.g. GSRC ibm05)
    if len(rows):
        rx0 = min(r.get('subroworigin', 0) for r in rows); ry0 = min(r.get('coordinate', 0) for r in rows)
        rx1 = max(r.get('subroworigin', 0) + r.get('numsites', 0) * r.get('sitespacing', r.get('sitewidth', 1)) for r in rows)
        ry1 = max(r.get('coordinate', 0) + r.get('height', 0) for r in rows)
        xx, yy = np.nan_to_num(x), np.nan_to_num(y)
        outside = (xx + w <= rx0) | (xx >= rx1) | (yy + h <= ry0) | (yy >= ry1)
    else:
        outside = np.zeros(N, bool)
    io = (term > 0) & (((w == 0) & (h == 0)) | (outside & np.isfinite(x)))
    # macros: fixed blocks taller than a row, and movable nodes taller than 4 rows (mixed-size benchmarks: MMS,
    # IBM-HB+); shorter movable nodes are (multi-row) standard cells
    macro = ~io & (h > hmin + 1e-9) & (fixed | (h > 4 * hmin + 1e-9))
    movable = ~fixed
    movable_macros = int((macro & movable).sum())
    blocks = read_blocks(files['blocks']) if 'blocks' in files else {}
    soft = sum(1 for b in blocks.values() if b[0] == 'soft')
    unplaced = movable & (np.nan_to_num(x) == 0) & (np.nan_to_num(y) == 0)
    all_unplaced = bool(unplaced[movable].all()) if movable.any() else False

    flags = np.zeros(N, np.uint8)
    flags[fixed] |= FIXED; flags[macro] |= MACRO; flags[io] |= TERMINAL
    flags[~(all_unplaced & movable) & np.isfinite(x)] |= PLACED
    connected = np.zeros(N, bool); connected[pnode] = True
    flags[~connected & ~fixed] |= PHYSICAL

    # masters: one per node; master pins: the node's pins (zero-size boxes, lower-left based), grouped by node
    order = np.argsort(pnode, kind='stable')               # net-order pin -> sorted by node
    m_pin_ptr = np.zeros(N + 1, np.int64); np.add.at(m_pin_ptr, pnode + 1, 1); m_pin_ptr = np.cumsum(m_pin_ptr)
    dbu, dbu_err = _pick_dbu(w, h, pox, poy, x, y, row_h, [r.get('coordinate', 0) for r in rows],
                             [r.get('subroworigin', 0) for r in rows], [r.get('sitespacing', r.get('sitewidth', 1)) for r in rows])
    q = lambda v: globals()['q'](v, dbu, strict=dbu_err == 0)
    px = q(w[pnode] / 2 + pox); py = q(h[pnode] / 2 + poy)
    m_pin_box = np.stack([px, py, px, py], 1)[order].astype(np.int32)
    p_mpin = np.empty(len(pnode), np.int64); p_mpin[order] = np.arange(len(pnode))

    xy = np.stack([np.nan_to_num(x), np.nan_to_num(y)], 1)
    if all_unplaced:
        xy[movable] = 0
    c_xy = q(xy).astype(np.int32)
    r_xy = q(np.array([[r.get('subroworigin', 0), r.get('coordinate', 0)] for r in rows]).reshape(-1, 2)).astype(np.int32)
    r_site_w = q([r.get('sitespacing', r.get('sitewidth', 1)) for r in rows]).astype(np.int32)
    r_sites = np.array([int(r.get('numsites', 0)) for r in rows], np.int32)
    r_h = q(row_h).astype(np.int32)
    # die: bounding box of the rows and the fixed nodes
    bx = [r_xy[:, 0], r_xy[:, 1], r_xy[:, 0] + r_sites * r_site_w, r_xy[:, 1] + r_h]
    fx = np.nonzero(fixed & np.isfinite(x))[0]
    fw, fh = q(w[fx]), q(h[fx])
    die = [min(bx[0].min(initial=0), c_xy[fx, 0].min(initial=0)), min(bx[1].min(initial=0), c_xy[fx, 1].min(initial=0)),
           max(bx[2].max(initial=0), (c_xy[fx, 0] + fw).max(initial=0)), max(bx[3].max(initial=0), (c_xy[fx, 1] + fh).max(initial=0))]

    io_idx = np.nonzero(io)[0]
    # pin directions (I sink, O driver, B both, from the node's side; a terminal stands for the outside world)
    p_dir = np.array([{'I': D_IN, 'O': D_OUT}.get(v, D_INOUT) for v in pdir], np.uint8)
    has_dir = bool(len(p_dir)) and bool((p_dir != D_INOUT).any())        # all 'B': the file states no directions
    if not has_dir:
        p_dir[:] = D_UNKNOWN
    io_dir = np.full(len(io_idx), IO_DIR['INOUT'], np.uint8)          # a port: INPUT when it only drives its nets
    if has_dir and len(io_idx):
        tp = np.nonzero(io[pnode])[0]
        pos = {c: k for k, c in enumerate(io_idx.tolist())}
        drives = np.zeros(len(io_idx), np.int64); sinks = np.zeros(len(io_idx), np.int64)
        np.add.at(drives, [pos[c] for c in pnode[tp].tolist()], (p_dir[tp] == D_OUT).astype(np.int64))
        np.add.at(sinks, [pos[c] for c in pnode[tp].tolist()], (p_dir[tp] == D_IN).astype(np.int64))
        io_dir[(drives > 0) & (sinks == 0)] = IO_DIR['INPUT']; io_dir[(sinks > 0) & (drives == 0)] = IO_DIR['OUTPUT']
    core = {
        'm_size': np.stack([q(w), q(h)], 1).astype(np.int32), 'm_class': np.where(macro, M_BLOCK, M_CORE).astype(np.uint8),
        'm_pin_ptr': m_pin_ptr.astype(np.int32), 'm_pin_box': m_pin_box,
        'm_pin_use': np.zeros(len(pnode), np.uint8), 'm_pin_dir': p_dir[order], 'p_dir': p_dir,
        'c_master': np.arange(N, dtype=np.int32), 'c_xy': c_xy, 'c_orient': o, 'c_flags': flags,
        'p_cell': pnode.astype(np.int32), 'p_mpin': p_mpin.astype(np.int32),
        'n_ptr': ptr, 'n_flags': np.zeros(len(net_names), np.uint8), 'n_weight': wts,
        'r_xy': r_xy, 'r_sites': r_sites, 'r_site_w': r_site_w, 'r_h': r_h, 'r_orient': np.zeros(len(rows), np.uint8),
        'die_box': np.array([die], np.int32), 'r_die': np.zeros(len(rows), np.uint8), 'c_die': np.zeros(N, np.uint8),
        'io_cell': io_idx.astype(np.int32), 'io_dir': io_dir,
    }
    snap_meta = {}
    for sname, path in (snapshots or {}).items():
        sx, sy, so, _ = read_pl(path, nid)
        miss = ~np.isfinite(sx)
        sx[miss] = np.nan_to_num(x)[miss]; sy[miss] = np.nan_to_num(y)[miss]
        core['snap_%s_xy' % sname] = q(np.stack([sx, sy], 1)).astype(np.int32)
        core['snap_%s_orient' % sname] = so
        snap_meta[sname] = dict(source_ref(path), nodes_missing=int(miss.sum()))
    save_arrays(out_dir, core)

    deg = np.diff(ptr)
    src = {k: source_ref(v) for k, v in sorted(files.items()) if os.path.exists(v)}
    manifest = {
        'schema': SCHEMA, 'design': os.path.splitext(os.path.basename(aux))[0],
        'source': {'format': 'bookshelf', 'aux': source_ref(aux), 'files': src},
        'dbu': dbu, 'units': 'Bookshelf units (stored as 1/%d units%s)' % (dbu, ', rounded by up to %.2g' % dbu_err if dbu_err else ''),
        'dies': [{'name': 'die0', 'tech_profile': None}],
        'facts': dict([('netlist', 'bookshelf'),
                       ('cell_geometry', 'bookshelf' if not soft else 'bookshelf (soft blocks at their .nodes shape)'),
                       ('pin_offsets', 'bookshelf'), ('floorplan.die', 'bookshelf(rows+fixed)'), ('floorplan.io', 'bookshelf'),
                       ('rows', 'bookshelf')] + ([('floorplan.macros', 'bookshelf')] if not movable_macros else [])
                      + ([('pin_direction', 'bookshelf')] if has_dir else [])),
        'macros_unplaced': movable_macros,
        'not_stored': {'soft_blocks': soft} if soft else {},
        'fidelity': 'F3', 'reference_placement': None,
        'snapshots': snap_meta,
        'placed': not all_unplaced,
        'counts': {'cells': int(N - len(io_idx)), 'movable': int(movable.sum()), 'macros': int(macro.sum()),
                   'physical_only': int(((flags & PHYSICAL) > 0).sum()), 'terminals': int(len(io_idx)),
                   'fixed': int(fixed.sum()), 'nets': len(net_names), 'nets_2plus': int((deg >= 2).sum()),
                   'pins': int(len(pnode)), 'rows': len(rows), 'masters': int(N)},
        'pin_direction': pin_dir_counts(p_dir, ptr),
        'distill_seconds': round(time.time() - t0, 2),
    }
    if extra:
        manifest.update(extra)
    save_names_manifest(out_dir, {'design': manifest['design'], 'cells': names, 'nets': net_names, 'masters': names,
                                  'master_pins': None, 'layers': []}, manifest)
    return manifest
