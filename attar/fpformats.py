"""Readers for block-level floorplanning benchmarks -> Attar IR (no rows, no technology: tier C).

YAL (MCNC: apte, xerox, hp, ami33, ami49)
    Modules with a polygon outline and pins at given points; the PARENT module's outline is the die, its IOLIST the
    pads (on the outline), its NETWORK the instances, each listing one signal per module pin in IOLIST order.
    Facts: netlist, cell_geometry (rectilinear modules become their bounding box: counted in the manifest),
    pin_offsets, floorplan.die, floorplan.io. The blocks are movable macros, not placed: floorplan.macros is missing,
    so macro placement is the question. Power pins (PWR) and their nets are left out.
GSRC / UCSC blocks (.blocks .nets .pl, optional .fixeds outline; GSRC n10..n300, MCNC soft_modified)
    Hard blocks have a fixed shape; soft blocks only an area and an aspect-ratio range. Nets name blocks without pin
    offsets: the format connects at block centers (pin_offsets with that provenance). The .pl locations of the
    terminals are the pads (floorplan.io); block locations in the .pl are kept as the 'reference' snapshot.
    A design with soft blocks lacks cell_geometry (their shape is a decision, not a fact): it stores them as
    sqrt(area) squares only to have numbers in the arrays, plus the area and aspect range (soft_area, soft_aspect).
    No die unless a .fixeds outline is given.
"""
import os
import time

import numpy as np

from .bookshelf import _lines, read_blocks
from .distill import (FIXED, MACRO, PLACED, TERMINAL, M_BLOCK, SCHEMA, IO_DIR, source_ref, save_arrays,
                      save_names_manifest)


def _write(out_dir, design, fmt, sources, dbu, units, names, size, flags, xy, p_cell, p_off, nets, io_dir,
           die=None, facts=None, extra_arrays=None, extra=None):
    """Write the IR. Cells are lower-left based; IO pins are the zero-size cells flagged TERMINAL (pins with p_off
    None). nets: [(name, [pin index, ...])]. Every non-IO cell is its own master with its own master pins."""
    N = len(names)
    size = np.asarray(size, np.int64).reshape(-1, 2); flags = np.asarray(flags, np.uint8); xy = np.asarray(xy, np.int64).reshape(-1, 2)
    term = (flags & TERMINAL) != 0
    c_master = np.full(N, -1, np.int32)
    cells = np.nonzero(~term)[0]
    c_master[cells] = np.arange(len(cells))
    # pins in net order
    order = [k for _, pins in nets for k in pins]
    n_ptr = np.cumsum([0] + [len(p) for _, p in nets]).astype(np.int64)
    pc = np.array([p_cell[k] for k in order], np.int64)
    # master pins: one per non-IO pin, grouped by master
    is_cell_pin = ~term[pc] if len(pc) else np.zeros(0, bool)
    mp = np.nonzero(is_cell_pin)[0]
    mp = mp[np.argsort(c_master[pc[mp]], kind='stable')]
    M = len(cells)
    m_pin_ptr = np.zeros(M + 1, np.int64); np.add.at(m_pin_ptr, c_master[pc[mp]] + 1, 1); m_pin_ptr = np.cumsum(m_pin_ptr)
    off = np.array([p_off[order[k]] for k in mp], np.int64).reshape(-1, 2)
    m_pin_box = np.concatenate([off, off], 1).astype(np.int32)
    p_mpin = np.full(len(pc), -1, np.int64); p_mpin[mp] = np.arange(len(mp))
    io_idx = np.nonzero(term)[0]
    core = {
        'm_size': size[cells].astype(np.int32), 'm_class': np.full(M, M_BLOCK, np.uint8),
        'm_pin_ptr': m_pin_ptr.astype(np.int32), 'm_pin_box': m_pin_box, 'm_pin_use': np.zeros(len(mp), np.uint8),
        'c_master': c_master, 'c_xy': xy.astype(np.int32), 'c_orient': np.zeros(N, np.uint8), 'c_flags': flags,
        'p_cell': pc.astype(np.int32), 'p_mpin': p_mpin.astype(np.int32),
        'n_ptr': n_ptr, 'n_flags': np.zeros(len(nets), np.uint8), 'n_weight': np.ones(len(nets), np.float32),
        'c_die': np.zeros(N, np.uint8), 'io_cell': io_idx.astype(np.int32), 'io_dir': np.asarray(io_dir, np.uint8),
    }
    if die is not None:
        core['die_box'] = np.array([die], np.int32)
    core.update(extra_arrays or {})
    save_arrays(out_dir, core)
    movable = (flags & (FIXED | TERMINAL)) == 0
    deg = np.diff(n_ptr)
    manifest = {
        'schema': SCHEMA, 'design': design,
        'source': {'format': fmt, 'files': {k: source_ref(v) for k, v in sources.items()}},
        'dbu': dbu, 'units': units, 'dies': [{'name': 'die0', 'tech_profile': None}],
        'facts': facts, 'fidelity': 'F3', 'reference_placement': None,
        'placed': bool(((flags[movable] & PLACED) != 0).all()) if movable.any() else True,
        'counts': {'cells': int(len(cells)), 'movable': int(movable.sum()), 'macros': int(((flags & MACRO) != 0).sum()),
                   'physical_only': 0, 'terminals': int(len(io_idx)), 'nets': len(nets), 'nets_2plus': int((deg >= 2).sum()),
                   'pins': int(len(pc)), 'rows': 0, 'masters': int(M)},
    }
    manifest.update(extra or {})
    save_names_manifest(out_dir, {'design': design, 'cells': list(names), 'nets': [n for n, _ in nets],
                                  'masters': [names[i] for i in cells], 'master_pins': None, 'layers': []}, manifest)
    return manifest


# -- YAL ------------------------------------------------------------------------------------------------------------
def _yal_statements(path):
    """-> list of statements (token lists), ';'-terminated, comments /* */ removed."""
    with open(path, errors='replace') as f:
        text = f.read()
    while '/*' in text:
        a = text.index('/*'); b = text.find('*/', a)
        text = text[:a] + text[b + 2 if b >= 0 else len(text):]
    return [s.split() for s in text.split(';') if s.strip()]


def read_yal(path):
    """-> modules {name: {'type', 'poly' (k, 2), 'pins': [(name, kind, x, y)], 'network': [(inst, module, signals)]}}."""
    mods, cur, section = {}, None, None
    for t in _yal_statements(path):
        key = t[0].upper()
        if key == 'MODULE':
            cur = {'type': None, 'poly': None, 'pins': [], 'network': []}; mods[t[1]] = cur; section = None
        elif key == 'ENDMODULE':
            cur = None
        elif cur is None:
            continue
        elif key == 'TYPE':
            cur['type'] = t[1].upper()
        elif key == 'DIMENSIONS':
            cur['poly'] = np.array([float(v) for v in t[1:]]).reshape(-1, 2)
        elif key == 'IOLIST':
            section = 'io'
        elif key == 'NETWORK':
            section = 'net'
        elif key in ('ENDIOLIST', 'ENDNETWORK'):
            section = None
        elif section == 'io':
            cur['pins'].append((t[0], t[1].upper(), float(t[2]) if len(t) > 2 else 0.0, float(t[3]) if len(t) > 3 else 0.0))
        elif section == 'net':
            cur['network'].append((t[0], t[1], t[2:]))
    return mods


def distill_yal(path, out_dir, extra=None):
    t0 = time.time()
    mods = read_yal(path)
    parents = [m for m, v in mods.items() if v['type'] == 'PARENT']
    if len(parents) != 1:
        raise ValueError('%s: expected one PARENT module, found %d' % (path, len(parents)))
    top = mods[parents[0]]
    names, size, flags, xy, p_cell, p_off, io_dir = [], [], [], [], [], [], []
    net_pins, net_order = {}, []
    rectilinear = 0

    def connect(sig, k):
        if sig not in net_pins:
            net_pins[sig] = []; net_order.append(sig)
        net_pins[sig].append(k)

    power = {p[0] for p in top['pins'] if p[1] == 'PWR'}
    for inst, mname, sigs in top['network']:
        m = mods[mname]
        lo, hi = m['poly'].min(0), m['poly'].max(0)
        rectilinear += len(m['poly']) > 4
        i = len(names)
        names.append(inst); size.append(hi - lo); flags.append(MACRO); xy.append((0, 0))
        for (pname, kind, px, py), sig in zip(m['pins'], sigs):
            if kind == 'PWR' or sig in power:
                continue
            p_cell.append(i); p_off.append((px - lo[0], py - lo[1])); connect(sig, len(p_cell) - 1)
    for pname, kind, px, py in top['pins']:                  # pads: the parent's IO list, named by their signal
        if kind == 'PWR':
            continue
        i = len(names)
        names.append(pname); size.append((0, 0)); flags.append(TERMINAL | FIXED | PLACED); xy.append((px, py))
        io_dir.append({'PI': IO_DIR['INPUT'], 'PO': IO_DIR['OUTPUT']}.get(kind, IO_DIR['INOUT']))
        p_cell.append(i); p_off.append(None); connect(pname, len(p_cell) - 1)
    lo, hi = top['poly'].min(0), top['poly'].max(0)
    nets = [(n, net_pins[n]) for n in net_order]
    sizes = np.round(np.array(size)).astype(np.int64)
    m = _write(out_dir, os.path.splitext(os.path.basename(path))[0].replace('.yal', ''), 'yal', {'yal': path}, 1,
               'YAL units', names, sizes, flags, np.round(np.array(xy)).astype(np.int64), p_cell,
               [None if o is None else (round(o[0]), round(o[1])) for o in p_off], nets, io_dir,
               die=[int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])],
               facts={'netlist': 'yal', 'cell_geometry': 'yal' if not rectilinear else 'yal (rectilinear modules as bounding boxes)',
                      'pin_offsets': 'yal', 'floorplan.die': 'yal(parent outline)', 'floorplan.io': 'yal(parent iolist)'},
               extra=dict({'macros_unplaced': len(top['network']), 'rectilinear_modules': rectilinear,
                           'distill_seconds': round(time.time() - t0, 2)}, **(extra or {})))
    return m


# -- GSRC / UCSC blocks ---------------------------------------------------------------------------------------------
def read_block_nets(path):
    """-> [(net name, [node names])] (pin offsets, if any, are ignored: the format connects at block centers)."""
    nets = []
    for line in _lines(path):
        if line.startswith(('NumNets', 'NumPins')):
            continue
        if line.startswith('NetDegree'):
            nets.append(('net%d' % len(nets), [])); continue
        nets[-1][1].append(line.split()[0])
    return nets


def distill_blocks(blocks_path, out_dir, nets_path=None, pl_path=None, fixeds_path=None, extra=None):
    t0 = time.time()
    base = os.path.splitext(blocks_path)[0]
    nets_path = nets_path or base + '.nets'
    pl_path = pl_path or (base + '.pl' if os.path.exists(base + '.pl') else None)
    fixeds_path = fixeds_path or (base + '.fixeds' if os.path.exists(base + '.fixeds') else None)
    blocks = read_blocks(blocks_path)
    bnames = [n for n, b in blocks.items() if b[0] != 'terminal']
    tnames = [n for n, b in blocks.items() if b[0] == 'terminal']
    soft = [n for n in bnames if blocks[n][0] == 'soft']
    pl = {}
    if pl_path:
        for line in _lines(pl_path):
            t = line.split()
            if len(t) >= 3:
                pl[t[0]] = (float(t[1]), float(t[2]))
    vals = [v for n in bnames for v in blocks[n][1:3] if blocks[n][0] != 'soft'] + [v for p in pl.values() for v in p]
    dbu = next((d for d in (1, 10, 100, 1000) if all(abs(v * d - round(v * d)) < 1e-6 for v in vals)), 1000)
    names, size, flags, xy, ref = [], [], [], [], []
    soft_area = []; soft_aspect = []
    for n in bnames:
        b = blocks[n]
        if b[0] == 'soft':
            s = np.sqrt(b[1]); size.append((s * dbu, s * dbu)); soft_area.append(b[1]); soft_aspect.append(b[2:4])
        else:
            size.append((b[1] * dbu, b[2] * dbu)); soft_area.append(0.0); soft_aspect.append((0.0, 0.0))
        names.append(n); flags.append(MACRO); xy.append((0, 0)); ref.append(pl.get(n, (0.0, 0.0)))
    io_dir = []
    for n in tnames:
        names.append(n); size.append((0, 0)); xy.append(pl.get(n, (0.0, 0.0))); ref.append(pl.get(n, (0.0, 0.0)))
        flags.append(TERMINAL | FIXED | (PLACED if n in pl else 0)); io_dir.append(IO_DIR['INOUT'])
    nid = {n: i for i, n in enumerate(names)}
    p_cell, p_off, nets = [], [], []
    size = np.round(np.array(size, np.float64)).astype(np.int64).reshape(-1, 2)
    for name, nodes in read_block_nets(nets_path):
        pins = []
        for nd in nodes:
            i = nid[nd]
            p_cell.append(i); pins.append(len(p_cell) - 1)
            p_off.append(None if flags[i] & TERMINAL else (int(size[i, 0] // 2), int(size[i, 1] // 2)))
        nets.append((name, pins))
    die = None
    if fixeds_path:
        t = [float(v) for line in _lines(fixeds_path) for v in line.split()[:2]][:2]
        die = [0, 0, int(round(t[0] * dbu)), int(round(t[1] * dbu))]
    hard_only = not soft
    facts = {'netlist': 'blocks'}
    if hard_only:
        rect = sum(1 for n in bnames if blocks[n][0] == 'hard_rectilinear')
        facts['cell_geometry'] = 'blocks' if not rect else 'blocks (rectilinear blocks as bounding boxes)'
        facts['pin_offsets'] = 'blocks(center: the format has no pin offsets)'
    if die is not None:
        facts['floorplan.die'] = 'fixeds(outline)'
    if tnames and all(n in pl for n in tnames):
        facts['floorplan.io'] = 'pl'
    arrays = {'soft_area': np.array(soft_area + [0.0] * len(tnames), np.float32),
              'soft_aspect': np.array(soft_aspect + [(0.0, 0.0)] * len(tnames), np.float32).reshape(-1, 2)}
    if pl_path and all(n in pl for n in bnames) and any(pl[n] != (0.0, 0.0) for n in bnames):
        arrays['ref_xy'] = np.round(np.array(ref) * dbu).astype(np.int32)
        arrays['ref_orient'] = np.zeros(len(names), np.uint8)
    srcs = {k: v for k, v in (('blocks', blocks_path), ('nets', nets_path), ('pl', pl_path), ('fixeds', fixeds_path)) if v}
    return _write(out_dir, os.path.basename(base), 'blocks', srcs, dbu, 'blocks units' + (' (stored as 1/%d)' % dbu if dbu > 1 else ''),
                  names, size, flags, np.round(np.array(xy, np.float64) * dbu).astype(np.int64), p_cell, p_off, nets, io_dir,
                  die=die, facts=facts, extra_arrays=arrays,
                  extra=dict({'macros_unplaced': len(bnames), 'not_stored': {'soft_blocks': len(soft)} if soft else {},
                              'reference_placement': 'pl' if 'ref_xy' in arrays else None,
                              'distill_seconds': round(time.time() - t0, 2)}, **(extra or {})))
