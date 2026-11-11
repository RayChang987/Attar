"""Distill a DEF + LEF design into the Attar Core IR: a folder of core.safetensors, names.json.gz,
tech_profile.safetensors and manifest.json.

Disk conventions (spec 5.1): integer DBU; cell location = lower-left corner (as in DEF); pin geometry stored
per master in its original (N) orientation, relative to the master's lower-left; the effective offset is computed
on load. Numbers and strings are separated: names only go to names.json.gz.
"""
import gzip
import hashlib
import json
import os
import time

import numpy as np

from . import st
from .deffile import ORIENT_ID, netlist_fingerprint, pg_shapes, read_def, split_frame, write_framed
from .lef import read_lef

SCHEMA = 'attar-ir/0.9'
PDN_LEVELS = 3      # PDN metal kept on the 3 lowest routing layers (OpenROAD dpl's kMaxPinLevel)
IO_DIR = {'INPUT': 0, 'OUTPUT': 1, 'INOUT': 2, 'FEEDTHRU': 3}
# pin direction as seen from the net (p_dir, m_pin_dir): a sink, a driver, both, or not stated by the source. An IO
# terminal stands for the outside world, so an INPUT port drives its net (D_OUT), as Bookshelf terminals do.
D_IN, D_OUT, D_INOUT, D_UNKNOWN = 0, 1, 2, 3
LEF_DIR = {'INPUT': D_IN, 'OUTPUT': D_OUT, 'INOUT': D_INOUT, 'FEEDTHRU': D_INOUT}
PORT_DIR = {'INPUT': D_OUT, 'OUTPUT': D_IN, 'INOUT': D_INOUT, 'FEEDTHRU': D_INOUT}

# c_flags bits
FIXED, MACRO, PLACED, TERMINAL, PHYSICAL, ASIDE = 1, 2, 4, 8, 16, 32
# ASIDE: a movable cell without connections (fill, spacer) set aside by Design.unplace: engines do not place it,
# it is no obstacle and no utilization; export writes it UNPLACED
# m_class / pin use / net flags
M_CORE, M_BLOCK, M_PAD, M_OTHER = 0, 1, 2, 3
U_SIGNAL, U_POWER, U_GROUND, U_CLOCK = 0, 1, 2, 3
N_CLOCK, N_POWER = 1, 2
_USE = {'SIGNAL': U_SIGNAL, 'POWER': U_POWER, 'GROUND': U_GROUND, 'CLOCK': U_CLOCK}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pin_dir_counts(p_dir, n_ptr):
    """Manifest counts for a pin direction array: pins per direction, and nets by their number of drivers."""
    p_dir = np.asarray(p_dir, np.uint8); n_ptr = np.asarray(n_ptr, np.int64)
    if not (p_dir != D_UNKNOWN).any():                # the source states no directions: nothing to count per net
        return {'unknown': int(len(p_dir))}
    drv = np.zeros(len(n_ptr) - 1, np.int64)
    if len(p_dir):
        np.add.at(drv, np.repeat(np.arange(len(n_ptr) - 1), np.diff(n_ptr)), (p_dir == D_OUT).astype(np.int64))
    deg = np.diff(n_ptr)
    return {'sinks': int((p_dir == D_IN).sum()), 'drivers': int((p_dir == D_OUT).sum()),
            'inout': int((p_dir == D_INOUT).sum()), 'unknown': int((p_dir == D_UNKNOWN).sum()),
            'nets_one_driver': int(((drv == 1) & (deg >= 2)).sum()), 'nets_no_driver': int(((drv == 0) & (deg >= 2)).sum()),
            'nets_multi_driver': int(((drv > 1) & (deg >= 2)).sum())}


def _sparse(values):
    """{index: text} as {'default': the most common text, 'except': {index: text}} (most cells and nets share one)."""
    from collections import Counter
    default = Counter(values.values()).most_common(1)[0][0] if values else ''
    return {'default': default, 'except': {str(k): v for k, v in values.items() if v != default}}


def _check_frame(d, frame, comps, owned_idx, cell_attrs, net_attrs, net_names, core, cell_names, mpin_flat, n_comps, out_dir):
    """Reassemble the source DEF from the frame and the IR (owned components at their source placement) and compare
    it with the source: '' if it is the same design, else what differs (then export needs a template)."""
    import tempfile
    from .deffile import comp_statement, net_statements, same_design
    ca = lambda i: cell_attrs['except'].get(str(i), cell_attrs['default'])
    na = lambda k: net_attrs['except'].get(str(k), net_attrs['default'])
    lines = [comp_statement(comps[i][0], comps[i][1], comps[i][2], comps[i][3], comps[i][4], comps[i][5], ca(i)) for i in owned_idx]
    is_term = np.arange(len(cell_names)) >= n_comps
    nets = net_statements(net_names, core['n_ptr'], core['p_cell'], core['p_mpin'], cell_names, is_term, mpin_flat, na)
    with tempfile.TemporaryDirectory(dir=out_dir) as t:
        write_framed(frame, os.path.join(t, 'x.def'), lines, nets)
        return same_design(d, read_def(os.path.join(t, 'x.def')))


def save_arrays(out_dir, core, tp=None):
    os.makedirs(out_dir, exist_ok=True)
    st.save(os.path.join(out_dir, 'core.safetensors'), core, {'schema': SCHEMA})
    if tp is not None:
        st.save(os.path.join(out_dir, 'tech_profile.safetensors'), tp, {'schema': SCHEMA})


def source_ref(path):
    """How a manifest records a source file: its name and sha256, never a path (the IR is self-contained; a file
    is handed over again only where it is used, e.g. the DEF Design.export stamps the design onto)."""
    return {'name': os.path.basename(path), 'sha256': _sha256(path)}


def save_names_manifest(out_dir, names, manifest):
    with gzip.open(os.path.join(out_dir, 'names.json.gz'), 'wt') as f:
        json.dump(names, f, separators=(',', ':'))
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)


def distill(def_path, lef_paths, out_dir, lib_paths=(), extra=None, reference=None, facts=None):
    """reference: if given (a label such as 'orfs 3_3 global_place'), the movable cells' locations in the DEF are moved
    to ref_xy / ref_orient and the cells are stored unplaced, so engines start without the source placement."""
    t0 = time.time()
    macros, tech = read_lef(lef_paths)
    d = read_def(def_path)
    dbu = d['dbu']
    q = lambda v: int(round(v * dbu))

    # masters actually used
    used = sorted({c[1] for c in d['comps']})
    missing = [m for m in used if m not in macros]
    if missing:
        raise ValueError('masters not in the LEF files: %s' % ', '.join(missing[:10]))
    mid = {m: i for i, m in enumerate(used)}
    m_size = np.array([[q(macros[m].w), q(macros[m].h)] for m in used], np.int32).reshape(-1, 2)
    cls = lambda c: M_CORE if c == 'CORE' else M_BLOCK if c == 'BLOCK' else M_PAD if c == 'PAD' else M_OTHER
    m_class = np.array([cls(macros[m].cls) for m in used], np.uint8)
    # LEF SYMMETRY as bits: 1 X (may mirror about the x axis), 2 Y (about the y axis: N <-> FN), 4 R90
    m_sym = np.array([(1 if 'X' in macros[m].sym else 0) | (2 if 'Y' in macros[m].sym else 0) | (4 if 'R90' in macros[m].sym else 0)
                      for m in used], np.uint8)
    # routing layers (the tech profile's order); each master pin's layer is an index into them (-1: not a routing layer)
    layers = [l for l in tech['layers'] if 'pitch' in l]
    lid = {l['name']: i for i, l in enumerate(layers)}
    m_pin_ptr = [0]; m_pin_box = []; m_pin_use = []; m_pin_layer = []; m_pin_dir = []; mpin_names = []; mpin_id = {}
    for m in used:
        names = sorted(macros[m].pins)
        for p in names:
            box, use = macros[m].pins[p]
            mpin_id[(m, p)] = len(m_pin_box)
            m_pin_box.append([q(v) for v in box]); m_pin_use.append(_USE.get(use, U_SIGNAL))
            m_pin_layer.append(lid.get(macros[m].pin_layer.get(p), -1))
            m_pin_dir.append(LEF_DIR.get(macros[m].pin_dir.get(p), D_UNKNOWN))
        m_pin_ptr.append(len(m_pin_box)); mpin_names.append(names)

    # cells: components, then IO pins as zero-size fixed terminals
    comps = d['comps']; pins = d['pins']
    N = len(comps) + len(pins)
    c_master = np.full(N, -1, np.int32); c_xy = np.zeros((N, 2), np.int32)
    c_orient = np.zeros(N, np.uint8); c_flags = np.zeros(N, np.uint8)
    cell_names = []; cid = {}
    for i, (name, master, status, x, y, o) in enumerate(comps):
        cid[name] = i; cell_names.append(name)
        c_master[i] = mid[master]; c_xy[i] = (x, y); c_orient[i] = ORIENT_ID.get(o, 0)
        f = 0
        if status in ('FIXED', 'COVER'): f |= FIXED
        if status != 'UNPLACED': f |= PLACED
        if m_class[mid[master]] == M_BLOCK:
            f |= MACRO
        if m_class[mid[master]] != M_CORE and status == 'PLACED':
            f |= FIXED                      # a placed macro, pad, corner, endcap is part of the floorplan: frozen
        c_flags[i] = f
    tid = {}
    io_dir = np.array([IO_DIR.get(p[5], 2) for p in pins], np.uint8)
    io_unplaced = sum(1 for p in pins if p[2] == 'UNPLACED')  # IOs not placed yet: no floorplan.io
    macros_unplaced = sum(1 for c in comps if c[2] == 'UNPLACED' and macros[c[1]].cls == 'BLOCK')  # no floorplan.macros
    for j, (name, net, status, x, y, _) in enumerate(pins):
        i = len(comps) + j
        tid[name] = i; cell_names.append(name)
        c_xy[i] = (x, y)
        c_flags[i] = TERMINAL | FIXED | (PLACED if status != 'UNPLACED' else 0)

    # nets (CSR over pins, ordered by net)
    p_cell = []; p_mpin = []; p_dir = []; n_ptr = [0]; n_flags = []; net_names = []
    port_dir = {p[0]: PORT_DIR.get(p[5], D_INOUT) for p in pins}
    skipped = 0; has_conn = np.zeros(N, bool)
    for name, conns, use in d['nets']:
        if use in ('POWER', 'GROUND'):
            continue
        for comp, pin in conns:
            if comp == 'PIN':
                i = tid.get(pin)
                if i is None: skipped += 1; continue
                p_cell.append(i); p_mpin.append(-1); p_dir.append(port_dir[pin])
            else:
                i = cid.get(comp)
                k = mpin_id.get((comps[i][1], pin)) if i is not None else None
                if k is None: skipped += 1; continue
                p_cell.append(i); p_mpin.append(k); p_dir.append(m_pin_dir[k])
            has_conn[i] = True
        n_ptr.append(len(p_cell)); net_names.append(name)
        n_flags.append(N_CLOCK if use == 'CLOCK' else 0)
    c_flags[~has_conn & (c_flags & (MACRO | TERMINAL) == 0)] |= PHYSICAL

    # rows
    site = lambda s: tech['sites'].get(s, (0.0, 0.0))
    rows = d['rows']
    r_xy = np.array([[r[1], r[2]] for r in rows], np.int32).reshape(-1, 2)
    r_sites = np.array([r[4] * r[5] for r in rows], np.int32)
    r_site_w = np.array([r[6] if r[6] else q(site(r[0])[0]) for r in rows], np.int32)
    r_h = np.array([q(site(r[0])[1]) for r in rows], np.int32)
    r_orient = np.array([ORIENT_ID.get(r[3], 0) for r in rows], np.uint8)

    ref = {}
    if reference:
        mov = (c_flags & (FIXED | TERMINAL)) == 0
        ref = {'ref_xy': c_xy.copy(), 'ref_orient': c_orient.copy()}
        c_xy[mov] = 0; c_orient[mov] = 0; c_flags[mov] &= ~np.uint8(PLACED)
    core = {
        'm_size': m_size, 'm_class': m_class, 'm_pin_ptr': np.array(m_pin_ptr, np.int32),
        'm_pin_box': np.array(m_pin_box, np.int32).reshape(-1, 4), 'm_pin_use': np.array(m_pin_use, np.uint8),
        'c_master': c_master, 'c_xy': c_xy, 'c_orient': c_orient, 'c_flags': c_flags,
        'p_cell': np.array(p_cell, np.int32), 'p_mpin': np.array(p_mpin, np.int32),
        'n_ptr': np.array(n_ptr, np.int64), 'n_flags': np.array(n_flags, np.uint8),
        'n_weight': np.ones(len(net_names), np.float32),
        'r_xy': r_xy, 'r_sites': r_sites, 'r_site_w': r_site_w, 'r_h': r_h, 'r_orient': r_orient,
        # dies: a 2D design is one die (DEF DIEAREA); 3D designs add dies, and rows/cells carry their die id
        'die_box': np.array([d['die']], np.int32).reshape(-1, 4),
        'r_die': np.zeros(len(rows), np.uint8),
        'c_die': np.zeros(N, np.uint8),
        'io_cell': np.arange(len(comps), N, dtype=np.int32),
        'io_dir': io_dir,
    }
    # routing tracks of the DEF (pin access: macro pins must sit on them), one row per (axis, layer)
    trk = [t for t in d['tracks'] if t[4] in lid]
    core.update({'m_pin_layer': np.array(m_pin_layer, np.int16), 'm_pin_dir': np.array(m_pin_dir, np.uint8), 'm_sym': m_sym,
                 'p_dir': np.array(p_dir, np.uint8),
                 'tr_axis': np.array([0 if t[0] == 'X' else 1 for t in trk], np.uint8),
                 'tr_start': np.array([t[1] for t in trk], np.int32), 'tr_num': np.array([t[2] for t in trk], np.int32),
                 'tr_step': np.array([t[3] for t in trk], np.int32), 'tr_layer': np.array([lid[t[4]] for t in trk], np.int16),
                 'tr_die': np.zeros(len(trk), np.uint8)})
    core.update(ref)
    tp = {'layer_dir': np.array([0 if l.get('direction') == 'HORIZONTAL' else 1 for l in layers], np.uint8),
          'layer_pitch': np.array([q(l['pitch']) for l in layers], np.int32),
          'layer_width': np.array([q(l.get('width', 0)) for l in layers], np.int32),
          'layer_offset': np.array([q(l.get('offset', 0)) for l in layers], np.int32),
          'layer_spacing': np.array([q(l.get('spacing', 0)) for l in layers], np.int32)}
    # PDN metal on the lowest routing layers (the pin layers OpenROAD's detailed placer checks): wires, vias, fill
    with open(def_path, errors='replace') as f:
        text = f.read()
    pg = [s_ for s_ in pg_shapes(text, tech.get('vias', {}), dbu) if s_[0] in lid and lid[s_[0]] < PDN_LEVELS]
    pg_nets = sorted({s_[3] for s_ in pg})
    core.update({'pg_layer': np.array([lid[s_[0]] for s_ in pg], np.int16),
                 'pg_rect': np.array([s_[1] for s_ in pg], np.int32).reshape(-1, 4),
                 'pg_kind': np.array([s_[2] for s_ in pg], np.uint8),
                 'pg_net': np.array([pg_nets.index(s_[3]) for s_ in pg], np.uint8)})

    save_arrays(out_dir, core, tp)
    # the frame (deffile.split_frame): the source DEF minus the components Attar may move and the signal nets, kept
    # verbatim (PDN, IO pin shapes, fixed macros, ...); Design.export writes those back around it, from the IR
    owned_idx = [i for i in range(len(comps)) if not (c_flags[i] & FIXED)]
    frame, comp_attr, net_attr = split_frame(def_path, {comps[i][0] for i in owned_idx})
    cell_attrs = _sparse({i: comp_attr.get(comps[i][0], '') for i in owned_idx})
    net_attrs = _sparse({k: net_attr.get(n, '') for k, n in enumerate(net_names)})
    mpin_flat = [p for names in mpin_names for p in names]
    frame_bad = _check_frame(d, frame, comps, owned_idx, cell_attrs, net_attrs, net_names, core, cell_names, mpin_flat,
                             len(comps), out_dir)
    if not frame_bad:
        with gzip.open(os.path.join(out_dir, 'frame.def.gz'), 'wt') as f:
            f.write(frame)
    elif os.path.exists(os.path.join(out_dir, 'frame.def.gz')):
        os.remove(os.path.join(out_dir, 'frame.def.gz'))
    with gzip.open(os.path.join(out_dir, 'names.json.gz'), 'wt') as f:
        json.dump({'design': d['design'], 'cells': cell_names, 'nets': net_names, 'masters': used,
                   'master_pins': mpin_names, 'layers': [l['name'] for l in layers],
                   'cell_attrs': cell_attrs, 'net_attrs': net_attrs, 'pg_nets': pg_nets}, f, separators=(',', ':'))

    deg = np.diff(core['n_ptr'])
    movable = ((c_flags & (FIXED | TERMINAL)) == 0)
    manifest = {
        'schema': SCHEMA,
        'design': d['design'],
        'source': {'format': 'lefdef', 'def': source_ref(def_path), 'lef': [source_ref(p) for p in lef_paths],
                   'lib': [source_ref(p) for p in lib_paths]},
        'netlist_sha256': netlist_fingerprint(d),
        'dbu': dbu,
        'dies': [{'name': 'die0', 'tech_profile': 'tech_profile.safetensors'}],
        # facts this source provides, with provenance (spec 5.1); missing facts are absent, never filled in.
        # Placement blockages are counted but not stored yet, so 'constraints' is not claimed.
        'facts': dict([('netlist', 'def'), ('cell_geometry', 'lef'), ('pin_offsets', 'lef'), ('floorplan.die', 'def'),
                       ('cell_library', 'lef')] + ([('rows', 'def')] if rows else [])
                      + ([('tracks', 'def+lef')] if trk else [])
                      + ([('pin_direction', 'lef+def')] if any(v != D_UNKNOWN for v in m_pin_dir) else [])
                      + ([('def_frame', 'def')] if not frame_bad else [])
                      + ([('pdn', 'def')] if pg else [])
                      + ([('floorplan.io', 'def')] if io_unplaced == 0 else [])
                      + ([('floorplan.macros', 'def')] if macros_unplaced == 0 else [])),
        'io_unplaced': io_unplaced,
        'macros_unplaced': macros_unplaced,
        # counted, not stored yet, so 'constraints' is not claimed
        'not_stored': {k: int(v) for k, v in (('placement_blockages', d['blockages']), ('fence_regions', d['regions']),
                                              ('region_groups', d['groups'])) if v},
        'fidelity': 'F1' if lib_paths else 'F2',
        'reference_placement': reference,
        'placed': bool((c_flags[movable] & PLACED).all()) if movable.any() else True,
        'counts': {'cells': int(len(comps)), 'movable': int(movable.sum()), 'macros': int(((c_flags & MACRO) > 0).sum()),
                   'physical_only': int(((c_flags & PHYSICAL) > 0).sum()), 'terminals': int(len(pins)), 'pg_pins_skipped': int(d['pg_pins']),
                   'nets': int(len(net_names)), 'nets_2plus': int((deg >= 2).sum()), 'pins': int(len(p_cell)),
                   'rows': int(len(rows)), 'masters': int(len(used)), 'skipped_pin_refs': int(skipped)},
        'pin_direction': pin_dir_counts(p_dir, n_ptr),
        'def_frame': {'bytes': len(frame), 'check': frame_bad or 'ok'},
        'distill_seconds': round(time.time() - t0, 2),
    }
    # data-quality warnings, e.g. generated macro LEFs with pins outside the SIZE outline
    warnings = []
    for m in used:
        mac = macros[m]
        out = [p for p, (b, u) in mac.pins.items() if u not in ('POWER', 'GROUND')  # rails overhang by design
               and (b[0] < -1e-6 or b[1] < -1e-6 or b[2] > mac.w + 1e-6 or b[3] > mac.h + 1e-6)]
        if out:
            warnings.append('LEF %s: %d signal pin(s) outside the %gx%g outline (%s)' % (m, len(out), mac.w, mac.h, ', '.join(sorted(out)[:3])))
    if warnings:
        manifest['warnings'] = warnings
    if facts:  # facts the caller knows about the source, e.g. {'slow_loop': 'orfs:asap7'}
        manifest['facts'].update(facts)
    if extra:
        manifest.update(extra)
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    return manifest
