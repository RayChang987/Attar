"""DEF write-back: a Design written into its source DEF's frame (fact 'def_frame').

The frame is everything of the source DEF but the components Attar may move and the signal nets, kept verbatim
(header, die, rows, tracks, IO pins with their shapes, PDN, blockages, fixed macros, ...). What an engine changed of
the floorplan (IO pins, die, rows, macros) is stamped onto the frame first; then every component Attar owns and every
signal net are written from the IR. Design.export, Design.pin_layers call these."""
import gzip
import os
import tempfile

import numpy as np
import torch

from . import st
from .deffile import ORIENTS, comp_statement, net_statements, write_def, write_framed
from .distill import ASIDE, FIXED, MACRO, TERMINAL


def export(d, out_def):
    """Write d as a DEF: the source DEF's frame with the design written into it from the IR: every component Attar
    owns (PLACED where placed, FIXED where an engine fixed it, UNPLACED otherwise or when set aside) and every signal
    net (connectivity; any wiring of the source is dropped: it is stale once cells move). Needs the facts netlist,
    cell_library and def_frame (LEF/DEF sources). Returns counts of what was written."""
    d.require('export')
    with gzip.open(os.path.join(d.path, 'frame.def.gz'), 'rt') as f:
        frame = f.read()
    dbu = d.dbu
    xy = torch.round(d.pos * dbu).to(torch.int64).cpu().numpy()
    o = d.orient.cpu().numpy(); fl = d.flags.cpu().numpy(); placed = d.placed.cpu().numpy()
    a = d.raw; names = d.names; cells = names['cells']
    n_comp = int(len(cells) - len(d.io)) if d.io is not None else len(cells)
    raw_fl = a['c_flags']
    owned = [i for i in range(n_comp) if not (raw_fl[i] & FIXED)]
    # what an engine changed of the floorplan goes onto the frame first (IO pins, die, rows, fixed macros)
    site = next((ln.split()[2] for ln in frame.splitlines() if ln.startswith('ROW ')), 'core')
    comps, pins, die, rows = floorplan_changes(d, dbu, d.tracks(), site)
    owned_names = {cells[i] for i in owned}
    comps = {k: v for k, v in comps.items() if k not in owned_names}   # owned macros are written below anyway
    done = {'pins': 0, 'die': 0, 'rows': 0, 'tracks': 0}
    if comps or pins or die is not None or rows is not None:
        with tempfile.TemporaryDirectory() as t:
            with open(os.path.join(t, 'in.def'), 'w') as fo:
                fo.write(frame)
            done = write_def(os.path.join(t, 'in.def'), os.path.join(t, 'out.def'), comps=comps, pins=pins, die=die,
                             rows=rows)
            with open(os.path.join(t, 'out.def')) as fi:
                frame = fi.read()
    ca = names.get('cell_attrs') or {'default': '', 'except': {}}
    na = names.get('net_attrs') or {'default': '', 'except': {}}
    masters = names['masters']; cm = a['c_master']
    lines = []
    for i in owned:
        if fl[i] & ASIDE or not placed[i]:
            state = 'UNPLACED'
        else:
            state = 'FIXED' if fl[i] & FIXED else 'PLACED'
        lines.append(comp_statement(cells[i], masters[cm[i]], state, int(xy[i, 0]), int(xy[i, 1]), ORIENTS[o[i]],
                                    ca['except'].get(str(i), ca['default'])))
    mpin_flat = [p for ms in names['master_pins'] for p in ms]
    is_term = (raw_fl & TERMINAL) != 0
    nets = net_statements(names['nets'], a['n_ptr'], a['p_cell'], a['p_mpin'], cells, is_term, mpin_flat,
                          lambda k: na['except'].get(str(k), na['default']))
    n = write_framed(frame, out_def, lines, nets)
    done.update({'components': sum(1 for ln in lines if '+ UNPLACED' not in ln), 'nets': n['NETS']})
    return done


def floorplan_changes(d, dbu, tracks, site):
    """What an engine changed of the floorplan, as write_def arguments: macros {name: (x, y, orient, 'FIXED')}, IO
    pins with their shapes, die and rows (None when unchanged)."""
    xy = torch.round(d.pos * dbu).to(torch.int64).cpu().numpy()
    o = d.orient.cpu().numpy(); fl = d.flags.cpu().numpy(); cells = d.names['cells']
    comps = {}
    if str(d.facts.get('floorplan.macros', '')).startswith('naive'):     # only what an engine changed
        comps = {cells[i]: (int(xy[i, 0]), int(xy[i, 1]), ORIENTS[o[i]], 'FIXED') for i in np.nonzero((fl & MACRO) != 0)[0]}
    pins = {}
    if str(d.facts.get('floorplan.io', '')).startswith('naive') and d.io is not None and len(d.io):
        pins = pin_shapes(d, xy, dbu, tracks)
    die = rows = None
    if str(d.facts.get('floorplan.die', '')).startswith('naive'):
        die = [int(round(v * dbu)) for v in d.die_box[0].tolist()]
        r_xy = torch.round(d.row_xy * dbu).long().tolist(); sw = torch.round(d.row_site_w * dbu).long().tolist()
        rows = [('ROW_%d' % k, site, r_xy[k][0], r_xy[k][1], ORIENTS[int(d.row_orient[k])], int(d.row_sites[k]), sw[k])
                for k in range(len(r_xy))]
    return comps, pins, die, rows


def pin_layers(d):
    """(horizontal, vertical) layer for IO pins: ORFS's IO_PLACER_H/V when known (asap7: M4, M5), else the first
    horizontal layer above the third and the first vertical one above it, from the tech profile."""
    if d.manifest.get('orfs', {}).get('platform') == 'asap7':
        return 'M4', 'M5'
    tp, _ = st.load(os.path.join(d.path, 'tech_profile.safetensors'))
    names = d.names['layers']; dr = tp['layer_dir']
    h = next(i for i in range(3, len(names)) if dr[i] == 0)
    v = next(i for i in range(h + 1, len(names)) if dr[i] == 1)
    return names[h], names[v]


def pin_shapes(d, xy, dbu, tracks):
    """IO pins as DEF pins, as ORFS's place_pins makes them: on the nearest die edge, a shape on the vertical layer
    (bottom, top) or horizontal layer (left, right), half of it inside the die, centered on a track of that layer
    (tracks: {(axis, layer): (start, step)} in DBU), at least 2 tracks from the next pin on the same edge."""
    tp, _ = st.load(os.path.join(d.path, 'tech_profile.safetensors'))
    names = d.names['layers']
    h, v = pin_layers(d)
    width = {n: int(tp['layer_width'][i]) for i, n in enumerate(names)}
    die = [int(round(t * dbu)) for t in d.die_box[0].tolist()]
    cells = d.names['cells']
    sides = {0: [], 1: [], 2: [], 3: []}
    for i in d.io.tolist():
        x, y = int(xy[i, 0]), int(xy[i, 1])
        sides[int(np.argmin([y - die[1], die[2] - x, die[3] - y, x - die[0]]))].append((i, x, y))
    out = {}
    for side, pins in sides.items():
        layer = v if side in (0, 2) else h
        axis = 'X' if side in (0, 2) else 'Y'
        start, step = tracks.get((axis, layer), (0, 1))
        lo, hi = (die[0], die[2]) if axis == 'X' else (die[1], die[3])
        wd = max(2, width.get(layer, 20)); wd -= wd % 2
        ln = 2 * ((7 * wd // 2) // 2)
        along = sorted(((x if axis == 'X' else y), i) for i, x, y in pins)
        k_lo = -(-(lo + step - start) // step); k_hi = (hi - step - start) // step   # tracks inside the edge
        last = None
        for u, i in along:
            k = min(max(round((u - start) / step), k_lo), k_hi)
            if last is not None and k < last + 2:
                k = last + 2                 # ORFS place_pins' default minimum distance: 2 tracks
            last = k
            c = start + k * step
            if side == 0:
                p = (c, die[1] + ln // 2, (-wd // 2, -ln // 2, wd // 2, ln // 2))
            elif side == 2:
                p = (c, die[3] - ln // 2, (-wd // 2, -ln // 2, wd // 2, ln // 2))
            elif side == 1:
                p = (die[2] - ln // 2, c, (-ln // 2, -wd // 2, ln // 2, wd // 2))
            else:
                p = (die[0] + ln // 2, c, (-ln // 2, -wd // 2, ln // 2, wd // 2))
            out[cells[i]] = (p[0], p[1], 'N', layer, p[2])
    return out
