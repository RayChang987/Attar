"""The 'timing' fact: Liberty, SDC and wire RC distilled into an IR (spec 5.1: the fast loop reads only the IR).

attach(ir_dir, lib_paths, sdc_path, rc_tcl) writes two files next to core.safetensors:
  timing.safetensors  per master pin (aligned with the IR's m_pin_* arrays): rise / fall capacitance (and the lower
                      bounds of their ranges, for min analysis), max capacitance,
                      Liberty direction and clock flag; per Liberty timing arc of a master used by the design: from /
                      to master pin, timing type and sense, and its NLDM tables (padded to K x K) in ps / fF.
  timing.json         clocks (period, waveform, ideal latency, uncertainty, source IO cells), input / output delays per
                      IO cell, clock groups, wire RC per um (signal, clock) and per layer, what the SDC had that is not
                      modeled, and the sources (name + sha256, never paths).
and records the fact in manifest.json. The timer is attar.sta.
"""
import gzip
import json
import os
import re

import numpy as np

from . import sdc as sdc_mod, st
from .distill import source_ref

# arc timing types
T_COMB, T_RISE_EDGE, T_FALL_EDGE, T_SETUP_R, T_HOLD_R, T_SETUP_F, T_HOLD_F, T_PRESET, T_CLEAR, T_TRISTATE, T_OTHER = range(11)
TYPE_CODE = {'combinational': T_COMB, 'combinational_rise': T_COMB, 'combinational_fall': T_COMB,
             'rising_edge': T_RISE_EDGE, 'falling_edge': T_FALL_EDGE,
             'setup_rising': T_SETUP_R, 'hold_rising': T_HOLD_R, 'setup_falling': T_SETUP_F, 'hold_falling': T_HOLD_F,
             'preset': T_PRESET, 'clear': T_CLEAR, 'three_state_enable': T_TRISTATE, 'three_state_disable': T_TRISTATE,
             'three_state_enable_rise': T_TRISTATE, 'three_state_enable_fall': T_TRISTATE,
             'three_state_disable_rise': T_TRISTATE, 'three_state_disable_fall': T_TRISTATE}
# arc sense: positive (in rise -> out rise), negative (in rise -> out fall), non (both)
S_POS, S_NEG, S_NON = 0, 1, 2
SENSE_CODE = {'positive_unate': S_POS, 'negative_unate': S_NEG, 'non_unate': S_NON}
# table slots: delay arcs (cell_rise, cell_fall, rise_transition, fall_transition); checks (rise_, fall_constraint)
DELAY_SLOTS = ('cell_rise', 'cell_fall', 'rise_transition', 'fall_transition')
CHECK_SLOTS = ('rise_constraint', 'fall_constraint')
# Liberty pin directions
LD_IN, LD_OUT, LD_INOUT, LD_INTERNAL, LD_NONE = 0, 1, 2, 3, 4
LIB_DIR = {'input': LD_IN, 'output': LD_OUT, 'inout': LD_INOUT, 'internal': LD_INTERNAL}


def read_rc_tcl(path):
    """An OpenROAD RC script (ORFS platform setRC.tcl): set_layer_rc / set_wire_rc -> {'layer': {name: [r, c]},
    'via': {name: r}, 'signal': [r, c], 'clock': [r, c]} (r in the Liberty resistance unit per um, c in its capacitance
    unit per um: kohm and fF for ASAP7, so r * c * um^2 is ps). set_wire_rc -layers averages the layers' RC and
    names them ('signal_layers' / 'clock_layers'); set_routing_layers -signal gives 'routing'."""
    out = {'layer': {}, 'via': {}}
    text = open(path).read().replace('\\\n', ' ')
    for line in text.splitlines():
        t = re.findall(r'\{[^}]*\}|\S+', line.split('#')[0])
        if not t:
            continue
        o, i = {}, 1
        while i < len(t):
            nxt = t[i + 1] if i + 1 < len(t) else None
            if t[i].startswith('-') and nxt is not None and not re.match(r'-[a-z]', nxt):
                o[t[i]] = nxt.strip('{}'); i += 2
            else:
                o[t[i]] = True; i += 1
        if t[0] == 'set_layer_rc':
            if '-via' in o:
                out['via'][o['-via']] = float(o.get('-resistance', 0))
            elif '-layer' in o:
                out['layer'][o['-layer']] = [float(o.get('-resistance', 0)), float(o.get('-capacitance', 0))]
        elif t[0] == 'set_routing_layers' and isinstance(o.get('-signal'), str):
            out['routing'] = o['-signal'].split('-')             # set_routing_layers -signal M2-M7
        elif t[0] == 'set_wire_rc':
            kinds = [k for k in ('signal', 'clock') if o.get('-' + k)] or ['signal', 'clock']
            if '-layers' in o or '-layer' in o:
                ls = (o.get('-layers') or o.get('-layer')).split()
                rc = [float(np.mean([out['layer'][l][k] for l in ls])) for k in (0, 1)]
                for k in kinds:                                   # the tree's layer: pins connect up to it by vias
                    out[k + '_layers'] = ls
            else:
                rc = [float(o.get('-resistance', 0)), float(o.get('-capacitance', 0))]
            for k in kinds:
                out[k] = rc
    return out


def _pack_tables(arcs, K):
    """arcs' table dicts -> ix1 [A, 4, K], ix2 [A, 4, K], val [A, 4, K, K], n [A, 4, 2] (n = 0: no table)."""
    A = len(arcs)
    ix1 = np.zeros((A, 4, K), np.float32); ix2 = np.zeros((A, 4, K), np.float32)
    val = np.zeros((A, 4, K, K), np.float32); n = np.zeros((A, 4, 2), np.int8)
    for a, arc in enumerate(arcs):
        slots = CHECK_SLOTS if TYPE_CODE.get(arc['type'], T_OTHER) in (T_SETUP_R, T_HOLD_R, T_SETUP_F, T_HOLD_F) else DELAY_SLOTS
        for s, kind in enumerate(slots):
            if kind not in arc['tables']:
                continue
            i1, i2, v = arc['tables'][kind]
            n1, n2 = len(i1), len(i2)
            ix1[a, s, :n1] = i1; ix2[a, s, :n2] = i2
            ix1[a, s, n1:] = i1[-1]; ix2[a, s, n2:] = i2[-1]
            vv = np.asarray(v, np.float32).reshape(n1, n2)
            val[a, s, :n1, :n2] = vv
            n[a, s] = (n1, n2)
    return ix1, ix2, val, n


def _sdc_units(s, tu, cu):
    """SDC values from the first Liberty file's units (OpenSTA's user units) to ps and fF, in place."""
    for c in s['clocks'].values():
        for k in ('period', 'latency', 'source_latency', 'uncertainty', 'uncertainty_setup', 'uncertainty_hold'):
            if k in c:
                c[k] *= tu
        c['waveform'] = [v * tu for v in c['waveform']]
        if 'transition' in c:
            c['transition'] = [v * tu for v in c['transition']]
    for w in ('input_delay', 'output_delay', 'path_delays'):
        for e in s[w]:
            e['value'] *= tu
    s['input_transition'] = {p: v * tu for p, v in s['input_transition'].items()}
    s['load'] = {p: v * cu for p, v in s['load'].items()}


def find_bundle(ir_dir, name):
    """A shared library bundle's folder: <zoo>/libraries/<name> next to the design's zoo, else the default zoo's."""
    from . import paths
    zoo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ir_dir))))
    for root in (zoo, paths.zoo_dir()):
        f = os.path.join(root, 'libraries', name)
        if os.path.exists(os.path.join(f, 'library.safetensors')):
            return f
    raise FileNotFoundError('library bundle %r (zoo/libraries/%s) not found for %s' % (name, name, ir_dir))


_BUNDLES = {}


def _load_library(folder):
    from .library import Library
    key = os.path.abspath(folder)
    if key not in _BUNDLES:
        _BUNDLES[key] = Library.load(folder)
    return _BUNDLES[key]


def chain(ir_dir, manifest):
    """The design's cell library chain: its local library (the cells only it has), then the shared bundle."""
    ref = manifest.get('library') or {}
    out = []
    if ref.get('local'):
        out.append(_load_library(ir_dir))
    if ref.get('bundle'):
        out.append(_load_library(find_bundle(ir_dir, ref['bundle'])))
    return out


def arrays(d, c_master=None):
    """The timer's per-design arrays, functions and arc `when` conditions: looked up along the design's library chain
    (attar.library.design_arrays) when it references one, else timing.safetensors and timing.json."""
    info = json.load(open(os.path.join(d.path, 'timing.json')))
    if 'library' not in d.manifest:
        t, _ = st.load(os.path.join(d.path, 'timing.safetensors'))
        return t, info.get('functions') or {}, info.get('arc_when') or {}
    from .library import design_arrays
    cm = d.raw['c_master'] if c_master is None else c_master
    used = set(np.asarray(cm)[np.asarray(cm) >= 0].tolist())
    t, functions, arc_when, _ = design_arrays(chain(d.path, d.manifest), d.names['masters'], d.names['master_pins'],
                                              d.raw['m_pin_ptr'], used)
    return t, functions, arc_when


def attach(ir_dir, lib_paths, sdc_path=None, rc_tcl=None, sdc_text=None, bundle=None):
    """Distill timing facts into the IR at ir_dir (see the module doc). Returns a summary dict.
    bundle: a shared library folder (attar.library, zoo/libraries/<name>). The design's Liberty files that are in it
    are taken from it, the others become the design's local library (library.safetensors in ir_dir), and the design
    references the bundle instead of storing timing.safetensors, provided the arrays come out identical to its own
    Liberty files' (else timing.safetensors as without a bundle, and the summary says why)."""
    from .library import Library, design_arrays
    with gzip.open(os.path.join(ir_dir, 'names.json.gz'), 'rt') as f:
        names = json.load(f)
    a, _ = st.load(os.path.join(ir_dir, 'core.safetensors'))
    masters, mpins = names['masters'], names['master_pins']
    m_ptr = a['m_pin_ptr'].astype(np.int64)
    used = set(a['c_master'][a['c_master'] >= 0].tolist())
    # the design's own Liberty files: the first definition of a cell wins (OpenSTA warns on duplicates)
    own = Library.build(lib_paths, source_ref=source_ref)
    tens, functions, arc_when, missing_cells = design_arrays([own], masters, mpins, m_ptr, used)
    lib_refs = own.meta['sources']['lib']
    default_thr = own.meta['default_thresholds']                 # OpenSTA: ports use the first library read
    units = own.meta['units'] or {}
    sdc_tu, sdc_cu = units.get('time_ps', 1.0), units.get('cap_ff', 1.0)   # and SDC values are in its units
    lib_ref, bundle_note = None, None
    for f in ('library.safetensors', 'library.json'):
        if os.path.exists(os.path.join(ir_dir, f)):
            os.remove(os.path.join(ir_dir, f))
    if bundle is not None:
        B = _load_library(bundle)
        have = {r.get('sha256') for r in B.meta['sources']['lib']}
        local = [p for p, r in zip(lib_paths, lib_refs) if r.get('sha256') not in have]
        L = Library.build(local, source_ref=source_ref, name='local') if local else None
        t2, f2, w2, m2 = design_arrays(([L] if L else []) + [B], masters, mpins, m_ptr, used)
        same = (all(np.array_equal(tens[k], t2[k]) for k in tens) and f2 == functions and w2 == arc_when
                and m2 == missing_cells)
        if same:
            if L is not None:
                L.save(ir_dir)
            lib_ref = {'bundle': B.meta['name'], 'sha256': source_ref(os.path.join(bundle, 'library.safetensors'))['sha256'],
                       'local': [os.path.basename(p) for p in local]}
        else:
            bundle_note = 'arrays differ from the design\'s own Liberty files with bundle %s' % B.meta['name']
    if lib_ref is None:
        st.save(os.path.join(ir_dir, 'timing.safetensors'), tens, {'schema': 'attar-timing/0.1'})
    elif os.path.exists(os.path.join(ir_dir, 'timing.safetensors')):
        os.remove(os.path.join(ir_dir, 'timing.safetensors'))

    # constraints, by IO cell index
    io_cell, io_dir = a['io_cell'], a['io_dir']
    cell_names = names['cells']
    pin_in = [cell_names[i] for i, d in zip(io_cell, io_dir) if d == 0]
    pin_out = [cell_names[i] for i, d in zip(io_cell, io_dir) if d == 1]
    port_idx = {cell_names[i]: int(i) for i in io_cell}
    info = {'clocks': {}, 'input_delay': [], 'output_delay': [], 'input_transition': {}, 'load': {},
            'clock_groups': [], 'path_delays': [], 'unsupported': {}}
    if sdc_path or sdc_text:
        s = sdc_mod.read(sdc_text if sdc_text else sdc_path, pin_in, pin_out, names.get('design', ''), is_text=bool(sdc_text))
        _sdc_units(s, sdc_tu, sdc_cu)
        for k, c in s['clocks'].items():
            info['clocks'][k] = dict(c, sources=[port_idx[p] for p in c['sources'] if p in port_idx])
        for w in ('input_delay', 'output_delay'):
            info[w] = [dict(e, cell=port_idx[e['port']]) for e in s[w] if e['port'] in port_idx]
        info['input_transition'] = {str(port_idx[p]): v for p, v in s['input_transition'].items() if p in port_idx}
        info['load'] = {str(port_idx[p]): v for p, v in s['load'].items() if p in port_idx}
        info['clock_groups'] = s['clock_groups']
        # path delays: endpoints of -from / -to as cell indices (cells: instances; ports: their IO cell)
        cell_idx = {n.replace('\\', ''): i for i, n in enumerate(cell_names)}    # DEF escapes ('a\\[0\\]'), SDC does not
        def cells_of(objs):
            if objs is None:
                return None
            out = []
            for k, n in objs:
                n = n.replace('\\', '')
                if k in ('cell', 'port') and n in cell_idx:
                    out.append(cell_idx[n])
                elif k == 'pin' and '/' in n and n.rsplit('/', 1)[0] in cell_idx:
                    out.append(cell_idx[n.rsplit('/', 1)[0]])
            return out
        info['path_delays'] = [dict(e, **{'from': cells_of(e['from']), 'to': cells_of(e['to'])}) for e in s['path_delays']]
        info['unsupported'] = s['unsupported']
    # IO pins' layers (the first LAYER of each pin in the source DEF's PINS): OpenROAD connects a port to its net's
    # estimated tree through the vias from that layer
    info['io_layer'] = {}
    frame = os.path.join(ir_dir, 'frame.def.gz')
    if os.path.exists(frame):
        with gzip.open(frame, 'rt') as f:
            text = f.read()
        i, j = text.find('\nPINS'), text.find('\nEND PINS')
        if i >= 0 and j > i:
            for m in re.finditer(r'^\s*-\s+(\S+)(.*?);\s*$', text[i:j], re.M | re.S):
                lay = re.search(r'\+\s*LAYER\s+(\S+)', m.group(2))
                if lay and m.group(1) in port_idx:
                    info['io_layer'][str(port_idx[m.group(1)])] = lay.group(1)
    # Liberty `when` conditions of the conditional arcs, by arc index (OpenSTA's sim disables them under constants),
    # output pin functions of the masters used (constant propagation from tie cells, as OpenSTA's sim does)
    info['arc_when'] = arc_when
    info['functions'] = functions
    info['rc'] = read_rc_tcl(rc_tcl) if rc_tcl else None
    # OpenROAD est::computeAverageCutResistance: the mean resistance of the cut layers between the block's lowest and
    # highest routing layer (via Vk joins Mk and Mk+1); every pin hangs off the estimated tree through it
    # (the routing layers: the manifest's, else the RC script's set_routing_layers -signal)
    rl = (json.load(open(os.path.join(ir_dir, 'manifest.json'))).get('routing') or {}).get('layers')
    rl = rl or (info['rc'] or {}).get('routing')
    if info['rc'] and rl and info['rc'].get('via'):
        lo, hi = (int(re.sub(r'\D', '', x)) for x in rl)
        via_k = {int(m.group(1)): r for v, r in info['rc']['via'].items() for m in [re.fullmatch(r'V(?:IA)?(\d+)', v)] if m}
        vs = [via_k[k] for k in range(lo, hi) if k in via_k]
        info['rc']['cut_res'] = float(np.mean(vs)) if vs else 0.0
    info['units'] = {'time': 'ps', 'cap': 'fF', 'res': 'kohm'}
    info['default_thresholds'] = default_thr
    info['missing_cells'] = missing_cells
    info['sources'] = {'lib': lib_refs, 'sdc': source_ref(sdc_path) if sdc_path else None,
                       'rc': source_ref(rc_tcl) if rc_tcl else None}
    with open(os.path.join(ir_dir, 'timing.json'), 'w') as f:
        json.dump(info, f, indent=1)

    mf = os.path.join(ir_dir, 'manifest.json')
    m = json.load(open(mf))
    m.setdefault('facts', {})['timing'] = 'liberty+sdc' + ('+rc' if rc_tcl else '')
    if lib_ref is not None:
        m['facts']['library'] = lib_ref['bundle']
        m['library'] = lib_ref
    else:
        m['facts'].pop('library', None); m.pop('library', None)
    if sdc_path or sdc_text:                                  # the SDC is read now
        (m.get('unread_sources') or {}).pop('sdc', None)
        if 'unread_sources' in m and not m['unread_sources']:
            del m['unread_sources']
    m['timing'] = {'arcs': len(tens['a_master']), 'clocks': list(info['clocks']), 'missing_cells': missing_cells,
                   'unsupported_sdc': info['unsupported'], 'sources': info['sources']}
    if bundle_note:
        m['timing']['bundle_note'] = bundle_note
    json.dump(m, open(mf, 'w'), indent=1)
    return m['timing']
