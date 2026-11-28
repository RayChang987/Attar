"""The 'timing' fact: Liberty, SDC and wire RC distilled into an IR (spec 5.1: the fast loop reads only the IR).

attach(ir_dir, lib_paths, sdc_path, rc_tcl) writes two files next to core.safetensors:
  timing.safetensors  per master pin (aligned with the IR's m_pin_* arrays): rise / fall capacitance (and the lower
                      bounds of their ranges, for min analysis), max capacitance,
                      Liberty direction and clock flag; per Liberty timing arc of a master used by the design: from /
                      to master pin, timing type and sense, and its NLDM tables (padded to K x K) in ps / fF.
  timing.json         clocks (period, waveform, ideal latency, uncertainty, source IO cells), input / output delays per
                      IO cell, clock groups, wire RC per um (signal, clock) and per layer, what the SDC had that is not
                      modeled, and the sources (name + sha256, never paths).
and records the fact in manifest.json. The timer is attar.timing.sta.
"""
import gzip
import json
import os
import re

import numpy as np

from ..io import sdc as sdc_mod, st
from ..io.distill import source_ref

# arc timing types
T_COMB, T_RISE_EDGE, T_FALL_EDGE, T_SETUP_R, T_HOLD_R, T_SETUP_F, T_HOLD_F, T_PRESET, T_CLEAR, T_TRISTATE, T_OTHER = range(11)
TYPE_CODE = {'combinational': T_COMB, 'combinational_rise': T_COMB, 'combinational_fall': T_COMB,
             'rising_edge': T_RISE_EDGE, 'falling_edge': T_FALL_EDGE,
             'setup_rising': T_SETUP_R, 'hold_rising': T_HOLD_R, 'setup_falling': T_SETUP_F, 'hold_falling': T_HOLD_F,
             # asynchronous (preset / clear) pin checks: OpenSTA times them as setup / hold checks (TimingRole recovery
             # / removal: the same cycle accounting; sta_recovery_removal_checks_enabled is on by default)
             'recovery_rising': T_SETUP_R, 'recovery_falling': T_SETUP_F, 'removal_rising': T_HOLD_R,
             'removal_falling': T_HOLD_F,
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
    with open(path) as f:
        text = f.read().replace('\\\n', ' ')
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
    for e in s.get('pin_latency', []):
        e['value'] *= tu
    for k in ('input_transition', 'input_transition_min'):
        if k in s:
            s[k] = {p: v * tu for p, v in s[k].items()}
    s['load'] = {p: v * cu for p, v in s['load'].items()}
    for kind, u in (('max_transition', tu), ('max_capacitance', cu)):
        L = s.get('limits', {}).get(kind)
        if L:
            L['design'] = None if L['design'] is None else L['design'] * u
            L['ports'] = {p: v * u for p, v in L['ports'].items()}
            L['clocks'] = {c: {k: v * u for k, v in d.items()} for c, d in L['clocks'].items()}


def find_bundle(ir_dir, name):
    """A shared library bundle's folder: <zoo>/libraries/<name> next to the design's zoo, else the default zoo's."""
    from .. import paths
    zoo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ir_dir))))
    for root in (zoo, paths.zoo_dir()):
        f = os.path.join(root, 'libraries', name)
        if os.path.exists(os.path.join(f, 'library.safetensors')):
            return f
    raise FileNotFoundError('library bundle %r (zoo/libraries/%s) not found for %s' % (name, name, ir_dir))


_BUNDLES = {}


def _load_library(folder, stem='library'):
    """A cell library (<folder>/<stem>.safetensors), loaded once per process and file version (a library fetched
    again, attar.fetch(force=True), is read again)."""
    from .library import Library
    f = os.path.join(folder, stem + '.safetensors')
    key = (os.path.abspath(folder), stem, os.path.getmtime(f) if os.path.exists(f) else None)
    if key not in _BUNDLES:
        _BUNDLES[key] = Library.load(folder, stem)
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
    (attar.timing.library.design_arrays) when it references one, else timing.safetensors and timing.json."""
    if 'library' not in d.manifest:
        with open(os.path.join(d.path, 'timing.json')) as f:
            info = json.load(f)
        t, _ = st.load(os.path.join(d.path, 'timing.safetensors'))
        return t, info.get('functions') or {}, info.get('arc_when') or {}
    from .library import design_arrays
    cm = d.raw['c_master'] if c_master is None else c_master
    used = set(np.unique(np.asarray(cm)[np.asarray(cm) >= 0]).tolist())
    # kept on the design: a timer rebuilt after a netlist edit or a resize with the same masters needs the same arrays
    key = (len(d.names['masters']), tuple(sorted(used)))
    hit = getattr(d, '_timing_arrays', None)
    if hit is not None and hit[0] == key:
        return hit[1]
    t, functions, arc_when, _ = design_arrays(chain(d.path, d.manifest), d.names['masters'], d.names['master_pins'],
                                              d.raw['m_pin_ptr'], used)
    d._timing_arrays = (key, (t, functions, arc_when))
    return t, functions, arc_when


def chain_min(ir_dir, manifest):
    """The library chain of the min analysis (hold, earliest arrivals), when the design has one of its own (manifest
    'library_min': an Early library, e.g. ICCAD 2015's; its local cells in ir_dir/library_min.safetensors), else
    None."""
    ref = manifest.get('library_min')
    if not ref:
        return None
    out = []
    if ref.get('local'):
        out.append(_load_library(ir_dir, 'library_min'))
    if ref.get('bundle'):
        out.append(_load_library(find_bundle(ir_dir, ref['bundle'])))
    return out


ALIGN = ('a_master', 'a_from', 'a_to', 'a_type', 'a_sense', 'a_n')     # what must match arc for arc across libraries


def arrays_min(d):
    """The min analysis's arrays (as arrays(d)) from the design's Early library chain (chain_min), or None when it has
    none: the timer then uses one library for both analyses. Same masters and arcs as arrays(d), arc for arc."""
    ch = chain_min(d.path, d.manifest)
    if ch is None:
        return None
    from .library import design_arrays
    cm = d.raw['c_master']
    used = set(np.unique(np.asarray(cm)[np.asarray(cm) >= 0]).tolist())
    key = (len(d.names['masters']), tuple(sorted(used)))
    hit = getattr(d, '_timing_arrays_min', None)
    if hit is not None and hit[0] == key:
        return hit[1]
    t, _, _, _ = design_arrays(ch, d.names['masters'], d.names['master_pins'], d.raw['m_pin_ptr'], used)
    d._timing_arrays_min = (key, t)
    return t


def attach_min_library(ir_dir, lib_paths, bundle=None):
    """Give a design (timing attached, with a library chain) an Early library for the min analysis: lib_paths its
    Liberty files, bundle a shared folder (zoo/libraries/<name>) holding the standard cells among them; the others go
    to ir_dir/library_min. Additive: the manifest gains 'library_min', nothing else changes (older Attar ignores it).
    Refused unless every arc lines up with the design's Late library (ALIGN) and the bundle gives the same arrays as
    the files themselves. The local cells go to ir_dir/library_min.safetensors and .json (flat files, as the zoo
    release packs a design folder). Returns the manifest entry."""
    from .library import Library, design_arrays
    with gzip.open(os.path.join(ir_dir, 'names.json.gz'), 'rt') as f:
        names = json.load(f)
    a, _ = st.load(os.path.join(ir_dir, 'core.safetensors'))
    with open(os.path.join(ir_dir, 'manifest.json')) as f:
        m = json.load(f)
    if 'library' not in m:
        raise ValueError('%s: no library chain (attach timing with a bundle first)' % ir_dir)
    masters, mpins = names['masters'], names['master_pins']
    m_ptr = a['m_pin_ptr'].astype(np.int64)
    used = set(a['c_master'][a['c_master'] >= 0].tolist())
    late, _, _, _ = design_arrays(chain(ir_dir, m), masters, mpins, m_ptr, used)
    own = Library.build(lib_paths, source_ref=source_ref, power=False)
    tens, _, _, _ = design_arrays([own], masters, mpins, m_ptr, used)
    bad = [k for k in ALIGN if not np.array_equal(late[k], tens[k])]
    if bad:
        raise ValueError('%s: the Early library does not line up with the Late one (%s)' % (ir_dir, ', '.join(bad)))
    for f in ('library_min.safetensors', 'library_min.json'):
        if os.path.exists(os.path.join(ir_dir, f)):
            os.remove(os.path.join(ir_dir, f))
    ref = {'local': [os.path.basename(p) for p in lib_paths]}
    chain_ = [own]
    if bundle is not None:
        B = _load_library(bundle)
        have = {r.get('sha256') for r in B.meta['sources']['lib']}
        refs = own.meta['sources']['lib']
        local = [p for p, r in zip(lib_paths, refs) if r.get('sha256') not in have]
        L = Library.build(local, source_ref=source_ref, name='local') if local else None
        chain_ = ([L] if L else []) + [B]
        ref = {'bundle': B.meta['name'], 'sha256': source_ref(os.path.join(bundle, 'library.safetensors'))['sha256'],
               'local': [os.path.basename(p) for p in local]}
    else:
        L = Library.build(lib_paths, source_ref=source_ref, name='local')
        chain_ = [L]
    t2, _, _, _ = design_arrays(chain_, masters, mpins, m_ptr, used)
    eq = lambda x, y: x.shape == y.shape and bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))  # noqa: E731
                                                            if x.dtype.kind == 'f' else x == y))
    if not all(k in t2 and eq(tens[k], t2[k]) for k in tens):
        raise ValueError('%s: the Early bundle gives other arrays than the Liberty files' % ir_dir)
    if L is not None:
        L.save(ir_dir, 'library_min')
    m['library_min'] = ref
    mf = os.path.join(ir_dir, 'manifest.json')
    with open(mf + '.tmp', 'w') as f:                # a new file: a hard link to it elsewhere (zoo_hub) keeps the old
        json.dump(m, f, indent=1)
    os.replace(mf + '.tmp', mf)
    return ref


def pin_locations(d):
    """Per master pin, OpenROAD's pin location (attar.timing.library.pin_locations: the mean of its LEF rectangles'
    centers, microns from the master's lower-left, N) from the design's library chain; NaN where unknown (no chain,
    a bundle without lp_loc)."""
    from .library import pin_locations as look
    n = int(d.raw['m_pin_ptr'][-1])
    if 'library' not in d.manifest:
        return np.full((n, 2), np.nan)
    return look(chain(d.path, d.manifest), d.names['masters'], d.names['master_pins'], d.raw['m_pin_ptr'])


def attach(ir_dir, lib_paths, sdc_path=None, rc_tcl=None, sdc_text=None, bundle=None):
    """Distill timing facts into the IR at ir_dir (see the module doc). Returns a summary dict.
    bundle: a shared library folder (attar.timing.library, zoo/libraries/<name>). The design's Liberty files that are in it
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
    own = Library.build(lib_paths, source_ref=source_ref, power=False)   # timing arrays only: the bundle check
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
        L = Library.build(local, source_ref=source_ref, name='local') if local else None   # with power
        t2, f2, w2, m2 = design_arrays(([L] if L else []) + [B], masters, mpins, m_ptr, used)
        eq = lambda x, y: x.shape == y.shape and bool(np.all((x == y) | (np.isnan(x) & np.isnan(y))  # noqa: E731
                                                                if x.dtype.kind == 'f' else x == y))
        same = (all(k in t2 and eq(tens[k], t2[k]) for k in tens) and f2 == functions and w2 == arc_when
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
        info['input_transition_min'] = {str(port_idx[p]): v for p, v in s['input_transition_min'].items()
                                        if p in port_idx}
        info['load'] = {str(port_idx[p]): v for p, v in s['load'].items() if p in port_idx}
        info['clock_groups'] = s['clock_groups']
        # design rule limits (ps, fF, fanout): the design's, by IO cell index for ports, by name for clocks
        info['limits'] = {k: {'design': L['design'], 'ports': {str(port_idx[p]): v for p, v in L['ports'].items()
                                                             if p in port_idx}, 'clocks': L['clocks']}
                          for k, L in s['limits'].items()}
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
        # pin clock latencies (set_clock_latency on pins / ports): by pin index
        info['pin_latency'] = []
        if s.get('pin_latency'):
            p_cell, p_mpin = a['p_cell'].astype(np.int64), a['p_mpin'].astype(np.int64)
            m_of = np.searchsorted(m_ptr, np.arange(int(m_ptr[-1])), side='right') - 1    # master of a master pin
            want = {}
            for e in s['pin_latency']:
                n = e['pin'].replace('\\', '')
                if e['port']:
                    c = port_idx.get(n)
                    key = (c, None)
                else:
                    inst, _, pn = n.rpartition('/')
                    c = cell_idx.get(inst)
                    key = (c, pn)
                want.setdefault(key, []).append(e)
            cells = np.asarray(sorted({c for c, _ in want if c is not None}), np.int64)
            hit = np.nonzero(np.isin(p_cell, cells))[0] if len(cells) else np.zeros(0, np.int64)
            pin_of = {}
            for p in hit.tolist():
                k = int(p_mpin[p])
                pn = None if k < 0 else mpins[int(m_of[k])][k - int(m_ptr[int(m_of[k])])]
                pin_of[(int(p_cell[p]), pn)] = p
            for key, es in want.items():
                p = pin_of.get(key) if key[0] is not None else None
                for e in es:
                    if p is None:
                        info['unsupported']['set_clock_latency on unknown pins'] = (
                            info['unsupported'].get('set_clock_latency on unknown pins', 0) + 1)
                    else:
                        info['pin_latency'].append({'pin': int(p), 'clock': e['clock'], 'value': e['value'],
                                                    'rf': e['rf'], 'mm': e['mm']})
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
    with open(os.path.join(ir_dir, 'manifest.json')) as f:
        rl = (json.load(f).get('routing') or {}).get('layers')
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
    with open(mf) as f:
        m = json.load(f)
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
