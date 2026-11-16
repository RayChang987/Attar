"""Cell libraries as tensors (the 'library' fact): Liberty files, and the LEF of their cells, distilled once.

A Library holds every cell of a set of Liberty files (the first definition of a cell wins, as in OpenSTA):
  library.safetensors  per cell: area, leakage (nW), sequential kind, the Liberty thresholds of its file (input /
                       output, slew lower / upper, derate), LEF size and class; per pin (CSR l_pin_ptr): direction,
                       rise / fall capacitance (and the lower bounds of their ranges), max capacitance, clock flag,
                       macro clock tree path, LEF box / use / layer and location (the mean of its rectangles'
                       centers, OpenROAD's; NaN when unknown); per timing arc: cell, from / to pin, type, sense,
                       NLDM tables padded to K x K, in ps / fF.
                       Design rules: per pin max_transition, max_fanout, fanout_load; per cell its file's defaults
                       (default_max_transition, _capacitance, _fanout, default_fanout_load); NaN when absent.
                       Power (OpenSTA's report_power): per cell cell_leakage_power (NaN when absent), the default
                       operating voltage, macro / memory / pad flags; per pin the voltage of its related_power_pin;
                       per leakage_power group (lk_*) cell, value (nW), pg pin; per internal_power group (ip_*) cell,
                       pin, related pin, pg pin and rise / fall energy tables (fJ, on input slew x load).
  library.json         cell and pin names, output functions, arc `when` conditions, dont_use, footprints, equivalence
                       classes (cells that can replace each other: same signal pins, same per-output logic, same
                       sequentials and statetable, as OpenSTA's equivCells; same height), buffers and inverters,
                       sequentials and statetables, pg pins per cell, `when` conditions of the leakage and internal
                       power groups, sources (name + sha256), units.
Designs reference a shared bundle per platform and corner (zoo/libraries/<name>) and keep the cells only they have
(macros) in a local library in their own folder; design_arrays() looks a design's masters up along that chain and
returns the per-master-pin and per-arc arrays the timer reads (attar.timing.ir.attach writes the same arrays).
"""
import itertools
import json
import os

import numpy as np

from ..io import liberty, st

SCHEMA = 'attar-library/0.1'
# Liberty pin directions, timing types and senses: attar.timing.ir's codes
from .ir import LD_NONE, LIB_DIR, S_NON, SENSE_CODE, T_OTHER, TYPE_CODE  # noqa: E402
from ..io.distill import D_UNKNOWN, LEF_DIR, M_BLOCK, M_CORE, M_OTHER, M_PAD, U_SIGNAL, _USE  # noqa: E402
SEQ_CODE = {'': 0, 'ff': 1, 'latch': 2}
KIND_MACRO, KIND_PAD = 1, 2          # lc_kind bits: OpenSTA's report_power groups (macro / memory, pad)


def _pack_power(tabs):
    """Internal power groups' {'rise' | 'fall': (index_1, index_2, values)} -> ix1 [I, 2, K], ix2 [I, 2, K],
    val [I, 2, K, K], n [I, 2, 2] as attar.timing.ir._pack_tables packs delay tables (n = 0: no table)."""
    K = max([max(len(t[0]), len(t[1])) for g in tabs for t in g.values()] + [1])
    I = len(tabs)
    ix1 = np.zeros((I, 2, K), np.float32); ix2 = np.zeros((I, 2, K), np.float32)
    val = np.zeros((I, 2, K, K), np.float32); n = np.zeros((I, 2, 2), np.int8)
    for i, g in enumerate(tabs):
        for s, rf in enumerate(('rise', 'fall')):
            if rf not in g:
                continue
            i1, i2, v = g[rf]
            n1, n2 = len(i1), len(i2)
            ix1[i, s, :n1] = i1; ix2[i, s, :n2] = i2
            ix1[i, s, n1:] = i1[-1]; ix2[i, s, n2:] = i2[-1]
            val[i, s, :n1, :n2] = np.asarray(v, np.float32).reshape(n1, n2)
            n[i, s] = (n1, n2)
    return ix1, ix2, val, n


def _thr4(t, key):
    return np.array([[t[key][i], t['slew_lower'][i], t['slew_upper'][i], t['slew_derate']] for i in (0, 1)],
                    np.float32)


def _truth(expr, inputs):
    """A Liberty function's truth table over the given input names (missing names: internal states, left symbolic)."""
    vs, f = liberty.parse_function(expr)
    extra = sorted(v for v in vs if v not in inputs)
    rows = []
    for bits in itertools.product((0, 1), repeat=len(inputs) + len(extra)):
        rows.append(f(dict(zip(list(inputs) + extra, bits))))
    return tuple(extra), tuple(rows)


def _seq_key(s, inputs):
    """What OpenSTA's equivCellSequentials and equivCellStatetables compare: per sequential (in order) its kind,
    internal state names, clock, data, clear and preset functions (truth tables here); the statetable's input and
    internal node names and rows. A rising-edge flop is no replacement for a falling-edge one, nor a high latch for a low one."""
    if not s:
        return ()
    t = lambda e: _truth(e, inputs) if e else None  # noqa: E731
    seqs = tuple((q['kind'], tuple(q['vars']), t(q['clock']), t(q['data']), t(q['clear']), t(q['preset']))
                 for q in s.get('seq', []))
    st = s.get('statetable')
    return seqs, (tuple(st['inputs']), tuple(st['internals']), tuple(st['rows'])) if st else None


class Library:
    def __init__(self, arr, meta):
        self.a, self.meta = arr, meta
        self.cells, self.pins = meta['cells'], meta['pins']
        self.cid = {n: i for i, n in enumerate(self.cells)}
        ptr = arr['l_pin_ptr'].astype(np.int64)
        self.pid = [{p: int(ptr[c]) + j for j, p in enumerate(ps)} for c, ps in enumerate(self.pins)]
        ac = arr['la_cell'].astype(np.int64)
        o = np.argsort(ac, kind='stable')
        self.arcs_of = np.split(o, np.searchsorted(ac[o], np.arange(1, len(self.cells))))

    # -- building -----------------------------------------------------------------------------------------------------
    @classmethod
    def build(cls, lib_paths, lef_paths=(), name='', source_ref=None, power=True):
        """Read Liberty files (and the LEF files of their cells) into a Library. source_ref(path) -> {name, sha256}.
        power=False skips the power groups (faster; the lk_* / ip_* arrays come out empty)."""
        cells, cell_thr, cell_file, refs, file_lib = {}, {}, {}, [], []
        default_thr, units = None, None
        for i, p in enumerate(lib_paths):
            lib = liberty.read(p, power=power)
            refs.append(source_ref(p) if source_ref else {'name': os.path.basename(p)})
            default_thr = default_thr or lib['thresholds']
            units = units or {'time_ps': lib['time_unit_ps'], 'cap_ff': lib['cap_unit_ff']}
            file_lib.append(lib)
            for k, v in lib['cells'].items():
                if k not in cells:
                    cells[k] = v; cell_thr[k] = lib['thresholds']; cell_file[k] = i
        macros, tech = {}, {'sites': {}, 'layers': [], 'vias': {}}
        if lef_paths:
            from ..io.lef import read_lef
            read_lef(list(lef_paths), macros, tech)
        layers = [l['name'] for l in tech['layers']] if tech['layers'] else []
        lid = {n: i for i, n in enumerate(layers)}
        names = list(cells) + sorted(m for m in macros if m not in cells)      # LEF-only cells (fillers, taps, ...)
        C = len(names)
        pin_names, ptr = [], [0]
        for c in names:
            lp = list(cells[c]['pins']) if c in cells else []
            lp += sorted(p for p in (macros[c].pins if c in macros else {}) if p not in lp)
            pin_names.append(lp); ptr.append(ptr[-1] + len(lp))
        L = ptr[-1]
        a = {'l_pin_ptr': np.asarray(ptr, np.int32),
             'lc_area': np.zeros(C, np.float32), 'lc_leak': np.zeros(C, np.float32), 'lc_seq': np.zeros(C, np.uint8),
             'lc_file': np.full(C, -1, np.int16), 'lc_thr_in': np.zeros((C, 2, 4), np.float32),
             'lc_thr_out': np.zeros((C, 2, 4), np.float32), 'lc_size': np.zeros((C, 2), np.float32),
             'lc_lef': np.zeros(C, np.uint8), 'lc_class': np.full(C, M_OTHER, np.uint8), 'lc_sym': np.zeros(C, np.uint8),
             'lp_lib': np.zeros(L, np.uint8), 'lp_cap': np.zeros((L, 2), np.float32),
             'lp_cap_min': np.zeros((L, 2), np.float32), 'lp_maxcap': np.zeros(L, np.float32),
             'lp_dir': np.full(L, LD_NONE, np.uint8), 'lp_clock': np.zeros(L, np.uint8),
             'lp_ctp': np.zeros((L, 2, 2), np.float32), 'lp_has_ctp': np.zeros((L, 2), np.uint8),
             'lp_lef': np.zeros(L, np.uint8), 'lp_box': np.zeros((L, 4), np.float32),
             'lp_loc': np.full((L, 2), np.nan, np.float32),
             'lp_use': np.full(L, U_SIGNAL, np.uint8), 'lp_layer': np.full(L, -1, np.int16),
             'lp_lefdir': np.full(L, D_UNKNOWN, np.uint8),
             # design rules (NaN: absent)
             'lp_maxtran': np.full(L, np.nan, np.float32), 'lp_maxfo': np.full(L, np.nan, np.float32),
             'lp_fanload': np.full(L, np.nan, np.float32), 'lc_dlim': np.full((C, 4), np.nan, np.float32),
             # power
             'lc_cleak': np.full(C, np.nan, np.float32), 'lc_volt': np.zeros(C, np.float32),
             'lc_kind': np.zeros(C, np.uint8), 'lp_volt': np.full(L, np.nan, np.float32)}
        arcs, la = [], {'la_cell': [], 'la_from': [], 'la_to': [], 'la_type': [], 'la_sense': []}
        functions, whens, dont_use, footprint, lef_class, sequentials = {}, [], [], {}, {}, {}
        lk = {'lk_cell': [], 'lk_val': [], 'lk_pg': [], 'lk_primary': []}
        ip, ip_tabs = {'ip_cell': [], 'ip_pin': [], 'ip_rel': [], 'ip_pg': []}, []
        lk_when, ip_when, pg_names = {}, {}, {}
        for c, cn in enumerate(names):
            lc = cells.get(cn)
            base = ptr[c]
            if lc is not None:
                a['lc_area'][c] = lc.get('area', 0.0); a['lc_leak'][c] = lc.get('leakage', 0.0)
                a['lc_seq'][c] = SEQ_CODE.get(lc.get('seq', ''), 0); a['lc_file'][c] = cell_file[cn]
                a['lc_thr_in'][c] = _thr4(cell_thr[cn], 'input'); a['lc_thr_out'][c] = _thr4(cell_thr[cn], 'output')
                flib = file_lib[cell_file[cn]]
                lim = flib.get('limits') or {}
                a['lc_dlim'][c] = [np.nan if lim.get(k) is None else lim[k] for k in
                                   ('default_max_transition', 'default_max_capacitance', 'default_max_fanout',
                                    'default_fanout_load')]
                a['lc_cleak'][c] = np.nan if lc.get('cell_leakage') is None else lc['cell_leakage']
                a['lc_volt'][c] = flib.get('default_voltage') or 0.0
                a['lc_kind'][c] = (KIND_MACRO if lc.get('macro') or lc.get('memory') else 0) | (KIND_PAD if lc.get('pad') else 0)
                vmap = flib.get('voltage_map') or {}
                pgs = lc.get('pg_pins') or {}
                if pgs:
                    pg_names[cn] = {k: v.get('type') for k, v in pgs.items()}
                pg_idx = {k: i for i, k in enumerate(pgs)}

                def pg_volt(pg):
                    vn = (pgs.get(pg) or {}).get('voltage_name')
                    return vmap.get(vn, np.nan) if vn else np.nan
                for g in lc.get('leakage_groups', ()):
                    if g['when']:
                        lk_when[str(len(lk['lk_cell']))] = g['when']
                    lk['lk_cell'].append(c); lk['lk_val'].append(g['value'])
                    lk['lk_pg'].append(pg_idx.get(g['pg'], -1))
                    lk['lk_primary'].append(1 if g['pg'] is None or (pgs.get(g['pg']) or {}).get('type') == 'primary_power'
                                            else 0)
                if lc.get('dont_use'):
                    dont_use.append(cn)
                if lc.get('footprint'):
                    footprint[cn] = lc['footprint']
                if lc.get('sequentials') or lc.get('statetable'):
                    sequentials[cn] = {'seq': lc.get('sequentials', []), 'statetable': lc.get('statetable')}
                fs = {}
                for j, pn in enumerate(pin_names[c]):
                    pa = lc['pins'].get(pn)
                    if pa is None:
                        continue
                    k = base + j
                    a['lp_lib'][k] = 1
                    a['lp_cap'][k] = pa['cap']; a['lp_cap_min'][k] = pa.get('cap_min', pa['cap'])
                    a['lp_maxcap'][k] = pa['max_cap'] or 0.0
                    a['lp_dir'][k] = LIB_DIR.get(pa['dir'], LD_NONE); a['lp_clock'][k] = pa['clock']
                    for key, arr in (('max_transition', 'lp_maxtran'), ('max_fanout', 'lp_maxfo'),
                                     ('fanout_load', 'lp_fanload')):
                        if pa.get(key) is not None:
                            a[arr][k] = pa[key]
                    if pa.get('related_power_pin'):
                        a['lp_volt'][k] = pg_volt(pa['related_power_pin'])
                    for i, mm in enumerate(('min', 'max')):
                        if mm in pa.get('ctp', {}):
                            a['lp_ctp'][k, i] = pa['ctp'][mm]; a['lp_has_ctp'][k, i] = 1
                    if pa.get('function') and pa['dir'] in ('output', 'inout'):
                        fs[pn] = pa['function']
                if fs:
                    functions[cn] = fs
                pidx = {p: base + j for j, p in enumerate(pin_names[c])}
                for g in lc.get('internal_power', ()):
                    if g['pin'] not in pidx or (g['related'] is not None and g['related'] not in pidx):
                        continue
                    if g['when']:
                        ip_when[str(len(ip['ip_cell']))] = g['when']
                    ip['ip_cell'].append(c); ip['ip_pin'].append(pidx[g['pin']])
                    ip['ip_rel'].append(pidx[g['related']] if g['related'] is not None else -1)
                    ip['ip_pg'].append(pg_idx.get(g['pg'], -1))
                    ip_tabs.append(g['tables'])
                for arc in lc['arcs']:
                    t = TYPE_CODE.get(arc['type'], T_OTHER)
                    if t == T_OTHER or arc['from'] not in pidx or arc['to'] not in pidx:
                        continue
                    arcs.append(arc)
                    la['la_cell'].append(c); la['la_from'].append(pidx[arc['from']]); la['la_to'].append(pidx[arc['to']])
                    la['la_type'].append(t); la['la_sense'].append(SENSE_CODE.get(arc['sense'], S_NON))
                    whens.append(arc.get('when', '') or '')
            if cn in macros:
                m = macros[cn]
                # LEF geometry in attar.io.distill's codes (a design adds the cell as distill would have)
                a['lc_size'][c] = (m.w, m.h); a['lc_lef'][c] = 1; lef_class[cn] = m.cls
                a['lc_class'][c] = {'CORE': M_CORE, 'BLOCK': M_BLOCK, 'PAD': M_PAD}.get(m.cls, M_OTHER)
                a['lc_sym'][c] = (1 if 'X' in m.sym else 0) | (2 if 'Y' in m.sym else 0) | (4 if 'R90' in m.sym else 0)
                for j, pn in enumerate(pin_names[c]):
                    if pn in m.pins:
                        box, use = m.pins[pn]
                        k = base + j
                        a['lp_lef'][k] = 1; a['lp_box'][k] = box; a['lp_use'][k] = _USE.get(use, U_SIGNAL)
                        if pn in m.pin_loc:
                            a['lp_loc'][k] = m.pin_loc[pn]
                        a['lp_layer'][k] = lid.get(m.pin_layer.get(pn), -1)
                        a['lp_lefdir'][k] = LEF_DIR.get(m.pin_dir.get(pn), D_UNKNOWN)
        from .ir import _pack_tables
        K = max([len(t[0]) for arc in arcs for t in arc['tables'].values()]
                + [len(t[1]) for arc in arcs for t in arc['tables'].values()] + [1])
        ix1, ix2, val, n = _pack_tables(arcs, K)
        a.update({k: np.asarray(v, np.int32 if k in ('la_cell', 'la_from', 'la_to') else np.uint8) for k, v in la.items()})
        a.update({'la_ix1': ix1, 'la_ix2': ix2, 'la_val': val, 'la_n': n})
        a.update({'lk_cell': np.asarray(lk['lk_cell'], np.int32), 'lk_val': np.asarray(lk['lk_val'], np.float32),
                  'lk_pg': np.asarray(lk['lk_pg'], np.int16), 'lk_primary': np.asarray(lk['lk_primary'], np.uint8)})
        a.update({'ip_cell': np.asarray(ip['ip_cell'], np.int32), 'ip_pin': np.asarray(ip['ip_pin'], np.int32),
                  'ip_rel': np.asarray(ip['ip_rel'], np.int32), 'ip_pg': np.asarray(ip['ip_pg'], np.int16)})
        a.update(dict(zip(('ip_ix1', 'ip_ix2', 'ip_val', 'ip_n'), _pack_power(ip_tabs))))
        meta = {'schema': SCHEMA, 'name': name, 'cells': names, 'pins': pin_names, 'functions': functions,
                'when': {str(i): w for i, w in enumerate(whens) if w}, 'dont_use': dont_use, 'footprint': footprint,
                'pg_pins': pg_names, 'leak_when': lk_when, 'ip_when': ip_when, 'power': bool(power),
                'lef_class': lef_class, 'sequentials': sequentials, 'layers': layers, 'units': units or {},
                'default_thresholds': default_thr, 'sources': {'lib': refs}}
        lib = cls(a, meta)
        lib.meta.update(lib._classes())
        return lib

    def _classes(self):
        """Equivalence classes (cells that can replace each other), buffers and inverters."""
        groups = {}
        a = self.a
        for c, cn in enumerate(self.cells):
            if not self.a['lc_lef'][c] and a['lc_file'][c] < 0:
                continue
            ptr = a['l_pin_ptr']
            sig = [(pn, int(a['lp_dir'][ptr[c] + j])) for j, pn in enumerate(self.pins[c])
                   if a['lp_lib'][ptr[c] + j] and a['lp_dir'][ptr[c] + j] != LD_NONE]
            if not sig:
                continue
            ins = tuple(sorted(p for p, d in sig if d != LIB_DIR['output']))
            fs = self.meta['functions'].get(cn, {})
            try:
                logic = tuple(sorted((o, _truth(f, ins)) for o, f in fs.items()))
                seqk = _seq_key(self.meta.get('sequentials', {}).get(cn), ins)
            except Exception:  # noqa: BLE001  (an unparsable function: the cell is its own class)
                logic, seqk = None, cn
            key = (tuple(sorted(sig)), logic, seqk, int(a['lc_seq'][c]), round(float(a['lc_size'][c][1]), 4))
            groups.setdefault(key, []).append(cn)
        classes = [sorted(v) for v in groups.values()]
        classes.sort()
        bufs, invs = [], []
        for key, v in groups.items():
            sig, logic, seqk, seq, _h = key
            if logic is None or seq or seqk or len(sig) != 2 or len(logic) != 1 or logic[0][1][0]:
                continue
            rows = logic[0][1][1]
            if rows == (0, 1):
                bufs += v
            elif rows == (1, 0):
                invs += v
        return {'classes': classes, 'buffers': sorted(bufs), 'inverters': sorted(invs)}

    # -- files --------------------------------------------------------------------------------------------------------
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        st.save(os.path.join(folder, 'library.safetensors'), self.a, {'schema': SCHEMA})
        with open(os.path.join(folder, 'library.json'), 'w') as f:
            json.dump(self.meta, f)
        return folder

    @classmethod
    def load(cls, folder):
        a, _ = st.load(os.path.join(folder, 'library.safetensors'))
        with open(os.path.join(folder, 'library.json')) as f:
            meta = json.load(f)
        return cls(a, meta)

    def class_of(self, cell):
        """The equivalence class (list of cell names) a cell belongs to, or None."""
        if not hasattr(self, '_class_of'):
            self._class_of = {c: k for k, cl in enumerate(self.meta.get('classes', [])) for c in cl}
        k = self._class_of.get(cell)
        return None if k is None else self.meta['classes'][k]

    def __len__(self):
        return len(self.cells)

    def __repr__(self):
        return 'Library(%r: %d cells, %d arcs, %d classes)' % (self.meta.get('name'), len(self.cells),
                                                             len(self.a['la_cell']), len(self.meta.get('classes', [])))


def design_arrays(chain, masters, mpins, m_ptr, used):
    """The timer's per-design arrays (as attar.timing.ir.attach writes timing.safetensors) for a design's masters, each
    looked up along chain (a list of Library; the first that has a cell wins), its arcs only for the used masters.
    Returns (tensors, functions {master index: {pin: function}}, arc_when {arc index: expr}, missing used masters)."""
    m_ptr = np.asarray(m_ptr, np.int64)
    n_mp = int(m_ptr[-1])
    mp_cap = np.zeros((n_mp, 2), np.float32); mp_cap_min = np.zeros((n_mp, 2), np.float32)
    mp_maxcap = np.zeros(n_mp, np.float32); mp_ctp = np.zeros((n_mp, 2, 2), np.float32)
    mp_thr = np.tile(np.array([.5, .2, .8, 1.0], np.float32), (n_mp, 2, 1))
    mp_dir = np.full(n_mp, LD_NONE, np.uint8); mp_clock = np.zeros(n_mp, np.uint8)
    # design rules: per master pin its limits, the defaults of its cell's library resolved as OpenSTA's findLimit
    # (default_max_* for outputs only) and fanoutLoad (default_fanout_load); NaN: no limit / no fanout load
    mp_maxtran = np.full(n_mp, np.nan, np.float32); mp_caplim = np.full(n_mp, np.nan, np.float32)
    mp_maxfo = np.full(n_mp, np.nan, np.float32); mp_fanload = np.full(n_mp, np.nan, np.float32)
    mp_lib = np.zeros(n_mp, np.uint8)                       # 1: the pin has a Liberty port
    sel, a_master, a_from, a_to, a_thr = [], [], [], [], []
    functions, missing = {}, []
    for m, mname in enumerate(masters):
        hit = next(((L, L.cid[mname]) for L in chain if mname in L.cid and L.a['lc_file'][L.cid[mname]] >= 0), None)
        local = {pn: int(m_ptr[m]) + i for i, pn in enumerate(mpins[m])}
        if hit is None:
            if m in used:
                missing.append(mname)
            continue
        L, c = hit
        A = L.a
        for pn, k_lib in L.pid[c].items():
            if not A['lp_lib'][k_lib] or pn not in local:
                continue
            k = local[pn]
            mp_cap[k] = A['lp_cap'][k_lib]; mp_cap_min[k] = A['lp_cap_min'][k_lib]; mp_maxcap[k] = A['lp_maxcap'][k_lib]
            mp_dir[k] = A['lp_dir'][k_lib]; mp_clock[k] = A['lp_clock'][k_lib]
            for i in (0, 1):
                if A['lp_has_ctp'][k_lib, i]:
                    mp_ctp[k, i] = A['lp_ctp'][k_lib, i]
            mp_thr[k] = A['lc_thr_in'][c]
            mp_lib[k] = 1
            dl = A['lc_dlim'][c] if 'lc_dlim' in A else np.full(4, np.nan, np.float32)
            out = A['lp_dir'][k_lib] in (LIB_DIR['output'], LIB_DIR['inout'])
            own = lambda arr: A[arr][k_lib] if arr in A else np.nan  # noqa: E731
            mt, mc, mf, fl = own('lp_maxtran'), A['lp_maxcap'][k_lib], own('lp_maxfo'), own('lp_fanload')
            mp_maxtran[k] = mt if np.isfinite(mt) else (dl[0] if out else np.nan)
            mp_caplim[k] = mc if mc > 0 else (dl[1] if out else np.nan)
            mp_maxfo[k] = mf if np.isfinite(mf) else (dl[2] if out else np.nan)
            mp_fanload[k] = fl if np.isfinite(fl) else dl[3]
        if m not in used:
            continue
        fs = L.meta['functions'].get(mname)
        if fs:
            functions[str(m)] = fs
        inv = {v: p for p, v in L.pid[c].items()}
        for ai in L.arcs_of[c]:
            fp, tp = inv[int(A['la_from'][ai])], inv[int(A['la_to'][ai])]
            if fp not in local or tp not in local:
                continue
            sel.append((L, int(ai))); a_master.append(m); a_from.append(local[fp]); a_to.append(local[tp])
            a_thr.append(A['lc_thr_out'][c])
    nn = np.stack([L.a['la_n'][ai] for L, ai in sel]) if sel else np.zeros((0, 4, 2), np.int8)
    K = max(int(nn.max()) if len(nn) else 1, 1)

    def fit(x, axes):
        """A table (index or values) at the design's K: cut, or padded as attar.timing.ir._pack_tables pads (indices
        repeat their last value, values are 0)."""
        for ax in axes:
            k = x.shape[ax]
            if k > K:
                x = np.take(x, np.arange(K), axis=ax)
            elif k < K:
                pad = [(0, 0)] * x.ndim; pad[ax] = (0, K - k)
                x = np.pad(x, pad, mode='edge' if len(axes) == 1 else 'constant')
        return x
    ix1 = np.stack([fit(L.a['la_ix1'][ai], (1,)) for L, ai in sel]) if sel else np.zeros((0, 4, K), np.float32)
    ix2 = np.stack([fit(L.a['la_ix2'][ai], (1,)) for L, ai in sel]) if sel else np.zeros((0, 4, K), np.float32)
    val = np.stack([fit(L.a['la_val'][ai], (1, 2)) for L, ai in sel]) if sel else np.zeros((0, 4, K, K), np.float32)
    tens = {'mp_cap': mp_cap, 'mp_cap_min': mp_cap_min, 'mp_maxcap': mp_maxcap, 'mp_ctp': mp_ctp, 'mp_dir': mp_dir,
            'mp_clock': mp_clock,
            'a_master': np.asarray(a_master, np.int32), 'a_from': np.asarray(a_from, np.int32),
            'a_to': np.asarray(a_to, np.int32),
            'a_type': np.asarray([L.a['la_type'][ai] for L, ai in sel], np.uint8),
            'a_sense': np.asarray([L.a['la_sense'][ai] for L, ai in sel], np.uint8),
            'a_ix1': ix1.astype(np.float32), 'a_ix2': ix2.astype(np.float32), 'a_val': val.astype(np.float32),
            'a_n': nn.astype(np.int8),
            'a_thr': np.asarray(a_thr, np.float32).reshape(-1, 2, 4), 'mp_thr': mp_thr,
            'mp_maxtran': mp_maxtran, 'mp_caplim': mp_caplim, 'mp_maxfo': mp_maxfo, 'mp_fanload': mp_fanload,
            'mp_lib': mp_lib}
    arc_when = {}
    for i, (L, ai) in enumerate(sel):
        w = L.meta['when'].get(str(ai))
        if w:
            arc_when[str(i)] = w
    return tens, functions, arc_when, missing


def design_power(chain, masters, mpins, m_ptr, used):
    """What OpenSTA's report_power reads for a design's masters (attar.timing.power), looked up along chain (the first
    library that has a cell wins), the groups for the used masters only:
      per master: cell_leakage_power (nW, NaN: absent), default voltage, kind (KIND_* bits), has Liberty;
      per master pin: the voltage of its related_power_pin (NaN: none: the master's default voltage applies);
      leakage groups lk_master, lk_val (nW), lk_pg (pg pin within the master, -1: none), lk_primary;
      internal power groups ip_master, ip_pin, ip_rel (master pins, -1: no related pin), ip_pg, ip_* tables;
    and the groups' `when` conditions {group index: expr}. Raises ValueError when the chain has no power data
    (a bundle distilled before power: tools/distill_library.py)."""
    m_ptr = np.asarray(m_ptr, np.int64)
    M, n_mp = len(masters), int(m_ptr[-1])
    for L in chain:
        if 'ip_cell' not in L.a:
            raise ValueError('library %r has no power data: it was downloaded or distilled before attar 1.0.2. A zoo '
                             'design: download it again, attar.fetch(<its suite>, force=True) (e.g. '
                             "'after_fp/orfs_asap7'); your own: distill it again" % L.meta.get('name'))
    m_cleak = np.full(M, np.nan, np.float32); m_volt = np.zeros(M, np.float32); m_kind = np.zeros(M, np.uint8)
    m_lib = np.zeros(M, np.uint8); mp_volt = np.full(n_mp, np.nan, np.float32)
    lk = {'lk_master': [], 'lk_val': [], 'lk_pg': [], 'lk_primary': []}
    ip = {'ip_master': [], 'ip_pin': [], 'ip_rel': [], 'ip_pg': []}
    ip_src, lk_when, ip_when = [], {}, {}
    for m, mname in enumerate(masters):
        hit = next(((L, L.cid[mname]) for L in chain if mname in L.cid and L.a['lc_file'][L.cid[mname]] >= 0), None)
        if hit is None:
            continue
        L, c = hit
        A = L.a
        m_lib[m] = 1
        m_cleak[m] = A['lc_cleak'][c]; m_volt[m] = A['lc_volt'][c]; m_kind[m] = A['lc_kind'][c]
        local = {pn: int(m_ptr[m]) + i for i, pn in enumerate(mpins[m])}
        lib_to_local = {k_lib: local[pn] for pn, k_lib in L.pid[c].items() if pn in local}
        for k_lib, k in lib_to_local.items():
            mp_volt[k] = A['lp_volt'][k_lib]
        if m not in used:
            continue
        for g in np.nonzero(A['lk_cell'] == c)[0]:
            w = L.meta.get('leak_when', {}).get(str(int(g)))
            if w:
                lk_when[str(len(lk['lk_master']))] = w
            lk['lk_master'].append(m); lk['lk_val'].append(A['lk_val'][g])
            lk['lk_pg'].append(A['lk_pg'][g]); lk['lk_primary'].append(A['lk_primary'][g])
        for g in np.nonzero(A['ip_cell'] == c)[0]:
            pin, rel = int(A['ip_pin'][g]), int(A['ip_rel'][g])
            if pin not in lib_to_local or (rel >= 0 and rel not in lib_to_local):
                continue
            w = L.meta.get('ip_when', {}).get(str(int(g)))
            if w:
                ip_when[str(len(ip['ip_master']))] = w
            ip['ip_master'].append(m); ip['ip_pin'].append(lib_to_local[pin])
            ip['ip_rel'].append(lib_to_local[rel] if rel >= 0 else -1); ip['ip_pg'].append(A['ip_pg'][g])
            ip_src.append((L, int(g)))
    K = max([int(L.a['ip_n'][g].max()) for L, g in ip_src] + [1])

    def fit(x, axes):                                     # as design_arrays: cut or pad to K
        for ax in axes:
            k = x.shape[ax]
            if k > K:
                x = np.take(x, np.arange(K), axis=ax)
            elif k < K:
                pad = [(0, 0)] * x.ndim; pad[ax] = (0, K - k)
                x = np.pad(x, pad, mode='edge' if len(axes) == 1 else 'constant')
        return x
    st_ = lambda key, ax: (np.stack([fit(L.a[key][g], ax) for L, g in ip_src]) if ip_src  # noqa: E731
                           else np.zeros((0, 2) + (K,) * len(ax), np.float32))
    out = {'m_cleak': m_cleak, 'm_volt': m_volt, 'm_kind': m_kind, 'm_lib': m_lib, 'mp_volt': mp_volt,
           'lk_master': np.asarray(lk['lk_master'], np.int64), 'lk_val': np.asarray(lk['lk_val'], np.float32),
           'lk_pg': np.asarray(lk['lk_pg'], np.int64), 'lk_primary': np.asarray(lk['lk_primary'], np.uint8),
           'ip_master': np.asarray(ip['ip_master'], np.int64), 'ip_pin': np.asarray(ip['ip_pin'], np.int64),
           'ip_rel': np.asarray(ip['ip_rel'], np.int64), 'ip_pg': np.asarray(ip['ip_pg'], np.int64),
           'ip_ix1': st_('ip_ix1', (1,)), 'ip_ix2': st_('ip_ix2', (1,)), 'ip_val': st_('ip_val', (1, 2)),
           'ip_n': (np.stack([L.a['ip_n'][g] for L, g in ip_src]) if ip_src else np.zeros((0, 2, 2), np.int8))}
    return out, {'leak_when': lk_when, 'ip_when': ip_when}


def pin_locations(chain, masters, mpins, m_ptr):
    """Per master pin of a design (m_ptr CSR over masters, mpins their pin names), its LEF location in microns from
    the master's lower-left in orientation N: the mean of its rectangles' centers, as OpenROAD's dbITerm::getAvgXY
    (estimate_parasitics puts the pin there), looked up along chain; NaN where no library has it (a bundle distilled
    before lp_loc, a cell without LEF)."""
    m_ptr = np.asarray(m_ptr, np.int64)
    out = np.full((int(m_ptr[-1]), 2), np.nan)
    for m, mname in enumerate(masters):
        hit = next(((L, L.cid[mname]) for L in chain if mname in L.cid and 'lp_loc' in L.a), None)
        if hit is None:
            continue
        L, c = hit
        for i, pn in enumerate(mpins[m]):
            k = L.pid[c].get(pn)
            if k is not None:
                out[m_ptr[m] + i] = L.a['lp_loc'][k]
    return out
