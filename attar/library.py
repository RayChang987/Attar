"""Cell libraries as tensors (the 'library' fact): Liberty files, and the LEF of their cells, distilled once.

A Library holds every cell of a set of Liberty files (the first definition of a cell wins, as in OpenSTA):
  library.safetensors  per cell: area, leakage (nW), sequential kind, the Liberty thresholds of its file (input /
                       output, slew lower / upper, derate), LEF size and class; per pin (CSR l_pin_ptr): direction,
                       rise / fall capacitance (and the lower bounds of their ranges), max capacitance, clock flag,
                       macro clock tree path, LEF box / use / layer; per timing arc: cell, from / to pin, type, sense,
                       NLDM tables padded to K x K, in ps / fF.
  library.json         cell and pin names, output functions, arc `when` conditions, dont_use, footprints, equivalence
                       classes (cells that can replace each other: same signal pins, same per-output logic, same
                       sequential kind, same height), buffers and inverters, sources (name + sha256), units.
Designs reference a shared bundle per platform and corner (zoo/libraries/<name>) and keep the cells only they have
(macros) in a local library in their own folder; design_arrays() looks a design's masters up along that chain and
returns the per-master-pin and per-arc arrays the timer reads (attar.timing.attach writes the same arrays).
"""
import itertools
import json
import os

import numpy as np

from . import liberty, st

SCHEMA = 'attar-library/0.1'
# Liberty pin directions, timing types and senses: attar.timing's codes
from .timing import LD_NONE, LIB_DIR, S_NON, SENSE_CODE, T_OTHER, TYPE_CODE  # noqa: E402
from .distill import D_UNKNOWN, LEF_DIR, M_BLOCK, M_CORE, M_OTHER, M_PAD, U_SIGNAL, _USE  # noqa: E402
SEQ_CODE = {'': 0, 'ff': 1, 'latch': 2}


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
    def build(cls, lib_paths, lef_paths=(), name='', source_ref=None):
        """Read Liberty files (and the LEF files of their cells) into a Library. source_ref(path) -> {name, sha256}."""
        cells, cell_thr, cell_file, refs = {}, {}, {}, []
        default_thr, units = None, None
        for i, p in enumerate(lib_paths):
            lib = liberty.read(p)
            refs.append(source_ref(p) if source_ref else {'name': os.path.basename(p)})
            default_thr = default_thr or lib['thresholds']
            units = units or {'time_ps': lib['time_unit_ps'], 'cap_ff': lib['cap_unit_ff']}
            for k, v in lib['cells'].items():
                if k not in cells:
                    cells[k] = v; cell_thr[k] = lib['thresholds']; cell_file[k] = i
        macros, tech = {}, {'sites': {}, 'layers': [], 'vias': {}}
        if lef_paths:
            from .lef import read_lef
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
             'lp_use': np.full(L, U_SIGNAL, np.uint8), 'lp_layer': np.full(L, -1, np.int16),
             'lp_lefdir': np.full(L, D_UNKNOWN, np.uint8)}
        arcs, la = [], {'la_cell': [], 'la_from': [], 'la_to': [], 'la_type': [], 'la_sense': []}
        functions, whens, dont_use, footprint, lef_class = {}, [], [], {}, {}
        for c, cn in enumerate(names):
            lc = cells.get(cn)
            base = ptr[c]
            if lc is not None:
                a['lc_area'][c] = lc.get('area', 0.0); a['lc_leak'][c] = lc.get('leakage', 0.0)
                a['lc_seq'][c] = SEQ_CODE.get(lc.get('seq', ''), 0); a['lc_file'][c] = cell_file[cn]
                a['lc_thr_in'][c] = _thr4(cell_thr[cn], 'input'); a['lc_thr_out'][c] = _thr4(cell_thr[cn], 'output')
                if lc.get('dont_use'):
                    dont_use.append(cn)
                if lc.get('footprint'):
                    footprint[cn] = lc['footprint']
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
                    for i, mm in enumerate(('min', 'max')):
                        if mm in pa.get('ctp', {}):
                            a['lp_ctp'][k, i] = pa['ctp'][mm]; a['lp_has_ctp'][k, i] = 1
                    if pa.get('function') and pa['dir'] in ('output', 'inout'):
                        fs[pn] = pa['function']
                if fs:
                    functions[cn] = fs
                pidx = {p: base + j for j, p in enumerate(pin_names[c])}
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
                # LEF geometry in attar.distill's codes (a design adds the cell as distill would have)
                a['lc_size'][c] = (m.w, m.h); a['lc_lef'][c] = 1; lef_class[cn] = m.cls
                a['lc_class'][c] = {'CORE': M_CORE, 'BLOCK': M_BLOCK, 'PAD': M_PAD}.get(m.cls, M_OTHER)
                a['lc_sym'][c] = (1 if 'X' in m.sym else 0) | (2 if 'Y' in m.sym else 0) | (4 if 'R90' in m.sym else 0)
                for j, pn in enumerate(pin_names[c]):
                    if pn in m.pins:
                        box, use = m.pins[pn]
                        k = base + j
                        a['lp_lef'][k] = 1; a['lp_box'][k] = box; a['lp_use'][k] = _USE.get(use, U_SIGNAL)
                        a['lp_layer'][k] = lid.get(m.pin_layer.get(pn), -1)
                        a['lp_lefdir'][k] = LEF_DIR.get(m.pin_dir.get(pn), D_UNKNOWN)
        from .timing import _pack_tables
        K = max([len(t[0]) for arc in arcs for t in arc['tables'].values()]
                + [len(t[1]) for arc in arcs for t in arc['tables'].values()] + [1])
        ix1, ix2, val, n = _pack_tables(arcs, K)
        a.update({k: np.asarray(v, np.int32 if k in ('la_cell', 'la_from', 'la_to') else np.uint8) for k, v in la.items()})
        a.update({'la_ix1': ix1, 'la_ix2': ix2, 'la_val': val, 'la_n': n})
        meta = {'schema': SCHEMA, 'name': name, 'cells': names, 'pins': pin_names, 'functions': functions,
                'when': {str(i): w for i, w in enumerate(whens) if w}, 'dont_use': dont_use, 'footprint': footprint,
                'lef_class': lef_class, 'layers': layers, 'units': units or {}, 'default_thresholds': default_thr,
                'sources': {'lib': refs}}
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
            except Exception:  # noqa: BLE001  (an unparsable function: the cell is its own class)
                logic = (cn,)
            key = (tuple(sorted(sig)), logic, int(a['lc_seq'][c]), round(float(a['lc_size'][c][1]), 4))
            groups.setdefault(key, []).append(cn)
        classes = [sorted(v) for v in groups.values()]
        classes.sort()
        bufs, invs = [], []
        for key, v in groups.items():
            sig, logic, seq, _h = key
            if seq or len(sig) != 2 or len(logic) != 1 or logic[0][1][0]:
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
    """The timer's per-design arrays (as attar.timing.attach writes timing.safetensors) for a design's masters, each
    looked up along chain (a list of Library; the first that has a cell wins), its arcs only for the used masters.
    Returns (tensors, functions {master index: {pin: function}}, arc_when {arc index: expr}, missing used masters)."""
    m_ptr = np.asarray(m_ptr, np.int64)
    n_mp = int(m_ptr[-1])
    mp_cap = np.zeros((n_mp, 2), np.float32); mp_cap_min = np.zeros((n_mp, 2), np.float32)
    mp_maxcap = np.zeros(n_mp, np.float32); mp_ctp = np.zeros((n_mp, 2, 2), np.float32)
    mp_thr = np.tile(np.array([.5, .2, .8, 1.0], np.float32), (n_mp, 2, 1))
    mp_dir = np.full(n_mp, LD_NONE, np.uint8); mp_clock = np.zeros(n_mp, np.uint8)
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
        """A table (index or values) at the design's K: cut, or padded as attar.timing._pack_tables pads (indices
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
            'a_thr': np.asarray(a_thr, np.float32).reshape(-1, 2, 4), 'mp_thr': mp_thr}
    arc_when = {}
    for i, (L, ai) in enumerate(sel):
        w = L.meta['when'].get(str(ai))
        if w:
            arc_when[str(i)] = w
    return tens, functions, arc_when, missing
