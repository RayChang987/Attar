"""Gate sizing on a Design: swap cells to other masters of their equivalence class (drive strengths, threshold-voltage
flavors), as OpenROAD's replace_cell. Masters the design does not have yet come from its cell library chain
(attar.timing.chain). Design.library, Design.equivalents and Design.resize call these."""
import numpy as np
import torch


def library(d):
    """The design's cell library chain (attar.timing.chain): its local library (macros), then the shared bundle."""
    d.require('resize')
    from .timing import chain
    return chain(d.path, d.manifest)


def lib_cell(d, name):
    """(library, cell id) of the first library in the chain that has `name` with Liberty data, or (None, None)."""
    for L in library(d):
        c = L.cid.get(name)
        if c is not None and L.a['lc_file'][c] >= 0:
            return L, c
    return None, None


def cell_index(d, cell):
    """A cell given by index or instance name -> its index."""
    if isinstance(cell, str):
        if getattr(d, '_cell_id', None) is None:
            d._cell_id = {n: i for i, n in enumerate(d.names['cells'])}
        return d._cell_id[cell]
    return int(cell)


def equivalents(d, cell, dont_use=False):
    """The masters `cell` (index or name) can be resized to: its master's equivalence class in the library, without
    the library's dont_use cells unless dont_use=True."""
    c = cell_index(d, cell)
    m = d.names['masters'][int(d.raw['c_master'][c])]
    L, _ = lib_cell(d, m)
    cl = (L.class_of(m) if L is not None else None) or [m]
    skip = set() if dont_use or L is None else set(L.meta.get('dont_use', []))
    return [x for x in cl if x == m or x not in skip]


def add_master(d, name):
    """Append a library cell to the design's masters (LEF geometry in dbu, pins sorted by name, as distill)."""
    L, c = lib_cell(d, name)
    if L is None or not L.a['lc_lef'][c]:
        raise KeyError('%s: not a cell with LEF geometry in the design\'s library' % name)
    a, n = d.raw, d.names
    q = lambda v: int(round(float(v) * d.dbu))  # noqa: E731
    ptr = int(L.a['l_pin_ptr'][c])
    pins = sorted(pn for j, pn in enumerate(L.pins[c]) if L.a['lp_lef'][ptr + j])
    k_of = {pn: ptr + j for j, pn in enumerate(L.pins[c])}
    lid = {ln: i for i, ln in enumerate(n.get('layers', []))}
    lay = lambda k: lid.get(L.meta['layers'][L.a['lp_layer'][k]], -1) if L.a['lp_layer'][k] >= 0 else -1  # noqa: E731
    add = {'m_size': np.array([[q(L.a['lc_size'][c][0]), q(L.a['lc_size'][c][1])]], a['m_size'].dtype),
           'm_class': np.array([L.a['lc_class'][c]], a['m_class'].dtype),
           'm_sym': np.array([L.a['lc_sym'][c]], a['m_sym'].dtype),
           'm_pin_box': np.array([[q(v) for v in L.a['lp_box'][k_of[pn]]] for pn in pins], a['m_pin_box'].dtype).reshape(-1, 4),
           'm_pin_use': np.array([L.a['lp_use'][k_of[pn]] for pn in pins], a['m_pin_use'].dtype),
           'm_pin_layer': np.array([lay(k_of[pn]) for pn in pins], a['m_pin_layer'].dtype),
           'm_pin_dir': np.array([L.a['lp_lefdir'][k_of[pn]] for pn in pins], a['m_pin_dir'].dtype)}
    for k, v in add.items():
        if k in a:
            a[k] = np.concatenate([a[k], v])
    a['m_pin_ptr'] = np.concatenate([a['m_pin_ptr'], np.array([a['m_pin_ptr'][-1] + len(pins)], a['m_pin_ptr'].dtype)])
    n['masters'].append(name); n['master_pins'].append(pins)
    return len(n['masters']) - 1


def resize(d, cells, masters, check=True):
    """Swap cells (indices or names) to other masters (one name, or one per cell). check: each new master must be in
    its cell's equivalence class. The cells keep their position and orientation; size and pin offsets follow the new
    master; the timer is rebuilt on the next timing(). Returns the previous master names."""
    d.require('resize')
    cells = [cells] if isinstance(cells, (str, int, np.integer)) else list(cells)
    idx = [cell_index(d, c) for c in cells]
    masters = [masters] * len(idx) if isinstance(masters, str) else list(masters)
    if len(masters) != len(idx):
        raise ValueError('%d cells but %d masters' % (len(idx), len(masters)))
    a, n = d.raw, d.names
    a['c_master'] = a['c_master'].copy(); a['p_mpin'] = a['p_mpin'].copy()
    mid = {m: i for i, m in enumerate(n['masters'])}
    p_cell = a['p_cell'].astype(np.int64)
    order = np.argsort(p_cell, kind='stable'); start = np.searchsorted(p_cell[order], np.arange(len(a['c_master']) + 1))
    old_names, touched = [], []
    for c, new in zip(idx, masters):
        om = int(a['c_master'][c])
        old_names.append(n['masters'][om])
        if new == n['masters'][om]:
            continue
        if check and new not in equivalents(d, c, dont_use=True):
            raise ValueError('%s: %s is not in the equivalence class of %s' % (n['cells'][c], new, n['masters'][om]))
        if new not in mid:
            mid[new] = add_master(d, new)
        nm = mid[new]
        pos_new = {pn: int(a['m_pin_ptr'][nm]) + j for j, pn in enumerate(n['master_pins'][nm])}
        for p in order[start[c]:start[c + 1]]:
            k = int(a['p_mpin'][p])
            if k < 0:
                continue
            pn = n['master_pins'][om][k - int(a['m_pin_ptr'][om])]
            if pn not in pos_new:
                raise ValueError('%s: pin %s of %s not in %s' % (n['cells'][c], pn, n['masters'][om], new))
            a['p_mpin'][p] = pos_new[pn]
            touched.append(p)
        a['c_master'][c] = nm
    # derived geometry of the cells and pins that changed
    s = d.scale
    cm = torch.as_tensor(a['c_master'].astype(np.int64), device=d.device)
    msize = torch.as_tensor(a['m_size'].astype(np.float64) * s, dtype=torch.float32, device=d.device)
    ii = torch.as_tensor(idx, device=d.device)
    d.size_n[ii] = msize[cm[ii]]
    if touched and 'pin_offsets' in d.facts:
        tp = np.asarray(touched); box = a['m_pin_box'][a['p_mpin'][tp]].astype(np.float64) * s
        ctr = np.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], -1)
        d._p_ctr_n[torch.as_tensor(tp, device=d.device)] = torch.as_tensor(ctr, dtype=torch.float32, device=d.device)
    d.update_orient()
    d._timer = None
    return old_names
