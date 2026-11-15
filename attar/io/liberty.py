"""Liberty (.lib, .lib.gz) reader: what static timing, design rule checks and power analysis need, nothing else.

Per cell: pin direction, capacitance (rise / fall), clock flag, the design rule limits (max_capacitance,
max_transition, max_fanout, fanout_load), and every timing group (related_pin, timing_sense, timing_type, when) with
its NLDM tables: cell_rise / cell_fall / rise_transition / fall_transition for delay arcs, rise_constraint /
fall_constraint for checks; its sequentials (ff, latch and their banks) and statetable, which decide what may replace
the cell. Per library: the default limits (default_max_transition, ...), voltage_map, nom_voltage.
With power=True also what OpenSTA's report_power reads (power/Power.cc): internal_power groups (related_pin,
related_pg_pin, when, rise_power / fall_power / power tables), leakage_power groups (value, when, related_pg_pin),
cell_leakage_power, pg_pin groups (pg_type, voltage_name), related_power_pin and the default operating conditions'
voltage; else those groups are skipped while parsing. Noise and CCS groups are always skipped.
Units are normalized to ps, fF, nW and fJ (internal power energy: the library's voltage unit squared times its
capacitance unit, OpenSTA's LibertyReader::energyScale); table axes to (input slew, load) for delays and internal
power, and (constrained pin slew, related pin slew) for checks, whatever order the library's templates use.
"""
import gzip
import re

_TOK = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:[^"\\]|\\.)*"|\\[ \t\r]*\n|[{}();:,]|[^\s{}();:,"]+', re.S)
# groups never needed for timing (large: power tables, noise, CCS)
SKIP = {'internal_power', 'power', 'ccsn_first_stage', 'ccsn_last_stage', 'output_current_rise',
        'output_current_fall', 'receiver_capacitance', 'pg_pin', 'dynamic_current', 'leakage_current',
        'intrinsic_parasitic', 'gate_leakage', 'normalized_driver_waveform', 'test_cell',
        'power_lut_template', 'output_voltage', 'input_voltage', 'operating_conditions', 'wire_load',
        'wire_load_selection', 'bundle', 'memory'}
SEQUENTIALS = ('ff', 'latch', 'ff_bank', 'latch_bank')
# kept with power=True (OpenSTA's report_power reads them)
POWER_GROUPS = {'internal_power', 'power', 'pg_pin', 'power_lut_template', 'operating_conditions'}
TABLES = ('cell_rise', 'cell_fall', 'rise_transition', 'fall_transition', 'rise_constraint', 'fall_constraint')
POWER_TABLES = ('rise_power', 'fall_power', 'power')
TIME_SCALE = {'1ps': 1.0, '10ps': 10.0, '100ps': 100.0, '1ns': 1000.0, '10ns': 1e4, '100ns': 1e5}
CAP_SCALE = {'ff': 1.0, 'pf': 1000.0, 'nf': 1e6}
VOLT_SCALE = {'1v': 1.0, '100mv': 0.1, '10mv': 0.01, '1mv': 1e-3}
POWER_SCALE = {'1pw': 1e-3, '10pw': 1e-2, '100pw': 1e-1, '1nw': 1.0, '10nw': 10.0, '100nw': 100.0, '1uw': 1e3,
               '1mw': 1e6, '1w': 1e9}


class Group:
    __slots__ = ('kind', 'args', 'attrs', 'groups')

    def __init__(self, kind, args):
        self.kind, self.args, self.attrs, self.groups = kind, args, {}, []

    def sub(self, kind):
        return [g for g in self.groups if g.kind == kind]


def _tokens(text):
    for m in _TOK.finditer(text):
        t = m.group(0)
        if t[0] == '/' and len(t) > 1 and t[1] in '*/' or (t[0] == '\\' and t[-1] == '\n'):    # comments, line continuations
            continue
        yield t


def _strip(t):
    return t[1:-1] if t[:1] == '"' else t


def parse(text, skip=SKIP):
    """Liberty text -> the library Group (kinds in skip are dropped with their contents)."""
    it = _tokens(text)
    root = None
    stack = []
    skip_depth = 0
    pending = []                           # tokens of the statement being read
    for t in it:
        if t == '{':
            # pending: kind ( args ), possibly after a simple attribute written without ';' (ASAP7: 'area : 0.2'
            # on its own line): that statement ends where the group's name starts
            j = len(pending) - 1 - pending[::-1].index('(') if '(' in pending else 1
            if j >= 2 and not skip_depth and stack:
                finish(stack[-1], pending[:j - 1])
            kind = pending[max(j - 1, 0)]
            args = [_strip(a) for a in pending[j + 1:-1] if a != ',']
            pending = []
            if skip_depth or kind in skip:
                if not skip_depth and stack:               # remember that the parent had one (e.g. a cell's ff)
                    stack[-1].attrs.setdefault('@skipped', set()).add(kind)
                skip_depth += 1
                continue
            g = Group(kind, args)
            if stack:
                stack[-1].groups.append(g)
            else:
                root = g
            stack.append(g)
        elif t == '}':
            if not skip_depth and stack and pending:
                finish(stack[-1], pending)                                  # last statement without ';'
            if skip_depth:
                skip_depth -= 1
            else:
                stack.pop()
            pending = []
        elif t == ';':
            if not skip_depth and stack and pending:
                finish(stack[-1], pending)
            pending = []
        else:
            # a complex attribute 'name ( ... )' without ';' ends where the next statement starts
            if len(pending) >= 3 and pending[1] == '(' and pending[-1] == ')' and depth(pending) == 0:
                if not skip_depth and stack:
                    finish(stack[-1], pending)
                pending = []
            pending.append(t)
    return root


def depth(tokens):
    return tokens.count('(') - tokens.count(')')


def finish(g, pending):
    if len(pending) >= 3 and pending[1] == ':':                             # simple attribute
        g.attrs[pending[0]] = _strip(' '.join(pending[2:]))
    elif len(pending) >= 2 and pending[1] == '(':                           # complex attribute
        vals = [_strip(a) for a in pending[2:] if a not in (',', ')')]
        g.attrs.setdefault(pending[0], []).append(vals)


def _sequentials(cell):
    """A cell's ff / latch / ff_bank / latch_bank groups, in file order (OpenSTA's Sequential: clock, data, clear,
    preset and the internal state names)."""
    out = []
    for g in cell.groups:
        if g.kind in SEQUENTIALS:
            a, ff = g.attrs, g.kind.startswith('ff')
            out.append({'kind': g.kind, 'vars': list(g.args),
                        'clock': a.get('clocked_on' if ff else 'enable'),
                        'data': a.get('next_state' if ff else 'data_in'),
                        'clear': a.get('clear'), 'preset': a.get('preset')})
    return out


def _statetable(cell):
    """A cell's statetable: input and internal node names, and its rows with blanks removed (None without one)."""
    for g in cell.groups:
        if g.kind == 'statetable':
            names = [x.split() for x in g.args] + [[], []]
            rows = [re.sub(r'[\s\\]+', '', r) for r in g.attrs.get('table', '').split(',')]
            return {'inputs': names[0], 'internals': names[1], 'rows': [r for r in rows if r]}
    return None


def _floats(s):
    return [float(v) for v in re.split(r'[,\s\\]+', s.strip()) if v]      # a '\' line continuation inside the string


def _table(g, templates, kind):
    """A table group -> (index_1, index_2, values 2D) on the axes (slew, load) for delays and internal power, or
    (constrained, related) for checks."""
    tmpl = templates.get(g.args[0] if g.args else '', {})
    ix = {}
    for k in ('index_1', 'index_2'):
        v = g.attrs.get(k) or tmpl.get(k)
        ix[k] = _floats(v[0][0]) if v else []
    flat =[x for row in g.attrs.get('values', [[]])[0] for x in _floats(row)]
    n1, n2 = max(1, len(ix['index_1'])), max(1, len(ix['index_2']))
    if len(flat) != n1 * n2:
        if len(flat) == 1:
            n1 = n2 = 1
        else:
            raise ValueError('table %s: %d values for %dx%d' % (kind, len(flat), n1, n2))
    v = [flat[i * n2:(i + 1) * n2] for i in range(n1)]
    i1, i2 = ix['index_1'] or [0.0], ix['index_2'] or [0.0]
    var1, var2 = tmpl.get('variable_1', ''), tmpl.get('variable_2', '')
    delay = kind in TABLES[:4] or kind in POWER_TABLES
    # wanted axis order: delay and power (input transition, output load); check (constrained pin, related pin)
    first = ('input_net_transition', 'input_transition_time') if delay else ('constrained_pin_transition',)
    if var1 and var1 not in first and (var2 in first or not var2):
        i1, i2 = i2, i1
        v = [list(r) for r in zip(*v)] if len(v) and len(v[0]) else v
    return i1, i2, v


def read(path, power=False):
    """A Liberty file -> {'name', 'time_unit_ps', 'cap_unit_ff', 'cells': {cell: {'pins': {...}, 'arcs': [...], 'area',
    'leakage' (nW), 'dont_use', 'footprint', 'seq' ('ff' | 'latch' | ''), 'sequentials', 'statetable', 'macro'}},
    'thresholds', 'limits', 'voltage_map', 'nom_voltage'}.
    Pin: {'dir': 'input'|'output'|'inout'|'internal', 'cap': (rise, fall) fF, 'cap_min': the *_capacitance_range lower
    bounds (min analysis), 'clock': bool, 'max_cap': fF, 'max_transition': ps, 'max_fanout', 'fanout_load' (each None
    when absent), 'related_power_pin'}.
    Arc: {'from', 'to', 'sense', 'type', 'when', 'tables': {kind: (index_1, index_2, values)}} in ps / fF.
    Sequentials, in file order, as OpenSTA's Sequential: {'kind': 'ff' | 'latch' | 'ff_bank' | 'latch_bank', 'vars'
    (the internal state names, e.g. IQ IQN; a bank's width last), 'clock' (clocked_on / enable), 'data' (next_state /
    data_in), 'clear', 'preset'} (functions as written, None when absent). Statetable: {'inputs', 'internals', 'rows'}
    (the table's rows with blanks removed) or None.
    Limits: the library's default_max_transition (ps), default_max_capacitance (fF), default_max_fanout,
    default_fanout_load (None when absent). voltage_map: {name: volts}.
    power=True adds per cell 'pg_pins' {name: {'type', 'voltage_name'}}, 'leakage_groups' [{'value' nW, 'when',
    'pg'}], 'cell_leakage' (nW, None when absent), 'internal_power' [{'pin', 'related' (a pin or None), 'pg', 'when',
    'tables': {'rise' | 'fall': (index_1 slews ps, index_2 loads fF, energies fJ)}}] (a group with several related
    pins is one entry per pin, as OpenSTA's makeInternalPower), and per library 'default_voltage' (the default
    operating conditions' voltage, None without them)."""
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', errors='replace') as f:
        lib = parse(f.read(), SKIP - POWER_GROUPS if power else SKIP)
    tu = TIME_SCALE.get(lib.attrs.get('time_unit', '1ns').replace(' ', '').lower(), 1000.0)
    cu = 1.0
    if 'capacitive_load_unit' in lib.attrs:
        n, u = lib.attrs['capacitive_load_unit'][0][:2]
        cu = float(n) * CAP_SCALE.get(u.lower(), 1.0)
    lu = POWER_SCALE.get(lib.attrs.get('leakage_power_unit', '1nW').replace(' ', '').lower(), 1.0)   # -> nW
    vu = VOLT_SCALE.get(lib.attrs.get('voltage_unit', '1V').replace(' ', '').lower(), 1.0)
    eu = vu * vu * cu                                   # internal power energy unit -> fJ (V^2 fF)
    templates = {}
    for g in lib.groups:
        if g.kind in ('lu_table_template', 'power_lut_template') and g.args:
            t = dict((k, v) for k, v in g.attrs.items() if k.startswith('variable'))
            for k in ('index_1', 'index_2'):
                if k in g.attrs:
                    t[k] = g.attrs[k]
            templates[g.args[0]] = t
    types = {g.args[0]: g.attrs for g in lib.groups if g.kind == 'type' and g.args}
    fnum = lambda a, k, scale=1.0: _num(a[k]) * scale if k in a else None   # noqa: E731
    cells = {}
    for c in lib.sub('cell'):
        pins, arcs, ipower = {}, [], []
        ctypes = dict(types, **{g.args[0]: g.attrs for g in c.groups if g.kind == 'type' and g.args})
        buses = {}                                     # bus name -> its bit names (a related_pin may name a bus)
        groups = []                                    # (group, pin names it defines, attrs inherited from a bus)
        for p in c.sub('pin'):
            groups.append((p, p.args, {}))
        for b in c.sub('bus'):
            ty = ctypes.get(b.attrs.get('bus_type', ''), {})
            lo, hi = int(ty.get('bit_from', 0)), int(ty.get('bit_to', 0))
            bits = ['%s[%d]' % (b.args[0], i) for i in range(min(lo, hi), max(lo, hi) + 1)]
            buses[b.args[0]] = bits
            groups.append((b, bits, {}))
            for p in b.sub('pin'):                     # per-bit overrides inside the bus
                groups.append((p, p.args, b.attrs))

        def expand(names):
            return [x for n in names for x in buses.get(n, [n])]
        for g, names, inherit in groups:
            a = dict(inherit, **g.attrs)
            cap = float(a.get('capacitance', 0.0)) * cu
            rc = float(a.get('rise_capacitance', cap / cu)) * cu
            fc = float(a.get('fall_capacitance', cap / cu)) * cu
            rng = lambda k, v: float(a[k][0][0]) * cu if k in a else v      # lower bound: OpenSTA's min analysis
            cmin = (rng('rise_capacitance_range', rc), rng('fall_capacitance_range', fc))
            for name in names:
                pins[name] = {'dir': a.get('direction', 'input'), 'cap': (rc, fc), 'cap_min': cmin,
                              'clock': a.get('clock', 'false') == 'true',
                              'max_cap': fnum(a, 'max_capacitance', cu), 'max_transition': fnum(a, 'max_transition', tu),
                              'max_fanout': fnum(a, 'max_fanout'), 'fanout_load': fnum(a, 'fanout_load'),
                              'related_power_pin': a.get('related_power_pin'), 'function': a.get('function')}
            for t in g.sub('timing'):
                ta = t.attrs
                if ta.get('timing_type') in ('min_clock_tree_path', 'max_clock_tree_path'):
                    # a macro's internal clock insertion delay at its clock pin (no related pin)
                    v = {}
                    for tg in t.groups:
                        if tg.kind in ('cell_rise', 'cell_fall'):
                            flat = [x for row in tg.attrs.get('values', [[]])[0] for x in _floats(row)]
                            v[tg.kind] = max(flat) * tu if flat else 0.0
                    for name in names:
                        pins[name].setdefault('ctp', {})[ta['timing_type'][:3]] = (v.get('cell_rise', 0.0), v.get('cell_fall', 0.0))
                    continue
                tables = {}
                for tg in t.groups:
                    if tg.kind in TABLES:
                        i1, i2, v = _table(tg, templates, tg.kind)
                        i1 = [x * tu for x in i1]
                        i2 = [x * (tu if tg.kind in TABLES[4:] else cu) for x in i2]
                        tables[tg.kind] = (i1, i2, [[x * tu for x in r] for r in v])
                for rp in expand(ta.get('related_pin', '').split()):
                    for name in names:
                        arcs.append({'from': rp, 'to': name, 'sense': ta.get('timing_sense', ''),
                                     'type': ta.get('timing_type', 'combinational'), 'when': ta.get('when', ''),
                                     'tables': tables})
            # internal power (LibertyReader::readInternalPowerGroups): rise_power / fall_power, or power for both
            for ip in g.sub('internal_power'):
                ia = ip.attrs
                tabs = {}
                for tg in ip.groups:
                    if tg.kind in POWER_TABLES:
                        i1, i2, v = _table(tg, templates, tg.kind)
                        tab = ([x * tu for x in i1], [x * cu for x in i2], [[x * eu for x in r] for r in v])
                        for rf in (('rise', 'fall') if tg.kind == 'power' else (tg.kind[:4],)):
                            tabs[rf] = tab
                rel = expand(ia.get('related_pin', '').split()) or [None]
                for name in names:
                    for rp in rel:
                        ipower.append({'pin': name, 'related': rp, 'pg': ia.get('related_pg_pin'),
                                       'when': ia.get('when') or None, 'tables': tabs})
        seqs = _sequentials(c)
        kinds = {q['kind'].split('_')[0] for q in seqs}
        cells[c.args[0]] = {'pins': pins, 'arcs': arcs, 'area': _num(c.attrs.get('area', 0.0)),
                            'leakage': _leakage(c) * lu,
                            'dont_use': c.attrs.get('dont_use', 'false') == 'true',
                            'footprint': c.attrs.get('cell_footprint', ''),
                            'seq': 'ff' if 'ff' in kinds else 'latch' if 'latch' in kinds else '',
                            'sequentials': seqs, 'statetable': _statetable(c),
                            'macro': c.attrs.get('is_macro_cell', 'false') == 'true',
                            'memory': c.attrs.get('is_memory', 'false') == 'true' or 'memory' in c.attrs.get('@skipped', ()),
                            'pad': c.attrs.get('is_pad', c.attrs.get('pad_cell', 'false')) == 'true'}
        if power:
            cells[c.args[0]].update({
                'pg_pins': {g.args[0]: {'type': g.attrs.get('pg_type'), 'voltage_name': g.attrs.get('voltage_name')}
                            for g in c.sub('pg_pin') if g.args},
                'leakage_groups': [{'value': _num(g.attrs['value']) * lu, 'when': g.attrs.get('when') or None,
                                    'pg': g.attrs.get('related_pg_pin')}
                                   for g in c.sub('leakage_power') if 'value' in g.attrs],
                'cell_leakage': fnum(c.attrs, 'cell_leakage_power', lu),
                'internal_power': ipower})
    # thresholds (fractions of the supply), OpenSTA's defaults when absent: 50% delay, 20% / 80% slew, derate 1
    pct = lambda k, d: float(lib.attrs[k]) / 100.0 if k in lib.attrs else d
    thr = {'input': (pct('input_threshold_pct_rise', .5), pct('input_threshold_pct_fall', .5)),
           'output': (pct('output_threshold_pct_rise', .5), pct('output_threshold_pct_fall', .5)),
           'slew_lower': (pct('slew_lower_threshold_pct_rise', .2), pct('slew_lower_threshold_pct_fall', .2)),
           'slew_upper': (pct('slew_upper_threshold_pct_rise', .8), pct('slew_upper_threshold_pct_fall', .8)),
           'slew_derate': float(lib.attrs.get('slew_derate_from_library', 1.0))}
    la = lib.attrs
    limits = {'default_max_transition': fnum(la, 'default_max_transition', tu),
              'default_max_capacitance': fnum(la, 'default_max_capacitance', cu),
              'default_max_fanout': fnum(la, 'default_max_fanout'), 'default_fanout_load': fnum(la, 'default_fanout_load')}
    vmap = {}
    for v in la.get('voltage_map', []):
        if len(v) >= 2:
            try:
                vmap[v[0]] = float(v[1]) * vu
            except ValueError:
                pass
    out = {'name': lib.args[0] if lib.args else '', 'time_unit_ps': tu, 'cap_unit_ff': cu, 'cells': cells,
           'thresholds': thr, 'limits': limits, 'voltage_map': vmap, 'nom_voltage': fnum(la, 'nom_voltage', vu)}
    if power:
        oc = {g.args[0]: g.attrs for g in lib.sub('operating_conditions') if g.args}
        dflt = oc.get(la.get('default_operating_conditions', ''))
        out['default_voltage'] = fnum(dflt, 'voltage', vu) if dflt else None
    return out


def _num(v):
    """A simple attribute's number (a value written without ';' runs into the next statement: its first token)."""
    try:
        return float(str(v).split()[0])
    except (ValueError, IndexError):
        return 0.0


def _leakage(c):
    """A cell's leakage in the library's unit: cell_leakage_power, else the mean of its leakage_power groups (states)."""
    if 'cell_leakage_power' in c.attrs:
        return _num(c.attrs['cell_leakage_power'])
    vals = [_num(g.attrs['value']) for g in c.sub('leakage_power') if 'value' in g.attrs]
    return sum(vals) / len(vals) if vals else 0.0


# -- Liberty boolean functions (for constant propagation) ---------------------------------------------------------------
_FTOK = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_\[\]\.]*|[01]|[()!'*&+|^])")


def parse_function(expr):
    """A Liberty 'function' -> (variables, evaluator(dict name -> 0/1) -> 0/1). Operators: ! and postfix ' (not),
    * & and juxtaposition (and), ^ (xor), + | (or), parentheses, constants 0 / 1."""
    toks = []
    s = expr.strip()
    i = 0
    while i < len(s):
        m = _FTOK.match(s, i)
        if not m:
            raise ValueError('function %r at %d' % (expr, i))
        toks.append(m.group(1)); i = m.end()
        while i < len(s) and s[i].isspace():
            i += 1
    pos = [0]
    names = []

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        pos[0] += 1
        return toks[pos[0] - 1]

    def primary():
        t = take()
        if t == '(':
            f = orr()
            take()                                   # ')'
        elif t == '!':
            g = primary()
            f = lambda v, g=g: 1 - g(v)
        elif t in ('0', '1'):
            f = (lambda v, c=int(t): c)
        else:
            if t not in names:
                names.append(t)
            f = (lambda v, n=t: v[n])
        while peek() == "'":
            take(); g = f
            f = lambda v, g=g: 1 - g(v)
        return f

    def xor():
        f = primary()
        while peek() == '^':
            take(); g = primary(); h = f
            f = lambda v, h=h, g=g: h(v) ^ g(v)
        return f

    def andd():
        f = xor()
        while peek() is not None and peek() not in ('+', '|', ')'):
            if peek() in ('*', '&'):
                take()
            g = xor(); h = f
            f = lambda v, h=h, g=g: h(v) & g(v)
        return f

    def orr():
        f = andd()
        while peek() in ('+', '|'):
            take(); g = andd(); h = f
            f = lambda v, h=h, g=g: h(v) | g(v)
        return f
    f = orr()
    return names, f
