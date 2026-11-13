"""Liberty (.lib, .lib.gz) reader: what static timing needs, nothing else.

Per cell: pin direction, capacitance (rise / fall), clock flag, max_capacitance, and every timing group (related_pin,
timing_sense, timing_type, when) with its NLDM tables: cell_rise / cell_fall / rise_transition / fall_transition for
delay arcs, rise_constraint / fall_constraint for checks. Power, noise and CCS groups are skipped while parsing.
Units are normalized to ps and fF; table axes to (input slew, load) for delays and (constrained pin slew, related pin
slew) for checks, whatever order the library's templates use.
"""
import gzip
import re

_TOK = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:[^"\\]|\\.)*"|\\\n|[{}();:,]|[^\s{}();:,"]+', re.S)
# groups never needed for timing (large: power tables, noise, CCS)
SKIP = {'internal_power', 'leakage_power', 'power', 'ccsn_first_stage', 'ccsn_last_stage', 'output_current_rise',
        'output_current_fall', 'receiver_capacitance', 'pg_pin', 'dynamic_current', 'leakage_current',
        'intrinsic_parasitic', 'gate_leakage', 'normalized_driver_waveform', 'test_cell', 'statetable',
        'power_lut_template', 'output_voltage', 'input_voltage', 'operating_conditions', 'wire_load',
        'wire_load_selection', 'ff', 'latch', 'bundle', 'memory'}
TABLES = ('cell_rise', 'cell_fall', 'rise_transition', 'fall_transition', 'rise_constraint', 'fall_constraint')
TIME_SCALE = {'1ps': 1.0, '10ps': 10.0, '100ps': 100.0, '1ns': 1000.0, '10ns': 1e4, '100ns': 1e5}
CAP_SCALE = {'ff': 1.0, 'pf': 1000.0, 'nf': 1e6}


class Group:
    __slots__ = ('kind', 'args', 'attrs', 'groups')

    def __init__(self, kind, args):
        self.kind, self.args, self.attrs, self.groups = kind, args, {}, []

    def sub(self, kind):
        return [g for g in self.groups if g.kind == kind]


def _tokens(text):
    for m in _TOK.finditer(text):
        t = m.group(0)
        if t[0] == '/' and len(t) > 1 and t[1] in '*/' or t == '\\\n':
            continue
        yield t


def _strip(t):
    return t[1:-1] if t[:1] == '"' else t


def parse(text):
    """Liberty text -> the library Group (skipped kinds are dropped with their contents)."""
    it = _tokens(text)
    root = None
    stack = []
    skip = 0
    pending = []                           # tokens of the statement being read
    for t in it:
        if t == '{':
            # pending: kind ( args )
            kind = pending[0]
            args = [_strip(a) for a in pending[2:-1] if a != ',']
            pending = []
            if skip or kind in SKIP:
                skip += 1
                continue
            g = Group(kind, args)
            if stack:
                stack[-1].groups.append(g)
            else:
                root = g
            stack.append(g)
        elif t == '}':
            if not skip and stack and pending:
                finish(stack[-1], pending)                                  # last statement without ';'
            if skip:
                skip -= 1
            else:
                stack.pop()
            pending = []
        elif t == ';':
            if not skip and stack and pending:
                finish(stack[-1], pending)
            pending = []
        else:
            # a complex attribute 'name ( ... )' without ';' ends where the next statement starts
            if len(pending) >= 3 and pending[1] == '(' and pending[-1] == ')' and depth(pending) == 0:
                if not skip and stack:
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


def _floats(s):
    return [float(v) for v in re.split(r'[,\s]+', s.strip()) if v]


def _table(g, templates, kind):
    """A table group -> (index_1, index_2, values 2D) on the axes (slew, load) or (constrained, related)."""
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
    delay = kind in TABLES[:4]
    # wanted axis order: delay (input transition, output load); check (constrained pin, related pin)
    first = ('input_net_transition', 'input_transition_time') if delay else ('constrained_pin_transition',)
    if var1 and var1 not in first and (var2 in first or not var2):
        i1, i2 = i2, i1
        v = [list(r) for r in zip(*v)] if len(v) and len(v[0]) else v
    return i1, i2, v


def read(path):
    """A Liberty file -> {'name', 'time_unit_ps', 'cap_unit_ff', 'cells': {cell: {'pins': {...}, 'arcs': [...]}}}.
    Pin: {'dir': 'input'|'output'|'inout'|'internal', 'cap': (rise, fall) fF, 'cap_min': the *_capacitance_range lower
    bounds (min analysis), 'clock': bool, 'max_cap': fF or None}.
    Arc: {'from', 'to', 'sense', 'type', 'when', 'tables': {kind: (index_1, index_2, values)}} in ps / fF."""
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt', errors='replace') as f:
        lib = parse(f.read())
    tu = TIME_SCALE.get(lib.attrs.get('time_unit', '1ns').replace(' ', '').lower(), 1000.0)
    cu = 1.0
    if 'capacitive_load_unit' in lib.attrs:
        n, u = lib.attrs['capacitive_load_unit'][0][:2]
        cu = float(n) * CAP_SCALE.get(u.lower(), 1.0)
    templates = {}
    for g in lib.groups:
        if g.kind in ('lu_table_template',) and g.args:
            t = dict((k, v) for k, v in g.attrs.items() if k.startswith('variable'))
            for k in ('index_1', 'index_2'):
                if k in g.attrs:
                    t[k] = g.attrs[k]
            templates[g.args[0]] = t
    types = {g.args[0]: g.attrs for g in lib.groups if g.kind == 'type' and g.args}
    cells = {}
    for c in lib.sub('cell'):
        pins, arcs = {}, []
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
                              'max_cap': float(a['max_capacitance']) * cu if 'max_capacitance' in a else None,
                              'function': a.get('function')}
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
        cells[c.args[0]] = {'pins': pins, 'arcs': arcs}
    # thresholds (fractions of the supply), OpenSTA's defaults when absent: 50% delay, 20% / 80% slew, derate 1
    pct = lambda k, d: float(lib.attrs[k]) / 100.0 if k in lib.attrs else d
    thr = {'input': (pct('input_threshold_pct_rise', .5), pct('input_threshold_pct_fall', .5)),
           'output': (pct('output_threshold_pct_rise', .5), pct('output_threshold_pct_fall', .5)),
           'slew_lower': (pct('slew_lower_threshold_pct_rise', .2), pct('slew_lower_threshold_pct_fall', .2)),
           'slew_upper': (pct('slew_upper_threshold_pct_rise', .8), pct('slew_upper_threshold_pct_fall', .8)),
           'slew_derate': float(lib.attrs.get('slew_derate_from_library', 1.0))}
    return {'name': lib.args[0] if lib.args else '', 'time_unit_ps': tu, 'cap_unit_ff': cu, 'cells': cells,
            'thresholds': thr}


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
