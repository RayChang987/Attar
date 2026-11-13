"""SDC reader: the constraints static timing needs, evaluated by a real Tcl interpreter (SDC is Tcl: variables, expr,
lsearch, loops all work), with the SDC commands bound to Python recorders.

Kept: clocks (period, waveform, source ports), ideal clock latency, uncertainty and transition, input / output delays,
input transitions, port loads, clock groups, set_max_delay / set_min_delay with -from / -to (no -through). Recorded but not
modeled (listed in 'unsupported'): set_false_path, set_multicycle_path, path delays with -through or rise/fall
qualifiers, driving cells, derates. Ignored: group_path,
design rule limits (set_max_fanout, ...), set_units, and other commands that do not change timing.
"""
import fnmatch

UNMODELED = ('set_false_path', 'set_multicycle_path', 'set_driving_cell',
             'set_timing_derate', 'set_case_analysis', 'set_disable_timing', 'set_propagated_clock',
             'create_generated_clock', 'set_clock_sense', 'set_data_check')
IGNORED = ('group_path', 'set_max_fanout', 'set_max_transition', 'set_max_capacitance', 'set_units', 'set_max_area',
           'set_dont_touch', 'set_dont_use', 'set_wire_load_model', 'set_operating_conditions', 'set_ideal_network',
           'set_max_dynamic_power', 'set_max_leakage_power', 'set_clock_gating_check', 'set_fanout_load',
           'set_logic_zero', 'set_logic_one', 'set_load_unit', 'set_time_unit')


def _opts(args, flags, valued):
    """Split SDC arguments into {flag: value or True} and positionals."""
    o, pos, i = {}, [], 0
    while i < len(args):
        a = args[i]
        if a in valued and i + 1 < len(args):
            o[a] = args[i + 1]; i += 2
        elif a in flags:
            o[a] = True; i += 1
        else:
            pos.append(a); i += 1
    return o, pos


def read(path_or_text, ports_in=(), ports_out=(), design='', is_text=False):
    """Evaluate an SDC. ports_in / ports_out: the design's port names (for get_ports patterns, all_inputs, ...).
    Returns {'clocks': {name: {'period', 'waveform', 'sources', 'latency', 'uncertainty', ['transition']}}, 'input_delay': [...],
    'output_delay': [...], 'input_transition': {port: ps}, 'load': {port: fF}, 'clock_groups': [...],
    'path_delays': [{'kind': 'max'|'min', 'value', 'from', 'to': [(kind, name)] or None, 'ignore_clock_latency'}],
    'unsupported': {command: count}} with times and capacitances in the SDC's units (OpenSTA: the first Liberty
    file's units; attar.timing.attach converts them to ps / fF)."""
    import tkinter
    tcl = tkinter.Tcl()
    ports = list(ports_in) + list(ports_out)
    port_set = set(ports)
    res = {'clocks': {}, 'input_delay': [], 'output_delay': [], 'input_transition': {}, 'load': {},
           'clock_groups': [], 'path_delays': [], 'unsupported': {}}

    def split(s):
        return list(tcl.splitlist(s)) if s else []

    KINDS = ('port', 'clock', 'cell', 'pin', 'net', 'lib_cell', 'lib_pin')

    def objs(s, kind=None):
        """Tcl list of tagged objects ('port:x', 'clock:y'), nested lists flattened -> [(kind, name)]."""
        out = []
        for x in split(s):
            k, sep, n = x.partition(':')
            if sep and k in KINDS:
                out.append((k, n))
            elif len(split(x)) > 1 or (split(x) and split(x)[0] != x):   # a nested list
                out += objs(x, kind)
            elif kind:                                   # a bare name: take it as the expected kind
                out.append((kind, x))
        return out

    def tag(kind, names):
        return tuple('%s:%s' % (kind, n) for n in names)      # _tkinter turns a tuple into a Tcl list

    def get_ports(*args):
        o, pos = _opts(args, ('-quiet', '-regexp', '-nocase'), ('-filter', '-of_objects'))
        names = []
        for pat in [p for a in pos for p in split(a)]:
            if pat in port_set:                      # bus bits ('key[0]') are names, not bracket globs
                names.append(pat)
            elif '*' in pat or '?' in pat:
                g = pat.replace('[', '[[]').replace(']', '[]]').replace('[[[]]', '[[]')
                names += [p for p in ports if fnmatch.fnmatchcase(p, g)]
        return tag('port', names)

    def get_clocks(*args):
        o, pos = _opts(args, ('-quiet',), ('-filter',))
        names = []
        for pat in [p for a in pos for p in split(a)]:
            names += [c for c in res['clocks'] if fnmatch.fnmatchcase(c, pat)]
        return tag('clock', names)

    def tagged(kind):
        def f(*args):
            o, pos = _opts(args, ('-quiet', '-hierarchical', '-regexp', '-nocase'), ('-filter', '-of_objects'))
            return tag(kind, [p for a in pos for p in split(a)])
        return f

    def create_clock(*args):
        o, pos = _opts(args, ('-add',), ('-name', '-period', '-waveform', '-comment'))
        period = float(o['-period'])
        wf = [float(v) for v in split(o['-waveform'])] if '-waveform' in o else [0.0, period / 2]
        src = [n for a in pos for k, n in objs(a, 'port')]
        name = o.get('-name') or (src[0] if src else 'clk')
        res['clocks'][name] = {'period': period, 'waveform': wf, 'sources': src, 'latency': 0.0, 'uncertainty': 0.0}
        return ''

    def set_clock_latency(*args):
        o, pos = _opts(args, ('-source', '-min', '-max', '-rise', '-fall', '-early', '-late'), ('-clock',))
        val = float(pos[0])
        for a in pos[1:]:
            for k, n in objs(a, 'clock'):
                if k == 'clock' and n in res['clocks'] and not o.get('-min'):
                    res['clocks'][n]['source_latency' if o.get('-source') else 'latency'] = val
        return ''

    def set_clock_uncertainty(*args):
        o, pos = _opts(args, ('-setup', '-hold', '-rise', '-fall'), ('-from', '-to'))
        val = float(pos[0])
        for a in pos[1:]:
            for k, n in objs(a, 'clock'):
                if n in res['clocks']:
                    key = 'uncertainty_hold' if o.get('-hold') else 'uncertainty_setup' if o.get('-setup') else 'uncertainty'
                    res['clocks'][n][key] = val
        return ''

    def set_clock_transition(*args):
        o, pos = _opts(args, ('-rise', '-fall', '-min', '-max'), ())
        val = float(pos[0])
        for a in pos[1:]:
            for k, n in objs(a, 'clock'):
                if n in res['clocks'] and not o.get('-min'):
                    tr = res['clocks'][n].setdefault('transition', [0.0, 0.0])
                    for i in ([0] if o.get('-rise') else [1] if o.get('-fall') else [0, 1]):
                        tr[i] = val
        return ''

    def remove_from_collection(*args):
        o, pos = _opts(args, ('-intersect',), ())
        if not pos:
            return ''
        flat =lambda v: ['%s:%s' % kn for kn in objs(v)] if objs(v) else split(v)   # noqa: E731
        drop = set(flat(pos[1])) if len(pos) > 1 else set()
        keep = [x for x in flat(pos[0]) if (x in drop) == bool(o.get('-intersect'))]
        return tuple(keep)

    def io_delay(which):
        def f(*args):
            o, pos = _opts(args, ('-add_delay', '-max', '-min', '-rise', '-fall', '-clock_fall', '-level_sensitive',
                                  '-network_latency_included', '-source_latency_included'), ('-clock', '-reference_pin'))
            val = float(pos[0])
            clk = objs(o['-clock'], 'clock')[0][1] if '-clock' in o else None
            for a in pos[1:]:
                for k, n in objs(a, 'port'):
                    res[which].append({'port': n, 'clock': clk, 'value': val, 'clock_fall': bool(o.get('-clock_fall')),
                                       'min': bool(o.get('-min')), 'max': bool(o.get('-max')),
                                       'rise': bool(o.get('-rise')), 'fall': bool(o.get('-fall')),
                                       'network_latency_included': bool(o.get('-network_latency_included'))})
            return ''
        return f

    def set_input_transition(*args):
        o, pos = _opts(args, ('-max', '-min', '-rise', '-fall'), ('-clock',))
        for a in pos[1:]:
            for k, n in objs(a, 'port'):
                if not o.get('-min'):
                    res['input_transition'][n] = float(pos[0])
        return ''

    def set_load(*args):
        o, pos = _opts(args, ('-pin_load', '-wire_load', '-max', '-min', '-subtract_pin_load'), ())
        for a in pos[1:]:
            for k, n in objs(a, 'port'):
                if not o.get('-min'):
                    res['load'][n] = float(pos[0])
        return ''

    def set_clock_groups(*args):
        o, pos = _opts(args, ('-logically_exclusive', '-physically_exclusive', '-asynchronous', '-allow_paths'),
                       ('-name', '-comment'))
        groups, i = [], 0
        while i < len(args):
            if args[i] == '-group' and i + 1 < len(args):
                groups.append([n for k, n in objs(args[i + 1], 'clock')]); i += 2
            else:
                i += 1
        kind = next((k[1:] for k in ('-logically_exclusive', '-physically_exclusive', '-asynchronous') if o.get(k)), '')
        res['clock_groups'].append({'kind': kind, 'groups': groups})
        return ''

    def path_delay(kind):
        def f(*args):
            o, pos = _opts(args, ('-ignore_clock_latency', '-reset_path', '-rise', '-fall'),
                           ('-from', '-to', '-through', '-rise_from', '-fall_from', '-rise_to', '-fall_to', '-comment'))
            if any(k in o for k in ('-through', '-rise_from', '-fall_from', '-rise_to', '-fall_to', '-rise', '-fall')) or not pos:
                res['unsupported']['set_%s_delay' % kind] = res['unsupported'].get('set_%s_delay' % kind, 0) + 1
                return ''
            res['path_delays'].append({'kind': kind, 'value': float(pos[0]),
                                       'from': objs(o['-from']) if '-from' in o else None,
                                       'to': objs(o['-to']) if '-to' in o else None,
                                       'ignore_clock_latency': bool(o.get('-ignore_clock_latency'))})
            return ''
        return f

    def unsupported(name):
        def f(*args):
            res['unsupported'][name] = res['unsupported'].get(name, 0) + 1
            return ''
        return f

    cmds = {'get_ports': get_ports, 'get_clocks': get_clocks, 'get_cells': tagged('cell'), 'get_pins': tagged('pin'),
            'get_nets': tagged('net'), 'get_lib_cells': tagged('lib_cell'), 'get_lib_pins': tagged('lib_pin'),
            'all_inputs': lambda *a: tag('port', ports_in), 'all_outputs': lambda *a: tag('port', ports_out),
            'all_clocks': lambda *a: tag('clock', list(res['clocks'])), 'all_registers': lambda *a: '',
            'current_design': lambda *a: design, 'current_instance': lambda *a: '',
            'create_clock': create_clock, 'set_clock_latency': set_clock_latency,
            'set_clock_uncertainty': set_clock_uncertainty, 'set_input_delay': io_delay('input_delay'),
            'set_output_delay': io_delay('output_delay'), 'set_input_transition': set_input_transition,
            'set_load': set_load, 'set_clock_groups': set_clock_groups, 'set_clock_transition': set_clock_transition,
            'remove_from_collection': remove_from_collection, 'delete_from_list': remove_from_collection,
            'set_max_delay': path_delay('max'), 'set_min_delay': path_delay('min'),
            'get_name': lambda x: x.partition(':')[2], 'get_full_name': lambda x: x.partition(':')[2],
            'get_object_name': lambda x: x.partition(':')[2]}
    for n in UNMODELED:
        cmds[n] = unsupported(n)
    for n in IGNORED:
        cmds[n] = lambda *a: ''
    errors = []

    def wrap(name, f):
        def g(*args):
            try:
                return f(*args)
            except Exception as e:  # noqa: BLE001  (a Python error would reach Tcl without its message)
                errors.append('%s %s: %r' % (name, ' '.join(args)[:200], e))
                raise
        return g
    for n, f in cmds.items():
        tcl.createcommand(n, wrap(n, f))
    text = path_or_text if is_text else open(path_or_text).read()
    try:
        tcl.eval(text)
    except tkinter.TclError as e:
        raise ValueError('SDC: %s' % (errors[-1] if errors else e)) from None
    return res
