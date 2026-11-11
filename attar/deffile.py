"""DEF reader (header, ROWS, TRACKS, COMPONENTS, PINS, NETS connectivity), the netlist fingerprint, and placement
write-back.

Routing in NETS and all of SPECIALNETS are skipped: the distillation never needs them.
"""
import re

ORIENTS = ['N', 'S', 'W', 'E', 'FN', 'FS', 'FW', 'FE']
ORIENT_ID = {o: i for i, o in enumerate(ORIENTS)}

_PLACE = re.compile(r'\+\s*(PLACED|FIXED|COVER)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*(\w+)')
_UNPLACED = re.compile(r'\+\s*UNPLACED')


def _statements(f, end):
    """Yield the ';'-terminated statements of a small section (COMPONENTS, PINS) as single strings."""
    buf = []
    for line in f:
        if line.startswith(end):
            return
        buf.append(line.strip())
        if line.rstrip().endswith(';'):
            yield ' '.join(buf); buf = []


def _pin_offset(orient, cx, cy):
    """Rotate a pin shape center (relative to the pin origin) by the IO pin orientation."""
    return {'N': (cx, cy), 'S': (-cx, -cy), 'E': (cy, -cx), 'W': (-cy, cx),
            'FN': (-cx, cy), 'FS': (cx, -cy), 'FE': (cy, cx), 'FW': (-cy, -cx)}.get(orient, (cx, cy))


def read_def(path):
    d = {'design': None, 'dbu': None, 'die': None, 'rows': [], 'comps': [], 'pins': [], 'nets': [], 'blockages': 0, 'pg_pins': 0,
         'regions': 0, 'groups': 0, 'tracks': []}
    with open(path, errors='replace') as f:
        for line in f:
            if line.startswith('DESIGN '):
                d['design'] = line.split()[1]
            elif line.startswith('UNITS DISTANCE MICRONS'):
                d['dbu'] = int(line.split()[3])
            elif line.startswith('DIEAREA'):
                v = [int(x) for x in re.findall(r'-?\d+', line)]
                d['die'] = (min(v[0::2]), min(v[1::2]), max(v[0::2]), max(v[1::2]))
            elif line.startswith('ROW '):
                t = line.split()  # ROW name site x y orient DO nx BY ny STEP sx sy
                nx, ny = int(t[7]), int(t[9])
                sx = int(t[11]) if len(t) > 11 and t[10] == 'STEP' else 0
                d['rows'].append((t[2], int(t[3]), int(t[4]), t[5], nx, ny, sx))
            elif line.startswith('TRACKS'):
                t = line.replace(';', ' ').split()   # TRACKS X start DO n STEP s LAYER l1 [l2 ...]
                layers = t[t.index('LAYER') + 1:] if 'LAYER' in t else ['']
                for l in layers:
                    d['tracks'].append((t[1], int(t[2]), int(t[4]), int(t[6]), l))     # (axis, start, num, step, layer)
            elif line.startswith('BLOCKAGES'):
                for s in _statements(f, 'END BLOCKAGES'):
                    if s.split()[1:2] == ['PLACEMENT']:
                        d['blockages'] += 1
            elif line.startswith('REGIONS'):
                d['regions'] = sum(1 for s in _statements(f, 'END REGIONS') if s.strip().startswith('-'))
            elif line.startswith('GROUPS'):
                d['groups'] = sum(1 for s in _statements(f, 'END GROUPS') if s.strip().startswith('-'))
            elif line.startswith('COMPONENTS'):
                for s in _statements(f, 'END COMPONENTS'):
                    t = s.split()
                    m = _PLACE.search(s)
                    st, x, y, o = (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)) if m else ('UNPLACED', 0, 0, 'N')
                    d['comps'].append((t[1], t[2], st, x, y, o))
            elif line.startswith('PINS'):
                for s in _statements(f, 'END PINS'):
                    t = s.split()
                    net = t[t.index('NET') + 1] if 'NET' in t else t[1]
                    m = _PLACE.search(s)
                    x = y = 0; st = 'UNPLACED'
                    if m:
                        st, x, y, o = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
                        r = re.search(r'\+\s*LAYER\s+\S+\s*(?:MASK\s+\d+\s*)?\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', s)
                        if r:
                            a, b, c, e = (int(v) for v in r.groups())
                            ox, oy = _pin_offset(o, (a + c) / 2, (b + e) / 2)
                            x, y = int(round(x + ox)), int(round(y + oy))
                    dr = t[t.index('DIRECTION') + 1] if 'DIRECTION' in t else 'INOUT'
                    use = t[t.index('USE') + 1] if 'USE' in t else 'SIGNAL'
                    if use in ('POWER', 'GROUND'):
                        d['pg_pins'] += 1  # PG pins are not part of the signal netlist
                        continue
                    d['pins'].append((t[1], net, st, x, y, dr))
            elif line.startswith('NETS'):
                _read_nets(f, d['nets'])
    return d


def netlist_fingerprint(d):
    """sha256 of a parsed DEF's netlist: components (name, master), signal IO pins, and every signal net with its
    connections. Positions, floorplan and routing do not count, so the same design at another placement (or written
    back by Attar) has the same fingerprint, and a different netlist (e.g. buffers ORFS inserted) does not."""
    import hashlib
    h = hashlib.sha256()
    for name, master, *_ in sorted(d['comps']):
        h.update(('C %s %s\n' % (name, master)).encode())
    for p in sorted(x[0] for x in d['pins']):
        h.update(('P %s\n' % p).encode())
    for name, conns, use in sorted(d['nets'], key=lambda n: n[0]):
        if use in ('POWER', 'GROUND'):
            continue
        h.update(('N %s %s\n' % (name, ' '.join('%s/%s' % c for c in sorted(conns)))).encode())
    return h.hexdigest()


def _read_nets(f, nets):
    """nets: list of (name, [(comp or 'PIN', pin)], use)."""
    cur = None; conn = False
    for line in f:
        if line.startswith('END NETS'):
            break
        if line.lstrip().startswith('- '):
            t = line.split()
            cur = [t[1], [], 'SIGNAL']; nets.append(cur); conn = True
            rest = t[2:]
        elif conn:
            rest = line.split()
        else:
            if '+ USE' in line:
                cur[2] = line.split('+ USE')[1].split()[0]
            continue
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == '(':
                cur[1].append((rest[i + 1], rest[i + 2])); i += 4
            elif tok in ('+', ';'):
                conn = False
                for j in range(i, len(rest) - 2):      # '+ USE x' anywhere among the attributes on this line
                    if rest[j] == '+' and rest[j + 1] == 'USE':
                        cur[2] = rest[j + 2]
                break
            else:
                i += 1


def write_placement(src, dst, place):
    """Copy DEF src to dst, replacing the location of the components in place {name: (x, y, orient)} (PLACED; FIXED
    components keep theirs). Returns the number of components rewritten. See write_def for the rest of a floorplan."""
    return write_def(src, dst, comps={k: (x, y, o, 'PLACED') for k, (x, y, o) in place.items()})['components']


def _status_sub(line, new, force):
    """Replace the placement attribute of a component line (FIXED only with force), or add one before ';'."""
    line2 = _PLACE.sub(lambda m: new if (m.group(1) != 'FIXED' or force) else m.group(0), line, count=1)
    if line2 == line:
        line2 = _UNPLACED.sub(new, line, count=1)
    if line2 == line and not _PLACE.search(line) and line.rstrip().endswith(';'):   # "- name master ;"
        k = line.rstrip().rfind(';')
        line2 = line[:k].rstrip() + ' ' + new + ' ;\n'
    return line2


def write_def(src, dst, comps=None, pins=None, die=None, rows=None):
    """Copy DEF src to dst with a new floorplan and placement; every other byte is kept.
    comps  {name: (x, y, orient, status)}: status PLACED or FIXED (FIXED overrides a FIXED source location)
    pins   {name: (x, y, orient, layer, (x0, y0, x1, y1))}: IO pin placement and its shape (relative to it)
    die    (x0, y0, x1, y1): new DIEAREA; the TRACKS counts are adjusted to cover it
    rows   [(name, site, x, y, orient, num_sites, step)]: replace all ROW statements
    Returns counts of what was rewritten."""
    comps = comps or {}; pins = pins or {}
    done = {'components': 0, 'pins': 0, 'rows': 0, 'die': 0, 'tracks': 0}
    section = None; name = None; seen = set(); rows_written = False
    pin_buf = []

    def flush_pin(fo):
        stmt = ''.join(pin_buf); pin_buf.clear()
        nm = stmt.split()[1]
        p = pins.get(nm)
        if p is None:
            fo.write(stmt); return
        # keep the header (name, net, direction, use, special), drop any old port/placement
        head = re.split(r'\+\s*(PORT|LAYER|PLACED|FIXED|COVER|UNPLACED)\b', stmt, maxsplit=1)[0].rstrip().rstrip(';').rstrip()
        x, y, o, layer, (a, b, c, e) = p
        fo.write('%s\n      + PORT\n        + LAYER %s ( %d %d ) ( %d %d )\n        + PLACED ( %d %d ) %s ;\n'
                 % (head, layer, a, b, c, e, x, y, o))
        done['pins'] += 1

    with open(src, errors='replace') as fi, open(dst, 'w') as fo:
        for line in fi:
            if section is None:
                if line.startswith('COMPONENTS'):
                    section = 'comp'
                elif line.startswith('PINS'):
                    section = 'pins'
                elif die is not None and line.startswith('DIEAREA'):
                    line = 'DIEAREA ( %d %d ) ( %d %d ) ;\n' % tuple(die); done['die'] += 1
                elif rows is not None and line.startswith('ROW '):
                    if not rows_written:
                        line = ''.join('ROW %s %s %d %d %s DO %d BY 1 STEP %d 0 ;\n' % r for r in rows)
                        done['rows'] = len(rows); rows_written = True
                    else:
                        line = ''
                elif die is not None and line.startswith('TRACKS'):
                    t = line.split()          # TRACKS X start DO n STEP s LAYER l ;
                    start, step = int(t[2]), int(t[6])
                    hi = die[2] if t[1] == 'X' else die[3]
                    t[4] = str(max(1, (hi - start) // step + 1))
                    line = ' '.join(t) + '\n'; done['tracks'] += 1
                fo.write(line)
                continue
            if section == 'comp':
                if line.startswith('END COMPONENTS'):
                    section = None
                else:
                    if line.lstrip().startswith('- '):
                        name = line.split()[1]
                    p = comps.get(name)
                    if p is not None and name not in seen:
                        x, y, o, st = p
                        new = '+ UNPLACED' if st == 'UNPLACED' else '+ %s ( %d %d ) %s' % (st, x, y, o)
                        line2 = _status_sub(line, new, st in ('FIXED', 'UNPLACED'))
                        if line2 != line:
                            seen.add(name); done['components'] += 1
                        line = line2
                fo.write(line)
            elif section == 'pins':
                if line.startswith('END PINS'):
                    if pin_buf:
                        flush_pin(fo)
                    section = None; fo.write(line)
                elif line.lstrip().startswith('- ') and not pin_buf:
                    pin_buf.append(line)
                    if line.rstrip().endswith(';'):
                        flush_pin(fo)
                elif pin_buf:
                    pin_buf.append(line)
                    if line.rstrip().endswith(';'):
                        flush_pin(fo)
                else:
                    fo.write(line)
    return done


# -- the frame: the DEF minus what Attar writes itself ----------------------------------------------------------------
# Attar owns the components it may move (and, later, resize or add) and every signal net (connectivity, and later
# buffering and routing). Everything else of the source DEF is the problem around them and is kept verbatim: header,
# DIEAREA, ROWS, TRACKS, VIAS, PINS (IO pin shapes), SPECIALNETS (PDN), BLOCKAGES, REGIONS, GROUPS, the fixed
# components (macros, tap cells, ...) and the POWER/GROUND entries of NETS. export = frame + what Attar writes.

_ROUTING = ('ROUTED', 'FIXED', 'COVER', 'NOSHIELD')        # net wiring: stale once cells move, never kept


def _groups(tokens):
    """Split '+ KEY ...' attribute tokens into groups (lists of tokens), each starting at a top-level '+'."""
    out, cur = [], None
    for tok in tokens:
        if tok == '+':
            cur = []; out.append(cur)
        elif cur is not None:
            cur.append(tok)
    return out


def _comp_attrs(stmt):
    """' + SOURCE TIMING ...' of a component statement: everything but its name, master and placement."""
    t = stmt.rstrip().rstrip(';').split()
    return ''.join(' + ' + ' '.join(g) for g in _groups(t[3:] if len(t) > 3 else []) if g and g[0] not in ('PLACED', 'FIXED', 'COVER', 'UNPLACED'))


def _net_attrs(stmt):
    """' + USE CLOCK ...' of a net statement: everything but its name, connections and wiring."""
    t = stmt.rstrip().rstrip(';').split()
    i = 2
    while i < len(t) and t[i] == '(':                     # connections: ( comp pin [+ SYNTHESIZED] )
        i = t.index(')', i) + 1
    return ''.join(' + ' + ' '.join(g) for g in _groups(t[i:]) if g and g[0] not in _ROUTING)


def _stmt_iter(lines):
    """(statement text, [its lines]) of a section body, up to and excluding its END line."""
    buf = []
    for line in lines:
        buf.append(line)
        if line.rstrip().endswith(';'):
            yield ''.join(buf), buf; buf = []
    if buf:
        yield ''.join(buf), buf


def split_frame(path, owned):
    """Split the DEF at path. owned: names of the components Attar writes itself. Returns (frame text, component
    attributes {name: ' + ...'}, net attributes {name: ' + ...'}) for the owned components and the signal nets."""
    with open(path, errors='replace') as f:
        lines = f.readlines()
    out, comp_attr, net_attr = [], {}, {}
    i = 0
    while i < len(lines):
        line = lines[i]
        key = line.split()[0] if line.strip() else ''
        if key in ('COMPONENTS', 'NETS') and line.strip().endswith(';'):
            end = 'END ' + key
            j = i + 1
            while j < len(lines) and not lines[j].startswith(end):
                j += 1
            out.append(line)
            for stmt, buf in _stmt_iter(lines[i + 1:j]):
                t = stmt.split()
                if len(t) < 2 or t[0] != '-':
                    out.extend(buf); continue
                if key == 'COMPONENTS':
                    if t[1] in owned:
                        comp_attr[t[1]] = _comp_attrs(stmt)
                    else:
                        out.extend(buf)
                else:
                    a = _net_attrs(stmt)
                    if re.search(r'\+\s*USE\s+(POWER|GROUND)\b', stmt):
                        out.extend(buf)               # PG entries stay with the frame (the IR has no PG nets)
                    else:
                        net_attr[t[1]] = a
            if j < len(lines):
                out.append(lines[j])
            i = j + 1
            continue
        out.append(line)
        i += 1
    return ''.join(out), comp_attr, net_attr


def write_framed(frame, dst, comp_lines, net_lines):
    """Write frame text with Attar's component and net statements appended to COMPONENTS and NETS (counts updated)."""
    lines = frame.splitlines(True)
    counts = {}
    sec = None
    for line in lines:                                    # statements the frame keeps per section
        k = line.split()[0] if line.strip() else ''
        if k in ('COMPONENTS', 'NETS') and line.strip().endswith(';'):
            sec = k; counts[k] = 0
        elif sec and line.startswith('END ' + sec):
            sec = None
        elif sec and line.lstrip().startswith('- '):
            counts[sec] += 1
    add = {'COMPONENTS': comp_lines, 'NETS': net_lines}
    with open(dst, 'w') as fo:
        sec = None
        for line in lines:
            k = line.split()[0] if line.strip() else ''
            if k in ('COMPONENTS', 'NETS') and line.strip().endswith(';'):
                sec = k
                fo.write('%s %d ;\n' % (k, counts[k] + len(add[k])))
                continue
            if sec and line.startswith('END ' + sec):
                fo.writelines(add[sec]); sec = None
            fo.write(line)
    return {k: len(v) for k, v in add.items()}


def comp_statement(name, master, status, x, y, orient, attrs=''):
    place = ' + UNPLACED' if status == 'UNPLACED' else ' + %s ( %d %d ) %s' % (status, x, y, orient)
    return '    - %s %s%s%s ;\n' % (name, master, attrs, place)


def net_statements(net_names, n_ptr, p_cell, p_mpin, cell_names, is_term, mpin_flat, attrs):
    """NETS statements from CSR arrays: pins of net k are n_ptr[k]:n_ptr[k+1]; an IO terminal cell is '( PIN name )',
    a cell pin '( cell master_pin )'. attrs(k) -> ' + USE ...' of net k."""
    out = []
    for k, name in enumerate(net_names):
        a, b = int(n_ptr[k]), int(n_ptr[k + 1])
        conns = ['( PIN %s )' % cell_names[c] if is_term[c] else '( %s %s )' % (cell_names[c], mpin_flat[m])
                 for c, m in zip(p_cell[a:b].tolist(), p_mpin[a:b].tolist())]
        body = '\n      '.join(' '.join(conns[i:i + 6]) for i in range(0, len(conns), 6))
        out.append('    - %s %s%s ;\n' % (name, body, attrs(k)))
    return out


def same_design(a, b):
    """'' when two parsed DEFs hold the same design (components with placement, IO pins, nets with connections and
    use, die, rows, tracks), else what differs first."""
    for key in ('design', 'dbu', 'die', 'rows', 'tracks'):
        if a[key] != b[key]:
            return '%s differs' % key
    for key in ('comps', 'pins'):
        x, y = sorted(a[key]), sorted(b[key])
        if x != y:
            d = sorted(set(x) ^ set(y))
            return '%s differ (%d vs %d; e.g. %s)' % (key, len(x), len(y), d[:2])
    na = {n: (sorted(c), u) for n, c, u in a['nets']}; nb = {n: (sorted(c), u) for n, c, u in b['nets']}
    if na != nb:
        d = sorted(k for k in set(na) | set(nb) if na.get(k) != nb.get(k))
        return 'nets differ (%d vs %d; e.g. %s)' % (len(na), len(nb), d[:2])
    return ''


# -- PG geometry: the special nets' metal (wires, vias, rects), in DBU ----------------------------------------------
PG_HWIRE, PG_VWIRE, PG_VIA, PG_FILL = 0, 1, 2, 3


def _section(text, name):
    """The text of a DEF section, from its header line up to (not including) its END line; '' if absent."""
    i = 0 if text.startswith(name + ' ') else text.find('\n' + name + ' ') + 1
    if i <= 0 and not text.startswith(name + ' '):
        return ''
    j = text.find('\nEND ' + name, i)
    return text[i:j if j >= 0 else len(text)]


def def_vias(text):
    """{via name: [(layer, (x0, y0, x1, y1))]} of the VIAS section (DBU): via rules (CUTSIZE, LAYERS, CUTSPACING,
    ENCLOSURE, ROWCOL, ORIGIN, OFFSET; the two metal layers) and fixed vias (+ RECT)."""
    out = {}
    for stmt in _section(text, 'VIAS').split(';'):
        t = stmt.split()
        if len(t) < 2 or t[0] != '-':
            continue
        name, g = t[1], {}
        for grp in _groups(t[2:]):
            if grp:
                g.setdefault(grp[0], []).append(grp[1:])
        rects = []
        if 'VIARULE' in g and 'CUTSIZE' in g and 'LAYERS' in g:
            cx, cy = (int(v) for v in g['CUTSIZE'][0][:2])
            bot, _, top = g['LAYERS'][0][:3]
            sx, sy = (int(v) for v in g.get('CUTSPACING', [[0, 0]])[0][:2])
            bx, by, tx, ty = (int(v) for v in g.get('ENCLOSURE', [[0, 0, 0, 0]])[0][:4])
            r, c = (int(v) for v in g.get('ROWCOL', [[1, 1]])[0][:2])
            ox, oy = (int(v) for v in g.get('ORIGIN', [[0, 0]])[0][:2])
            obx, oby, otx, oty = (int(v) for v in g.get('OFFSET', [[0, 0, 0, 0]])[0][:4])
            w, h = c * cx + (c - 1) * sx, r * cy + (r - 1) * sy
            rects.append((bot, (ox + obx - w // 2 - bx, oy + oby - h // 2 - by, ox + obx + w - w // 2 + bx, oy + oby + h - h // 2 + by)))
            rects.append((top, (ox + otx - w // 2 - tx, oy + oty - h // 2 - ty, ox + otx + w - w // 2 + tx, oy + oty + h - h // 2 + ty)))
        for grp in g.get('RECT', []):
            v = [int(x) for x in grp[1:] if x.lstrip('-').isdigit()][:4]
            if len(v) == 4:
                rects.append((grp[0], (min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3]))))
        out[name] = rects
    return out


def pg_shapes(text, lef_vias, dbu):
    """The special nets' metal: [(layer, (x0, y0, x1, y1), kind, net)] in DBU. kind: PG_HWIRE / PG_VWIRE (path
    segments, square ends extended by half the width), PG_VIA (each metal layer of a via), PG_FILL (+ SHAPE DRCFILL).
    lef_vias: {name: [(layer, rect in microns)]} from read_lef; DEF vias override them."""
    vias = {k: [(l, tuple(int(round(c * dbu)) for c in r)) for l, r in v] for k, v in lef_vias.items()}
    vias.update(def_vias(text))
    out = []
    for stmt in _section(text, 'SPECIALNETS').split(';'):
        t = stmt.replace('(', ' ( ').replace(')', ' ) ').split()
        if len(t) < 2 or t[0] != '-':
            continue
        net = t[1]
        i = 2
        layer = None; width = 0; shape = None; last = None
        while i < len(t):
            tok = t[i]
            if tok in ('ROUTED', 'FIXED', 'COVER', 'SHIELD', 'NEW'):
                j = i + 1 + (tok == 'SHIELD')                     # SHIELD <shieldNet> layer width ...
                if j + 1 < len(t) and t[j] != '+':
                    layer, width = t[j], int(t[j + 1]) if t[j + 1].lstrip('-').isdigit() else 0
                    i = j + 2; shape = None; last = None
                    continue
                i += 1; continue
            if tok == '+' and i + 1 < len(t) and t[i + 1] == 'SHAPE':
                shape = t[i + 2]; i += 3; continue
            if tok == '+' and i + 1 < len(t) and t[i + 1] == 'RECT':           # + RECT layer ( x y ) ( x y )
                v = [int(x) for x in t[i + 3:i + 11] if x.lstrip('-').isdigit()][:4]
                if len(v) == 4:
                    out.append((t[i + 2], (min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3])),
                                PG_FILL if shape == 'DRCFILL' else PG_HWIRE, net))
                i += 11; continue
            if tok == '+' and i + 1 < len(t) and t[i + 1] == 'MASK':
                i += 3; continue
            if tok == '(' and layer is not None:
                x = t[i + 1]; y = t[i + 2]
                px = last[0] if x == '*' and last else int(x)
                py = last[1] if y == '*' and last else int(y)
                k = t.index(')', i)
                if last is not None and width:
                    hw = width // 2
                    if py == last[1]:      # horizontal segment
                        out.append((layer, (min(px, last[0]) - hw, py - hw, max(px, last[0]) + hw, py + hw),
                                    PG_FILL if shape == 'DRCFILL' else PG_HWIRE, net))
                    elif px == last[0]:    # vertical
                        out.append((layer, (px - hw, min(py, last[1]) - hw, px + hw, max(py, last[1]) + hw),
                                    PG_FILL if shape == 'DRCFILL' else PG_VWIRE, net))
                last = (px, py); i = k + 1
                continue
            if last is not None and tok in vias:               # a via at the last point
                for l, (a, b, c, e) in vias[tok]:
                    out.append((l, (last[0] + a, last[1] + b, last[0] + c, last[1] + e), PG_VIA, net))
                i += 1; continue
            i += 1
    return out
