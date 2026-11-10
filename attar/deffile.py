"""DEF reader (header, ROWS, COMPONENTS, PINS, NETS connectivity) and placement write-back.

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
         'regions': 0, 'groups': 0}
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
                if tok == '+' and i + 2 < len(rest) and rest[i + 1] == 'USE':
                    cur[2] = rest[i + 2]
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
