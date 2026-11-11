"""LEF reader: only what the distillation keeps (macro size and class, pin boxes, sites, routing layers)."""


class Macro:
    __slots__ = ('name', 'cls', 'w', 'h', 'pins', 'pin_layer', 'pin_dir', 'sym')

    def __init__(self, name):
        self.name, self.cls, self.w, self.h = name, '', 0.0, 0.0
        self.pins = {}  # pin name -> (x0, y0, x1, y1) in microns: bbox of all port shapes; use
        self.pin_layer = {}  # pin name -> layer of its first port shape
        self.pin_dir = {}  # pin name -> DIRECTION (INPUT, OUTPUT, INOUT, FEEDTHRU) when the LEF states it
        self.sym = set()   # SYMMETRY: 'X', 'Y', 'R90'


def read_lef(paths, macros=None, tech=None):
    """Read LEF files into macros {name: Macro} and tech {'sites': {}, 'layers': [...], 'vias': {name: [(layer,
    (x0, y0, x1, y1))]}} (microns)."""
    macros = {} if macros is None else macros
    tech = {'sites': {}, 'layers': [], 'vias': {}} if tech is None else tech
    tech.setdefault('vias', {})
    for path in paths:
        mac = pin = layer = site = via = None; via_layer = None
        box = None; use = 'SIGNAL'; props = False
        with open(path) as f:
            for line in f:
                t = line.split()
                if not t:
                    continue
                k = t[0]
                if props:  # PROPERTYDEFINITIONS lists 'MACRO <prop> <type> ;' entries, not macros
                    props = not (k == 'END' and t[1:2] == ['PROPERTYDEFINITIONS'])
                    continue
                if k == 'PROPERTYDEFINITIONS':
                    props = True; continue
                if mac is not None:
                    if pin is not None:
                        if k == 'RECT':
                            v = [float(x) for x in t[1:5]] if t[1] != 'MASK' else [float(x) for x in t[3:7]]
                            x0, y0, x1, y1 = min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3])
                            box = [x0, y0, x1, y1] if box is None else [min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)]
                        elif k == 'USE':
                            use = t[1].rstrip(';')
                        elif k == 'DIRECTION' and len(t) > 1:
                            mac.pin_dir[pin] = t[1].rstrip(';')
                        elif k == 'LAYER' and len(t) > 1 and pin not in mac.pin_layer:
                            mac.pin_layer[pin] = t[1].rstrip(';')
                        elif k == 'END' and len(t) > 1 and t[1] == pin:
                            mac.pins[pin] = (tuple(box) if box else (0.0, 0.0, 0.0, 0.0), use)
                            pin = None
                    elif k == 'PIN':
                        pin, box, use = t[1], None, 'SIGNAL'
                        # attributes may follow on the PIN line (ISPD 2015: 'PIN o DIRECTION OUTPUT ;')
                        for kw in ('DIRECTION', 'USE'):
                            if kw in t[2:] and t.index(kw, 2) + 1 < len(t):
                                v = t[t.index(kw, 2) + 1].rstrip(';')
                                if kw == 'USE':
                                    use = v
                                else:
                                    mac.pin_dir[pin] = v
                    elif k == 'CLASS':
                        mac.cls = t[1].rstrip(';')
                    elif k == 'SIZE':
                        mac.w, mac.h = float(t[1]), float(t[3])
                    elif k == 'SYMMETRY':
                        mac.sym = {v.rstrip(';') for v in t[1:] if v.rstrip(';')}
                    elif k == 'END' and len(t) > 1 and t[1] == mac.name:
                        macros[mac.name] = mac; mac = None
                elif via is not None:
                    if k == 'LAYER' and len(t) > 1:
                        via_layer = t[1].rstrip(';')
                    elif k == 'RECT' and via_layer:
                        v = [float(x) for x in (t[3:7] if t[1] == 'MASK' else t[1:5])]
                        via[1].append((via_layer, (min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]), max(v[1], v[3]))))
                    elif k == 'END' and len(t) > 1 and t[1] == via[0]:
                        tech['vias'][via[0]] = via[1]; via = None
                elif layer is not None:
                    if k in ('DIRECTION', 'PITCH', 'WIDTH', 'OFFSET', 'TYPE'):
                        v = t[1].rstrip(';')
                        layer[k.lower()] = v if k in ('DIRECTION', 'TYPE') else float(v)
                    elif k == 'SPACING' and 'spacing' not in layer and len(t) > 1:   # the plain minimum spacing
                        layer['spacing'] = float(t[1].rstrip(';'))
                    elif k == 'END' and len(t) > 1 and t[1] == layer['name']:
                        if layer.get('type') == 'ROUTING':
                            tech['layers'].append(layer)
                        layer = None
                elif site is not None:
                    if k == 'SIZE':
                        tech['sites'][site] = (float(t[1]), float(t[3]))
                    elif k == 'END' and len(t) > 1 and t[1] == site:
                        site = None
                elif k == 'MACRO':
                    mac = Macro(t[1])
                elif k == 'VIA' and len(t) >= 2:
                    via = (t[1], []); via_layer = None
                elif k == 'LAYER' and len(t) == 2:
                    layer = {'name': t[1]}
                elif k == 'SITE' and len(t) == 2:
                    site = t[1]
    return macros, tech
