"""Design reports: status (which facts a design has (✅) or lacks (❌), what it can do, and what is left before the
answer goes out as a DEF) and summary (a few lines of numbers). Everything here follows from the IR; nothing on disk
outside it is looked at. The text is plain; styled() colors it for a terminal."""
import re
import sys

import torch

from .facts import CAPABILITIES, DESCRIBE, FACTS, INTERNAL_FACTS, PLACE_STEPS

OK, NO = '✅', '❌'   # ✅ ❌
PROV_WIDTH = 18       # provenance column; longer provenances push their detail right
SECTIONS = ('Problem facts', 'Solution state', 'Capabilities')


def _detail(d, fact):
    """A one-line description of what the design holds for a fact."""
    c = d.manifest.get('counts', {})
    u = d.units
    if fact == 'netlist':
        return '%d nets, %d pins' % (c.get('nets', 0), c.get('pins', 0))
    if fact == 'cell_geometry':
        return '%d cells (%d movable, %d macros), %d masters' % (c.get('cells', 0), c.get('movable', 0), c.get('macros', 0), c.get('masters', 0))
    if fact == 'floorplan.die' and d.die_box is not None:
        return '%d die(s), %s %s' % (d.num_dies, ', '.join('%.2f x %.2f' % (w, h) for w, h in zip(d.die_w.tolist(), d.die_h.tolist())), u)
    if fact == 'floorplan.io':
        return '%d IO pins placed' % c.get('terminals', 0)
    if fact == 'floorplan.macros':
        return '%d macros placed' % c.get('macros', 0)
    if fact == 'rows' and d.row_h is not None:
        hs = sorted({round(h, 4) for h in d.row_h.tolist()})
        return '%d rows, height %s %s' % (len(d.row_h), '/'.join('%g' % h for h in hs), u)
    if fact == 'cell_library':
        lef = d.manifest.get('source', {}).get('lef', [])
        return '%d LEF file(s): %s' % (len(lef), ', '.join(l.get('name', '?') for l in lef[:3]))
    if fact == 'pin_direction':
        c = d.manifest.get('pin_direction', {})
        return '%d drivers, %d sinks; %d nets with one driver, %d with none, %d with several' % (
            c.get('drivers', 0), c.get('sinks', 0), c.get('nets_one_driver', 0), c.get('nets_no_driver', 0),
            c.get('nets_multi_driver', 0))
    if fact == 'pdn':
        a = d.raw
        names = d.names['layers']
        lay = sorted({names[i] for i in a['pg_layer'].tolist()}) if 'pg_layer' in a else []
        nv = int((a['pg_kind'] == 2).sum()) if 'pg_kind' in a else 0
        return '%d wire and %d via shapes on %s' % (len(a.get('pg_rect', [])) - nv, nv, ', '.join(lay))
    if fact == 'def_frame':
        f = d.manifest.get('def_frame', {})
        return 'PDN, pins, fixed cells of the source DEF, kept verbatim (%.0f kB)' % (f.get('bytes', 0) / 1e3)
    if fact == 'tracks':
        t = d.tracks()
        return '%d track grids over %d layers' % (len(t), len({k[1] for k in t}))
    return ''


def _why_missing(d, fact):
    dropped = d.dropped
    if fact in dropped:
        return 'dropped: ' + dropped[fact]
    if fact == 'floorplan.io' and d.manifest.get('io_unplaced'):
        return '%d IO pins unplaced in the source' % d.manifest['io_unplaced']
    if fact == 'floorplan.macros' and d.manifest.get('macros_unplaced'):
        return '%d macros unplaced in the source' % d.manifest['macros_unplaced']
    if fact == 'constraints' and d.manifest.get('not_stored', {}).get('placement_blockages'):
        return '%d placement blockages in the source, not stored yet' % d.manifest['not_stored']['placement_blockages']
    if fact == 'tracks' and 'cell_library' in d.facts:
        return 'not in this IR (attar-ir < 0.5): distill again; macros go on the row grid meanwhile'
    return ''


def _line(mark, w, name, note='', pw=None, prov=None):
    """One checklist line: mark, name padded to w, [provenance padded to pw,] note; no trailing blanks."""
    if pw is None:
        return ('  %s %-*s  %s' % (mark, w, name, note)).rstrip()
    return ('  %s %-*s  %-*s  %s' % (mark, w, name, pw, prov, note)).rstrip()


def styled(text, stream=None):
    """The report text with ANSI styles when `stream` (default stdout) is a terminal: section titles bold, the
    reasons on ❌ lines dimmed, the closing '->' advice bold. Unchanged otherwise."""
    stream = stream or sys.stdout
    if not getattr(stream, 'isatty', lambda: False)():
        return text
    B, D, R = '\033[1m', '\033[2m', '\033[0m'
    out = []
    for i, ln in enumerate(text.split('\n')):
        if i == 0 or ln in SECTIONS:
            ln = B + ln + R
        elif ln.lstrip().startswith('->'):
            ln = B + ln + R
        elif ln.startswith('  ' + NO):
            m = re.match(r'(  %s \S+)(.*)' % NO, ln)
            ln = m.group(1) + D + m.group(2) + R if m else ln
        out.append(ln)
    return '\n'.join(out)


def _next_step(d, have, fp_ok, fp_issues):
    """The closing advice: what is left before the answer goes out as a DEF (into the user's own flow)."""
    ex = 'd.export("out.def")'
    if 'cell_library' not in d.facts:
        return '  -> no cell library: Attar is the end point for this design (a DEF of it would mean nothing)'
    gap = sorted({'floorplan.die', 'floorplan.io', 'floorplan.macros'} - set(d.facts))
    if gap:
        return '  -> missing %s: d.floorplan() places it; then %s' % (', '.join(gap), ex)
    if not fp_ok:
        return '  -> the floorplan is not legal (%s)' % '; '.join(fp_issues)
    if d.missing('export'):
        return '  -> export needs %s' % ', '.join(d.missing('export'))
    if 'placement' in have:
        return '  -> ready: %s writes the design for your flow' % ex
    if d.mode == 'before_fp':
        return '  -> ready: %s writes the floorplan for your flow (it draws the PDN and places)' % ex
    return '  -> only a placement is missing: place the design (or restore a snapshot), then %s' % ex


def status(d):
    m = d.manifest
    have = d.present
    src = m.get('suite', m['source'].get('format'))
    L = ['%s  (%s, fidelity %s)' % (m.get('design'), src, m.get('fidelity', '?'))]
    if m.get('target_density'):
        L.append('  placement target density %.3g (%s)' % (m['target_density'], m.get('target_density_source', 'the source')))
    L += ['', 'Problem facts']
    w = max(len(f) for f in FACTS if f not in INTERNAL_FACTS)
    pw = min(PROV_WIDTH, max([len(str(v)) for k, v in d.facts.items() if k not in INTERNAL_FACTS] + [1]))
    for f in FACTS:
        if f == 'placement' or f in INTERNAL_FACTS:
            continue
        if f in d.facts:
            L.append(_line(OK, w, f, _detail(d, f), pw, d.facts[f]))
        else:
            L.append(_line(NO, w, f, _why_missing(d, f), pw, '-'))
    L += ['', 'Solution state']
    n_unplaced = int((~d.placed & d.movable).sum())
    snaps = ', '.join(d.snapshots) or 'none'
    if 'placement' in have:
        L.append(_line(OK, w, 'placement', 'all movable cells placed  (snapshots: %s)' % snaps))
    else:
        L.append(_line(NO, w, 'placement', '%d movable cells unplaced  (snapshots: %s)' % (n_unplaced, snaps)))
    fp_ok, fp_issues = d.floorplan_legal()
    L.append(_line(OK if fp_ok else NO, w, 'floorplan', 'legal (checked now: macros in place and apart, IO pins in the '
                                                         'die)' if fp_ok else 'not legal: ' + '; '.join(fp_issues)))
    L += ['', 'Capabilities']
    shown = {c: req for c, req in CAPABILITIES.items() if c not in PLACE_STEPS}
    cw = max(len(DESCRIBE.get(c, c)) for c in shown)
    for c, req in shown.items():
        miss, why = sorted(req - have), d.blocked(c)
        note = why or ('needs ' + ', '.join(miss) if miss else '')
        L.append(_line(NO if note else OK, cw, DESCRIBE.get(c, c), note))
    L += ['', _next_step(d, have, fp_ok, fp_issues)]
    return '\n'.join(L)


def summary(d):
    """A few lines: counts, each die with its core, rows and utilization, unplaced cells, IO pins, facts, capabilities."""
    c = d.manifest['counts']
    n_io = len(d.io) if d.io is not None else c.get('terminals', 0)
    lines = ['%s: %d cells (%d movable, %d macros), %d nets, %d IO pins, %s'
             % (d.manifest['design'], c['cells'], c['movable'], c['macros'], c['nets'], n_io,
                '%d die(s)' % d.num_dies if d.die_box is not None else 'no die outline')]
    if d.die_box is not None:
        util = d.utilization()
        for k in range(d.num_dies):
            die, core = d.die_box[k].tolist(), d.core_box[k].tolist()
            rows = 'no rows'
            if d.row_h is not None:
                hs = sorted({round(h, 4) for h in d.row_h[d.row_die == k].tolist()})
                rows = '%d rows (height %s um)' % (int((d.row_die == k).sum()), '/'.join('%g' % h for h in hs))
            lines.append('  die %d: %.2f x %.2f um, core (%.2f, %.2f)-(%.2f, %.2f), %s, util %.1f%%'
                         % (k, die[2] - die[0], die[3] - die[1], *core, rows, 100 * util[k]))
    unplaced = int((~d.placed).sum())
    if unplaced:
        hint = " (d.restore('reference') restores the kept placement)" if 'reference' in d.snapshots else ''
        lines.append('  %d cells unplaced%s' % (unplaced, hint))
    if d.io_dir is not None and len(d.io_dir):
        n = torch.bincount(d.io_dir, minlength=4).tolist()
        lines.append('  IO: ' + ', '.join('%d %s' % (n[i], d.IO_DIRS[i].lower()) for i in range(4) if n[i]))
    facts = sorted((k, v) for k, v in d.facts.items() if k not in INTERNAL_FACTS)
    lines.append('  facts: ' + ', '.join('%s<%s>' % kv for kv in facts))
    lines.append('  can do: ' + ', '.join(d.capabilities()))
    return '\n'.join(lines)

