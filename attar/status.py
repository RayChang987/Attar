"""Design status: which facts a design has (✅) or lacks (❌), what it can do, and what a hand-off to ORFS still needs."""
import os

from .facts import CAPABILITIES, DESCRIBE, FACTS, HINTS, DEFAULT_HINT

OK, NO = '✅', '❌'   # ✅ ❌

TIERS = {
    'A': 'can be validated in ORFS (slow loop)',
    'B': 'can be written back to DEF, but cannot be validated in a slow loop (yet)',
    'C': 'Attar is the end point (no technology to hand off to)',
}


def tier(d):
    """A: ORFS hand-off possible (given a placement); B: DEF write-back only; C: neither (spec 3, principle 5)."""
    static = CAPABILITIES['orfs_handoff'] - {'placement'}
    if static <= set(d.facts) and _orfs_run_ok(d)[0]:
        return 'A'
    return 'B' if 'writeback' in d.facts else 'C'


def _upgrade(d, t):
    """For a tier-B design with a recorded ORFS run: what keeps it from tier A."""
    if t != 'B' or 'slow_loop' not in d.facts or not _orfs_run_ok(d)[0]:
        return ''
    gap = sorted(CAPABILITIES['orfs_handoff'] - {'placement'} - set(d.facts))
    return ' (tier A once it has %s)' % ', '.join(gap) if gap else ''


def _orfs_run_ok(d):
    o = d.manifest.get('orfs')
    if not o:
        return False, 'no ORFS run recorded'
    odb = os.path.join(o['run'], 'results', o['platform'], o['design'], 'base', o['stage'] + '.odb')
    if not os.path.exists(odb):
        return False, 'ORFS run state missing: %s' % odb
    return True, 'run %s @ %s' % (o['run'], o['stage'])


def _unit(d):
    return d.units


def _detail(d, fact):
    """A one-line description of what the design holds for a fact."""
    c = d.manifest.get('counts', {})
    u = _unit(d)
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
    if fact == 'writeback':
        return d.manifest['source']['def']['path']
    if fact == 'slow_loop':
        return _orfs_run_ok(d)[1]
    return ''


def _why_missing(d, fact):
    dropped = getattr(d, 'dropped', {})
    if fact in dropped:
        return 'dropped: ' + dropped[fact]
    if fact == 'floorplan.io' and d.manifest.get('io_unplaced'):
        return '%d IO pins unplaced in the source' % d.manifest['io_unplaced']
    if fact == 'floorplan.macros' and d.manifest.get('macros_unplaced'):
        return '%d macros unplaced in the source' % d.manifest['macros_unplaced']
    if fact == 'constraints' and d.manifest.get('not_stored', {}).get('placement_blockages'):
        return '%d placement blockages in the source, not stored yet' % d.manifest['not_stored']['placement_blockages']
    return ''


def status(d, color=True):
    m = d.manifest
    t = tier(d)
    have = d.present
    L = ['%s  (%s, fidelity %s)' % (m.get('design'), m.get('suite', m['source'].get('format')), m.get('fidelity', '?')),
         '  tier %s: %s' % (t, TIERS[t] if not getattr(d, 'dropped', None) else
                            'Attar is the end point for this state (write-back dropped)') + _upgrade(d, t), '', 'Problem facts']
    w = max(len(f) for f in FACTS)
    for f in FACTS:
        if f == 'placement':
            continue
        if f in d.facts:
            extra = _detail(d, f)
            if f == 'slow_loop' and not _orfs_run_ok(d)[0]:
                L.append('  %s %-*s  %-10s %s' % (NO, w, f, d.facts[f], extra)); continue
            L.append('  %s %-*s  %-10s %s' % (OK, w, f, d.facts[f], extra))
        else:
            L.append('  %s %-*s  %-10s %s' % (NO, w, f, '-', _why_missing(d, f)))
    L += ['', 'Solution state']
    n_unplaced = int((~d.placed & d.movable).sum())
    snaps = ', '.join(d.snapshots) or 'none'
    if 'placement' in have:
        L.append('  %s %-*s  all movable cells placed  (snapshots: %s)' % (OK, w, 'placement', snaps))
    else:
        L.append('  %s %-*s  %d movable cells unplaced  (snapshots: %s)' % (NO, w, 'placement', n_unplaced, snaps))
    L += ['', 'Capabilities']
    cw = max(len(DESCRIBE.get(c, c)) for c in CAPABILITIES)
    for c, req in CAPABILITIES.items():
        miss = sorted(req - have)
        if c == 'orfs_handoff' and 'slow_loop' in d.facts and not _orfs_run_ok(d)[0]:
            miss = sorted(set(miss) | {'slow_loop'})
        L.append('  %s %-*s  %s' % (OK if not miss else NO, cw, DESCRIBE.get(c, c), '' if not miss else 'needs ' + ', '.join(miss)))
    # hand-off to ORFS: the requirement list, then what to do
    L += ['', 'Hand-off to ORFS needs']
    req = sorted(CAPABILITIES['orfs_handoff'])
    for f in req:
        ok = f in have and (f != 'slow_loop' or _orfs_run_ok(d)[0])
        hint = '' if ok else (_why_missing(d, f) or HINTS.get(f, DEFAULT_HINT))
        L.append('  %s %-*s  %s' % (OK if ok else NO, w, f, hint))
    if t == 'A':
        if 'placement' in have:
            L.append('  -> ready: d.export("out.def"), then  %s' % handoff_command(d, 'out.def'))
        else:
            L.append('  -> only a placement is missing: place the design (or restore a snapshot), then export and hand off')
    elif getattr(d, 'dropped', None):
        L.append('  -> export writes the cells only, not the die, IO pins or macros an engine changed: no hand-off '
                 'from this state (for validation, make the floorplan in the flow and distill again)')
    elif t == 'B' and 'slow_loop' in d.facts and _orfs_run_ok(d)[0]:
        gap = sorted(CAPABILITIES['orfs_handoff'] - {'placement'} - set(d.facts))
        if d.manifest.get('orfs', {}).get('stage') == '2_1_floorplan':
            L.append('  -> missing %s: d.floorplan() places it; then d.export("out.def") and hand the floorplan to '
                     'ORFS (tools/orfs_handoff.py)' % ', '.join(gap))
        else:
            L.append('  -> an ORFS run is recorded, but the source lacks %s; an engine can place it, but this ORFS '
                     'stage has built tapcells and PDN already: distill from 2_1 to hand off a floorplan' % ', '.join(gap))
    elif t == 'B':
        L.append('  -> no slow loop for this source: the placement can be written back to DEF, not validated here')
    else:
        L.append('  -> no technology to hand off to: Attar is the end point for this design')
    return '\n'.join(L)


def handoff_command(d, out_def):
    """The command that injects an exported DEF into the recorded ORFS run and finishes the flow (spec 7.2)."""
    o = d.manifest.get('orfs')
    if not o:
        raise RuntimeError('%s has no ORFS run to hand off to (tier %s)' % (m_name(d), tier(d)))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if o['stage'] == '2_1_floorplan':      # the floorplan goes to ORFS, which places and routes from there
        return 'python3 %s %s %s %s' % (os.path.relpath(os.path.join(root, 'tools', 'orfs_handoff.py')),
                                        o['design'], os.path.basename(o['run']), out_def)
    return 'STAGE=%s python3 %s %s %s def=%s' % (o['stage'], os.path.relpath(os.path.join(root, 'tools', 'orfs_roundtrip.py')),
                                                  o['design'], os.path.basename(o['run']), out_def)


def m_name(d):
    return d.manifest.get('design', d.path)
