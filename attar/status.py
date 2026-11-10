"""Design status: which facts a design has (✅) or lacks (❌), what it can do, and what a hand-off to ORFS still needs.
Everything here follows from the IR; nothing on disk outside it is looked at."""
import os

from .facts import CAPABILITIES, DESCRIBE, FACTS, HINTS, DEFAULT_HINT

OK, NO = '✅', '❌'   # ✅ ❌

TIERS = {
    'A': 'can be validated in ORFS (slow loop), given an ORFS run of it',
    'B': 'can be stamped onto a DEF of it, but cannot be validated in a slow loop (yet)',
    'C': 'Attar is the end point (no cell library: nothing to hand off to)',
}


def tier(d):
    """A: ORFS hand-off possible (given a placement); B: a cell library, so export onto a DEF; C: neither."""
    if CAPABILITIES['orfs_handoff'] - {'placement'} <= set(d.facts):
        return 'A'
    return 'B' if 'cell_library' in d.facts else 'C'


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
    if fact == 'tracks':
        t = d.tracks()
        return '%d track grids over %d layers' % (len(t), len({k[1] for k in t}))
    if fact == 'slow_loop':
        o = d.manifest.get('orfs', {})
        return 'ORFS %s, stage %s' % (o.get('design', '?'), o.get('stage', '?')) if o else ''
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
    if fact == 'tracks' and 'cell_library' in d.facts:
        return 'not in this IR (attar-ir < 0.5): distill again; macros go on the row grid meanwhile'
    return ''


def status(d, color=True):
    m = d.manifest
    t = tier(d)
    have = d.present
    head = TIERS[t] if not getattr(d, 'dropped', None) else 'no slow loop for this state (see slow_loop below)'
    if t == 'B' and 'slow_loop' in d.facts:
        head += ' (tier A once it has %s)' % ', '.join(sorted(CAPABILITIES['orfs_handoff'] - {'placement'} - set(d.facts)))
    L = ['%s  (%s, fidelity %s)' % (m.get('design'), m.get('suite', m['source'].get('format')), m.get('fidelity', '?')),
         '  tier %s: %s' % (t, head), '', 'Problem facts']
    w = max(len(f) for f in FACTS)
    for f in FACTS:
        if f == 'placement':
            continue
        if f in d.facts:
            L.append('  %s %-*s  %-10s %s' % (OK, w, f, d.facts[f], _detail(d, f)))
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
        L.append('  %s %-*s  %s' % (OK if not miss else NO, cw, DESCRIBE.get(c, c), '' if not miss else 'needs ' + ', '.join(miss)))
    # hand-off to ORFS: the requirement list, then what to do
    L += ['', 'Hand-off to ORFS needs']
    for f in sorted(CAPABILITIES['orfs_handoff']):
        ok = f in have
        L.append('  %s %-*s  %s' % (OK if ok else NO, w, f, '' if ok else (_why_missing(d, f) or HINTS.get(f, DEFAULT_HINT))))
    stage = m.get('orfs', {}).get('stage')
    ex = 'd.export("<a DEF of this design>", "out.def")'
    if t == 'A':
        if 'placement' in have or stage == '2_1_floorplan':      # 2_1: the floorplan alone goes to ORFS
            L.append('  -> ready: %s, then  %s' % (ex, handoff_command(d, 'out.def')))
        else:
            L.append('  -> only a placement is missing: place the design (or restore a snapshot), then export and hand off')
    elif 'slow_loop' in d.facts and stage == '2_1_floorplan':
        gap = sorted(CAPABILITIES['orfs_handoff'] - {'placement'} - set(d.facts))
        L.append('  -> missing %s: d.floorplan() places it; then %s and hand the floorplan to ORFS '
                 '(tools/orfs_handoff.py)' % (', '.join(gap), ex))
    elif 'slow_loop' in d.facts:
        gap = sorted(CAPABILITIES['orfs_handoff'] - {'placement'} - set(d.facts))
        L.append('  -> the source lacks %s; an engine can place it, but this ORFS stage has built tapcells and PDN '
                 'already: distill from 2_1 to hand off a floorplan' % ', '.join(gap))
    elif getattr(d, 'dropped', None):
        L.append('  -> an engine changed the floorplan after ORFS %s built tapcells and PDN around it: no hand-off from '
                 'this state (distill from 2_1 to hand off a floorplan)' % stage)
    elif t == 'B':
        L.append('  -> no slow loop for this source: %s writes the result' % ex)
    else:
        L.append('  -> no cell library: Attar is the end point for this design')
    return '\n'.join(L)


def handoff_command(d, out_def, run_dir=None):
    """The command that hands an exported DEF to an ORFS run of this design (run_dir: its WORK_HOME) and finishes the
    flow (spec 7.2)."""
    o = d.manifest.get('orfs')
    if not o:
        raise RuntimeError('%s does not come from an ORFS stage (tier %s)' % (d.manifest.get('design', d.path), tier(d)))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = run_dir or '<ORFS WORK_HOME of %s>' % o['design']
    if o['stage'] == '2_1_floorplan':      # the floorplan goes to ORFS, which places and routes from there
        return 'python3 %s %s %s %s' % (os.path.relpath(os.path.join(root, 'tools', 'orfs_handoff.py')), o['design'], run, out_def)
    return 'python3 %s %s %s def=%s --stage %s' % (os.path.relpath(os.path.join(root, 'tools', 'orfs_roundtrip.py')),
                                                  o['design'], run, out_def, o['stage'])
