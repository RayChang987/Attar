"""Internal: handing a design back to the slow loop it came from (ORFS), for Attar's own validation.

Not part of the user API. Users take the result out with Design.export (a DEF) into their own flow (OpenROAD,
Innovus, ICC2, ...). Attar keeps this only to check itself against ORFS: tools/orfs_roundtrip.py (a placement into an
ORFS run), tools/orfs_handoff.py (a floorplan into ORFS 2_1) and their tests use it.

The fact 'slow_loop' says that a design comes from an ORFS stage on a platform ORFS knows; an engine that changes the
floorplan after ORFS built tapcells and PDN around it drops the fact (Design._floorplan_changed).
"""
import os

from .facts import MissingFacts

# what a hand-off needs, beyond what export needs
CAPABILITIES = {
    'orfs_handoff': {'cell_library', 'def_frame', 'slow_loop', 'floorplan.io', 'floorplan.macros', 'placement'},
    'orfs_floorplan_handoff': {'cell_library', 'def_frame', 'slow_loop', 'floorplan.io', 'floorplan.macros'},  # 2_1
}


def missing(d, capability='orfs_handoff'):
    """The facts a hand-off still needs (sorted)."""
    return sorted(CAPABILITIES[capability] - d.present)


def can(d, capability='orfs_handoff'):
    return not missing(d, capability)


def tier(d):
    """A: an ORFS hand-off is possible (given a placement); B: a cell library, so export onto a DEF; C: neither."""
    if CAPABILITIES['orfs_handoff'] - {'placement'} <= set(d.facts):
        return 'A'
    return 'B' if 'cell_library' in d.facts else 'C'


def handoff_command(d, out_def, run_dir=None):
    """The command that hands an exported DEF to an ORFS run of this design (run_dir: its WORK_HOME) and finishes the
    flow there (spec 7.2). A floorplan (from ORFS 2_1) goes to ORFS only when it is legal (Design.floorplan_legal)."""
    o = d.manifest.get('orfs')
    fp = (o or {}).get('stage') == '2_1_floorplan'
    cap = 'orfs_floorplan_handoff' if fp else 'orfs_handoff'
    miss = missing(d, cap)
    if miss:
        raise MissingFacts(d.manifest.get('design', d.path), cap, miss)
    if fp:
        ok, issues = d.floorplan_legal()
        if not ok:
            raise ValueError('%s: the floorplan is not legal (%s); ORFS would not take it'
                             % (d.manifest.get('design'), '; '.join(issues)))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    key = o['design'] if o.get('platform', 'asap7') == 'asap7' else '%s/%s' % (o['platform'], o['design'])
    run = run_dir or '<ORFS WORK_HOME of %s>' % key
    if fp:                                 # the floorplan goes to ORFS, which places and routes from there
        return 'python3 %s %s %s %s' % (os.path.relpath(os.path.join(root, 'tools', 'orfs_handoff.py')), key, run,
                                        out_def)
    return 'python3 %s %s %s def=%s --stage %s' % (os.path.relpath(os.path.join(root, 'tools', 'orfs_roundtrip.py')),
                                                  key, run, out_def, o['stage'])
