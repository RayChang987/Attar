"""Facts, capabilities and provenance (spec 5.1).

A Design holds a set of facts; each capability (an algorithm or evaluator) requires a set of facts. What a design can
do follows from what it has. Missing facts are never filled in silently: deriving them is an explicit operation of the
future `attar.derive` package (TODO), and derived facts carry the provenance 'derived:<method>'.
"""

# problem facts (the "question"); 'placement' is solution state and is tracked by the Design itself
FACTS = {
    'netlist': 'cells, nets and pins (connectivity), net weights',
    'cell_geometry': 'cell width/height and flags (movable, fixed, macro)',
    'pin_offsets': 'pin locations relative to their cell (master offsets + orientation)',
    'floorplan.die': 'the die array (outline of each die)',
    'floorplan.io': 'IO pin locations (on the die boundary)',
    'floorplan.macros': 'macro and fixed-cell (e.g. tap cell) locations',
    'rows': 'placement rows: origin, site width, height, die',
    'constraints': 'fence regions, placement blockages',
    'routing': 'routing layers, GCell grid and capacities, pin access',
    'timing': 'Liberty arcs and pin capacitances, clocks, I/O delays',
    'stack': '3D: per-die technology, hybrid bonding terminals',
    'writeback': 'a source DEF (path + hash) the placement can be written back into',
    'slow_loop': 'a slow-loop flow state to hand the placement to (an ORFS run and stage, on a platform ORFS knows)',
    'placement': 'locations and orientations of all movable cells (solution state)',
}

CAPABILITIES = {
    'hpwl': {'netlist', 'cell_geometry', 'pin_offsets', 'placement'},
    'wirelength_model': {'netlist', 'cell_geometry', 'pin_offsets'},          # WA etc. on caller-given positions
    'die_size': {'cell_geometry'},                         # on the rows' site grid when there are rows
    'io_place': {'netlist', 'floorplan.die'},
    'macro_place': {'netlist', 'cell_geometry', 'pin_offsets', 'floorplan.die', 'floorplan.io'},
    'global_place': {'netlist', 'cell_geometry', 'pin_offsets', 'floorplan.die', 'floorplan.io', 'floorplan.macros'},
    'legalize': {'netlist', 'cell_geometry', 'pin_offsets', 'floorplan.die', 'rows', 'placement'},
    'detailed_place': {'netlist', 'cell_geometry', 'pin_offsets', 'floorplan.die', 'rows', 'placement'},
    'global_route': {'netlist', 'cell_geometry', 'pin_offsets', 'floorplan.die', 'routing', 'placement'},
    'sta': {'netlist', 'timing'},
    'export': {'writeback'},                               # the placement, when there is one, and the floorplan
    'orfs_handoff': {'writeback', 'slow_loop', 'floorplan.io', 'floorplan.macros', 'placement'},
    'orfs_floorplan_handoff': {'writeback', 'slow_loop', 'floorplan.io', 'floorplan.macros'},   # from ORFS 2_1
    'plot': {'cell_geometry', 'floorplan.die'},
}

HINTS = {
    'placement': "run a placer, assign d.pos, or d.restore('reference') if the design keeps a reference placement",
    'writeback': 'distill from a DEF (Bookshelf and other formats without a DEF cannot be written back)',
    'slow_loop': 'distill from an ORFS run (tools/distill_orfs.py); designs without a real cell library end in Attar',
    'floorplan.io': 'place the IO pins: attar.naive.place_io (or a source with placed pins; ORFS: GPL_PLACE_IOS=0)',
    'floorplan.macros': 'place the macros: attar.naive.place_macros (or a source with placed macros)',
}

# what each capability means, for status()
DESCRIBE = {
    'hpwl': 'evaluate HPWL', 'wirelength_model': 'WA wirelength on given positions',
    'die_size': 'size the die for a utilization', 'io_place': 'IO pin placement', 'macro_place': 'macro placement', 'global_place': 'global placement', 'legalize': 'legalization', 'detailed_place': 'detailed placement',
    'global_route': 'global routing', 'sta': 'static timing analysis', 'export': 'write back to DEF',
    'orfs_handoff': 'hand off to ORFS for validation', 'orfs_floorplan_handoff': 'hand the floorplan to ORFS (2_1)',
    'plot': 'plot the design',
}
DEFAULT_HINT = 'load a source that provides it (deriving it, attar.derive, is not implemented yet)'


class MissingFacts(RuntimeError):
    def __init__(self, design, capability, missing):
        self.capability, self.missing = capability, sorted(missing)
        lines = ['%s cannot do %r: missing %s' % (design, capability, ', '.join(self.missing))]
        for m in self.missing:
            lines.append('  %s (%s): %s' % (m, FACTS.get(m, '?'), HINTS.get(m, DEFAULT_HINT)))
        super().__init__('\n'.join(lines))


def infer_facts(manifest, raw):
    """Facts of an IR written before manifests recorded them (schema <= 0.2): read off the stored arrays."""
    src = manifest.get('source', {}).get('format', 'unknown')
    f = {}
    if 'n_ptr' in raw and 'p_cell' in raw:
        f['netlist'] = 'def' if src == 'lefdef' else src
    if 'm_size' in raw:
        f['cell_geometry'] = 'lef' if src == 'lefdef' else src
    if 'm_pin_box' in raw:
        f['pin_offsets'] = 'lef' if src == 'lefdef' else src
    if 'die_box' in raw or 'die' in raw:
        f['floorplan.die'] = 'def' if src == 'lefdef' else src
        f['floorplan.io'] = f['floorplan.die']
        f['floorplan.macros'] = f['floorplan.die']
    if 'r_xy' in raw and len(raw['r_xy']):
        f['rows'] = 'def' if src == 'lefdef' else src
    if manifest.get('writeback'):
        f['writeback'] = 'def'
    return f


def upgrade_facts(facts, manifest):
    """Schema <= 0.3 had one 'floorplan.fixed' fact (IO pins and macros together): split it. Without it, the IO pins
    were unplaced (io_unplaced > 0) while the macros came placed with the floorplan."""
    f = dict(facts)
    if 'floorplan.fixed' in f:
        v = f.pop('floorplan.fixed')
        f.setdefault('floorplan.io', v)
        f.setdefault('floorplan.macros', v)
    elif 'floorplan.die' in f and 'floorplan.macros' not in f and manifest.get('io_unplaced') and \
            not manifest.get('macros_unplaced'):
        f['floorplan.macros'] = f['floorplan.die']
    return f
