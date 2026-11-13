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
    'pin_direction': 'which pins drive their net and which are driven (LEF pin DIRECTION, Bookshelf I/O)',
    'floorplan.die': 'the die array (outline of each die)',
    'floorplan.io': 'IO pin locations (on the die boundary)',
    'floorplan.macros': 'macro and fixed-cell (e.g. tap cell) locations',
    'rows': 'placement rows: origin, site width, height, die',
    'constraints': 'fence regions, placement blockages',
    'routing': 'routing layers, GCell grid and capacities, pin access',
    'timing': 'Liberty arcs and pin capacitances, clocks, I/O delays',
    'library': 'a shared cell library (zoo/libraries/<name>): every cell of the platform corner with its Liberty data, LEF '
               'geometry and equivalence classes (sizing, buffering); the design\'s own macros in a local library',
    'stack': '3D: per-die technology, hybrid bonding terminals',
    'cell_library': 'cells are instances of a real cell library (LEF masters): a DEF of the design means something',
    'tracks': 'routing tracks and the routing layer of every master pin (pin access: macros go on-track)',
    'pdn': 'PDN metal (wires, vias) on the lowest routing layers: signal pins must keep clear of it',
    'def_frame': 'the source DEF minus what Attar writes (movable components, signal nets), kept verbatim: export needs no template',
    'slow_loop': 'the design comes from a slow-loop flow stage it can be handed back to (an ORFS stage on a platform ORFS knows)',
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
    'sta': {'netlist', 'pin_direction', 'timing'},
    'export': {'netlist', 'cell_library', 'def_frame'},    # the source DEF's frame + the IR
    'orfs_handoff': {'cell_library', 'def_frame', 'slow_loop', 'floorplan.io', 'floorplan.macros', 'placement'},
    'orfs_floorplan_handoff': {'cell_library', 'def_frame', 'slow_loop', 'floorplan.io', 'floorplan.macros'},   # from ORFS 2_1
    'plot': {'cell_geometry', 'floorplan.die'},
}

HINTS = {
    'placement': "run a placer, assign d.pos, or d.restore('reference') if the design keeps a reference placement",
    'cell_library': 'this source has no cell library (Bookshelf, YAL, GSRC): a DEF of it would not mean anything',
    'tracks': 'distill from LEF/DEF (attar-ir >= 0.5): macros then go on the routing tracks, else on the row grid',
    'pin_direction': 'the source states no pin directions (or the IR is older than attar-ir 0.6: distill it again)',
    'pdn': 'the source has no special nets on the pin layers (or the IR is older than attar-ir 0.8: distill it again)',
    'def_frame': 'distill from LEF/DEF (attar-ir >= 0.7): the frame of the source DEF is what export writes into',
    'slow_loop': 'distill from an ORFS run (its 3_3 or 2_1 DEF); designs without a real cell library end in Attar',
    'floorplan.io': 'place the IO pins: attar.naive.place_io (or a source with placed pins; ORFS: GPL_PLACE_IOS=0)',
    'floorplan.macros': 'place the macros: attar.naive.place_macros (or a source with placed macros)',
}

# what each capability means, for status()
DESCRIBE = {
    'hpwl': 'evaluate HPWL', 'wirelength_model': 'WA wirelength on given positions',
    'die_size': 'size the die for a utilization', 'io_place': 'IO pin placement', 'macro_place': 'macro placement', 'global_place': 'global placement', 'legalize': 'legalization', 'detailed_place': 'detailed placement',
    'global_route': 'global routing', 'sta': 'static timing analysis', 'export': 'write a DEF (source frame + IR)',
    'orfs_handoff': 'hand off to ORFS for validation', 'orfs_floorplan_handoff': 'hand the floorplan to ORFS (2_1)',
    'plot': 'plot the design',
}
DEFAULT_HINT = 'load a source that provides it (deriving it, attar.derive, is not implemented yet)'


class MissingFacts(RuntimeError):
    def __init__(self, design, capability, missing, reasons=None):
        self.capability, self.missing = capability, sorted(missing)
        lines = ['%s cannot do %r: missing %s' % (design, capability, ', '.join(self.missing))]
        for m in self.missing:
            lines.append('  %s (%s): %s' % (m, FACTS.get(m, '?'), (reasons or {}).get(m) or HINTS.get(m, DEFAULT_HINT)))
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
    if src == 'lefdef':
        f['cell_library'] = 'lef'
    return f


def upgrade_facts(facts, manifest):
    """Older schemas: 'writeback' (<= 0.4) becomes 'cell_library'; one 'floorplan.fixed' fact (<= 0.3, IO pins and
    macros together) is split. Without it, the IO pins were unplaced (io_unplaced > 0) while the macros came placed."""
    f = dict(facts)
    if 'writeback' in f:                  # schema <= 0.4: 'writeback' meant a LEF/DEF source, i.e. a cell library
        f.pop('writeback')
        if manifest.get('source', {}).get('format') == 'lefdef':
            f.setdefault('cell_library', 'lef')
    if 'floorplan.fixed' in f:
        v = f.pop('floorplan.fixed')
        f.setdefault('floorplan.io', v)
        f.setdefault('floorplan.macros', v)
    elif 'floorplan.die' in f and 'floorplan.macros' not in f and manifest.get('io_unplaced') and \
            not manifest.get('macros_unplaced'):
        f['floorplan.macros'] = f['floorplan.die']
    return f
