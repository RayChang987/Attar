# Concepts

Attar has three core objects: a **Design** holds the state, an **Evaluator** measures it, and **attar.naive**
holds basic tensorized engines that change it.

## Facts, capabilities, provenance

A design is a set of *facts*. Each fact has a *provenance*: the source it came from, or the engine that made it.

| fact | what it is |
| --- | --- |
| `netlist` | cells, nets, pins, net weights |
| `cell_geometry` | cell sizes and flags (movable, fixed, macro) |
| `pin_offsets`, `pin_direction` | pin locations relative to their cell; which pins drive their net |
| `floorplan.die`, `floorplan.io`, `floorplan.macros` | die outline; IO pins; macro and fixed-cell locations |
| `rows` | placement rows: origin, site width, height, orientation |
| `constraints` | fence regions, placement blockages (counted, not stored yet) |
| `routing` | what blocks routing capacity; the routing layer range |
| `timing` | Liberty arcs, pin capacitances, clocks, I/O delays (from the design's Liberty and SDC) |
| `library` | a shared cell library (`zoo/libraries/<name>`) for sizing and buffering |
| `cell_library` | cells are instances of LEF macros, so the result can go back to DEF |
| `tracks` | routing tracks per layer and the layer of every macro pin |
| `pdn` | PDN metal on the three lowest routing layers |
| `def_frame` | the source DEF minus what Attar writes itself (what `export` writes into) |
| `placement` | positions of the movable cells |

Each *capability* (an engine or a metric) needs a set of facts, for example:

| capability | requires |
| --- | --- |
| `hpwl` | netlist, cell_geometry, pin_offsets, placement |
| `global_place` | netlist, cell_geometry, pin_offsets, floorplan.die, floorplan.io, floorplan.macros |
| `legalize`, `detailed_place` | netlist, cell_geometry, pin_offsets, floorplan.die, rows, placement |
| `global_route` | netlist, cell_geometry, pin_offsets, floorplan.die, routing, placement |
| `sta` | netlist, pin_direction, timing |
| `resize`, `netlist_edit` | netlist, cell_library, pin_offsets, timing, library |
| `export` | netlist, cell_library, def_frame |

`d.capabilities()`, `d.missing('global_place')`, `d.require('legalize')` and `d.status()` answer what a design can
do. Calling an engine without its facts raises `MissingFacts` with a hint.

**Missing facts are never filled in silently.** A Bookshelf benchmark has no cell library, so it does not export
to DEF. Engines add facts with their own provenance, e.g. `floorplan.io  naive:place_io(nets)` after
`d.floorplan()`.

## Modes

Zoo names are `<mode>/<suite>/<design>`. The mode says what the user solves:

- **`before_fp`, a floorplan problem:** netlist and cell library are given; the die, IO pins and macros are the
  answer. Attar draws no power grid: your flow draws it from the exported DEF.
- **`after_fp`, a placement problem:** die, IO pins, macros and power grid are given; the cells are the answer.

## Self-contained designs and export

A distilled design holds everything the fast loop reads, so a copy of its folder works on any machine. Manifests
name each source file by its sha256, never by path, and Attar reads no environment variables.

`d.export(out_def)` writes a DEF. The *frame* (fact `def_frame`) keeps the source DEF verbatim except for what
Attar writes from the IR:

| in the frame (verbatim) | written from the IR |
| --- | --- |
| header, `DIEAREA`, `ROW`, `TRACKS`, `VIAS`, `PINS`, `SPECIALNETS`, `BLOCKAGES`, `REGIONS`, `GROUPS`, fixed components, power nets | components Attar may move, signal nets (source wiring dropped) |

Floorplan changes (IO pins, die, rows) are stamped onto the frame first. The fact is only claimed when frame plus
IR give back the source design. Bookshelf, YAL and GSRC designs have no frame and do not export.

## The PDN as a placement rule

From the `pdn` fact come OpenROAD dpl's rules: PG via metal, grown by the layer spacing, must stay clear of signal
pins on its layer, and vertical PG stripes on the second and third layer block cells with pins there.
`Evaluator.legality()` counts the cells that break them (`pdn_blocked`).

A second check, `Evaluator.pin_access`, flags signal pins the router cannot reach: every access point (a track of
the layer above crossing the pin) has a via landing closer than the layer spacing to PG metal. On routed NS3K runs
these are exactly the pins with a signal-to-PG DRV ([validation](validation/pin_access.md)).

The legalizer clears both last:

1. a row-wise dynamic program on the GPU: each single-row cell may shift up to 8 sites and mirror where the LEF
   `SYMMETRY` allows, at a cost for blocked pins, displacement and mirroring (`pin_access=False`: dpl's rule only);
2. cells still blocked, multi-row ones included, mirrored in place where that clears them;
3. the few left shifted along their rows or moved to the nearest free spot.

On NS3K a few hundred cells per 60k need it, nearly all fixed by mirroring; on ASAP7 nothing is blocked.

## Design

`attar.load(name_or_path, device)` returns a `Design`. Everything is a tensor on `device`:

| attribute | shape | meaning |
| --- | --- | --- |
| `pos` | (N, 2) | lower-left corners, microns |
| `size`, `orient` | (N, 2), (N,) | size and orientation (DEF order N, S, W, E, FN, FS, FW, FE) |
| `flags`, `movable`, `placed` | (N,) | FIXED, MACRO, PLACED, TERMINAL, PHYSICAL, ASIDE bits; masks |
| `p_cell`, `p_net`, `pin_off`, `pin_dir` | (P, ...) | pins: owner cell, net, offset from the cell center, direction |
| `n_ptr`, `n_weight` | (E+1,), (E,) | nets as CSR over pins |
| `die_box`, `core_box` | (D, 4) | per die |
| `row_xy`, `row_h`, `row_site_w`, `row_sites`, `row_orient` | (R, ...) | rows |

`d.snapshot('mine')` and `d.restore('reference')` save and restore placements. A design from a placed source keeps
that placement as the `reference` snapshot and starts unplaced. `d.unplace()` also sets aside unconnected movable
cells (fill, spacers: flag `ASIDE`); engines ignore them and export writes them `UNPLACED`.

## Evaluator

```python
ev = attar.Evaluator(d, fast=True)       # fast: CUDA-graph HPWL
ev.hpwl()                                 # exact pin-to-pin HPWL (0-d tensor, no host sync)
ev.hpwl_batch(P)                          # (B, N, 2) placements at once
ev.legality()                             # rows, sites, overlaps, rail parity, PDN
ev.evaluate(metrics=('hpwl', 'steiner_wirelength', 'rudy', 'density_overflow', 'legality'))
```

Every metric also takes explicit positions, so it works inside an optimizer loop.

## attar.naive

Simple, tensorized engines, each a starting point to replace: `global_place` (ePlace / DREAMPlace electrostatics),
`legalize` (Abacus by row class, then the PDN steps above), `place` (both), `detailed_place` (OpenROAD dpo's moves
on the GPU), `cts`; for `before_fp`: `make_die`, `place_io`, `place_macros` (mixed-size GP, then macros on the
grid that puts their pins on tracks; `packing=True` for block-level designs) and `floorplan` (all three).
[features.md](features.md) has a line on each; timing, routing and edits are design methods ([api.md](api.md)).
