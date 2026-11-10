# Concepts

Attar has three core objects: a **Design** holds the state, an **Evaluator** measures it, and **attar.naive**
holds a basic set of tensorized engines that change it. This page covers how they fit together.

## Facts, capabilities, provenance

A design is a set of *facts*. Each fact has a *provenance*: the source it came from, or the engine that produced it.

| fact | what it is |
| --- | --- |
| `netlist` | cells, nets, pins, net weights |
| `cell_geometry` | cell sizes and flags (movable, fixed, macro) |
| `pin_offsets` | pin locations relative to their cell |
| `pin_direction` | which pins drive their net and which are driven (LEF `DIRECTION`, Bookshelf `I`/`O`) |
| `floorplan.die` | the die outline (an array: 3D designs have several dies) |
| `floorplan.io` | IO pin locations on the die boundary |
| `floorplan.macros` | macro and fixed-cell locations |
| `rows` | placement rows: origin, site width, height, orientation |
| `constraints` | fence regions, placement blockages (not stored yet) |
| `routing`, `timing`, `stack` | routing resources, Liberty/SDC, 3D stack (not stored yet) |
| `cell_library` | the cells are instances of a real cell library (LEF macros): the result can go back to DEF |
| `tracks` | routing tracks per layer and the routing layer of every macro pin (macro pins go on their tracks) |
| `slow_loop` | the design comes from an ORFS stage, so ORFS can take the result |
| `placement` | positions of all movable cells (solution state) |

Each *capability* (an engine or a metric) requires a set of facts:

| capability | requires |
| --- | --- |
| `hpwl` | netlist, cell_geometry, pin_offsets, placement |
| `die_size` | cell_geometry |
| `io_place` | netlist, floorplan.die |
| `macro_place` | ... + floorplan.io |
| `global_place` | ... + floorplan.io, floorplan.macros |
| `legalize` | ... + rows, placement |
| `export` | netlist, cell_library |
| `orfs_floorplan_handoff` | cell_library, slow_loop, floorplan.io, floorplan.macros |
| `orfs_handoff` | ... + placement |

`d.capabilities()`, `d.missing('global_place')` and `d.require('legalize')` answer the questions; `d.status()`
prints all of it. Calling an engine without its facts raises `MissingFacts` with a hint on how to get them.

**Missing facts are never filled in silently.** A Bookshelf benchmark has no cell library, so it does not export
to DEF (`force=True` writes one anyway, with a warning).
A soft-block floorplan has no cell geometry, because the block shapes are the question, not a fact. Deriving a
missing fact will be an explicit operation (a future `attar.derive`) that records its provenance as
`derived:<method>`.

Engines add facts with their own provenance. After `d.floorplan()` on an ORFS 2_1 design:

```text
floorplan.die     def
floorplan.io      naive:place_io(nets)
floorplan.macros  naive:place_macros
```

## Tiers

What happens to a result depends on where the design came from:

- **A**: can be validated in the slow loop (it comes from an ORFS stage and the floorplan is complete).
- **B**: has a cell library, so the result can be stamped onto a DEF, but there is no slow loop for it.
- **C**: Attar is the end point (Bookshelf, YAL, GSRC: no cell library to hand off to).

## Self-contained designs and export

A distilled design holds everything the fast loop reads, macro pin layers and routing tracks included, so a copy of
its folder works on any machine. Manifests name each source file with its sha256, never a path, and Attar reads no
environment variables.

A DEF comes back through `d.export(out_def)`. A LEF/DEF design keeps its source DEF's *frame* (`frame.def.gz`,
fact `def_frame`): the source minus what Attar writes itself. The frame is the problem around the answer and is
kept verbatim: header, die, rows, tracks, IO pins with their shapes, the PDN (`SPECIALNETS`, `VIAS`), blockages,
regions, the fixed components (macros, tap cells) and the power entries of `NETS`. Attar writes the rest from the
IR: every component it may move (`PLACED`, `FIXED` when an engine fixed it, `UNPLACED` otherwise) and every signal
net (connectivity; wiring of the source is dropped, as it is stale once cells move). What an engine changed of
the floorplan (IO pins, die, rows) is stamped onto the frame first. At distillation the frame and the IR are put
back together and compared with the source; the fact is only claimed when that gives the same design.

| in the frame (verbatim) | written from the IR |
| --- | --- |
| header, `DIEAREA`, `ROW`, `TRACKS`, `VIAS`, `PINS`, `SPECIALNETS`, `BLOCKAGES`, `REGIONS`, `GROUPS`, fixed components, power nets | components Attar may move, signal nets |

So nets can change later (buffering, routing) without a template. To stamp onto a DEF of your own instead,
`d.export(out_def, template=def_path, force=False)` copies that DEF with the answers stamped on. Before writing, it
checks:

1. `cell_library`: the design is made of LEF masters (not Bookshelf, YAL, GSRC);
2. units: the template's DBU per micron equals the design's;
3. netlist: the template's fingerprint (components and masters, IO pins, signal nets) equals the one recorded at
   distillation;
4. die: the template's die equals the design's, unless an engine made the die.

A mismatch is refused with what differs; `force=True` writes anyway, with a warning.

## Design

`attar.load(name_or_path, device)` returns a `Design`. Everything is a tensor on `device`:

| attribute | shape | meaning |
| --- | --- | --- |
| `pos` | (N, 2) | lower-left corners, microns (assigning it makes the placement a fact) |
| `size`, `orient` | (N, 2), (N,) | effective size and orientation (DEF order N, S, W, E, FN, FS, FW, FE) |
| `flags` | (N,) | FIXED, MACRO, PLACED, TERMINAL, PHYSICAL, ASIDE bits |
| `movable`, `placed` | (N,) | masks |
| `p_cell`, `p_net`, `pin_off` | (P,), (P,), (P, 2) | pins: owner cell, net, offset from the cell center |
| `pin_dir` | (P,) | seen from the net: 0 sink, 1 driver, 2 both, 3 not stated (an INPUT port drives its net) |
| `n_ptr`, `n_weight` | (E+1,), (E,) | nets as CSR over pins |
| `die_box`, `core_box` | (D, 4) | per die |
| `row_xy`, `row_h`, `row_site_w`, `row_sites`, `row_orient` | (R, ...) | rows |
| `io`, `macros` | indices | IO pins (zero-size terminals) and macros |

Snapshots save and restore solution state: `d.snapshot('mine')`, `d.restore('reference')`. Designs distilled from
a placed source keep that placement as the `reference` snapshot and start unplaced, so engines start from scratch.

`d.unplace()` forgets the movable positions and sets aside movable cells without connections (fill and spacer
cells: the `ASIDE` flag). Engines don't place set-aside cells, they don't count as obstacles or utilization, and
export writes them as `UNPLACED`.

## Evaluator

```python
ev = attar.Evaluator(d, fast=True)       # fast: CUDA-graph HPWL
ev.hpwl()                                 # exact pin-to-pin HPWL (0-d tensor, no host sync)
ev.hpwl_batch(P)                          # (B, N, 2) placements at once
ev.legality()                             # rows, sites, overlaps, row gaps, power-rail parity
ev.evaluate(metrics=('hpwl', 'steiner_wirelength', 'rudy', 'density_overflow', 'legality'))
```

Every metric takes explicit positions too, so it works inside an optimizer loop without touching the design.

## attar.naive

The basic set of engines: simple, tensorized and fast, each one a starting point to replace.

| engine | method |
| --- | --- |
| `make_die` | core area = cell area / utilization (+ macros), on the rows' site grid, rows regenerated |
| `place_io` | die boundary as one line; targets from net centroids (or name order); parallel isotonic packing |
| `place_macros` | mixed-size electrostatic GP, then macro legalization on the *macro grid* (positions that put macro pins on routing tracks): sweep line from the center out, or parallel pairwise push, keeping the better; leftovers to the nearest free grid spot |
| `global_place` | ePlace/DREAMPlace electrostatics: WA wirelength, DCT Poisson solve, Nesterov with backtracking, CUDA graphs; rolls back to the best solution if it diverges |
| `legalize` | multi-row-height, power-rail aware: bands of rows per (height, rails) class, parallel isotonic packing, a fill step, and parallel displacement improvement |
| `floorplan`, `place` | `make_die` + `place_io` + `place_macros`; `global_place` + `legalize` |
