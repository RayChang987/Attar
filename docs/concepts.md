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
| `routing`, `timing`, `die_stack` | routing resources, Liberty/SDC, 3D die stack (not stored yet) |
| `cell_library` | the cells are instances of a real cell library (LEF macros): the result can go back to DEF |
| `tracks` | routing tracks per layer and the routing layer of every macro pin (macro pins go on their tracks) |
| `pdn` | PDN metal (wires, vias) on the three lowest routing layers: signal pins keep clear of it |
| `def_frame` | the source DEF minus what Attar writes itself, kept verbatim: what `export` writes into |
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
| `export` | netlist, cell_library, def_frame |
| `orfs_floorplan_handoff` | cell_library, def_frame, slow_loop, floorplan.io, floorplan.macros |
| `orfs_handoff` | ... + placement |

`d.capabilities()`, `d.missing('global_place')` and `d.require('legalize')` answer the questions; `d.status()`
prints all of it. Calling an engine without its facts raises `MissingFacts` with a hint on how to get them.

**Missing facts are never filled in silently.** A Bookshelf benchmark has no cell library, so it does not export
to DEF (`force=True` writes one anyway, with a warning).
A soft-block floorplan has no cell geometry, because the block shapes are the question, not a fact. Deriving a
missing fact will be an explicit operation (a future `attar.derive`) that records its provenance as
`derived:<method>`.

Engines add facts with their own provenance. After `d.floorplan()` on a `before_fp` design from ORFS 2_1:

```text
floorplan.die     def
floorplan.io      naive:place_io(nets)
floorplan.macros  naive:place_macros
```

## Modes and tiers

The zoo has two modes, by what the user solves (zoo names are `<mode>/<suite>/<design>`):

- **`before_fp`, a floorplan problem:** the netlist and cell library are given; the die (utilization), IO pins and
  macros are the answer. Attar draws no power grid: the floorplan is exported and handed to a flow that draws it
  (ORFS: `tools/orfs_handoff.py`, which applies the die, rows, tracks, pins and macros and lets ORFS go on).
- **`after_fp`, a placement problem:** the die, IO pins, macros and power grid are given; the cells are the answer.

What happens to a result depends on where the design came from (the same in both modes):

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

So nets can change later (buffering, routing), and export needs no raw file. Bookshelf, YAL and GSRC designs have
no cell library and no frame, so they do not export.

## The PDN as a placement rule

A LEF/DEF design with special nets also keeps the PDN metal on its three lowest routing layers (fact `pdn`). From it
come the rules OpenROAD's detailed placer checks (`check_placement`, blocked layers): PG via and fill metal that no
PG wire of the same layer covers, grown by the layer's spacing, must stay clear of every signal pin on that layer,
and vertical PG stripes on the second and third layer block the cells with pins there. `Evaluator.legality()`
counts the cells that break them (`pdn_blocked`). The legalizer fixes them last, in three steps:

1. a row-wise dynamic program on the GPU, after the PDN-aware refinement of pa_pdn (Xplace `pinaccess/dp_refine.py`):
   every single-row cell may shift by up to 8 sites and be mirrored left to right (N and FN, FS and S share their
   rails; only where the LEF `SYMMETRY` allows it), at a cost for blocked pins, displacement and mirroring, keeping
   the order of the cells in every row segment; all segments are solved exactly at once;
2. every cell still blocked, multi-row ones included, mirrored where it stands if that clears it (one GPU pass);
3. the few left shifted along their rows, pushing neighbors, or moved to the nearest free spot; every such move is
   checked against a site occupancy map first.
 On i3dPDK_NS3K, whose via stacks from the M0
rails up to M5 cross the M1 pin layer, a few hundred cells per 60k need it, nearly all fixed by mirroring; on ASAP7
the M1-M2 vias sit inside the M1 rails and nothing is blocked.

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
ev.legality()                             # rows, sites, overlaps, row gaps, power-rail parity, PDN
ev.evaluate(metrics=('hpwl', 'steiner_wirelength', 'rudy', 'density_overflow', 'legality'))
```

Every metric takes explicit positions too, so it works inside an optimizer loop without touching the design.

## attar.naive

The basic set of engines: simple, tensorized and fast, each one a starting point to replace.

| engine | method |
| --- | --- |
| `global_place` | ePlace/DREAMPlace electrostatics: WA wirelength, DCT Poisson solve, Nesterov with backtracking, CUDA graphs; rolls back to the best solution if it diverges. Optional: a keep-out band around macros (`macro_halo`), and routability (`routability=True`: DREAMPlace's cell inflation from RUDY over the track capacity the slow loop's router uses) |
| `legalize` | multi-row-height, power-rail aware: bands of rows per (height, rails) class, parallel isotonic packing, a fill step, and parallel displacement improvement; then signal pins clear of the PDN (fact `pdn`, above) |
| `place` | `global_place` + `legalize`, at the source's own target density (ORFS: the density its global placement used) |

For `before_fp` problems:

| engine | method |
| --- | --- |
| `make_die` | core area = cell area / utilization (+ macros), on the rows' site grid, rows regenerated |
| `place_io` | die boundary as one line; targets from net centroids (or name order); parallel isotonic packing |
| `place_macros` | mixed-size electrostatic GP, then macro legalization on the *macro grid* (positions that put macro pins on routing tracks): sweep line from the center out, or parallel pairwise push, keeping the better; leftovers to the nearest free grid spot. With scipy, the pair directions of that result become constraint graphs and an LP places the macros as close to the GP as they allow: never overlapping, inside the core when the graphs can be made to fit, else sticking out by as little as found (`d.plot()` and `d.status()` mark the floorplan legal or not legal, `d.floorplan_legal()`). `packing=True`, for block-level designs (no rows): no GP; the macros are packed by a B*-tree and simulated annealing, first into the core, then for HPWL among packings that fit |
| `floorplan` | `make_die` + `place_io` + `place_macros` |
