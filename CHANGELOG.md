# Changelog

## 1.0.3

**Upgrading.** Two new suites, `after_fp/iccad2015` and `before_fp/orfs_nangate45`. The routing fact of the ORFS,
NS3K and ICCAD 2025 designs now also records what global routing needs: Nangate45's per-layer capacity adjustments
and the cut layers' LEF resistance. Designs downloaded before 1.0.3 lack them, so on Nangate45 `d.global_route()`
and global-routing timing hold back too much capacity on metal4 and up and count vias as 0 ohm: fetch them again,
`attar.fetch('after_fp/orfs_nangate45', force=True)` (any suite, mode or design). Their arrays are unchanged.

New suites
- `after_fp/iccad2015`: the 8 ICCAD 2015 superblue designs (0.77M-1.93M cells) at the contest placement, with
  timing (shared `iccad2015_late` library bundle, contest wire RC); against OpenSTA: WNS within 0.33%, TNS within
  0.13%, hold MAE at most 0.30 ps. Late library only; no sizing classes (the contest Liberty has no functions).
- `before_fp/orfs_nangate45`: ORFS's macro-placement designs (ariane133/136, black_parrot, bp_be/fe/multi_top,
  swerv_wrapper, mempool_group) at ORFS 2_1, with timing. Each records ORFS's minimum macro channel
  (`macro_channel`), `d.floorplan()`'s default halo; handed back to ORFS, 15 of 16 Attar floorplans pass PDN and
  detailed placement (docs/validation/fp_handoff.md).

Placement
- `naive.legalize`: incremental mode, as OpenROAD's detailed placer with its placed cells kept: the cells legal where
  they stand keep their place, the others (and `cells=`, e.g. the ones an optimizer moved or resized) go to the
  nearest free spot their rails allow. A legal input comes out unchanged (DREAMPlace's adaptec1: was +12% HPWL,
  now 0). `incremental=None` chooses; the global mode is bitwise unchanged.
- `naive.detailed_place`: OpenROAD dpo's global swap, vertical swap, local reordering and cell flipping, batched;
  every move keeps the placement legal (rows, sites, rails, row gaps, PDN). 37-92% of OpenROAD `improve_placement`'s
  HPWL gain, 2-20x faster.
- `Evaluator.displacement(pos, ref)`: total / mean / max displacement as OpenROAD dpl reports it;
  `Evaluator.legality(per_cell=True)`: the illegal cells as a mask.
- `d.floorplan_check()`: macros that only touch no longer count as overlapping on large dies (an overlap must exceed
  two float32 steps at the die's size).

Global routing
- `d.global_route()` (`attar.route.global_route`): a GPU global router on OpenROAD's gcell grid and capacities
  (Prim-Dijkstra topology, negotiated Z / L pattern routing, chunked layer assignment). Against OpenROAD's
  `global_route` on 14 designs: wirelength 0.6-2.3% below, 16-41x faster from jpeg up, some overflow left where
  OpenROAD has none; across 7 placements of 7 designs it orders placements as OpenROAD's wirelength does in every pair
  1% or more apart (HPWL 74-88%) (docs/validation/global_routing.md).
- Global-routing parasitics: `d.timing(routes=...)` / `Timer(d, parasitics='global_routing', routes=...)`
  (`attar.timing.grpar`), OpenROAD's `estimate_parasitics -global_routing` ported (networks from the routes, OpenSTA's
  pi-Elmore reduction). On OpenROAD's own routes: every net's wire R and C equal OpenROAD's SPEF, endpoint slacks
  equal OpenSTA's (14 ORFS designs).
- `attar.route`: global routes as data (`GlobalRoutes`; `read_segments` reads OpenROAD's
  `write_global_route_segments`, `pins_from_shapes` is grt's pin model), wirelength as grt reports it.
- Routing fact: per-layer adjustments where the flow sets them (Nangate45's fastroute.tcl: metal2-3 0.5, metal4 and
  up 0.25; `routing.layer_adjustments`) and the cut layers' LEF resistance (`routing.cut_res`, what OpenROAD uses
  for vias without `set_layer_rc -via`). `d.global_route()`, `d.gr_overflow()` and GR timing read them from the
  design; `capacity(adjustment={layer: share})` still overrides.
- `attar.timing.steiner_big.pd_edges`: the edges of OpenROAD's Prim-Dijkstra tree for nets of any size.

Timing speed (results unchanged, bitwise)
- Timer build: SDC `set_load` ports are looked up per pin instead of one mask per port over all pins
  (superblue7: 97 s -> 10 s, superblue18: 20 s -> 4 s, on the CPU and the GPU alike).
- CPU timing: the delay calculation's Newton and root-finding loops stop once every element has converged or
  failed and step only the live ones (`attar.timing.dmp.HOST_EARLY_EXIT`; the GPU keeps its fixed-count,
  graph-capturable loops), and every net with a driver gets its exact host tree instead of the batched PyTorch
  buckets. ORFS aes 59 s -> 4.4 s, nangate45 jpeg 149 s -> 8.9 s, superblue18 over an hour -> 82 s per run.

- `d.resize()` updates the timer in place (`Timer.resized`): the graph, its levels and the CUDA graph stay; each
  instance arc points to its new master's Liberty arc and the pins take the new master pins' data. Only a master the
  design had no tables for yet (or arcs that do not line up) builds a new timer. Results equal a new timer's
  (bitwise); jpeg_x20, 10.5k cells resized: resize + timing 3.3 s, was 9.6 s.

Readers
- DEF: component and pin locations written as decimals (`7214680.000000`) are read; a fractional database unit is
  refused instead of the component silently reading as unplaced.

## 1.0.2

**Upgrading.** The zoo's cell libraries now carry design rule limits and power data, and the ORFS designs their SDC
limits. Designs downloaded by attar 1.0.0 / 1.0.1 stay as they were (`d.power()` asks for an update; `d.drv()` finds
no slew limits): `attar.fetch('after_fp/orfs_asap7', force=True)` (any suite, mode or design) downloads them again
with their libraries. Attar 1.0.0 / 1.0.1 read the new zoo as before.

- `attar.fetch(..., force=True)`: download again what is already in the local zoo.
- `d.drv()`: max slew / capacitance / fanout checks, as OpenSTA's `report_check_types`; SDC `set_max_transition`,
  `set_max_capacitance`, `set_max_fanout` are read now.
- `d.power()`: OpenROAD's `report_power` on the GPU (activity propagation, internal, switching and leakage power).
- Propagated clocks: `set_propagated_clock` in the SDC, or `d.propagate_clocks()`; clock RC on non-leaf clock nets.
- `attar.naive.cts`: buffered, balanced clock trees: OpenROAD's clock buffer inference (no hold buffers), repeaters
  at OpenROAD's max wire length, TritonCTS's sink clustering diameter, latency padding across clock gates, level by
  level balancing on the timer. On ORFS's asap7 gcd / aes / riscv32i / jpeg: a smaller clock spread than TritonCTS,
  WNS / TNS as good or better on three, 2.5-4x less clock power.
- Timer: with propagated clocks, a cycle through a clock gating latch is broken at the latch D -> Q arc (as OpenSTA's
  levelization), and arrivals entering a clock network from data pins (a gate's enable) are not clock arrivals.
- CRPR: with propagated clocks, register checks remove clock reconvergence pessimism as OpenSTA does by default
  (`attar.timing.crpr`; exact, every launch clock pin that can still be worst is kept); `d.timing(crpr=False)` for
  `sta::set_crpr_enabled 0`.
- Timer: slews from every arc of OpenSTA's delay calculation (constant-disabled and unclocked arcs too), ideal
  clock networks timed for their slews; agreement with OpenSTA improves (cva6 TNS error 0.96% -> 0.03%).
- Library bundles carry design rule limits and power data (rebuild with tools/distill_library.py); Liberty reader
  fixes (an attribute without ';' before a group, line continuations inside strings).

## 1.0.1

- `examples/timing_driven_colab.ipynb`: the README demo as a Colab notebook (Open in Colab from the README).
- `d.plot()` no longer switches matplotlib to Agg inside Jupyter or Colab: 1.0.0 did, which hid the notebook's own
  plots from the first `d.plot()` on.

## 1.0.0

The first release on PyPI (`pip install attar`).

**Data.** 246 designs in 22 suites, distilled into the Attar IR and published as the
[Attar zoo](https://huggingface.co/datasets/RayChang987/attar-zoo): ISPD 2005/2015/2018/2019/2024, ICCAD 2025, GSRC,
MCNC, IBM-HB+, OpenROAD-flow-scripts designs on ASAP7 and Nangate45, and open designs on the i3dPDK_NS3K kit.
`attar.load(name)` downloads a design on first use (zstd-compressed on the Hub, 731 MB in all), with the shared cell
library it references. Readers for LEF/DEF, Bookshelf, YAL and GSRC blocks (`attar.distill` for your own designs).

**Evaluation.** Exact HPWL and differentiable weighted-average wirelength, density, legality (rows, sites, power rails,
PDN blockages; as OpenROAD's `check_placement`), RUDY and global-routing overflow on OpenROAD's gcell capacities,
and static timing on the GPU (OpenSTA's delay calculation and OpenROAD's placement parasitics): `d.timing()`,
`d.timing_report()`, pin and net slack.

**Engines.** `attar.naive`: die sizing, IO pin and macro placement, B*-tree packing, ePlace-style global placement,
multi-row-height legalization; netlist edits (`insert_buffer`, `clone`, `debuffer`) and gate sizing (`d.resize`).
`d.export(path)` writes a complete DEF for your own flow.

**Layout of the package.** Raw formats live in `attar.io`, timing in `attar.timing` (the timing fact is
`attar.timing.ir`), physical rules and routing resources in `attar.rules`. The top-level API (`attar.load`, `zoo`,
`fetch`, `distill`, `Design`, `Evaluator`, `attar.naive`) is what 1.x keeps stable.
