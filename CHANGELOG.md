# Changelog

## Unreleased (branch gp-hooks-timing-api)

Placement
- Net weights changed during global placement reach the placer. The callback documented in 1.0.4 ("update
  d.n_weight in place, the wirelength reads it at the next evaluation") had no effect: the wirelength read a copy of the
  weights made at the start. Now the Evaluator follows `d.n_weight` (in place or a new tensor) and
  `naive.global_place` takes the callback's changes from the next evaluation, CUDA graphs included. On the CPU its
  trajectory equals, bit for bit, the patched copy P11 (attar_DAC2026) had to keep for net-weighted timing-driven
  placement.
- `global_place(..., precond='net_weight')` (default): the preconditioner adds, over a cell's pins, the change of
  their nets' weights since the start (DREAMPlace's sum of pin weights when weights start at 1), so cells on heavily
  weighted nets take proportionally smaller steps; `'pins'` keeps the pin count. A placement whose weights do not
  change is the same as before (bit-identical on the CPU).
- `callback(it, pos, info)`: a callback taking a third argument gets the iteration's `hpwl` (net-weighted, what the
  schedule follows), `overflow`, `lambda` and `gamma`.
- `d.hpwl(weighted=False)`: the plain wire length of a net-weighted placement.

Timing
- The timer's data for research code, read only (copies): `TimingResult.graph()` (levels, net / cell / launch / check
  arcs with their Liberty arcs, types and senses, which arcs the search follows, clock networks), `library()` (the
  NLDM tables per Liberty arc with their master and pins, every pin's rise / fall max / min capacitance, the
  timer's own table lookup), `arrivals()` (latest / earliest arrival and slew per pin and transition, driver loads),
  `parasitics()` (per net the wire capacitance and the driver's pi model, per net arc the Elmore delay, the pin
  locations the parasitics used) and `trees(nets=None)` (the estimate-parasitics Steiner trees, rooted at the
  driver: exactly the trees the timer reduced, their length times the unit capacitance is its wire capacitance).
  Code that read `d._timer` internals (P05's differentiable timer, GRC-TDGR's arc gradients) can use these.
- `attar.timing.steiner_big.pd_edges(..., drv=)`: root each tree at a given point (pd_rc's driver: the same tree).
- More of the timer's data, so differentiable timers need none of its private fields: `graph()` adds the launches
  under ideal clocks (arc, output transition, tag, launch time, clock slew), the launch tags (clock, edge, time), the
  net and cell arcs disabled to break combinational loops, and the clocked latch D -> Q arcs with their enables;
  `arrivals(per_tag=True)` keeps arrivals per launch tag; `endpoint_required()` gives every setup check's required
  time and arrival per tag (its slack before CRPR; per pin their minimum is the setup slack with ideal clocks). Every
  bundle carries `version` (`attar.timing.sta.TIMING_DATA_VERSION`, 1).
- Pin clock latencies on ideal clocks (`set_clock_latency` on register clock pins of a clock that is not
  `set_propagated_clock`: pre-CTS timing of an SDC with useful-skew pin latencies, e.g. the ICCAD 2025 SDCs without
  their `set_propagated_clock`) are modeled, as OpenSTA does: the pin's latency replaces the clock's at launch and at
  capture. Before, they were ignored with a warning. Only pin latencies given for both transitions and both min / max
  (the common form); others are still listed as not modeled. Checked against OpenROAD on the ICCAD 2025 IWLS designs
  (in the LATTE reproduction).
- `TimingResult.clock_arrivals()` gave -inf / inf at the register clock pins of ideal clocks. It now gives what the
  timer uses there: the clock edge plus its latency, and the clock's transition.
- Known gaps are reported: `TimingResult.limitations` lists what makes a timing differ from OpenSTA's on the
  design's constraints (SDC commands not applied, pin clock latencies not modeled, nets with a lumped wire model)
  and a benchmark's Early library not read (ICCAD 2015: hold on Late delays). `summary()['exact']` is False when
  there is one, and `report()` prints them. Before, the first three were a warning only.
- Timing constraints editable in memory: `d.timing_settings()` (a copy of `timing.json`'s constraints),
  `d.set_timing(clocks=..., input_delay=..., ...)`, `d.reset_timing()`, and `d.ideal_clocks()`, the reverse of
  `d.propagate_clocks()`. The files are not touched; the timer is rebuilt on the next `timing()`.
- `report()` names the parasitics (it said "placement estimate" for global-routing parasitics too).

Design
- Names: `d.pin_names(pins=None)` ('instance/PIN', ports by name, as OpenSTA prints them), `d.pin_index(names)`
  (OpenTimer's 'instance:PIN' too), `d.cell_index(names)`, `TimingResult.pin_names(pins)`. The index spaces (cells,
  pins, nets, masters, master pins, Liberty and timing arcs) are described in docs/api.md.
- Placements from other tools: `d.read_placement(def_path, fixed=False)` takes positions and orientations by
  component name from a DEF (any DBU); `d.set_positions(names, xy, orient=None)` sets them directly.

Examples
- Four Colab notebooks in `examples/colab/`, listed in the README: the timing-driven placement demo (moved from
  `examples/`), LATTE (DAC 2026 timing-driven detailed placement), and the DAC 2026 hybrid net-pin weighting with its
  DATE 2025 baseline on superblue18.

Docs
- docs/benchmarks.md: ICCAD 2015 timing against published (OpenTimer) numbers, and the GPU memory a superblue run
  needs. docs/features.md: the known gaps of the global placer against DREAMPlace and of the legalizer on cells left
  on macros. README: keeping the compiled kernels across Colab runtimes (`TRITON_CACHE_DIR`,
  `TORCHINDUCTOR_CACHE_DIR`), also said by the first-compile notice.

Routing
- The global router failed with an IndexError on gcell grids wider than tall (`after_fp/iccad2025/des3`, 35 x 34):
  pattern routing computes both L / Z costs of every branch and keeps one, and the discarded cost indexed the other
  axis with the kept pattern's bend. Each cost now clamps the bend into its own axis; routes on grids that worked
  before are unchanged (bit-identical on gcd and aes).

## 1.0.4

**Upgrading.** The zoo's timing data changed: `after_fp/iccad2025` was distilled again (signal pins on DEF power /
ground nets are now kept, as constants), the shared library bundles were rebuilt (recovery / removal arcs) and every
design's timing was attached again. Designs downloaded before 1.0.4 still load and time, without those: fetch the
suites you time again, e.g. `attar.fetch('after_fp/iccad2025', force=True)`.

Timing
- Equal to the ICCAD 2025 contest's own evaluation (`openRoad_eval_script/eval_def.tcl`: OpenROAD,
  `estimate_parasitics -placement`, TNS / WNS, `report_power`) on all 13 designs at their given placement: TNS / WNS
  within 0.01%, power within 0.06% (`tools/validate_iccad2025.py`, docs/validation/timing_routing.md). Ported from
  OpenSTA for it: pins tied to VDD / VSS as constants; `set_input_transition -min`; combinational loops broken where
  OpenSTA's levelization breaks them; latches (no D -> Q arrival without an enabling clock, the borrowing rules and
  the D pins' required times otherwise); one wire delay per load, the largest over its driver's arcs; recovery /
  removal checks; escaped port names in SDC patterns. Other suites gain too: ORFS asap7 aes TNS error 0.45% -> 0.01%,
  nangate45 ibex and NS3K aes endpoint MAE 0.5 -> 0.00 ps.
- `TimingResult.critical_paths(endpoints=None, violating=True, max_paths=None)`: the worst path into each of many
  endpoints at once, as `report_checks -path_delay max` picks it (the endpoint's worst transition and launch tag,
  then back through the latest arrivals), from the run's recorded delays: a `CriticalPaths` in CSR form with
  `pin_count()` (paths through each pin) and `edges()` (net and arc steps). 53 of 54 sampled ICCAD 2025 paths are
  OpenSTA's to the pin and transition (the other starts at another register: CRPR is not part of the choice);
  96,506 paths of a 1M-cell design in 0.13 s on the GPU. `path()` uses it: it no longer launches a rising-edge
  flip-flop from its clock's falling edge, and it is 20-100x faster.
- CPU: the C++ helpers use PyTorch's thread count (torch.get_num_threads). With one thread per logical CPU they
  fought PyTorch's own OpenMP threads: CPU timing on iccad2025/fpu 19.4 -> 0.8 s, mempool 8.9 -> 1.6 s.
- Estimated parasitics: pins stacked at one location (an unplaced design) no longer make the Steiner trees quadratic,
  with the same trees (a 95k-pin net: 12.5 -> 1.5 s).

Power
- `d.power()` is 10-20x faster: the register passes of the activity propagation visit only the pins that changed,
  as OpenSTA's breadth-first search, in C++ (also for GPU timers); mempool (184k cells) 74 -> 3.8 s on the CPU,
  10.9 -> 1.0 s on the GPU. The first call's setup is 5-9x faster (mempool 41 -> 4.6 s).
- Less memory: the expression tables stay in host memory, compact; the instance powers are evaluated on the GPU
  when they fit, else on the CPU. 1M-cell designs now fit a GPU with 7 GB free; CPU peak on jpeg_x20 14.7 -> 10.4 GB.
- A resize in place releases the timer's power analysis: it was kept with the old masters' functions and tables.

Placement
- `naive.global_place(..., extra=fn, callback=cb, callback_every=k)`: hooks for research objectives such as
  timing-driven placement. `extra(pos) -> (value, grad)` is added to the objective at every evaluation, with the
  wirelength's preconditioner, outside the CUDA graph (it may time the design); `callback(it, pos)` runs every k
  iterations (time the placement, update `d.n_weight` in place; returning True stops). Nothing changes when unused.

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
- CPU timing: the delay calculation in C++ (`attar.timing.dmp_host`, attar/timing/csrc/dmp.cpp, built with g++ on first
  use like the Steiner trees): OpenSTA's dmp_ceff per element in double precision, each loop stopping when its
  element converges or fails, OpenMP; results equal the PyTorch reference bitwise (ORFS asap7 / nangate45, NS3K,
  ICCAD 2025). Per run on 10 CPU threads: superblue18 12-26 s, jpeg_x20 13 s (was 317 s), superblue7 21-30 s with a
  9.5 s build (OpenROAD, 10 threads: 23 s per re-timing after 15 s of reading). Without a compiler the timer keeps
  the PyTorch path.

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
