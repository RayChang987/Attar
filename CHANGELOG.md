# Changelog

## Unreleased

Fixes
- `naive.detailed_place` ran out of memory on designs with long PG stripes over the pin layers (ispd19_test4: OOM
  on a 24 GB GPU and at 98 GB on the CPU); it now takes 3.6 s and 1.2 GB there (GPU; CPU 7 s, 1.7 GB). Two causes:
  - the PDN pin-blockage index (`attar.rules.pdn.Index`) sized its bins by the largest blockage, so a stripe a whole
    chip high made every bin the whole chip and every query a scan of all the blockages. Bins are now small and a
    blockage is listed in every bin it covers; queries run in slices of bounded memory;
  - global and vertical swap checked the PDN for every candidate move; now only for the moves that gain.
  Results are unchanged where it ran before (test1, iccad2025 aes: bit for bit).
- The same index missed blockages under the middle of a query box larger than two bins. Pin boxes were never that
  large, but the legalizer's "cells near the PDN" filter (a cell grown by its shift range) was: on NS3K aes it found
  2,294 of 5,533 such cells, and the cells it missed were left to the final PDN repair. The legalizer's output can
  therefore change slightly on designs with pin blockages (NS3K aes: 6 cells; still legal, same HPWL).

## 1.0.6

**Upgrading.** Nothing to do: results are the same as 1.0.5's (timing bit for bit), only faster when a design is
timed or routed again after a change. `d.global_route(previous=routes)` is new and opt-in. The zoo is unchanged.

Errors
- A CUDA out-of-memory error in `d.timing`, `d.place`, `d.global_route`, `d.drv`, `d.power`, `naive.global_place`,
  `naive.legalize` or `naive.detailed_place` says which step ran out on which design, how much of the GPU was free and
  held by other processes, and the ways out (free memory, a larger GPU, `device='cpu'`): `attar.gpu.GPUOutOfMemory`,
  still a `torch.cuda.OutOfMemoryError`.

Timing
- Timing with global-route parasitics (`d.timing(routes=...)`) is 9-17x faster on large designs, with bit-identical
  results: netcard_fast (300k cells) 67 -> 3.8 s per call (83 -> 10 s the first time), superblue18 88 -> 8.8 s.
  `attar.timing.grpar.reduce` went over every node once per tree depth (thousands of gcell-long nodes deep), now once
  per node on the breadth-first order's depth runs; `grpar.build` sorts its node keys as packed int64 instead of
  structured records.
- Re-timing recomputes only the parasitics that changed, bit for bit as a new timer would:
  - placement parasitics: `d.timing()` redoes the Steiner trees and their RC of the nets with a pin that moved or
    changed (a resize) since the last call, up to 30 % of the nets (`Timer.INCREMENTAL_WIRES`), else all; the wires
    now run before the CUDA graph's replay. netcard_fast 352 -> 222-251 ms per call, superblue18 590 -> 277-317 ms
    (nothing moved / 0.1-1 % of the cells moved);
  - global-routing parasitics: `d.timing(routes=other)` keeps its timer and rebuilds and reduces only the nets whose
    route or pins changed (`Timer.set_routes`, `grpar.net_signatures`). netcard_fast after 1 % of the cells moved
    and a new route: 11 -> 5 s; superblue18: 19 -> 12 s. A new route changes 30-47 % of the nets' routes there.
  - Timing again on the same routes no longer reduces them again (its cache key was a temporary tensor's address):
    netcard_fast 3.8 -> 0.45 s.

Routing
- Incremental global routing: `d.global_route(previous=routes)` routes again only the nets with a pin in another
  gcell or on another layer since `routes`, against the other nets' routes, which stay as they were through the
  negotiation and the layer assignment (as OpenROAD's `grt -incremental`); more than 30 % of the nets changed
  (`max_changed`), or other parameters: a full route. Not what a full route gives (the other nets do not make room):
  after 1 % of the cells moved, overflow ends a little higher (netcard_fast: 2D 5 against 0, 3D +10 %), timing about
  the same. The unchanged nets keep their segments bit for bit, so `d.timing(routes=new)` then redoes only the
  rerouted nets' parasitics. superblue18: 16-25 s -> 4.6-4.8 s for a new route after 0.1-1 % of the cells moved,
  timing 3.5 s; netcard_fast after 0.1 %: 12 -> 3.6 s (after 1 % about as long as a full route: its new branches
  need as many negotiation rounds). A full route is unchanged (the same segments).
- `d.global_route()` on designs whose tech stops at four routing layers (ICCAD 2015 superblue18: metal1-4)
  failed choosing the IO pins' layers (`export.pin_layers` looked for a horizontal layer above the third only); it
  now takes the highest horizontal layer with a vertical one above.

Examples
- The Colab notebooks are in English and walk through their methods step by step, each piece in its own cell with
  its explanation. `p11_attar_superblue18.ipynb` (the DAC 2026 net-pin weighting) and `date25_attar_superblue18.ipynb`
  (DATE 2025's pin-pair weighting) are independent of each other. Their DATE25 pin pairs are updated at every timing
  update: the code they came from kept the pairs of the first update only, which made DATE25 end at TNS -5.3e6 or,
  more often, -8.2e6 to -9.0e6 on superblue18; updated, it ends near -3.45e6, and P11 near -2.59e6.
  `latte_on_attar.ipynb` splits LATTE into its differentiable timer, the independent sets, the set optimizer and the
  round loop, with a check that the timer equals the exact one at the anchor; its results are unchanged. The README
  lists the notebooks without a runtime column.

Docs
- docs/features.md: the global placer's gap to DREAMPlace in a timing-driven loop is withdrawn. It came from a
  DATE'25 port that, by a bug, kept its first pin pairs; fixed, the port's TNS on the 8 superblue designs is 0.93x
  DATE25-TDP's own (Attar-timed). The wirelength-only gap stays.

## 1.0.5

**Upgrading.** `naive.legalize` (and so `d.place()` and `naive.flow`) uses Abacus by default: placements differ from
1.0.4's, mostly with less displacement and HPWL (`method='columns'` gives the 1.0.4 legalizer). It needs g++, like
the other C++ helpers; without it the 1.0.4 legalizer runs with a warning. The zoo's 8 `after_fp/iccad2015` designs
gained their Early libraries (`library_min`): fetch them again for hold checks on Early delays,
`attar.fetch('after_fp/iccad2015', force=True)`. Without that they time as in 1.0.4 (the result's `limitations` says
so), and 1.0.4 reads the new files unchanged.

Placement
- Net weights changed during global placement reach the placer. The callback documented in 1.0.4 ("update
  d.n_weight in place, the wirelength reads it at the next evaluation") had no effect: the wirelength read a copy of the
  weights made at the start. Now the Evaluator follows `d.n_weight` (in place or a new tensor) and
  `naive.global_place` takes the callback's changes from the next evaluation, CUDA graphs included. On the CPU (same
  thread count) its trajectory equals, bit for bit, the patched copy P11 (attar_DAC2026) had to keep for net-weighted
  timing-driven placement: superblue18, D25Nns, 573 iterations.
- `global_place(..., precond='net_weight')` (default): the preconditioner adds, over a cell's pins, the change of
  their nets' weights since the start (DREAMPlace's sum of pin weights when weights start at 1), so cells on heavily
  weighted nets take proportionally smaller steps; `'pins'` keeps the pin count. A placement whose weights do not
  change is the same as before (bit-identical on the CPU).
- `callback(it, pos, info)`: a callback taking a third argument gets the iteration's `hpwl` (net-weighted, what the
  schedule follows), `overflow`, `lambda` and `gamma`.
- `d.hpwl(weighted=False)`: the plain wire length of a net-weighted placement.
- `naive.legalize` uses Abacus (Spindler et al., ISPD 2008) by default in the global mode (`method='abacus'`), in C++
  (`attar/naive/csrc/abacus.cpp`, built with g++ on first use; without g++ it falls back to `'columns'`, the earlier
  default, with a warning): per height class, cells in order of x go to the band and segment where their own
  displacement is least, segments packed by cluster collapse; rails, fill, improve and the PDN repair as before.
  Cells global placement leaves on fixed macros go to the nearest free space instead of wherever their column has
  room (P11: 1-3 mm moves on superblue18). Against `'columns'` on the same global placements, all legal:

  | design | HPWL after LG (columns -> abacus) | max displacement (um) | other |
  |---|---|---|---|
  | aes | +5.98% -> +4.78% | | mean 0.224 -> 0.192 um, LG 2.6 -> 1.9 s |
  | jpeg | +13.66% -> +10.44% | | LG 13.7 -> 6.9 s |
  | ibex | +9.52% -> +8.14% | | |
  | cva6 | +11.25% -> +7.83% | 46.6 -> 10.4 (cells on macros) | TNS -8.54e4 -> -7.96e4 |
  | ariane | +21.8% -> +9.1% | | WNS -815 -> -347, TNS -6.6e5 -> -2.7e5 |
  | dynamic_node | +33.2% -> +14.75% | 96 -> 5.5 | |
  | ethmac | | 192 -> 3.8 | |
  | mgc_fft_a | +25.8% -> +11.8% | 699 -> 120 | |
  | mgc_des_perf_a | | 551 -> 49 | |
  | mgc_matrix_mult_b | +12.7% -> +8.4% | 783 -> 153 | LG 54 -> 79 s (host under load) |
  | ac97_top | lower | | WNS -294 -> -345 (the one worse result) |
  | superblue18 (745k cells, 653 macros) | +10.13% -> +2.74% | 3062 -> 317 | cells on macros mean 161 -> 16.7 um; WNS -24.1 -> -12.3 ns, TNS -7.82e6 -> -6.92e6; LG 393 -> 189 s |

  `d.place()` and `naive.flow` results change accordingly.
- `naive.global_place(d, deterministic=True)`: on the GPU, the same placement on every run with the same inputs and
  seed (under `torch.use_deterministic_algorithms`: scatter adds sorted, and `attar.kernels.cumsum`, a fixed-order
  scan, for the density map's prefix sums, which CUDA's cumsum adds in any order; cuBLAS products, one stream, are
  reproducible on the same GPU model). Slower: aes 2.9 -> 3.9 s, jpeg 3.8 -> 14 s on an RTX 3080 Ti. Timing feedback
  amplifies the run-to-run noise of the default mode (DATE'25-weighted superblue18 runs end at TNS about -5.3e6 or,
  when they drift at the first weight update, at -8.2e6 to -9.0e6; 6 of 7 runs drifted on an RTX 4090, 1 of 4 on
  another GPU), so compare methods with it. On the CPU a run is reproducible with the same thread count.

Timing
- Early libraries: the min analysis (hold checks, earliest arrivals, the min side of propagated clocks) reads a
  design's own Early Liberty where it has one, as OpenSTA's `read_liberty -min`: manifest `library_min`, the cells in
  `<design>/library_min.safetensors` (`attar.timing.ir.attach_min_library`; additive: older Attar ignores it). The
  zoo's 8 ICCAD 2015 superblue designs have theirs now. The timer stacks the Early tables after the Late ones
  (`TimingResult.library().min_offset`); its min lanes, on the CPU, the GPU's fused stage kernel and the per-op
  path, read Liberty arc a at a + offset. Against OpenSTA with `-max` Late and `-min` Early at the ICCAD 2015
  placements, hold slack MAE per endpoint (Late only before): superblue1 0.20 ps (67.9), superblue3 0.13 (52.1),
  superblue4 0.11 (38.5), superblue5 0.15 (63.7), superblue7 0.16 (54.9), superblue10 0.21 (77.2), superblue16
  0.17 (65.2), superblue18 0.18 (64.3); WHS 8.4 ps as OpenSTA on all 8 (14.9 before); setup unchanged.
  Designs without an Early library time as before (bit-identical setup and hold slacks, CPU and GPU).
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
  DATE 2025 baseline on superblue18. On 1.0.5 they need no patched copies of Attar: the P11 notebooks drop their copy
  of the global placer, their Abacus and the relegalization step (1,930 -> 650 lines; the placement is bit-identical
  to the copy's on the CPU), and LATTE drops its runtime patches and the pre-CTS design copy (`d.ideal_clocks()`;
  the same results on des).

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
