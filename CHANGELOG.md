# Changelog

## Unreleased

New
- Incremental re-timing on the GPU (`attar.timing.incremental`): after cells move, `d.timing()` re-times only the
  cone of the nets whose parasitics changed, as OpenSTA's incremental search: the drivers of those nets, then only
  the arcs whose input arrival or slew changed, level by level, until nothing changes. Each recomputed pin takes
  every arc into it with the same kernels as a full run, so the result is bit for bit a full run's (tested on
  ASAP7, Nangate45, NS3K, ICCAD 2015 and 2025 designs). The timer keeps the average time of both kinds of run and
  uses the faster one. Moving 1 / 10 / 100 cells of jpeg_x20 (1.05M cells, RTX 4090): 54-67 / 78-82 / 92 ms
  instead of 490 ms; superblue18 27-51 / 65-78 / 143-169 ms instead of 292 ms. Small designs, the CPU, latch time
  borrowing, pin clock latencies, global-routing parasitics, a resize and changes on a propagated clock network
  run in full as before.
- The timer follows netlist edits in place (`attar.timing.inplace`), as OpenSTA updates its graph instead of
  building it again: after `d.edit()` (`insert_buffer`, `clone`, `remove_cell`, `move`, ...) or `d.debuffer()`, the
  pin-indexed arrays follow the new pin numbers, the new cells' arcs join the graph, levels and stages are made
  again, and the last run's arrivals are kept, so the next `d.timing()` re-times only the cone of the edited nets.
  The result is a new timer's bit for bit (tested on ASAP7, NS3K, ICCAD 2015 and 2025 designs, up to jpeg_x20).
  One buffer inserted on jpeg_x20: 1.3-1.7 s for the edit and 65-70 ms for the timing, instead of a 9-10 s
  rebuild and a 2.3 s first run. Edits on a clock network, a constant net, a net an input port drives, latches, pin clock
  latencies, combinational loops or global-routing parasitics still rebuild the timer (`d._timer_edit` says why).
- Building a timer: its stages, levels and reachability are made on the device (jpeg_x20: 5-6 s instead of 9-10 s).
- The incremental re-timing sweeps small cones on the CPU (`attar/timing/csrc/inc.cpp`): a worklist of driver pins
  in level order, OpenSTA's search order, with a C++ twin of the GPU kernels that gives the same bits; the timer's
  state is mirrored on the host and the recomputed rows are copied back. Cones over 10,000 arcs (or 5% of the
  design) go to the GPU sweep as before. jpeg_x20, one cell moved: the sweep takes 4-6 ms instead of 21-34 ms, the
  whole `d.timing()` about 25 ms instead of 55-65 ms.
- Netlist edits no longer rebuild the timer's levels (`attar.timing.inplace.update_light`, `attar.timing.cpu_inc`):
  on a timed GPU timer the CPU engine follows the edit under its own pin handles (the design renumbers its pins;
  the engine only remaps its handle maps), adds the new cells' arcs and the edited nets' loads, and gives each new
  driver a level key between its fanin and its readers, as OpenSTA updates its graph. The levels and stages are
  dropped and rebuilt only when something reads them (required times, `graph()`, `stages()`) or a run is too
  large for the CPU (over a fifth of the arcs); once the levels are read between edits, later edits keep them up
  to date instead. jpeg_x20, one buffer inserted: the edit takes 130-150 ms and the next `d.timing()` 27-30 ms
  (was 1.3-1.7 s and 65-70 ms; 9-10 s and 2.3 s before in-place edits); the first edit after a full run also
  builds the engine (about 2 s). Every result is a new timer's bit for bit (tested on aes, NS3K aes, jpeg_x20:
  buffers, clones, removed buffers, cells moved between edits). Propagated clocks (CRPR) still take the full
  in-place update.
- The CPU and the GPU timer give the same bits: the C++ delay calculation keeps the driver waveform in float64
  between an arc and its loads, as the GPU kernels do, and the PyTorch reference, the C++ and the Triton kernels
  agree bit for bit (tested).

Fixes
- The driver loads of a timing run (`arrivals().load`, read by `d.drv()` and `d.power()`) are summed in a fixed
  order on the GPU; CUDA's atomic adds made their last bits differ from run to run. They are summed when read.
- `Design.resize` kept the incremental re-timing's copies of the old Liberty arcs: after the full run that follows
  a resize, a cone re-timing used the old cells' tables. They are made again.
- Netlist edits on large designs: `Edit.apply` moves the pin arrays by slices (the stable sort by net moves blocks)
  and updates the manifest's pin direction counts from what changed; timing.json is read once per design.
- The GPU delay kernels in float64 used float32 values of OpenSTA's constants (1.4, 0.01, 1e-3, ...: Triton reads a
  float literal as float32) and float32 differences of table axis values, and contracted multiply-adds into FMAs;
  they now use the double constants and separate operations, as OpenSTA on x86. GPU arrivals change in the last
  bits (under 1e-4 ps on aes).

## 1.0.9

**Upgrading.** Nothing to do: results are the same as 1.0.8's and the zoo is unchanged.

New
- `TimingResult.stages()`: the level schedule of the last timing run, with the delays the timer computed. Stages
  come in run order: the ideal clocks' launch arcs, then one stage per level. Each holds its cell arcs (Liberty arc,
  pins, transitions, whether searched or a latch D -> Q arc), the wire arcs leaving them, and the gate and wire
  delays of both analyses. The arrivals of `arrivals(per_tag=True)` follow from them exactly (tested on CPU and
  GPU). With it, the LATTE notebook reads no private timer fields.

Fixes
- The GPU timer's first-compile notice asks Triton where its kernel cache is, so it honours `TRITON_CACHE_DIR`.
  Before, it checked Triton's default folder only and printed the notice on every run with the variable set,
  though the kernels were cached. (Attar itself still reads no environment variable.)
- Tests: `test_examples` and `test_paths` read UTF-8 explicitly, so the suite passes under a non-UTF-8 locale.

Docs
- The docs are about half as long: one line per call in api.md, no tables repeated across pages, and a
  summary table of every validation in docs/validation/README.md. Earlier CHANGELOG entries are condensed.

## 1.0.8

**Upgrading.** Refresh the zoo designs downloaded by an older attar in five suites:
`attar.fetch('after_fp/ispd2015', 'after_fp/ispd2018', 'after_fp/ispd2019', 'after_fp/iccad2025', 'after_fp/orfs_asap7',
'before_fp/orfs_asap7', force=True)`. Their layer widths were misread (fixes below), and ISPD 2018/2019 now route
from metal1. Without the refresh, `pin_access` leaves out the misread layers with a warning, and placement behaves
as in 1.0.7.

On NS3K designs, legalization and detailed placement now keep signal pins reachable past the PDN. The placements
differ (HPWL +0.01-0.24%), and `pin_access=False` gives 1.0.7's. ASAP7 and Nangate45 placements are bit for bit
1.0.7's. Global routing changes on ISPD 2018/2019 only (metal1). Timing is unchanged.

New
- `Evaluator.pin_access` (metric `'pin_access'`, facts `pdn` and `tracks`): the signal pins the router cannot reach
  past the PDN. Each track of the layer above that crosses a pin is an access point; the via lands a square of the
  layer width on both layers; a pin is blocked when every landing comes closer than the layer spacing to PG metal
  (Euclidean); pins below the lowest signal routing layer only (a pin on a routing layer can be left along it).
  Geometry comes from the IR alone (layer directions, widths, spacings, every DEF TRACKS pattern), so
  it runs on any PDK: 0.2 s on 62k cells, 2.6 s on bp_quad (1.18M cells, GPU). On 988 routed Innovus runs of
  i3dPDK_NS3K (pa_pdn) the blocked pins are exactly the pins with a signal-to-PG M1/M2 DRV (156,665 of 156,665,
  no others). OpenROAD dpl's PDN rule (`legality()['pdn_blocked']`, `plot(tiles='pin_access')`) is unchanged;
  on those runs it flags 66 cells, 23 of them failing: a single-access M1 pin one M1 spacing from a PG via is
  legal, but its M2 landing is 0.040 um from the via's M2 pad (M2 spacing 0.048). Signal-to-signal DRVs (58% of
  the M2 markers there) are not modelled.
- `tools/validate_pin_access.py` reproduces that check from the pa_pdn archive; the tables are in
  `docs/validation/pin_access.md`.
- `plot(tiles='unreachable_pins')`: those pins per tile. The README gallery shows them on ariane136 legalized with
  dpl's PDN rule only (1,254 pins); the roadmap marks blocked pin checking done.

Changed
- `naive.legalize` and `naive.detailed_place` keep signal pins reachable (`pin_access=True`, the default, with facts
  `pdn` and `tracks`): the PDN-aware row DP counts a pin the router cannot reach like a pin on a blockage, the
  in-place mirroring of multi-row cells clears them too, and the detailed placer rejects moves that make one
  unreachable.
  NS3K u80, from the same global placement: unreachable pins 66 -> 0 (aes), 603 -> 0 (jpeg), 1,235 -> 0
  (ariane136), 4,360 -> 4 (bp_quad); HPWL +0.01-0.24%; the PDN step of legalization +0.2-0.6 s (bp_quad +2 s),
  detailed placement +1.5 s on 60k cells (bp_quad +4 s). Where no pin can be blocked (no pin layer below the lowest
  routing layer), nothing is checked and nothing costs more.
  ASAP7 and Nangate45: the same placements as before (no pin is unreachable there; a strip test skips every
  pin), at most +0.4 s. Legality is still OpenROAD dpl's; `pin_access=False` gives 1.0.7's placements.

Fixes
- LEF reader: a layer's width was taken from the last line starting with WIDTH, which in a SPACINGTABLE is a table
  row (ISPD 2015/2018/2019: 4.1 um for a 0.1 um wire; ASAP7: 1-12 um for M8, M9, Pad). Only a one-value WIDTH
  statement counts now. The width went into the tech profile (`layer_width`), used by DEF export and now by
  `pin_access`; the zoo's 73 IRs concerned (ispd2015, ispd2018, ispd2019, iccad2025, orfs_asap7) were rewritten
  from their LEFs. `pin_access` leaves out pins on layers whose width exceeds the pitch, with a warning, so an IR
  distilled before this fix does not stall the detailed placer.
- Zoo: the ISPD 2018 and 2019 designs route from metal1 (the contests route on every metal layer; their guides use
  metal1), not from metal2 (a default). Their pins are then never unreachable for `pin_access` (ispd19_test4 had
  28k at +0.86% HPWL). Global routing there: wirelength within 0.1%, half the vias on ispd19_test4, overflow
  slightly lower.

## 1.0.7

**Upgrading.** Nothing to do. Two fixes found while reproducing SROA (ICCAD 2025) on Attar: detailed placement no
longer runs out of memory on designs with long PG stripes, and incremental legalization no longer takes many
minutes when a few cells are far from free space. Results are the same as 1.0.6's except where noted below (the
legalizer on designs with PDN pin blockages, and incremental legalization where windows used to grow). Timing,
routing and the zoo are unchanged.

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
- `naive.legalize(incremental=True)` took 1,250 s (RTX 4090) to put back the 2,038 cells ABCDPlace leaves outside
  the rows of ispd19_test8; now 15 s (RTX 3080 Ti). When any cell saw no free spot in its window, every cell's
  window doubled, so the few cells far from free space made all the others search hundreds of rows by thousands
  of sites, a few cells at a time (544 rounds, as they compete for the same spots). Now each cell keeps its own
  window; a single-row cell that sees no free spot in its first window looks at every row at once, from the free
  runs of each row (exact: the same distance as a dense search over the whole chip); a taller cell doubles its own
  window. Placements that only need nearby spots are unchanged in speed (ispd19_test8 with 1 % of the cells moved
  up to 100 um: 0.4 -> 0.3 s); where the window used to grow, the cells can land on different spots
  (test4 repair: 6-9 cells differ, HPWL within 0.001 %). Full legalization does not use this step unless a cell
  does not fit its class (ispd19 test1 / test8, ASAP7 aes, NS3K ariane133: bit for bit).

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

**Upgrading.** `naive.legalize` uses Abacus by default (needs g++; `method='columns'` gives 1.0.4's legalizer), so
placements differ, mostly with less displacement and HPWL. For hold on Early delays, fetch the ICCAD 2015 designs
again: `attar.fetch('after_fp/iccad2015', force=True)`.

- Abacus legalization in C++: cells left on macros go to the nearest free space (superblue18: HPWL +10.13% ->
  +2.74%, max displacement 3062 -> 317 um).
- Net weights changed during global placement reach the placer (the 1.0.4 callback had no effect);
  `precond='net_weight'`; `callback(it, pos, info)`; `global_place(deterministic=True)`; `d.hpwl(weighted=False)`.
- Early libraries for the min analysis (`library_min`), as OpenSTA `read_liberty -min`: superblue hold MAE
  0.11-0.21 ps (52-78 ps before).
- The timer's data for research code: `graph()`, `library()`, `arrivals()`, `endpoint_required()`,
  `parasitics()`, `trees()`, with `TIMING_DATA_VERSION`.
- Pin clock latencies on ideal clocks modeled; `clock_arrivals()` fixed on ideal clocks; `limitations` and
  `summary()['exact']`; constraints editable in memory (`timing_settings`, `set_timing`, `ideal_clocks`).
- `d.pin_names`, `pin_index`, `cell_index`; `d.read_placement`, `d.set_positions`.
- Four Colab notebooks in `examples/colab/`.
- Global router: IndexError on grids wider than tall fixed.

## 1.0.4

**Upgrading.** The zoo's timing data changed (ICCAD 2025 distilled again, library bundles rebuilt): fetch the
suites you time again, e.g. `attar.fetch('after_fp/iccad2025', force=True)`.

- Timing equals the ICCAD 2025 contest evaluation on all 13 designs (TNS / WNS within 0.01%, power within 0.06%):
  tie cells, `set_input_transition -min`, loop breaking, latches, wire-delay merging and recovery / removal checks
  ported from OpenSTA.
- `TimingResult.critical_paths()`: the worst path into many endpoints at once; `path()` 20-100x faster.
- C++ helpers use PyTorch's thread count (CPU timing up to 24x faster); `d.power()` 10-20x faster with less memory.
- `naive.global_place(extra=..., callback=...)`: hooks for research objectives.

## 1.0.3

**Upgrading.** New suites `after_fp/iccad2015` and `before_fp/orfs_nangate45`. The routing fact gained Nangate45's
layer adjustments and via resistances: fetch older designs again (`force=True`) for global routing on Nangate45.

- `d.global_route()`: a GPU global router on OpenROAD's grid, 16-41x faster; `d.timing(routes=...)`:
  global-routing parasitics equal to OpenROAD's SPEF.
- `naive.legalize` incremental mode; `naive.detailed_place` (OpenROAD dpo's moves, batched);
  `Evaluator.displacement`, `legality(per_cell=True)`.
- CPU timing in C++ (superblue18: over an hour -> 12-26 s per run); `d.resize()` updates the timer in place.
- DEF reader: decimal coordinates.

## 1.0.2

**Upgrading.** Libraries now carry design-rule limits and power data: `attar.fetch(<suite>, force=True)`.

- `d.drv()`, `d.power()`, propagated clocks with CRPR, `naive.cts`, `attar.fetch(force=True)`.

## 1.0.1

- The README demo as a Colab notebook; `d.plot()` no longer hides a notebook's own plots.

## 1.0.0

The first release on PyPI (`pip install attar`): 246 designs in 22 suites on the
[Attar zoo](https://huggingface.co/datasets/RayChang987/attar-zoo); readers for LEF/DEF, Bookshelf, YAL and GSRC
blocks; exact HPWL, legality, RUDY, global-routing overflow and GPU static timing; `attar.naive` engines; netlist
edits, gate sizing and DEF export. The top-level API (`attar.load`, `zoo`, `fetch`, `distill`, `Design`,
`Evaluator`, `attar.naive`) is what 1.x keeps stable.
