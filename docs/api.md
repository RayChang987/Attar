# API

```python
import attar
from attar import naive
```

## Loading

| call | returns |
| --- | --- |
| `attar.zoo(suite=None, remote=False, root=None)` | names `<mode>/<suite>/<design>` of the distilled designs (`'after_fp/ispd2005/adaptec1'`, ...; `suite`: a prefix such as `'before_fp'` or `'after_fp/ns3k'`) in the local zoo, or on the Hugging Face Hub |
| `attar.load(name_or_path, device='cpu', download=True, root=None)` | a `Design`; a zoo name missing locally is downloaded first |
| `attar.fetch(*names, root=None)` | download designs or whole suites (`'ispd2015'`) into the local zoo; everything when no name is given |
| `attar.distill(def_path, lef_paths, out_dir, lib_paths=(), ...)` | distill a LEF/DEF design; returns the manifest |

The zoo is `zoo/` in a source checkout, else `~/.cache/attar/zoo`; `root=` picks another folder. Attar reads no
environment variables. A name that is not in the zoo (nor on the Hub) raises `FileNotFoundError` with the closest
names; an unknown fact, capability, metric or snapshot raises `KeyError` with the known ones.

## Design

| method | what it does |
| --- | --- |
| `status(show=True)` | facts, solution state, capabilities, what is left before export; printed (in color on a terminal) and returned as plain text |
| `summary()` | a few lines: counts, dies with core, rows and utilization, IO pins, facts |
| `capabilities()`, `missing(cap)`, `require(cap)`, `has(fact)` | facts and what follows from them |
| `place(target_density=None, macro_halo=0, **kw)` | `naive.place`: global placement + legalization (default density: the source's own, else 0.9) |
| `floorplan(utilization=None, aspect_ratio=1.0, packing=False, **kw)` | `naive.floorplan`, for `before_fp` problems (see below) |
| `floorplan_legal(pos=None)` | `(legal, problems)`: macros in their region and apart, IO pins in the die, all placed; checked from the positions on every call, not stored |
| `plot(path=None, dpi=150, nets=None, max_nets=None, tiles=None, bins=None, **kw)` | die, core, rows, cells (an area-density image above 30k cells), macros, IO pins by direction; the whole floorplan, also what sticks out of the die (in red); 'legal' / 'not legal' with the problems; block-level designs also show the strongest connections between macros and IO pins (`nets`, default 2 per macro); in the logo's colors. `tiles=` draws a tile map in place of the cells, with a color bar: `'gr_overflow'` (per gcell, the worst RUDY demand / capacity of its edges, as `gr_overflow`; red above 1), `'pin_access'` (the PDN-aware DP's penalty, w_pa per cell with a signal pin on a PDN pin blockage, summed per tile; needs `pdn`), `'density'` (movable area / free area per bin, as global placement sees it; red above 1); `bins=` sets the grid of the last two (default: global placement's, halved until a bin spans two rows, so the rows do not alias into stripes); cream to amber up to capacity, rust to espresso above it |
| `export(out_def)` | write the design as a DEF: its source frame (PDN, pins, fixed cells, verbatim) with the cells and signal nets from the IR |
| `hpwl(pos=None, per_net=False)` | exact HPWL (0-d tensor) |
| `wa_wirelength(pos=None, gamma=1.0)` | weighted-average wirelength, differentiable |
| `pin_pos(pos=None)` | (P, 2) pin locations |
| `utilization()` | per die: standard-cell area / (core - macro area inside the core) |
| `snapshot(name)`, `restore(name)`, `unplace()` | solution state |
| `timing(pos=None, crpr=True)`, `timing_report(pos=None, endpoints=10, path=True)` | static timing (`attar.timing.sta`): a `TimingResult` (`summary()`, `path()`, `report()`, `pin_slack()`, `net_slack()`); the report prints WNS / TNS, the worst endpoints and path; `pin_slack()` / `net_slack()` give every pin's / net's setup slack (inf where unconstrained), as OpenSTA's, from the required times of the run (computed on first use) |
| `gr_overflow(pos=None, adjustment=None, maps=False)` | global-routing overflow (GROV): RUDY demand against OpenROAD's gcell capacities |
| `drv(pos=None)` | electrical design rules of a timing run (`attar.timing.drv`): max slew, capacitance and fanout per pin, as OpenSTA's `report_check_types` (limits from the SDC, the Liberty pins and the libraries' defaults); `summary()`, `report()`, `violators(kind)` |
| `power(pos=None)` | power of a timing run (`attar.timing.power`), as OpenROAD's `report_power`: propagated activities, internal / switching / leakage power per instance (W); `summary()`, `report()` (the report_power groups), `instances()` |
| `propagate_clocks(clocks=None)` | time clocks through their trees (`set_propagated_clock`): clock arrivals and slews at the register clock pins come from the timed clock network; checks use them, less the clock reconvergence pessimism (`attar.timing.crpr`, OpenSTA's default; `timing(crpr=False)` as `sta::set_crpr_enabled 0`; pin and net slacks do not remove it) |
| `equivalents(cell)`, `resize(cells, masters)` | gate sizing within a master's equivalence class (`attar.timing.sizing`) |
| `edit()` | a batch of netlist edits (`attar.netlist.Edit`), applied when its `with` block ends: `insert_buffer(net, loads=None, master=None, xy=None)` and `clone(cell, loads)` return (new cell, new net); `add_cell`, `add_net`, `connect`, `move`, `remove_cell` are the primitives. New cells and nets are appended (no cell or net index moves); pins are renumbered (`e.pin_map[old]`); a TimingResult of before is stale; new cells sit where put (`insert_buffer`: on its loads), legalize them. Clock nets are refused (clocks are ideal) |
| `debuffer(buffers=True, inverters=True)` | remove the netlist's buffers and double inversions (OpenROAD's `remove_buffers`, plus inverter pairs: an even chain goes whole, an odd one keeps one inverter); loads go back onto the net that fed them, which keeps the name of the net with the IO pin, else of the driver's. Not touched: clock nets, fixed cells, joins that would put two IO pins on one net. Removed cells stay as tombstones (flags `ASIDE | REMOVED`, not placed, not exported). Returns counts |
| `buffers()`, `inverters()` | the library's buffer / inverter cells (one input, one output; no dont_use, no tri-state), smallest first; `insert_buffer` defaults to the smallest buffer the design already uses |
| `macro_grid(mac)` | step and per-macro offset that put macro pins on their routing tracks (from the IR) |
| `tracks()` | `{(axis, layer): (start, step)}` routing tracks |
| `set_io(xy, provenance)`, `fix_macros(provenance)`, `release_macros()`, `set_floorplan(die, rows, provenance)` | for engines: write an answer as a fact |

## Evaluator

`attar.Evaluator(design, fast=False, compile=False)`. Every metric takes optional positions `pos`.

| method | returns |
| --- | --- |
| `hpwl(pos)` | exact HPWL |
| `hpwl_batch(P)` | HPWL of a batch of placements (B, N, 2) |
| `hpwl_per_net(pos)` | HPWL of every net |
| `star_wirelength(pos)`, `steiner_wirelength(pos)` | star model; rectilinear Steiner estimate |
| `rudy(pos)`, `density_overflow(pos)` | congestion and density maps / scalars |
| `legality(pos)` | counts: wrong_rail, off_row, outside_rows, off_site, overlapping, pdn_blocked (fact `pdn`); `legal` |
| `evaluate(pos, metrics=(...))` | a report dict (`metrics`: names, or one name) |
| `status(show=False)`, `available()`, `missing(metric)` | which metrics the design's facts allow |

## attar.naive

| function | key arguments |
| --- | --- |
| `global_place(d, target_density=0.9, stop_overflow=0.07, max_iters=2000, movable=None, macro_halo=0, routability=False, route_args=None)` | |
| `legalize(d, max_rounds=5, window=8, improve=True, passes=3, macro_halo=0)` | |
| `place(d, target_density=None, gp_args=None, lg_args=None, macro_halo=0)` | |
| `cts(d, leaf_size=20, fanout=8, target_slew=None, buffers=None, balance_rounds=None, max_diameter=50.0)` | buffered clock trees on every ideal clock net: OpenROAD's inferred clock buffers, clusters bounded by size and TritonCTS's sink clustering diameter, repeaters on edges longer than OpenROAD's max wire length (as many on every edge into a level), buffers kept out of macros, only the new buffers legalized; clocks set propagated; register trees behind clock gates latency-padded, then balanced level by level from the root on the timer (resizes predicted from measured delays); nets get `USE CLOCK` for `d.export` |

Floorplan engines, for the `before_fp` problems (the die, IO pins and macros are the answer; the result is exported
and handed to a flow that draws the power grid):

| function | key arguments |
| --- | --- |
| `floorplan(d, utilization=None, aspect_ratio=1.0, io_pitch=None, halo=None, packing=False, packing_args=None)` | |
| `make_die(d, utilization, aspect_ratio=1.0)` | |
| `place_io(d, pitch=None, margin=None, mode='auto' \| 'nets' \| 'order')` | |
| `place_macros(d, halo=None, target_density=0.9, legalizer='auto' \| 'sweep' \| 'push' \| 'lp', packing=False, packing_args=None)` | default: GP, then `lp` (= `auto`): constraint graphs from the better of sweep / push, positions by linear programming (HiGHS, scipy); never overlapping, in the core when it can (`res['fits_outline']`, `res['outside_core']`). `packing=True`: no GP, B*-tree packing + annealing (into the core, then HPWL), block-level designs only (no rows); `packing_args`: `moves` (per macro and phase, default 200), `seed` |

Each engine checks its facts first (`MissingFacts` otherwise) and writes its answer back into the design.

## Where the code is

| module | what |
| --- | --- |
| `attar/design.py` | `Design`: the state (facts, positions, floorplan), checks, wirelength; `load`, `zoo` |
| `attar/status.py`, `facts.py` | `status`, `summary`; facts and the capabilities that follow from them |
| `attar/netlist.py` | netlist edits (`Edit`: buffers, clones, cells, nets, pins; `debuffer`; `check`) |
| `attar/evaluator.py`, `metrics.py`, `kernels.py` | metrics and the tensor kernels they share with the engines |
| `attar/hub.py`, `paths.py` | the zoo: where it lives, downloads from the Hugging Face Hub |
| `attar/viz.py` | `plot` |
| `attar/io/` | raw files in and out: `lef.py`, `deffile.py`, `bookshelf.py`, `fpformats.py`, `liberty.py`, `sdc.py`, `st.py` (safetensors); `distill.py` (LEF/DEF to the IR), `export.py` (DEF write-back into the source frame) |
| `attar/timing/` | `ir.py` (the timing fact), `library.py` (cell libraries), `sta.py` (static timing, ideal and propagated clocks), `crpr.py` (clock reconvergence pessimism removal), `drv.py` (design rules), `power.py` (power), `dmp.py` / `dmp_triton.py` (delay calculation), `steiner*.py` + `csrc/` (placement parasitics), `sizing.py` (`resize`, `equivalents`) |
| `attar/rules/` | `routing.py` (global-routing capacity and overflow), `rails.py` (power-rail parity), `pdn.py` (PDN blockages) |
| `attar/naive/` | engines: `floorplan.py` (die, IO, macros), `gp.py`, `lg.py`, `flow.py` (`floorplan`, `place`) |
