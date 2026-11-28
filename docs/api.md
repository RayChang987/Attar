# API

```python
import attar
from attar import naive
```

One line per call; the docstrings have the details (`help(d.timing)`).

## Loading

| call | returns |
| --- | --- |
| `attar.zoo(suite=None, remote=False, root=None)` | names `<mode>/<suite>/<design>` in the local zoo, or on the Hugging Face Hub; `suite` is a prefix such as `'after_fp/ns3k'` |
| `attar.load(name_or_path, device='cpu', download=True, root=None)` | a `Design`; a zoo name missing locally is downloaded first |
| `attar.fetch(*names, root=None, force=False)` | download designs or suites, with their shared libraries; `force=True` downloads again (after a zoo update) |
| `attar.distill(def_path, lef_paths, out_dir, lib_paths=(), ...)` | distill a LEF/DEF design; returns the manifest |

The zoo is `zoo/` in a source checkout, else `~/.cache/attar/zoo`; `root=` picks another folder. Attar reads no
environment variables.

## Design

| method | what it does |
| --- | --- |
| `status(show=True)`, `summary()` | facts, solution state, capabilities, what is left before export; a short summary |
| `capabilities()`, `missing(cap)`, `require(cap)`, `has(fact)` | facts and what follows from them |
| `place(target_density=None, macro_halo=0, **kw)` | global placement + legalization (default density: the source's own, else 0.9) |
| `floorplan(utilization=None, aspect_ratio=1.0, packing=False, **kw)`, `floorplan_legal(pos=None)` | `before_fp`: die, IO pins, macros; `(legal, problems)` of a floorplan |
| `plot(path=None, dpi=150, nets=None, tiles=None, bins=None, **kw)` | the floorplan and cells; `tiles=` draws a map instead: `'gr_overflow'`, `'pin_access'` (dpl's PDN rule), `'unreachable_pins'` (`Evaluator.pin_access`), `'density'` |
| `export(out_def)` | write a DEF: the source frame verbatim, cells and signal nets from the IR |
| `hpwl(pos=None, per_net=False, weighted=True)`, `wa_wirelength(pos=None, gamma=1.0)` | exact HPWL (net weights `d.n_weight`); differentiable weighted-average wirelength |
| `pin_pos(pos=None)`, `utilization()` | (P, 2) pin locations; per-die standard-cell utilization |
| `snapshot(name)`, `restore(name)`, `unplace()` | solution state |
| `read_placement(def_path, fixed=False)`, `set_positions(names, xy, orient=None)` | positions by instance name, from another tool's DEF or given |
| `pin_names(pins=None)`, `pin_index(names)`, `cell_index(names)` | names as OpenSTA prints them, and back to indices |
| `timing(pos=None, crpr=True, routes=None)`, `timing_report(pos=None, endpoints=10, path=True)` | static timing as OpenSTA, placement parasitics or (`routes=`) global-routing parasitics; a re-time redoes only the nets that changed. Returns a `TimingResult` (below) |
| `gr_overflow(pos=None, adjustment=None, maps=False)` | global-routing overflow: RUDY demand against OpenROAD's gcell capacities |
| `global_route(pos=None, verbose=True, previous=None, **kw)` | `(routes, stats)` on OpenROAD's gcell grid; `previous=routes` reroutes only the nets that moved |
| `drv(pos=None)` | max slew, capacitance and fanout per pin, as OpenSTA `report_check_types` |
| `power(pos=None)` | internal, switching and leakage power per instance, as OpenROAD `report_power` |
| `timing_settings()`, `set_timing(**settings)`, `reset_timing()`, `ideal_clocks(clocks=None)` | edit the timing constraints in memory (the files stay), e.g. `s = d.timing_settings(); s['clocks']['clk']['period'] = 300; d.set_timing(clocks=s['clocks'])` |
| `propagate_clocks(clocks=None)` | time clocks through their trees (`set_propagated_clock`), with CRPR |
| `equivalents(cell)`, `resize(cells, masters)` | gate sizing within a master's equivalence class |
| `edit()` | a `with` block of netlist edits: `insert_buffer`, `clone`, `add_cell`, `add_net`, `connect`, `move`, `remove_cell`; no cell or net index moves |
| `debuffer(buffers=True, inverters=True)` | remove buffers and double inversions, as OpenROAD `remove_buffers`; returns counts |
| `buffers()`, `inverters()` | the library's buffer and inverter cells, smallest first |
| `macro_grid(mac)`, `tracks()` | positions that put macro pins on tracks; `{(axis, layer): (start, step)}` |
| `set_io`, `fix_macros`, `release_macros`, `set_floorplan` | for engines: write an answer as a fact |

### TimingResult

`summary()`, `report()`, `path()`, `endpoint_slacks()`; `pin_slack()`, `net_slack()` (every pin's and net's setup
slack); `critical_paths(endpoints=None, violating=True, max_paths=None)` (the worst path into many endpoints at
once); `clock_arrivals()`; `limitations` (known gaps that affect this timing; `summary()['exact']` is False when
there is one).

The timer's data, read only (copies with `version` = `attar.timing.sta.TIMING_DATA_VERSION`), for research code:

| call | what |
| --- | --- |
| `graph()` | levels; net, cell, launch and check arcs with their Liberty arcs; launch tags; loop-breaking and latch arcs; clock networks |
| `library()` | NLDM tables per Liberty arc, pin capacitances, `lookup(arc, slot, x, y)` |
| `arrivals(per_tag=False)` | latest and earliest arrival and slew per pin and transition (per launch tag if asked), driver loads |
| `endpoint_required()` | per setup check: required time and arrival per launch tag, slack before CRPR |
| `stages()` | the level schedule of the last run with the gate and wire delays the timer computed; the arrivals follow from them |
| `parasitics()`, `trees(nets=None)` | wire capacitance, pi model and Elmore delays per net; the placement parasitics' Steiner trees |

## Evaluator

`attar.Evaluator(design, fast=False, compile=False)`. Every metric takes optional positions `pos`.

| method | returns |
| --- | --- |
| `hpwl(pos)`, `hpwl_batch(P)`, `hpwl_per_net(pos)` | exact HPWL; of a batch (B, N, 2); per net |
| `star_wirelength(pos)`, `steiner_wirelength(pos)` | star model; rectilinear Steiner estimate |
| `rudy(pos)`, `density_overflow(pos)` | congestion and density |
| `legality(pos, per_cell=False, ignore=None)` | counts (wrong_rail, off_row, outside_rows, off_site, overlapping, pdn_blocked) and `legal` |
| `pin_access(pos=None, per_pin=False, per_cell=False)` | signal pins the router cannot reach past the PDN (facts `pdn`, `tracks`); any PDK ([validation](validation/pin_access.md)) |
| `displacement(pos, ref=None, cells=None)` | per-cell move from `ref`, as OpenROAD dpl reports it |
| `evaluate(pos, metrics=(...))` | a report dict |
| `status(show=False)`, `available()`, `missing(metric)` | which metrics the design's facts allow |

## attar.naive

| function | key arguments |
| --- | --- |
| `global_place(d, target_density=0.9, stop_overflow=0.07, max_iters=2000, movable=None, macro_halo=0, routability=False, extra=None, callback=None, callback_every=1, precond='net_weight', deterministic=False)` | `extra(pos) -> (value, grad)` adds an objective term; `callback(it, pos[, info])` may change `d.n_weight` (used from the next evaluation) or stop; `deterministic=True` repeats on the GPU (slower) |
| `legalize(d, cells=None, incremental=None, method=None, pin_access=True, ...)` | Abacus by default (`'columns'` without g++); incremental when `cells` is given or most cells are legal; `pin_access=False` keeps only OpenROAD dpl's PDN rule |
| `detailed_place(d, ops=('gs', 'vs', 'ro', 'flip'), passes=10, tol=0.01, window=3, seed=1, pin_access=True)` | OpenROAD dpo's global swap, vertical swap, reordering and flipping, batched; stays legal and keeps pins reachable |
| `place(d, target_density=None, gp_args=None, lg_args=None, macro_halo=0)` | `global_place` + `legalize` |
| `cts(d, leaf_size=20, fanout=8, target_slew=None, buffers=None, balance_rounds=None, max_diameter=50.0)` | buffered clock trees on the ideal clock nets, balanced on the timer |
| `floorplan(d, utilization=None, aspect_ratio=1.0, io_pitch=None, halo=None, packing=False, packing_args=None)` | `make_die` + `place_io` + `place_macros` |
| `make_die(d, utilization, aspect_ratio=1.0)` | die and rows |
| `place_io(d, pitch=None, margin=None, mode='auto')` | IO pins on the boundary (`'nets'` or `'order'`) |
| `place_macros(d, halo=None, target_density=0.9, legalizer='auto', packing=False, packing_args=None)` | GP, then macros legalized by LP on their track grid; `packing=True`: B*-tree packing for block-level designs |

Each engine checks its facts first and writes its answer back into the design.

## Errors

| error | when |
| --- | --- |
| `attar.facts.MissingFacts` | an engine or metric lacks facts; the message lists them and how to get them |
| `attar.gpu.GPUOutOfMemory` | CUDA ran out of memory; the message says where, how much was free, and the ways out (a `torch.cuda.OutOfMemoryError`) |
| `KeyError`, `FileNotFoundError` | an unknown fact, capability, metric, snapshot or design; the message lists the closest names |

## Where the code is

| module | what |
| --- | --- |
| `attar/design.py`, `status.py`, `facts.py` | `Design`, `load`, `zoo`; status; facts and capabilities |
| `attar/netlist.py` | netlist edits and `debuffer` |
| `attar/evaluator.py`, `metrics.py`, `kernels.py` | metrics and shared tensor kernels |
| `attar/hub.py`, `paths.py`, `gpu.py`, `viz.py` | the zoo and downloads; GPU out-of-memory messages; `plot` |
| `attar/io/` | LEF, DEF, Bookshelf, floorplan formats, Liberty, SDC readers; `distill.py`, `export.py` |
| `attar/timing/` | `sta.py` (static timing), `crpr.py`, `drv.py`, `power.py`, `dmp*.py` (delay calculation), `steiner*.py`, `grpar.py` (parasitics), `sizing.py`; `csrc/` C++ built on first use |
| `attar/route/` | `global_route` and `GlobalRoutes` |
| `attar/rules/` | routing capacity, rail parity, PDN blockages, pin access |
| `attar/naive/` | engines: floorplan, GP, LG (Abacus in `csrc/`), DP, CTS |
