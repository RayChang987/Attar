# API

```python
import attar
from attar import naive
```

## Loading

| call | returns |
| --- | --- |
| `attar.zoo(suite=None, remote=False, root=None)` | names of the distilled designs (`'ispd2005/adaptec1'`, ...) in the local zoo, or on the Hugging Face Hub |
| `attar.load(name_or_path, device='cpu', download=True, root=None)` | a `Design`; a zoo name missing locally is downloaded first |
| `attar.fetch(*names, root=None)` | download designs or whole suites (`'ispd2015'`) into the local zoo; everything when no name is given |

The zoo is `zoo/` in a source checkout, else `~/.cache/attar/zoo`; `root=` picks another folder. Attar reads no
environment variables.
| `attar.distill(def_path, lef_paths, out_dir, lib_paths=(), ...)` | distill a LEF/DEF design; returns the manifest |

## Design

| method | what it does |
| --- | --- |
| `status(show=True)` | facts, solution state, capabilities, what an ORFS hand-off needs |
| `capabilities()`, `missing(cap)`, `require(cap)`, `has(fact)`, `tier()` | facts and what follows from them |
| `floorplan(utilization=None, aspect_ratio=1.0, **kw)` | `naive.floorplan`: new die if a utilization is given, then IO pins and macros as needed |
| `place(target_density=None, **kw)` | `naive.place`: global placement + legalization |
| `plot(path=None, theme='light', dpi=150, **kw)` | die, core, rows, cells (an area-density image above 30k cells), macros, IO pins by direction; warm light theme by default |
| `export(out_def, template=None, force=False)` | write the design as a DEF: its source frame (PDN, pins, fixed cells, verbatim) with the cells and signal nets from the IR; or stamped onto `template` (checks cell library, units, netlist, die; `force` to override) |
| `handoff_command(out_def, run_dir=None)` | the command that hands an exported DEF to ORFS |
| `hpwl(pos=None, per_net=False)` | exact HPWL (0-d tensor) |
| `wa_wirelength(pos=None, gamma=1.0)` | weighted-average wirelength, differentiable |
| `pin_pos(pos=None)` | (P, 2) pin locations |
| `utilization()` | per die: standard-cell area / (core - macro area inside the core) |
| `snapshot(name)`, `restore(name)`, `unplace()` | solution state |
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
| `legality(pos)` | counts: wrong_rail, off_row, outside_rows, off_site, overlapping; `legal` |
| `evaluate(pos, metrics=(...))` | a report dict |
| `status()`, `available()`, `missing(metric)` | which metrics the design's facts allow |

## attar.naive

| function | key arguments |
| --- | --- |
| `floorplan(d, utilization=None, aspect_ratio=1.0, io_pitch=None, halo=None)` | |
| `make_die(d, utilization, aspect_ratio=1.0)` | |
| `place_io(d, pitch=None, margin=None, mode='auto' \| 'nets' \| 'order')` | |
| `place_macros(d, halo=None, target_density=0.9, legalizer='auto' \| 'sweep' \| 'push')` | |
| `global_place(d, target_density=0.9, stop_overflow=0.07, max_iters=2000, movable=None)` | |
| `legalize(d, max_rounds=5, window=8, improve=True, passes=3)` | |
| `place(d, target_density=None, gp_args=None, lg_args=None)` | |

Each engine checks its facts first (`MissingFacts` otherwise) and writes its answer back into the design.
