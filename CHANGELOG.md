# Changelog

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
