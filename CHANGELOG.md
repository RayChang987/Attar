# Changelog

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
