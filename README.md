<div align="center">

<img src="docs/images/logo.png" alt="Attar: the fast loop for physical design" width="640">

Diverse P&R benchmarks, distilled into one plug-and-play PyTorch environment on the GPU.<br>
Place and evaluate a 100k-cell design in seconds.

[![Python](https://img.shields.io/badge/python-3.8%2B-d9622b?style=flat-square)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-d99a1e?style=flat-square)](https://pytorch.org)
[![Validated](https://img.shields.io/badge/validated%20against-OpenROAD-8f6a3f?style=flat-square)](docs/validation/orfs_check.md)
[![Tests](https://img.shields.io/badge/tests-95%20passing-5f7f1e?style=flat-square)](tests)
[![License](https://img.shields.io/badge/license-MIT-a8946f?style=flat-square)](LICENSE)

[Why Attar](#why-attar) &nbsp;·&nbsp; [Roadmap](#roadmap) &nbsp;·&nbsp; [Two modes](#two-modes) &nbsp;·&nbsp;
[How it works](#how-it-works) &nbsp;·&nbsp; [Gallery](#gallery)

</div>

<br>

## Why Attar

1. **Speed decides quality.** When AI agents do the research, iteration speed is more than a metric: in our view it
   decides the quality of results (QoR). An agent that can propose, test and discard an idea in seconds finds what
   works. When every run takes hours, even the best algorithm starves for feedback.
2. **Data preparation is painful.** Traditional place-and-route (P&R) tools take minutes to hours per run, and every
   benchmark suite comes in its own format (LEF/DEF, Bookshelf, YAL, GSRC blocks) with its own gaps. Getting a design
   ready to experiment on is slow, fragmented work, repeated for every suite.
3. **Attar is the fast loop.** Attar distills diverse P&R benchmarks into one plug-and-play PyTorch environment:
   `attar.load` a name, and the design is on the GPU. Each reader hands over only the facts its source has, with
   their provenance, and nothing missing is made up ([a design is a state](#a-design-is-a-state)). The fast loop
   stays faithful to the full flows without their overhead: its HPWL equals OpenROAD's to 7 digits,
   its legality equals `check_placement`, its placements route in OpenROAD-flow-scripts with 0 DRC, and HPWL ranks
   placements as routed wirelength does in 951 routed Innovus runs, once they differ by 2% or more
   ([validation](docs/validation/orfs_check.md)).
4. **GPU-native.** Layouts and design features are tensors on the GPU, so researchers, developers and AI agents
   train, evaluate and debug in the same few lines of Python: exact HPWL of 8.9M pins in 3 ms, a 2.2M-cell design
   placed and legalized in 49 s. An algorithm is a function from tensors to positions: no database
   API, no TCL, no file formats.

## Roadmap

<p align="center"><img src="docs/images/roadmap.svg" alt="Roadmap mind map. Attar in the center. Left, Data, as tensors: contest benchmarks, one per suite (placement: ISPD 2005, 2015, 2018, 2019, 2024, ICCAD 2025, GSRC placement; floorplan: MCNC, GSRC blocks, IBM-HB+), ASAP7 (15 placement, 5 floorplan designs) and NS3K (99 placement designs). Right, Naive Engine: floorplan (packing, macro placement, IO placement) and placement (placer, legalizer, PDN-aware refinement), all done. Right, Evaluation: HPWL, global routing OVF and timing done; blocked pin checking not yet; detailed routing and DRV checks will not be built" width="100%"></p>

### Evaluation

Each evaluator against OpenROAD on the same netlist and placement; full tables in
[docs/validation](docs/validation/timing_routing.md).

| Evaluator | Call | Compared with | Accuracy |
| --- | --- | --- | --- |
| **HPWL** | `ev.hpwl` | OpenROAD's HPWL from its database; 23 ORFS designs (ASAP7, Nangate45) up to 1.05M cells | within **0.00002%** on every design ([table](docs/validation/orfs_check.md)) |
| **Legality** | `ev.legality` | OpenROAD `check_placement` | same verdict on all **23** ORFS designs and the **62** NS3K floorplans OpenROAD can check; placements broken by half a site rejected by both |
| **Timing** | `d.timing()` | OpenSTA after `estimate_parasitics -placement`; 30 designs (ASAP7, Nangate45, NS3K) up to 1.05M cells | WNS median error **0.06-0.20%**, max **1.19%**; hold slack MAE at most 2.6 ps; 5-30x faster than OpenSTA |
| **Pin and net slack** | `d.timing().pin_slack()`, `.net_slack()` | OpenSTA's `slack_max` of every pin; 8 designs (ASAP7, Nangate45) up to 384k pins | Pearson at least **0.9995**; on the pins with negative slack, MAE at most **1.03 ps**; 7-32 ms after a timing run |
| **Timing after sizing** | `d.resize` + `d.timing()` | OpenROAD `replace_cell` + OpenSTA, 20% of the cells swapped at random; 16 designs | WNS error at most **0.23%**, TNS at most 0.37% |
| **Global routing OVF** | `d.gr_overflow()` | OpenROAD's gcell capacities and global route; 13 ORFS designs | capacities exact without macros, within **0.16%** with them; ranks designs as OpenROAD does (Spearman **0.97**); a pre-route estimate: reads mild congestion high and heavy congestion low |
| **Steiner wirelength, RUDY, density overflow** | `ev.steiner_wirelength`, `ev.rudy`, `ev.density_overflow` |  | |

### Naive Engine

| Engine | Mode | Call | What it does |
| --- | --- | --- | --- |
| **Packing** | `before_fp` | `naive.place_macros(packing=True)` | block-level designs: B*-tree packing, annealed into the core, then for HPWL |
| **Macro placement** | `before_fp` | `naive.place_macros` | mixed-size global placement, then macros legalized on their routing tracks with a halo |
| **IO placement** | `before_fp` | `naive.place_io` | IO pins on the die boundary, net-aware once cells are placed |
| **Placer** | both | `naive.global_place` | electrostatic (ePlace/DREAMPlace style, Nesterov, CUDA graphs) at the source flow's own target density; `routability=True`: DREAMPlace's cell inflation |
| **Legalizer** | `after_fp` | `naive.legalize` | multi-row-height cells, power-rail parity, signal pins clear of the PDN's via metal |
| **PDN-aware refinement** | `after_fp` | inside `naive.legalize` | row DP and repair that keep cells clear of the PDN |

`d.floorplan(utilization=...)` sizes the die (`naive.make_die`), then places the IO pins and the macros;
`d.place()` is global placement plus legalization.

<div align="center">

<img src="docs/images/demo_placement.gif" alt="Timing-driven placement written in fifteen lines of Python, run on cva6 and swerv: place for wirelength, time it, weight the nets by their slack as OpenROAD's global placement does, place again, time it" width="920">

<sub><b>Your algorithm, in fifteen lines</b> (<a href="examples/timing_driven.py">examples/timing_driven.py</a>). Real
runs on an RTX 3080 Ti, the same code on two ASAP7 designs: place for wirelength, take every net's slack
(<code>net_slack()</code>, as OpenSTA's), weight the worst 10% of the nets from 1 to 5 by criticality as OpenROAD's
timing-driven global placement does, place again. <b>cva6</b> (121k cells, 8 SRAMs): WNS -195 to -75 ps, TNS
-85.7k to -5.1k ps, HPWL +6.7%, in 2.2 s. <b>swerv</b> (103k cells): WNS -205 to -117 ps, TNS -39.5k to -1.2k ps,
HPWL +1.1%, in 1.9 s. Both legal.</sub>

<img src="docs/images/demo_floorplan.gif" alt="Floorplan mode: GSRC n100 at 85% block area, floorplanned twice: GP plus LP legalization leaves 5 macros outside the core (not legal), B*-tree packing fits them all (legal)" width="920">

<sub><b>Floorplan mode</b> (<code>before_fp</code>). GSRC <b>n100</b>: 100 hard blocks, no die; Attar sizes one for 85%
block area, places the IO pins and the macros. GP + LP legalization never overlaps macros, but here 5 stick out of
the core: not legal. <code>packing=True</code> (B*-tree, annealed into the core, then for HPWL) fits them all in
5.1 s, at 8% more wire. The lines are the strongest connections between macros and IO pins.</sub>

</div>

## Two modes

In production chip design, the power delivery network (PDN) is the dividing line: once the power grid and the macros
are locked down, the design enters a different phase. Attar splits its problems at that line. It does not synthesize
a PDN, and it should not: a `before_fp` answer goes to a flow that draws one, and an `after_fp` problem comes
with the PDN its flow drew.

| | **Before-floorplan** (`before_fp`) | **After-floorplan** (`after_fp`) |
| --- | --- | --- |
| given | the netlist (an initial die for ORFS designs) | die, IO pins, macros, rows and PDN: a fixed floorplan |
| the answer | target utilization (the die), IO pin placement, macro placement | standard-cell placement; then timing optimization and global routing on the same floorplan |
| one call | `d.floorplan(utilization=0.6)` | `d.place()` |
| goes out as | a DEF: your flow draws the PDN and places | a DEF: your flow routes and times |
| in the zoo | 55 designs, 6 suites | 192 designs, 17 suites |

Attar measures wirelength, congestion (RUDY, GROV), density, legality and timing; see the [roadmap](#roadmap).

## How it works

<p align="center"><img src="docs/images/flow.svg" alt="Sources (LEF/DEF, Bookshelf, YAL, GSRC blocks) are distilled into the Attar IR and loaded as a Design on the GPU; attar.naive moves it and adds facts, the Evaluator measures it; export writes a DEF for your own flow (OpenROAD, Innovus, ICC2, ...)" width="100%"></p>

1. **Distill.** A reader turns a benchmark into the Attar IR: integer arrays in a safetensors file, plus a manifest
   that lists which facts the source provides and where each came from. Readers exist for LEF/DEF, Bookshelf, YAL
   and GSRC blocks.
2. **Load.** `attar.load` puts the arrays on the GPU as a `Design`. `d.status()` shows each fact, each capability,
   and what is left before export.
3. **Move.** Your algorithm, or the [built-in engines](#roadmap) (`attar.naive`), changes the positions:
   die, IO pins and macros for `before_fp`, cells for `after_fp`.
4. **Measure.** `attar.Evaluator` computes HPWL (exact, batched), Steiner wirelength, RUDY, density overflow and
   legality, on the current placement or on any positions you pass.
5. **Export.** `d.export(out_def)` writes the design back into its source DEF's frame: the PDN, IO pin shapes and
   fixed cells as they were, the cells and nets from the IR (a changed die, IO pins and macros included). Your own
   flow (OpenROAD, Innovus, ICC2, ...) takes it from there.

More: [concepts](docs/concepts.md) · [benchmarks](docs/benchmarks.md) · [API](docs/api.md) ·
[validation](docs/validation/timing_routing.md)

## Gallery

<table>
<tr>
<td width="50%"><img src="docs/images/gallery/fp_n300.png" alt="GSRC n300 floorplanned by attar.naive with B*-tree packing"></td>
<td width="50%"><img src="docs/images/gallery/fp_ami49.png" alt="MCNC ami49 floorplanned by attar.naive with B*-tree packing"></td>
</tr>
<tr>
<td><sub><b>GSRC n300</b> (floorplan mode): 300 hard blocks and 569 IO pins, no die given. Attar sizes one for 85% block
area and packs the blocks (B*-tree) in 46 s: legal, HPWL 565k.</sub></td>
<td><sub><b>MCNC ami49</b> (floorplan mode): 49 blocks of very different sizes in a die sized for 80% block area,
packed in 2.3 s: legal, HPWL 948k.</sub></td>
</tr>
<tr>
<td><img src="docs/images/gallery/cva6.png" alt="cva6 placed by attar.naive"></td>
<td><img src="docs/images/gallery/ariane136.png" alt="ariane136 on NS3K at 80% density placed by attar.naive"></td>
</tr>
<tr>
<td><sub><b>cva6</b> (ASAP7, ORFS global placement stage): 121k cells around 8 SRAMs in 2.4 s. ORFS routes it with
0 DRC and 7.4% less wire than its own placement.</sub></td>
<td><sub><b>ariane136</b> (NS3K, Innovus floorplan at 80% std-cell density): 170k cells around 136 SRAMs in 5.2 s,
legal including power rails and the PDN; OpenROAD's <code>check_placement</code> and global routing pass.</sub></td>
</tr>
</table>

## A design is a state

```text
>>> d = attar.load('after_fp/orfs_asap7/aes'); d.status()
aes_cipher_top  (orfs/asap7, fidelity F1)
  placement target density 0.71 (orfs 3_3 global_place (GPL-0023))
Problem facts
  ✅ netlist           def             12268 nets, 42885 pins
  ✅ cell_geometry     lef             12420 cells (12009 movable, 0 macros), 56 masters
  ✅ floorplan.die     def             1 die(s), 48.71 x 48.71 um
  ✅ floorplan.io      def             388 IO pins placed
  ✅ rows              def             164 rows, height 0.27 um
  ✅ pdn               def             1815 wire and 4785 via shapes on M1, M2, M3
  ✅ def_frame         def             PDN, pins, fixed cells of the source DEF, kept verbatim (849 kB)
  ...
Solution state
  ❌ placement         12009 movable cells unplaced  (snapshots: reference)
  ...
  -> only a placement is missing: place the design (or restore a snapshot), then d.export("out.def")
```

Every reader states which facts it provides, and every engine and metric states which facts it needs:
- a Bookshelf design has no cell library, so it does not export to DEF;
- a design with no special nets on the pin layers has no `pdn` rule to keep;
- a block floorplan has no rows;
- a soft-block benchmark has no cell geometry.

Each can do exactly what its facts allow. Engines add facts with provenance (`placement: naive`), so it is always
clear which part of a design came from where. See [docs/concepts.md](docs/concepts.md).

## Acknowledgements

Attar's global placement follows ePlace, DREAMPlace and Xplace, and its routability option DREAMPlace's cell
inflation; its rail and PDN checks follow OpenROAD's detailed placer, and its PDN repair the row DP of pa_pdn.
Attar is validated against [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) on
the ASAP7 and Nangate45 platforms. Benchmarks are from the ISPD, ICCAD and GSRC contests, MCNC and ORFS.

## License

<div align="center">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-a8946f?style=for-the-badge" alt="MIT License"></a>
<br><br>
<sub>Copyright &copy; 2026 RayChang987 &nbsp;·&nbsp; free to use, modify and distribute; see <a href="LICENSE">LICENSE</a></sub>
<br><br>
<img src="docs/images/logo.png" alt="Attar" width="180">
</div>
