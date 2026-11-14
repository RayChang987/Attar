<div align="center">

<img src="docs/images/logo.png" alt="Attar: the fast loop for physical design" width="640">

Diverse P&R benchmarks, distilled into one plug-and-play PyTorch environment on the GPU.<br>
Place and evaluate a 100k-cell design in seconds.

[![Python](https://img.shields.io/badge/python-3.8%2B-d9622b?style=flat-square)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-d99a1e?style=flat-square)](https://pytorch.org)
[![Validated](https://img.shields.io/badge/validated%20against-OpenROAD-8f6a3f?style=flat-square)](docs/validation/orfs_check.md)
[![Tests](https://img.shields.io/badge/tests-89%20passing-5f7f1e?style=flat-square)](tests)
[![License](https://img.shields.io/badge/license-MIT-a8946f?style=flat-square)](LICENSE)

[Why Attar](#why-attar) &nbsp;·&nbsp; [Roadmap](#roadmap) &nbsp;·&nbsp; [Two modes](#two-modes) &nbsp;·&nbsp;
[Quickstart](#quickstart) &nbsp;·&nbsp; [How it works](#how-it-works) &nbsp;·&nbsp; [Gallery](#gallery)

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
   stays faithful to the full flows (the *slow loop*) without their overhead: its HPWL equals OpenROAD's to 7 digits,
   its legality equals `check_placement`, its placements route in OpenROAD-flow-scripts with 0 DRC, and HPWL ranks
   placements as routed wirelength does in 951 routed Innovus runs, once they differ by 2% or more
   ([validation](docs/validation/orfs_check.md)).
4. **GPU-native.** Layouts and design features are tensors on the GPU, so researchers, developers and AI agents
   train, evaluate and debug in the same few lines of Python: exact HPWL of 8.9M pins in 3 ms, a 2.2M-cell design
   placed and legalized in 49 s. An algorithm is a function from tensors to positions: no database
   API, no TCL, no file formats.

## Roadmap

<p align="center"><img src="docs/images/roadmap.svg" alt="Roadmap mind map. Attar in the center. Left, Data, as tensors: contest benchmarks, one per suite (placement: ISPD 2005, 2015, 2018, 2019, 2024, ICCAD 2025, GSRC placement; floorplan: MCNC, GSRC blocks, IBM-HB+), ASAP7 (15 placement, 5 floorplan designs) and NS3K (99 placement designs). Right, Naive Engine: floorplan (packing, macro placement, IO placement) and placement (placer, legalizer, PDN-aware refinement), all done. Right, Evaluation: HPWL, global routing OVF and timing done; blocked pin checking not yet; detailed routing and DRV checks will not be built" width="100%"></p>

Attar's built-in GPU engines, `attar.naive`, let a flow run end to end even when you bring only one part of it: to
work on timing optimization, you need a placement, not a placer of your own. Every engine is fully tensorized,
states the facts it needs, and writes its answer back into the design with provenance (`placement: naive`).

| engine | mode | call | what it does |
| --- | --- | --- | --- |
| **Packing** | `before_fp` | `naive.place_macros(packing=True)` | block-level designs: B*-tree packing, annealed into the core, then for HPWL |
| **Macro placement** | `before_fp` | `naive.place_macros` | mixed-size global placement, then macros legalized on their routing tracks with a halo |
| **IO placement** | `before_fp` | `naive.place_io` | IO pins on the die boundary, net-aware once cells are placed |
| **Placer** | both | `naive.global_place` | electrostatic (ePlace/DREAMPlace style, Nesterov, CUDA graphs) at the slow loop's own target density; `routability=True`: DREAMPlace's cell inflation |
| **Legalizer** | `after_fp` | `naive.legalize` | multi-row-height cells, power-rail parity, signal pins clear of the PDN's via metal |
| **PDN-aware refinement** | `after_fp` | inside `naive.legalize` | row DP and repair that keep cells clear of the PDN |

`d.floorplan(utilization=...)` sizes the die (`naive.make_die`), then places the IO pins and the macros;
`d.place()` is global placement plus legalization.

<div align="center">

<img src="docs/images/demo_placement.gif" alt="Placement mode: eight lines of Python run on cva6 and ariane133: load, plot the floorplan, place, plot the placement, export" width="920">

<sub><b>Placement mode</b> (<code>after_fp</code>). Real runs on an RTX 3080 Ti, the same code on two designs. The
floorplan comes as the flow made it; only the cells are placed. <b>cva6</b> (ASAP7, ORFS global placement stage): 121k
cells around 8 SRAMs, legal in 2.5 s; handed back to ORFS it routes with 0 DRC and 7.4% less wire than ORFS's own
placement. <b>ariane133</b> (NS3K, Innovus floorplan at 80% std-cell density): 163k cells around 133 SRAMs, legal in
5.0 s.</sub>

<img src="docs/images/demo_floorplan.gif" alt="Floorplan mode: GSRC n100 at 85% block area, floorplanned twice: GP plus LP legalization leaves 5 macros outside the core (not legal), B*-tree packing fits them all (legal)" width="920">

<sub><b>Floorplan mode</b> (<code>before_fp</code>). GSRC <b>n100</b>: 100 hard blocks, no die; Attar sizes one for 85%
block area, places the IO pins and the macros. GP + LP legalization never overlaps macros, but here 5 stick out of
the core: not legal. <code>packing=True</code> (B*-tree, annealed into the core, then for HPWL) fits them all in
5.1 s, at 8% more wire. The lines are the strongest connections between macros and IO pins.</sub>

</div>

## Two modes

In production chip design, the power delivery network (PDN) is the dividing line: once the power grid and the macros
are locked down, the design enters a different phase. Attar splits its problems at that line. It does not synthesize
a PDN, and it should not: a `before_fp` answer goes to a flow that draws one (ORFS), and an `after_fp` problem comes
with the PDN its flow drew.

| | **Before-floorplan** (`before_fp`) | **After-floorplan** (`after_fp`) |
| --- | --- | --- |
| given | the netlist (an initial die for ORFS designs) | die, IO pins, macros, rows and PDN: a fixed floorplan |
| the answer | target utilization (the die), IO pin placement, macro placement | standard-cell placement; then timing optimization and global routing on the same floorplan |
| one call | `d.floorplan(utilization=0.6)` | `d.place()` |
| back to the slow loop | ORFS draws the PDN and places | ORFS or OpenROAD routes and times |
| in the zoo | 55 designs, 6 suites | 192 designs, 17 suites |

Attar measures wirelength, congestion (RUDY, GROV), density, legality and timing; see the [roadmap](#roadmap).

## Quickstart

```bash
git clone https://github.com/RayChang987/Attar.git attar && cd attar
pip install -e .                # torch (a CUDA build), numpy, matplotlib
```

```python
import attar

d = attar.load('after_fp/orfs_asap7/cva6', device='cuda')
d.status()                   # what it has, what it can do
d.plot('floorplan.png')      # die, macros, IO pins, PDN: given

d.place()                    # global placement + legalization
d.plot('placed.png')
d.export('placed.def')       # a DEF, back to the slow loop
```

`examples/quickstart_placement.py` is this program with a design name argument; `examples/quickstart_floorplan.py` is the same for
`before_fp` problems (`d.floorplan(utilization=0.6)` instead of `d.place()`, then `d.export` for ORFS). The designs come from the *zoo*, distilled from public
benchmarks and published on the Hugging Face Hub, where `attar.load` fetches a design on first use
([more](docs/benchmarks.md#building-the-zoo)); Attar ships the readers, the zoo the data.
`d.export` needs no raw file: the design keeps what it does not model of its source DEF (PDN, IO pin shapes, fixed
cells) and writes the cells and nets around it.

Your own algorithm takes the same few lines, because every engine is a function of the tensors:

```python
import torch, attar
from attar import naive

d = attar.load('after_fp/ispd2005/adaptec1', device='cuda').restore('dreamplace_lg')
ev = attar.Evaluator(d)

pos = d.pos.clone().requires_grad_(True)              # (N, 2) lower-left corners
opt = torch.optim.Adam([pos], lr=0.5)
for _ in range(100):                                  # e.g. wirelength descent on the differentiable WA model
    opt.zero_grad(); d.wa_wirelength(pos, gamma=2.0).backward(); opt.step()   # 100 steps: 0.2 s

print(ev.hpwl(pos.detach()))                          # exact HPWL in milliseconds
print(ev.legality(pos.detach()))                      # rows, sites, overlaps, power rails: not legal any more
d.pos = pos.detach(); naive.legalize(d)               # ... so legalize (multi-row-height, rail aware)
```

## How it works

<p align="center"><img src="docs/images/flow.svg" alt="Sources (LEF/DEF, Bookshelf, YAL, GSRC blocks) are distilled into the Attar IR and loaded as a Design on the GPU; attar.naive moves it and adds facts, the Evaluator measures it; export writes DEF, ORFS routes it, and the slow loop's truth is compared with the Evaluator" width="100%"></p>

1. **Distill.** A reader turns a benchmark into the Attar IR: integer arrays in a safetensors file, plus a manifest
   that lists which facts the source provides and where each came from. Readers exist for LEF/DEF, Bookshelf, YAL
   and GSRC blocks.
2. **Load.** `attar.load` puts the arrays on the GPU as a `Design`. `d.status()` shows each fact, each capability,
   and what a hand-off to ORFS still needs.
3. **Move.** Your algorithm, or the [built-in engines](#roadmap) (`attar.naive`), changes the positions:
   die, IO pins and macros for `before_fp`, cells for `after_fp`.
4. **Measure.** `attar.Evaluator` computes HPWL (exact, batched), Steiner wirelength, RUDY, density overflow and
   legality, on the current placement or on any positions you pass.
5. **Hand off.** `d.export(out_def)` writes the design back into its source DEF's frame: the PDN, IO pin shapes and
   fixed cells as they were, the cells and nets from the IR (a changed die, IO pins and macros included). For ORFS
   designs, `tools/orfs_roundtrip.py` takes a placement into the ORFS run, and ORFS routes and times it;
   `tools/orfs_handoff.py` takes a floorplan, and ORFS draws the PDN and goes on. Designs without an ORFS platform
   (i3dPDK_NS3K) go to OpenROAD with `tools/validate_openroad.py`.

More: [concepts](docs/concepts.md) · [benchmarks](docs/benchmarks.md) · [slow loop](docs/slow_loop.md) ·
[API](docs/api.md)

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
<td><sub><b>cva6</b> (ASAP7, ORFS global placement stage): 121k cells around 8 SRAMs in 2.5 s. ORFS routes it with
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
  ✅ slow_loop         orfs:asap7      ORFS aes, stage 3_3_place_gp
  ...
Solution state
  ❌ placement         12009 movable cells unplaced  (snapshots: reference)
  ...
  -> only a placement is missing: place the design (or restore a snapshot), then export and hand off
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
The slow loop is [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) on the ASAP7
PDK. Benchmarks are from the ISPD, ICCAD and GSRC contests, MCNC and ORFS.

## License

[MIT](LICENSE)
