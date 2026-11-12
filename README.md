<div align="center">

# Attar

### The fast loop for physical design

Any placement benchmark as GPU tensors. Place and evaluate a 100k-cell design in seconds,<br>
in a few lines of Python. Then hand the result back to OpenROAD and route it.

[![Python](https://img.shields.io/badge/python-3.8%2B-d9622b?style=flat-square)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-d99a1e?style=flat-square)](https://pytorch.org)
[![Validated](https://img.shields.io/badge/validated%20against-OpenROAD-8f6a3f?style=flat-square)](#does-the-fast-loop-agree-with-the-slow-loop)
[![Tests](https://img.shields.io/badge/tests-77%20passing-5f7f1e?style=flat-square)](tests)
[![License](https://img.shields.io/badge/license-MIT-a8946f?style=flat-square)](LICENSE)

[Quickstart](#quickstart) &nbsp;·&nbsp; [How it works](#how-it-works) &nbsp;·&nbsp; [Gallery](#gallery) &nbsp;·&nbsp;
[Speed](#speed) &nbsp;·&nbsp; [Validation](#does-the-fast-loop-agree-with-the-slow-loop) &nbsp;·&nbsp;
[Benchmarks](#benchmarks) &nbsp;·&nbsp; [Roadmap](#roadmap)

<br>

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

<br>

## Why a fast loop

Placement research lives in a loop: change an idea, place, look at the numbers, change again. The slow loop
(synthesis, place and route, signoff) takes minutes to hours per run. Attar is the other half: everything you need
to iterate on placement, **fast to run and fast to write**. The zoo holds two kinds of problems, and every answer goes
back to the slow loop:

- **`after_fp`, placement:** the floorplan and power grid are given, the cells are the answer (timing optimization and
  global routing later).
- **`before_fp`, floorplan:** the netlist is given; the die (utilization), the IO pins and the macros are the answer.
  Attar draws no power grid: the floorplan goes to a flow that does (ORFS).

<table>
<tr>
<td width="25%" valign="top"><b>Fast to run</b><br><sub>The design lives on the GPU as flat tensors. HPWL of 8.9M pins in 3 ms; 2.2M cells placed and legalized in 48 s; a design loads in under a second.</sub></td>
<td width="25%" valign="top"><b>Fast to write</b><br><sub>One <code>Design</code>, one <code>Evaluator</code>, a basic set of engines (<code>attar.naive</code>). An algorithm is a function from tensors to positions: no database API, no TCL, no file formats.</sub></td>
<td width="25%" valign="top"><b>Honest about what it knows</b><br><sub>A design is a set of <i>facts</i> with provenance. What it can do follows from what it has, and nothing missing is ever made up.</sub></td>
<td width="25%" valign="top"><b>Checked against the slow loop</b><br><sub>HPWL equals OpenROAD's to 7 digits, legality equals <code>check_placement</code>, the write-back round-trips through ORFS bit-identically, and Attar's placements route in ORFS with 0 DRC.</sub></td>
</tr>
</table>

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
3. **Move.** `attar.naive` is the basic, fully tensorized set of engines:
   - electrostatic global placement (ePlace/DREAMPlace style, Nesterov, CUDA graphs), at the target density the
     slow loop itself uses for the design; optionally routability-driven (DREAMPlace's cell inflation)
   - a multi-row-height legalizer that keeps power-rail parity and signal pins clear of the PDN's via metal

   For `before_fp` problems: die sizing for a utilization, IO pins, and macros on their routing tracks
   (`d.floorplan(utilization=...)`).
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

## Speed

RTX 3080 Ti (12 GB), PyTorch 2.4. Times include everything the call does.

| | cva6 (ASAP7) | bp_quad (NS3K) | bigblue4 (ISPD 05) |
| --- | ---: | ---: | ---: |
| cells / pins | 125k / 421k | 1.18M / 4.3M | 2.18M / 8.9M |
| load from disk to GPU | 0.36 s | 0.31 s | 0.99 s |
| HPWL, exact | **0.23 ms** | **1.5 ms** | **3.2 ms** |
| WA wirelength, forward + backward | 1.2 ms | 8.3 ms | 17 ms |
| global placement + legalization | **3.1 s** | **82 s** | **49 s** |
| GPU memory, peak | 1.3 GB | 2.9 GB | 4.3 GB |

bp_quad's legalization includes keeping signal pins clear of its PDN: a GPU row DP mirrors and shifts about 17k cells
in a few seconds; the 1.5k multi-row cells left take about a minute in a Python fallback ([roadmap](#roadmap)).

## Does the fast loop agree with the slow loop?

A fast loop is only useful if its numbers mean the same thing as the slow loop's. Six checks, each reproducible with
the tools in this repo (ORFS on ASAP7 and Nangate45; OpenROAD alone on i3dPDK_NS3K, which has no ORFS platform):

| | check | result |
| --- | --- | --- |
| **1** | Attar's HPWL vs the HPWL OpenROAD computes from its own database, same DEF, 51 ORFS designs (33 ASAP7, 18 Nangate45) up to 1.48M cells | equal to within **0.00005%** |
| **2** | Attar says legal, then OpenROAD `check_placement` on Attar's DEF; and 1% of the cells moved half a site | ORFS **51 / 51**, NS3K **11 / 17** (with bp_quad, 1.18M cells): legal on both, broken rejected by both. On the other 6 NS3K designs OpenROAD's `check_placement` crashes on the floorplan DEF itself ([which](docs/benchmarks.md#the-zoo-today)); Attar finds them legal |
| **3** | ORFS's own placement written back unchanged, then ORFS resize, DP, CTS, routing | routed WL, DRC, WNS, TNS **bit-identical**, **9 / 9** designs |
| **4** | Attar's own placement, at the design's target density, then ORFS to the end | **9 / 9** route with **0 DRC**; routed WL **0.2% to 7.4% shorter** than with ORFS's own placement |
| **5** | HPWL vs routed wirelength, 951 routed runs of 16 designs (NS3K + Innovus) | same order at **≥ 2%** HPWL difference; a coin flip below 0.25% |
| **6** | `before_fp`: Attar's floorplan (a new die at a looser or tighter utilization, IO pins, macros) handed to ORFS, which draws the PDN and places, up to detailed placement | **12 / 14** pass `check_placement` with no more slew / cap / fanout violations than ORFS's own floorplan, every macro and pin kept. The 2 that fail (tight cva6, tight swerv_wrapper: [why](docs/validation/fp_handoff.md)) are not in the zoo until the floorplan engines handle them<a id="before_fp-hand-off"></a> |

<p align="center"><img src="docs/images/proxy_agreement.png" alt="Agreement of routed wirelength with HPWL rises from 55% for HPWL differences under 0.1% to 99.8% above 5%" width="760"></p>

HPWL does *not* predict timing: WNS and TNS move independently of it, and Attar's placements are not timing-driven
(their WNS ranges from better to worse than ORFS's). Timing is the next proxy to build ([roadmap](#roadmap)).

<details>
<summary><b>Check 1 in detail: HPWL against OpenROAD, design by design</b></summary>
<br>

`tools/validate_orfs.py`, ORFS's global placement of each design:

| design | cells | HPWL, Attar (um) | HPWL, OpenROAD (um) | difference |
| --- | ---: | ---: | ---: | ---: |
| gcd | 412 | 579.38 | 579.38 | -0.00001% |
| riscv32i | 10,106 | 42,491.95 | 42,491.94 | +0.00000% |
| aes | 12,420 | 40,690.80 | 40,690.80 | -0.00000% |
| ibex | 21,076 | 66,060.55 | 66,060.55 | -0.00001% |
| tinyRocket | 32,292 | 119,581.85 | 119,581.86 | -0.00000% |
| jpeg | 57,891 | 126,376.97 | 126,376.96 | +0.00001% |
| ethmac | 59,653 | 141,021.00 | 141,020.97 | +0.00002% |
| swerv_wrapper | 114,086 | 732,124.62 | 732,124.70 | -0.00001% |
| cva6 | 124,618 | 475,600.19 | 475,600.21 | -0.00001% |
| jpeg_x20 | 1,052,787 | 5,979,077.00 | 5,979,076.93 | +0.00000% |

jpeg_x20 is 20 copies of the synthesized jpeg run through ORFS as one design: a million-cell scaling test.
The legality check covers rows, sites, overlaps, row segments cut around macros, power-rail parity, and signal pins
clear of the PDN's via metal (as OpenROAD's `check_placement` checks it).
All 51 ORFS designs, ASAP7 and Nangate45: [docs/validation/orfs_check.md](docs/validation/orfs_check.md).

</details>

<details>
<summary><b>Check 4 in detail: Attar's placement routed by ORFS</b></summary>
<br>

`tools/orfs_roundtrip.py <design> <run> def=<Attar's DEF>`: ORFS resizes, legalizes in detail, builds the clock tree,
routes and times Attar's placement. Target density: the one ORFS's own global placement used for the design.

| design | cells | target density | routed WL, ORFS's placement | routed WL, Attar's | difference | DRC | WNS, ORFS / Attar (ps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gcd | 412 | 0.75 | 812 | 792 | -2.5% | 0 | -37.7 / -35.4 |
| riscv32i | 10,106 | 0.749 | 55,422 | 53,144 | -4.1% | 0 | -16.1 / -48.9 |
| aes | 12,420 | 0.71 | 50,810 | 50,203 | -1.2% | 0 | -21.0 / -11.2 |
| ibex | 21,076 | 0.594 | 89,254 | 85,609 | -4.1% | 0 | -13.7 / -45.7 |
| tinyRocket | 32,292 | 0.506 | 157,727 | 151,074 | -4.2% | 0 | -90.1 / -100.3 |
| jpeg | 57,891 | 0.75 | 168,997 | 167,280 | -1.0% | 0 | -76.1 / -87.5 |
| ethmac | 59,653 | 0.79 | 196,952 | 196,594 | -0.2% | 0 | -75.9 / -80.2 |
| swerv_wrapper | 114,086 | 0.646 | 916,214 | 864,815 | -5.6% | 0 | -57.3 / -370.8 |
| cva6 | 124,618 | 0.84 | 642,595 | 595,183 | -7.4% | 0 | -57.1 / -32.2 |
Routed wirelength in um. tinyRocket runs with `LEC_CHECK=0`, as its ORFS baseline does: ORFS's equivalence checker
cannot read its generated memory macro. Attar's placements are not timing-driven; ORFS's are.
</details>

## Benchmarks

The zoo is distilled from public benchmarks, ORFS runs and Innovus runs on i3dPDK_NS3K, and published on the
Hugging Face Hub. Each reader hands over only the facts its format has. **193 designs in 17 suites, in two modes** (`attar.load('<mode>/<suite>/<design>')`).

<details open>
<summary><b><code>after_fp</code>: placement problems (138 designs, 11 suites)</b></summary>
<br>

The floorplan and power grid are given; the cells are the answer.

| suite | format | designs | cells | starts from | status |
| --- | --- | ---: | --- | --- | --- |
| ispd2005 | Bookshelf | 8 | 211k-2.2M | cells unplaced | placed legally |
| ispd2015 | LEF/DEF | 20 | 29k-1.3M | cells unplaced | placed legally (fence regions not enforced yet) |
| ispd2018, ispd2019 | LEF/DEF | 20 | 8k-900k | placed | re-placed legally |
| ispd2024 | LEF/DEF (Nangate45) | 5 | 120k-715k | placed | re-placed legally |
| iccad2025 | LEF/DEF (ASAP7) | 13 | 2k-298k | placed | re-placed legally |
| gsrc_place | Bookshelf | 4 | 0.8k-28k | cells unplaced | placed legally |
| ns3k | LEF/DEF (3nm-class) | 16 | 0.3k-170k | cells unplaced | placed legally; OpenROAD `check_placement` passes |
| orfs_asap7 | LEF/DEF from ORFS 3_3 | 33 | 0.4k-1.48M | cells unplaced | placed legally; `check_placement` passes; 9 routed by ORFS with 0 DRC |
| orfs_nangate45 | LEF/DEF from ORFS 3_3 | 18 | 0.7k-209k | cells unplaced | placed legally; `check_placement` passes |
| stress | LEF/DEF (3nm-class) | 1 | 1.18M | placed | bp_quad: re-placed legally; `check_placement` passes |

orfs_asap7 holds ORFS's asap7 designs, eight nangate45 designs ported to ASAP7 with the RTL unmodified (ariane133/136,
black_parrot, bp_be/fe/multi_top, bp_quad, swerv, dynamic_node; `tools/orfs_port_asap7.py`) and jpeg_x20 (20 x jpeg).

</details>

<details>
<summary><b><code>before_fp</code>: floorplan problems (55 designs, 6 suites)</b></summary>
<br>

The netlist is given; the die (utilization), IO pins and macros are the answer.

| suite | format | designs | starts from | status today |
| --- | --- | ---: | --- | --- |
| orfs_asap7 | LEF/DEF from ORFS 2_1 | 5 | die initialized, IO pins and macros unplaced, no PDN | floorplanned at a looser and a tighter utilization than ORFS's; handed to ORFS, which draws the PDN and places: all pass ([checked](#before_fp-hand-off)) |
| mcnc | YAL | 5 | blocks unplaced | floorplanned, no overlaps (no cell library: Attar is the end point) |
| gsrc_hard | GSRC blocks | 6 | no die | die sized, floorplanned, no overlaps |
| ispd2006_hb | Bookshelf | 28 | hard + soft blocks | runs; some block overlaps remain |
| gsrc_soft, mcnc_soft | GSRC blocks | 11 | soft blocks only | block shapes are part of the answer: no engine yet |

</details>

Details, data quirks and how to add a reader: [docs/benchmarks.md](docs/benchmarks.md).

## A design is a state

```text
>>> d = attar.load('after_fp/orfs_asap7/aes'); d.status()
aes_cipher_top  (orfs/asap7, fidelity F1)
  placement target density 0.71 (orfs 3_3 global_place (GPL-0023))
Problem facts
  ✅ netlist           def        12268 nets, 42885 pins
  ✅ cell_geometry     lef        12420 cells (12009 movable, 0 macros), 56 masters
  ✅ floorplan.die     def        1 die(s), 48.71 x 48.71 um
  ✅ floorplan.io      def        388 IO pins placed
  ✅ rows              def        164 rows, height 0.27 um
  ✅ pdn               def        1815 wire and 4785 via shapes on M1, M2, M3
  ✅ def_frame         def        PDN, pins, fixed cells of the source DEF, kept verbatim (849 kB)
  ✅ slow_loop         orfs:asap7 ORFS aes, stage 3_3_place_gp
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

## Roadmap

<p align="center"><img src="docs/images/roadmap.svg" alt="HPWL evaluation is done; routed wirelength evaluation is half way; timing evaluation has not started" width="100%"></p>

A proxy is lit once it is fast in Attar **and** shown to agree with the slow loop.

| | status | what is there | next |
| --- | --- | --- | --- |
| **HPWL evaluation** | done | exact on the GPU; equals OpenROAD; legality equals `check_placement` | |
| **Routed wirelength evaluation** | half way | ORFS hand-off and round trip; HPWL tracks routed WL above a 2% difference | a global-routing proxy, checked against ORFS routing |
| **Timing evaluation** | not yet | Liberty and SDC paths recorded with each design | a GPU timing proxy, checked against ORFS STA |

Also on the list:
- **`before_fp`**: a macro legalizer that leaves no channel narrower than the flow allows (ORFS:
  `RTLMP_MIN_CHANNEL_SIZE`), and utilization with headroom for the flow's own netlist growth (both behind the two
  failed hand-offs above); then ORFS up to its PDN (2_4) distilled back as an `after_fp` problem, so a floorplan
  and a placement can be solved one after the other; `before_fp` designs of the other ORFS designs;
- routability: DREAMPlace's cell inflation (RUDY branch) is in global placement (`routability=True`), with the track
  capacity ORFS's router itself uses; the pin-density branch is next;
- libraries without a one-site filler (i3dPDK_NS3K): OpenROAD forbids one-site gaps there, the NS3K flow does not;
- multi-row cells in the PDN row DP (today they fall back to a Python loop: about a minute on bp_quad);
- a legalizer that leaves an already-legal placement alone (today it re-spreads it: +12% HPWL on DREAMPlace's adaptec1 result);
- global placement that sees the gaps of rows cut around macros;
- fence regions and placement blockages (stored today only as counts);
- a block-level floorplanner for soft blocks (IBM-HB+, GSRC soft);
- 3D designs (per-die geometry) and a hosted zoo.

## Repository layout

```text
attar/            the package: Design, Evaluator, readers, kernels, attar.naive engines
examples/         quickstart_placement.py, quickstart_floorplan.py, floorplan_flow.py (the floorplan engines step by step)
tools/            distillation, ORFS round trip and hand-off, OpenROAD checks, validation, benchmarks, README figures
tests/            77 tests (python -m unittest discover -s tests); designs missing from the zoo are skipped
docs/             concepts, benchmarks, slow loop, API, the project spec (zh-TW), validation results, images
```

## Acknowledgements

Attar's global placement follows ePlace, DREAMPlace and Xplace, and its routability option DREAMPlace's cell
inflation; its rail and PDN checks follow OpenROAD's detailed placer, and its PDN repair the row DP of pa_pdn.
The slow loop is [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) on the ASAP7
PDK. Benchmarks are from the ISPD, ICCAD and GSRC contests, MCNC and ORFS.

## License

[MIT](LICENSE)
