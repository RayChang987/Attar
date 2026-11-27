<div align="center">

<img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/logo.png" alt="Attar: GPU-accelerated VLSI physical design in PyTorch" width="640">

Diverse P&R benchmarks, distilled into one plug-and-play PyTorch environment on the GPU.<br>
Evolve VLSI placement, timing-optimization and routing algorithms in minutes, not days.

[![Python](https://img.shields.io/badge/python-3.8%2B-d9622b?style=flat-square)](https://github.com/RayChang987/Attar/blob/main/pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-d99a1e?style=flat-square)](https://pytorch.org)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/attar?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/attar)
[![Validated](https://img.shields.io/badge/validated%20against-OpenROAD-8f6a3f?style=flat-square)](https://github.com/RayChang987/Attar/blob/main/docs/features.md#evaluators)
[![Tests](https://img.shields.io/badge/tests-192%20passing-5f7f1e?style=flat-square)](https://github.com/RayChang987/Attar/tree/main/tests)
[![License](https://img.shields.io/badge/license-MIT-a8946f?style=flat-square)](https://github.com/RayChang987/Attar/blob/main/LICENSE)

**English** &nbsp;·&nbsp; [繁體中文](README.zh-TW.md) &nbsp;·&nbsp; [简体中文](README.zh-CN.md)

[Quick start](#quick-start) &nbsp;·&nbsp; [Why Attar](#why-attar) &nbsp;·&nbsp; [What's inside](#whats-inside) &nbsp;·&nbsp;
[How it works](#how-it-works) &nbsp;·&nbsp; [Gallery](#gallery)

</div>

## Quick start

```bash
pip install attar
```

```python
import attar
d = attar.load('after_fp/orfs_asap7/aes', device='cuda')   # from the zoo, as tensors on the GPU
d.place()                                                   # or set d.pos from your own algorithm
print(attar.Evaluator(d).hpwl(), d.timing().wns)            # HPWL, and WNS as OpenSTA reports it
d.export('aes.def')                                         # back to OpenROAD, Innovus, ICC2, ...
```

Designs download from the [Attar zoo](https://huggingface.co/datasets/RayChang987/attar-zoo) on first use. The first
timing run on a GPU compiles its kernels (about two minutes, once per machine; in Colab, once per runtime unless
`TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` point to a folder that persists). No GPU? Everything also runs on the CPU, or try
the examples in Colab:

| notebook | what it runs |
| --- | --- |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayChang987/Attar/blob/main/examples/colab/timing_driven_colab.ipynb) | Timing-driven placement in fifteen lines, the demo below: place cva6 for wirelength, time it, weight nets by slack as OpenROAD does, place again |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayChang987/Attar/blob/main/examples/colab/latte_on_attar.ipynb) | **[DAC 2026]** LATTE, timing-driven detailed placement, step by step: a differentiable timer anchored on Attar's, independent sets, Adam with CAWP, judged after Attar's global routing, on an ICCAD 2025 contest design |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayChang987/Attar/blob/main/examples/colab/p11_attar_superblue18.ipynb) | **[DAC 2026]** Hybrid net-pin weighting for timing-driven global placement, step by step: DATE 2025's pin pairs plus net weights derived from a delay model, built on Attar's timer and placer, on superblue18 (745k cells) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayChang987/Attar/blob/main/examples/colab/sroa_on_attar.ipynb) | **[ICCAD 2025]** SROA, GPU detailed placement for quality, step by step: every permutation of a few neighbouring cells, a DP over the rows, HPWL then MST wirelength, judged by Attar's global routing, on ispd19_test1 |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RayChang987/Attar/blob/main/examples/colab/date25_attar_superblue18.ipynb) | **[DATE 2025]** Pin-pair weighting for timing-driven global placement, step by step, on superblue18 |

<div align="center">
<img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/demo_placement.gif" alt="Timing-driven placement written in fifteen lines of Python, run on cva6 and swerv: place for wirelength, time it, weight the nets by their slack as OpenROAD's global placement does, place again, time it" width="920">

<sub>Timing-driven placement in 15 lines (<a href="examples/timing_driven.py">examples/timing_driven.py</a>):
cva6 WNS -195 → -75 ps, in 2.2 s.</sub>
</div>

## Why Attar

- **Fast.** Timing inside the placement loop 21-54x faster than OpenROAD kept warm; a 2.2M-cell design placed
  and legalized in 49 s. An idea, from your code or an AI agent's, is tested in seconds, not in an overnight
  P&R run.

<p align="center"><img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/speed_vs_openroad.png" alt="Seconds per timing update in global placement, OpenROAD kept warm vs Attar on the GPU: netcard_fast 7.2 vs 0.34 s, superblue18 13.0 vs 0.56 s, superblue1 19.6 vs 0.83 s, superblue7 33.6 vs 1.2 s; Attar is 21-27 times faster per GP update and 21-54 times faster per DP round" width="860"></p>

<p align="center"><sub>Timing after every move, as a timing-driven placer needs it, against OpenROAD with the design already loaded: same machine, same rounds, TNS within 0.13%. <a href="https://github.com/RayChang987/Attar/blob/main/docs/validation/speed_vs_openroad.md">How it was measured</a>.</sub></p>

- **One environment.** 262 designs from 24 benchmark suites (ISPD 2005-2024, ICCAD 2015 and 2025, MCNC, GSRC,
  IBM-HB+, OpenROAD-flow-scripts) load with one call, in one format, from LEF/DEF, Bookshelf or YAL. An algorithm
  is a function from tensors to positions: no database API, no TCL.
- **Faithful.** Against OpenROAD and OpenSTA on the same placement: HPWL within 0.00002%, timing (WNS) within
  0.34%, power within 0.06%, global-routing wirelength within 3.6%.

## What's inside

<p align="center"><img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/roadmap.svg" alt="Roadmap in three cards. Data, 262 designs in the zoo: contest benchmarks ISPD 2005 (8), ISPD 2015 (20), ISPD 2018 (10), ISPD 2019 (10), ISPD 2024 (5), ICCAD 2015 (8), ICCAD 2025 (13), GSRC placement (4) for placement; MCNC (10), GSRC blocks (12), IBM-HB+ (28) for floorplan; ASAP7 (15 placement, 5 floorplan), Nangate45 (8 and 8) and NS3K (98 placement). Engines, done: die sizing, packing, macro placement, IO placement, global placer, legalizer, PDN-aware refinement, detailed placement, clock tree synthesis, global router, gate sizing, buffering and cloning, debuffering; not yet: delay-balanced clock trees. Evaluation, done: HPWL, Steiner wirelength, legality, displacement, density overflow, RUDY, global routing overflow, setup and hold timing, GR parasitics, propagated clocks and CRPR, slew, capacitance and fanout limits, power, blocked pin checking. Left to your P&amp;R tool: PDN synthesis, detailed routing, routing DRC, sign-off" width="100%"></p>

Every engine and evaluator, with its accuracy against OpenROAD and OpenSTA:
[docs/features.md](https://github.com/RayChang987/Attar/blob/main/docs/features.md).

## How it works

<p align="center"><img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/flow.svg" alt="Sources (LEF/DEF, Bookshelf, YAL, GSRC blocks) are distilled into the Attar IR and loaded as a Design on the GPU; attar.naive moves it and adds facts, the Evaluator measures it; export writes a DEF for your own flow (OpenROAD, Innovus, ICC2, ...)" width="100%"></p>

Attar splits physical design at the power grid, as production flows do:

- **`before_fp`**: given a netlist, find the die, IO pins and macros (`d.floorplan()`).
- **`after_fp`**: given a fixed floorplan with its PDN, place the cells, then size, buffer, route and time them
  (`d.place()`).

Every design states the facts it has and where each came from, and every engine says which facts it needs:

```text
>>> d.status()
  ✅ netlist           def             12268 nets, 42885 pins
  ✅ cell_geometry     lef             12420 cells (12009 movable, 0 macros), 56 masters
  ✅ pdn               def             1815 wire and 4785 via shapes on M1, M2, M3
  ...
  ❌ placement         12009 movable cells unplaced
  -> only a placement is missing: place the design, then d.export("out.def")
```

Docs: [concepts](https://github.com/RayChang987/Attar/blob/main/docs/concepts.md) ·
[engines and evaluators](https://github.com/RayChang987/Attar/blob/main/docs/features.md) ·
[benchmarks](https://github.com/RayChang987/Attar/blob/main/docs/benchmarks.md) ·
[API](https://github.com/RayChang987/Attar/blob/main/docs/api.md) ·
[validation](https://github.com/RayChang987/Attar/tree/main/docs/validation)

## Gallery

<table>
<tr>
<td colspan="2"><img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/demo_floorplan.gif" alt="Floorplan mode: GSRC n100 at 85% block area, floorplanned twice: GP plus LP legalization leaves 5 macros outside the core (not legal), B*-tree packing fits them all (legal)"></td>
</tr>
<tr>
<td colspan="2"><sub><b>Floorplanning</b> GSRC n100: B*-tree packing fits all 100 blocks into the core in 5.1 s.</sub></td>
</tr>
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/gallery/eval_grov.png" alt="cva6 after placement, colored by global-routing load on OpenROAD's gcells"></td>
<td width="50%"><img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/gallery/eval_pin_access.png" alt="ariane136 legalized with OpenROAD dpl's PDN rule only: 1,254 signal pins the router cannot reach past the PDN, spread over the standard-cell area"></td>
</tr>
<tr>
<td><sub><b>Routing congestion</b> cva6, placed in 2.4 s.</sub></td>
<td><sub><b>Pin access</b> ariane136 legalized with OpenROAD dpl's PDN rule only: 1,254 pins the router cannot reach past the power grid (signal-to-PG DRVs after Innovus routing). Attar's legalizer leaves none.</sub></td>
</tr>
<tr>
<td><img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/gallery/eval_placed.png" alt="ariane136 legalized: legal, rails and PDN included"></td>
<td><img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/gallery/fp_n300.png" alt="GSRC n300 floorplanned by B*-tree packing, with the strongest connections between blocks and IO pins"></td>
</tr>
<tr>
<td><sub><b>Legal placement</b> ariane136, rails and PDN included, in 2.2 s.</sub></td>
<td><sub><b>Floorplan</b> GSRC n300: 300 blocks, 569 IO pins, packed in 43 s.</sub></td>
</tr>
</table>

## Acknowledgements

Attar's placer follows ePlace, [DREAMPlace](https://github.com/limbo018/DREAMPlace) and [Xplace](https://github.com/cuhk-eda/Xplace); its timer follows [OpenSTA](https://github.com/the-openroad-project/opensta) and its checks follow OpenROAD.
It is validated against [OpenROAD-flow-scripts](https://github.com/the-openroad-project/openroad-flow-scripts) on ASAP7 and Nangate45. Benchmarks are from the ISPD, ICCAD and GSRC contests, MCNC and ORFS.

## License

<div align="center">
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-a8946f?style=for-the-badge" alt="MIT License"></a>
<br><br>
<sub>Copyright &copy; 2026 RayChang987 &nbsp;·&nbsp; free to use, modify and distribute; see <a href="LICENSE">LICENSE</a></sub>
<br><br>
<img src="https://raw.githubusercontent.com/RayChang987/Attar/main/docs/images/logo.png" alt="Attar" width="180">
</div>
