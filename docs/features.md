# Engines and evaluators

What Attar does to a design and how it measures one. [api.md](api.md) lists the arguments.

## Engines

| Engine | Mode | Call | What it does |
| --- | --- | --- | --- |
| **Die sizing** | `before_fp` | `naive.make_die` | die and rows for a target utilization and aspect ratio |
| **Packing** | `before_fp` | `naive.place_macros(packing=True)` | block-level designs: B*-tree packing, annealed into the core |
| **Macro placement** | `before_fp` | `naive.place_macros` | mixed-size global placement, then macros legalized on their tracks with a halo |
| **IO placement** | `before_fp` | `naive.place_io` | IO pins on the die boundary, net-aware once cells are placed |
| **Placer** | both | `naive.global_place` | electrostatic (ePlace / DREAMPlace style, Nesterov, CUDA graphs); `routability=True` inflates cells; net weights may change during placement. Legalized HPWL is 2.2-6.6% above DREAMPlace's on ISPD 2005 |
| **Legalizer** | `after_fp` | `naive.legalize` | Abacus (C++): multi-row-height cells, rail parity, signal pins clear of the PDN; incremental with `cells=` |
| **PDN-aware refinement** | `after_fp` | inside `naive.legalize` | row DP that keeps cells clear of the PDN and their signal pins reachable (`pin_access=True`); NS3K u80: unreachable pins 1,235 -> 0 for +0.12% HPWL |
| **Detailed placement** | `after_fp` | `naive.detailed_place` | OpenROAD dpo's swaps, reordering and flipping, batched on the GPU; stays legal |
| **Global router** | `after_fp` | `d.global_route()` | OpenROAD's gcell grid and capacities, pattern routing, layer assignment; `previous=routes` reroutes only the nets that moved |
| **Clock tree synthesis** | `after_fp` | `naive.cts` | buffered clock trees; against TritonCTS on 4 designs: 1.4-2.7x less spread, 2.3-3.8x less clock power |
| **Gate sizing** | `after_fp` | `d.resize`, `d.equivalents` | swap cells within their equivalence class, as OpenROAD's `replace_cell` |
| **Buffering and cloning** | `after_fp` | `d.edit()` | `insert_buffer`, `clone`, `add_cell`, `connect`, ... in one batch |
| **Debuffering** | `after_fp` | `d.debuffer()` | removes buffers and double inversions, as OpenROAD's `remove_buffers` |

## Evaluators

Each is compared with OpenROAD, OpenSTA or Innovus on the same netlist and placement; full tables in
[validation/](validation/).

| Evaluator | Call | Compared with | Accuracy |
| --- | --- | --- | --- |
| **HPWL** | `ev.hpwl` | OpenROAD; 23 ORFS designs up to 1.05M cells | within **0.00002%** ([table](validation/orfs_check.md)) |
| **Legality** | `ev.legality` | OpenROAD `check_placement` | same verdict on all **23** ORFS designs and **62** NS3K floorplans |
| **Timing** | `d.timing()` | OpenSTA, placement parasitics; 31 designs up to 1.05M cells | WNS error at most **0.00%** (ORFS), **0.01%** (NS3K), **0.33%** (ICCAD 2015 superblue) ([table](validation/timing_routing.md)); 5-30x faster |
| **Pin and net slack** | `.pin_slack()`, `.net_slack()` | OpenSTA `slack_max`; 8 designs | Pearson at least **0.9995**; MAE at most **1.03 ps** on negative-slack pins |
| **Timing after sizing** | `d.resize` + `d.timing()` | OpenROAD `replace_cell` + OpenSTA; 16 designs | WNS error at most **0.23%** |
| **Timing after netlist edits** | `d.edit()`, `d.debuffer()` | OpenSTA on the exported DEF; 20 designs | WNS error at most **0.04%** (debuffer), **1.18%** (NS3K edits) |
| **Design rules** | `d.drv()` | OpenSTA `report_check_types`; 21 designs | same violating pins; capacitance slack within **1.1 fF** ([table](validation/drv_power_cts.md)) |
| **Power** | `d.power()` | OpenSTA `report_power`; 20 designs | total within **0.01%**, every instance equal |
| **Propagated clocks and CRPR** | `d.propagate_clocks()`, `d.timing(crpr=...)` | OpenSTA on TritonCTS trees; 4 designs | setup slack MAE at most **0.03 ps** |
| **Global routing OVF** | `d.gr_overflow()` | OpenROAD capacities and global route; 13 designs | capacities exact without macros, within **0.16%** with them; Spearman **0.97** |
| **Global-routing parasitics** | `d.timing(routes=...)` | OpenSTA on OpenROAD's routes; 14 designs | wire R and C equal OpenROAD's SPEF; WNS error at most **0.08%** ([table](validation/global_routing.md)) |
| **Global routing** | `d.global_route()` | OpenROAD `global_route`; 14 designs | wirelength **0.5-3.6%** below; orders placements as OpenROAD does in **100%** of pairs 1% or more apart; 16-41x faster |
| **Pin access** | `ev.pin_access` | Innovus `verify_drc`; 988 routed NS3K runs | blocked pins = pins with a signal-to-PG M1/M2 DRV (**156,665** of 156,665) ([table](validation/pin_access.md)) |
| **Steiner wirelength, RUDY, density overflow** | `ev.steiner_wirelength`, `ev.rudy`, `ev.density_overflow` | not compared yet | |
