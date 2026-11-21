# Engines and evaluators

What Attar does to a design and how it measures one. The [roadmap](../README.md#whats-inside) shows the same at a
glance; [api.md](api.md) lists the arguments.

## Engines

`attar.naive` and the design's own edits. `d.floorplan(utilization=...)` is die sizing, IO placement and macro
placement in one call; `d.place()` is global placement plus legalization.

| Engine | Mode | Call | What it does |
| --- | --- | --- | --- |
| **Die sizing** | `before_fp` | `naive.make_die` | a die and rows for a target utilization and aspect ratio |
| **Packing** | `before_fp` | `naive.place_macros(packing=True)` | block-level designs: B*-tree packing, annealed into the core, then for HPWL |
| **Macro placement** | `before_fp` | `naive.place_macros` | mixed-size global placement, then macros legalized on their routing tracks with a halo |
| **IO placement** | `before_fp` | `naive.place_io` | IO pins on the die boundary, net-aware once cells are placed |
| **Placer** | both | `naive.global_place` | electrostatic (ePlace/DREAMPlace style, Nesterov, CUDA graphs) at the source flow's own target density; `routability=True`: DREAMPlace's cell inflation; net weights a callback changes are used from the next iteration, with DREAMPlace's net-weighted preconditioner. Known gap: in a timing-driven loop (DATE'25 net weighting ported, superblue) the final TNS was about 1.3x worse than with DREAMPlace's global placer; legalized HPWL is 2.2-6.6% above DREAMPlace's on ISPD 2005 |
| **Legalizer** | `after_fp` | `naive.legalize` | multi-row-height cells, power-rail parity, signal pins clear of the PDN's via metal; incremental on a placement to repair (`cells=`: only those move; a legal input comes out unchanged). Abacus (C++) by default (`method='columns'`: the earlier column balancing), so cells global placement leaves on fixed macros go to the nearest free space; on superblue18 after one global placement, those 8,171 cells moved 16.7 um on average with Abacus instead of 161 um with columns (max 0.3 mm instead of 3.1 mm), HPWL +2.7% instead of +10.1%, WNS -12.3 ns instead of -24.1 ns |
| **PDN-aware refinement** | `after_fp` | inside `naive.legalize` | row DP and repair that keep cells clear of the PDN |
| **Detailed placement** | `after_fp` | `naive.detailed_place` | OpenROAD dpo's global swap, vertical swap, local reordering and cell flipping, batched on the GPU; keeps the placement legal |
| **Global router** | `after_fp` | `d.global_route()` | on OpenROAD's gcell grid and capacities: Prim-Dijkstra trees, negotiated Z / L pattern routing, chunked layer assignment; time on its routes with `d.timing(routes=...)` |
| **Clock tree synthesis** | `after_fp` | `naive.cts` | buffered clock trees on every ideal clock net (OpenROAD's clock buffers, TritonCTS's clustering, repeaters on long edges), balanced level by level on the propagated-clock timer; against TritonCTS on 4 ORFS designs: 1.4-2.7x less spread, 2.3-3.8x less clock power ([table](validation/drv_power_cts.md)) |
| **Gate sizing** | `after_fp` | `d.resize`, `d.equivalents` | swap cells within their equivalence class (drive strength, threshold voltage) in place, as OpenROAD's `replace_cell`; the timer updates without a rebuild |
| **Buffering and cloning** | `after_fp` | `d.edit()`: `insert_buffer`, `clone`, `add_cell`, `connect`, ... | netlist edits in one batch; no cell or net index moves; the next `d.timing()` sees them |
| **Debuffering** | `after_fp` | `d.debuffer()` | removes buffers and double inversions, as OpenROAD's `remove_buffers` |

## Evaluators

Each against OpenROAD or OpenSTA on the same netlist and placement; full tables in [validation/](validation/).

| Evaluator | Call | Compared with | Accuracy |
| --- | --- | --- | --- |
| **HPWL** | `ev.hpwl` | OpenROAD's HPWL from its database; 23 ORFS designs (ASAP7, Nangate45) up to 1.05M cells | within **0.00002%** on every design ([table](validation/orfs_check.md)) |
| **Legality** | `ev.legality` | OpenROAD `check_placement` | same verdict on all **23** ORFS designs and the **62** NS3K floorplans OpenROAD can check; placements broken by half a site rejected by both |
| **Timing** | `d.timing()` | OpenSTA after `estimate_parasitics -placement`; 31 designs (ASAP7, Nangate45, NS3K) up to 1.05M cells | WNS median error **0.00%** (ORFS) and **0.11%** (NS3K), max **0.34%**; hold slack MAE at most 0.36 ps; 5-30x faster than OpenSTA |
| **Pin and net slack** | `d.timing().pin_slack()`, `.net_slack()` | OpenSTA's `slack_max` of every pin; 8 designs (ASAP7, Nangate45) up to 384k pins | Pearson at least **0.9995**; on the pins with negative slack, MAE at most **1.03 ps**; 7-32 ms after a timing run |
| **Timing after sizing** | `d.resize` + `d.timing()` | OpenROAD `replace_cell` + OpenSTA, 20% of the cells swapped at random; 16 designs | WNS error at most **0.23%**, TNS at most 0.37% |
| **Timing after netlist edits** | `d.edit()`, `d.debuffer()` + `d.timing()` | OpenSTA on the DEF Attar exports: after 1,548 buffers and 443 clones inserted (10 NS3K designs up to 163k cells), and after `d.debuffer()` removed 45,991 buffers and 83 inverters (10 ORFS designs) | setup Pearson at least **0.9997**, WNS error at most **0.04%** after debuffer and **1.18%** after the NS3K edits; hold MAE at most 0.36 ps |
| **Design rules** | `d.drv()` | OpenSTA `report_check_types -max_slew -max_capacitance -max_fanout`; 21 ORFS designs (ASAP7, Nangate45) up to 3.4M pins | the same violating pins on every design; capacitance slack within **1.1 fF**; slew slack exact on about half of them, up to 18 ps on a few pins of adders with tied inputs ([table](validation/drv_power_cts.md)) |
| **Power** | `d.power()` | OpenSTA `report_power`, `-instances`; 20 ORFS designs | total within **0.01%**, every instance equal; up to 7.6x faster on the large ones ([table](validation/drv_power_cts.md)) |
| **Propagated clocks and CRPR** | `d.propagate_clocks()`, `d.timing(crpr=...)` | OpenSTA with `set_propagated_clock` on ORFS's own clock trees (TritonCTS), CRPR on and off; 4 designs | clock arrivals within 0.24 ps; setup slack MAE at most **0.03 ps**, WNS within 0.03 ps ([table](validation/drv_power_cts.md)) |
| **Global routing OVF** | `d.gr_overflow()` | OpenROAD's gcell capacities and global route; 13 ORFS designs | capacities exact without macros, within **0.16%** with them; ranks designs as OpenROAD does (Spearman **0.97**); a pre-route estimate: reads mild congestion high and heavy congestion low |
| **Global-routing parasitics** | `d.timing(routes=...)` | OpenSTA after `estimate_parasitics -global_routing` on OpenROAD's own routes; 14 ORFS designs (ASAP7, Nangate45) | every net's wire R and C equal OpenROAD's SPEF; WNS error at most **0.08%**, setup slack MAE at most **0.02 ps** (one latch pin aside) ([table](validation/global_routing.md)) |
| **Global routing** | `d.global_route()` | OpenROAD's `global_route` on the same placement; 14 designs, and 7 placements each of 7 | wirelength **0.5-3.6%** below OpenROAD's, per net Pearson at least 0.998; orders placements as OpenROAD's routed wirelength does in **100%** of pairs 1% or more apart (HPWL: 74-88%), 91% below 1%; 16-41x faster from jpeg up; leaves some overflow OpenROAD does not ([table](validation/global_routing.md)) |
| **Steiner wirelength, RUDY, density overflow** | `ev.steiner_wirelength`, `ev.rudy`, `ev.density_overflow` | not compared yet | |
