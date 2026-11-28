# Global routing: parasitics and router against OpenROAD

Truth: OpenROAD (ORFS image `openroad/orfs:26Q4-368-g9ca2a9e76`) on the same netlist and placement: the zoo design
`after_fp/orfs_<platform>` at ORFS's `3_3_place_gp`. OpenROAD routes with the flow's settings (routing layers from
the manifest, the platform's `fastroute.tcl`, `global_route -congestion_iterations 30 -allow_congestion`) but without
`pin_access`, so pins connect from their shapes, as in the IR (`tools/gr_truth.py`, cached in `orfs_runs/gr_par/`).

## 1. Parasitics from global routes (`Timer(parasitics='global_routing')`, `d.timing(routes=...)`)

`attar.timing.grpar` ports OpenROAD's `estimate_parasitics -global_routing` (`est::MakeWireParasitics`) and OpenSTA's
reduction (`reduceToPiElmore`). Input: OpenROAD's routes (`write_global_route_segments`) and grt's pin model
(`attar.route.pins_from_shapes`, checked against `write_pin_locations`). Truth: OpenSTA after `estimate_parasitics
-global_routing` on the same routes, and its SPEF (`tools/validate_gr_parasitics.py`).

| design | pins | pin gcell = grt | loop nets | wire C (per net / total) | wire R (per net / total) | setup Pearson | setup MAE ps | max ps | WNS ours / OpenSTA ps | TNS err | hold MAE ps | re-time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| asap7 aes | 42,885 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -9.38 / -9.38 | 0.00% | 0.00 | 26 ms |
| asap7 gcd | 1,218 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -34.89 / -34.89 | 0.00% | 0.00 | 10 ms |
| asap7 jpeg | 184,248 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -95.50 / -95.50 | 0.00% | 0.00 | 79 ms |
| asap7 ethmac | 193,857 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -84.87 / -84.87 | 0.00% | 0.00 | 67 ms |
| asap7 cva6 | 420,842 | 100% | 2 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.005 | 18.69 | -62.81 / -62.81 | 0.00% | 0.00 | 202 ms |
| asap7 swerv | 365,817 | 100% | 2 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -110.48 / -110.48 | 0.00% | 0.00 | 216 ms |
| asap7 dynamic_node | 49,055 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -35.19 / -35.19 | 0.00% | 0.00 | 45 ms |
| asap7 riscv32i-mock-sram | 35,930 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -64.49 / -64.49 | 0.00% | 0.00 | 56 ms |
| asap7 picorv32 | 64,774 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -136.16 / -136.16 | 0.00% | 0.00 | 53 ms |
| nangate45 gcd | 1,947 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -153.90 / -153.90 | 0.00% | 0.00 | 17 ms |
| nangate45 ibex | 55,364 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | 29.56 / 29.55 | – | 0.00 | 39 ms |
| nangate45 jpeg | 193,096 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -82.88 / -82.88 | 0.00% | 0.00 | 59 ms |
| nangate45 tinyRocket | 101,959 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -117.33 / -117.33 | 0.00% | 0.00 | 46 ms |
| nangate45 picorv32 | 60,295 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -311.20 / -311.20 | 0.00% | 0.00 | 31 ms |

Re-time: one `Timer.run()` on a shared RTX 3080 Ti (Attar 1.0.6; `--device cuda`, the script defaults to the CPU).
The cva6 residual comes from its two nets whose route forms a resistor loop: OpenSTA reduces a loop over a
depth-first tree, Attar over a breadth-first one. Nangate45 cut resistances are the LEF's (`RESISTANCE`, zoo
`routing.cut_res`); the table uses the zoo alone (no `via_res=`, no `adjustment=`).

## 2. The global router (`d.global_route()`, `attar.route.global_route`)

Not a FastRoute port. Gcell grid and edge capacities are OpenROAD's (exact per layer; Nangate45 with its
fastroute.tcl per-layer adjustments). Per net: OpenROAD's Prim-Dijkstra tree (alpha 0) over its pins' gcells, Z / L
pattern routes with negotiated rip-up and reroute, then chunked layer assignment (runs of up to 10 gcells) and via
stacks.

`tools/validate_router.py` compares Attar's routes with OpenROAD's on the same placement. WL is as grt reports it
(each segment + one gcell). Per-net figures cover nets OpenROAD routes beyond two gcells. Overflow: Attar's 3D total;
OpenROAD's from its final congestion report. Timing: Attar's timer on Attar's routes vs OpenSTA on OpenROAD's
routes. Times: Attar on an RTX 3080 Ti after a warm-up run; OpenROAD's `global_route` with 4 threads.

| design | WL | WL OpenROAD | WL err | per-net Pearson | per-net median / p90 err | overflow (Attar / OpenROAD) | vias / OpenROAD's | time / OpenROAD's | WNS on Attar routes / OpenSTA | slack Pearson | slack MAE ps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| asap7 aes | 65,634 | 66,329 | -1.05% | 0.999 | 0.0% / 12.5% | 17 / 0 | 72,444 / 66,998 | 0.65 s / 1.29 s | -10.9 / -9.4 | 0.9996 | 1.51 |
| asap7 gcd | 1,071 | 1,096 | -2.28% | 0.998 | 0.0% / 20.7% | 0 / 0 | 1,601 / 1,608 | 0.16 s / 0.11 s | -34.7 / -34.9 | 1.0000 | 0.38 |
| asap7 jpeg | 199,560 | 200,857 | -0.65% | 1.000 | 0.0% / 12.5% | 16 / 0 | 233,710 / 226,541 | 0.69 s / 11.09 s | -105.3 / -95.5 | 0.9980 | 4.72 |
| asap7 ethmac | 238,225 | 241,005 | -1.15% | 1.000 | 0.0% / 12.5% | 24 / 0 | 263,281 / 253,609 | 0.81 s / 28.94 s | -83.0 / -84.9 | 0.9999 | 0.97 |
| asap7 cva6 | 690,789 | 699,395 | -1.23% | 1.000 | 0.0% / 11.1% | 341 / 0 | 646,052 / 595,922 | 1.50 s / 37.26 s | -63.7 / -62.8 | 0.9999 | 2.02 |
| asap7 swerv | 812,689 | 817,743 | -0.62% | 1.000 | 0.0% / 8.3% | 363 / 0 | 627,352 / 571,891 | 1.42 s / 57.60 s | -111.6 / -110.5 | 0.9994 | 6.19 |
| asap7 dynamic_node | 65,347 | 66,088 | -1.12% | 1.000 | 0.0% / 12.7% | 27 / 0 | 68,739 / 64,835 | 0.83 s / 2.26 s | -35.7 / -35.2 | 0.9942 | 3.62 |
| asap7 riscv32i-mock-sram | 63,232 | 64,584 | -2.09% | 0.999 | 0.0% / 11.1% | 12 / 0 | 58,755 / 54,074 | 0.92 s / 1.83 s | -72.6 / -64.5 | 0.9998 | 3.95 |
| asap7 picorv32 | 90,601 | 91,390 | -0.86% | 1.000 | 0.0% / 14.3% | 10 / 0 | 93,221 / 89,952 | 0.87 s / 3.14 s | -137.8 / -136.2 | 0.9999 | 0.72 |
| nangate45 gcd | 5,529 | 5,580 | -0.91% | 0.998 | 0.0% / 0.0% | 0 / 0 | 2,408 / 2,419 | 0.26 s / 0.14 s | -154.5 / -153.9 | 0.9999 | 0.42 |
| nangate45 ibex | 329,776 | 334,786 | -1.50% | 1.000 | 0.0% / 9.1% | 65 / 0 | 86,978 / 80,346 | 1.02 s / 1.92 s | -8.6 / 29.6 | 0.9965 | 15.18 |
| nangate45 jpeg | 736,115 | 745,733 | -1.29% | 1.000 | 0.0% / 10.0% | 78 / 0 | 247,569 / 236,900 | 1.11 s / 6.33 s | -86.7 / -82.9 | 0.9991 | 4.89 |
| nangate45 tinyRocket | 636,048 | 645,601 | -1.48% | 1.000 | 0.0% / 9.1% | 149 / 0 | 155,836 / 142,837 | 1.13 s / 4.95 s | -116.3 / -117.3 | 0.9996 | 4.82 |
| nangate45 picorv32 | 320,140 | 323,942 | -1.17% | 1.000 | 0.0% / 10.0% | 21 / 0 | 88,244 / 83,961 | 0.88 s / 1.84 s | -310.0 / -311.2 | 0.9998 | 2.36 |

Wirelength is 0.6-2.3% below OpenROAD's on every design (FLUTE trees vs Prim-Dijkstra). Per-net Pearson is
0.998-1.000; half the nets are exact. Attar is 16-41x faster on jpeg, ethmac, cva6 and swerv, and slower on gcd.
Limits: Attar leaves overflow where OpenROAD has none (no maze stage; a chunk keeps one layer) and uses 2-10% more
vias. Slacks differ by a few ps per endpoint: WNS within 10 ps on 12 of 14; nangate45 ibex 38 ps apart, as its worst
paths take other layers.

## 3. Ranking placements as OpenROAD's router does

`tools/rank_router.py`: 7 placements of the same netlist per design: ORFS's global placement, the same with 10% of
the cells moved, and Attar's `place()` at target density 0.60 / 0.75 (two seeds) / 0.90 / 0.95. OpenROAD and Attar
route each. 7 designs (asap7 aes, jpeg, dynamic_node, picorv32, riscv32i-mock-sram; nangate45 ibex, picorv32), 147
pairs. A pair agrees when a predictor orders the two placements as OpenROAD's wirelength does. Pairs are binned by
how far apart OpenROAD puts them.

| OpenROAD GR wirelength difference | pairs | HPWL agrees | Attar's router agrees |
|---|---|---|---|
| under 1% | 22 | 72.7% | **90.9%** |
| 1-2% | 19 | 73.7% | **100%** |
| 2-5% | 49 | 87.8% | **100%** |
| 5% and more | 57 | 100% | **100%** |

Attar's wirelength is 0.5-3.6% below OpenROAD's on all 49 placements; the offset is steady, so the order holds.
