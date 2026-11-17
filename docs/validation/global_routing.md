# Global routing: parasitics and router against OpenROAD

All numbers below are OpenROAD (the ORFS image, `openroad/orfs:26Q4-368-g9ca2a9e76`) on the same netlist and
placement as Attar: the zoo design (`after_fp/orfs_<platform>`) at its reference placement (ORFS `3_3_place_gp`).
OpenROAD routes with the flow's settings (routing layers from the manifest, the platform's `fastroute.tcl`,
`global_route -congestion_iterations 30 -allow_congestion`) but without `pin_access`: pins then connect from their
shapes, the information the IR has (`tools/gr_truth.py`, cached in `orfs_runs/gr_par/`).

## 1. Parasitics from global routes (`Timer(parasitics='global_routing')`, `d.timing(routes=...)`)

`attar.timing.grpar` ports `est::MakeWireParasitics` (OpenROAD's `estimate_parasitics -global_routing`): a node per
route point and layer, each segment a resistor with its capacitance split to its ends (a gcell holding pins stands
for the median of those pins), vias as cut resistances, each pin tied to its gcell's node one layer above its
connection layer; then OpenSTA's reduction (`reduceToPiElmore`: admittance moments, pi model, Elmore per load).
Input: OpenROAD's own routes (`write_global_route_segments`, read by `attar.route.read_segments`) and grt's pin
model ported from its source (`attar.route.pins_from_shapes`, checked against `write_pin_locations`).
Truth: OpenSTA after `estimate_parasitics -global_routing` on the same routes, and its SPEF of them
(`tools/validate_gr_parasitics.py`).

| design | pins | pin gcell = grt | loop nets | wire C (per net / total) | wire R (per net / total) | setup Pearson | setup MAE ps | max ps | WNS ours / OpenSTA ps | TNS err | hold MAE ps | re-time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| asap7 aes | 42,885 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.001 | 0.02 | -9.37 / -9.38 | 0.07% | 0.00 | 267 ms |
| asap7 gcd | 1,218 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.001 | 0.00 | -34.89 / -34.89 | 0.00% | 0.00 | 8 ms |
| asap7 jpeg | 184,248 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.002 | 0.07 | -95.50 / -95.50 | 0.01% | 0.00 | 66 ms |
| asap7 ethmac | 193,857 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.020 | 0.28 | -84.87 / -84.87 | 0.00% | 0.00 | 57 ms |
| asap7 cva6 | 420,842 | 100% | 2 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.021 | 18.71 | -62.80 / -62.81 | 0.03% | 0.00 | 171 ms |
| asap7 swerv | 365,817 | 100% | 2 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.021 | 0.97 | -110.48 / -110.48 | 0.06% | 0.00 | 184 ms |
| asap7 dynamic_node | 49,055 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.005 | 0.11 | -35.19 / -35.19 | 0.24% | 0.00 | 41 ms |
| asap7 riscv32i-mock-sram | 35,930 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.023 | 0.22 | -64.45 / -64.49 | 0.09% | 0.00 | 46 ms |
| asap7 picorv32 | 64,774 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.003 | 0.07 | -136.16 / -136.16 | 0.02% | 0.00 | 44 ms |
| nangate45 gcd | 1,947 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -153.90 / -153.90 | 0.00% | 0.00 | 9 ms |
| nangate45 ibex | 55,364 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 0.99732 | 0.524 | 1055 | 29.56 / 29.55 | – | 0.00 | 32 ms |
| nangate45 jpeg | 193,096 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.01 | -82.88 / -82.88 | 0.00% | 0.00 | 49 ms |
| nangate45 tinyRocket | 101,959 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.007 | 0.16 | -117.33 / -117.33 | 0.00% | 0.00 | 38 ms |
| nangate45 picorv32 | 60,295 | 100% | 0 | 1.00000 / 0.000% | 1.0000 / 0.000% | 1.00000 | 0.000 | 0.00 | -311.20 / -311.20 | 0.00% | 0.00 | 26 ms |

Re-time: one `Timer.run()` on an RTX 3080 Ti (GPU 1), the networks reduced once per set of pin capacitances.
Residuals: nangate45 ibex, one clock-gate latch D pin (latches are not modeled; the placement-parasitics timer has
the same residual there, table in [timing_routing.md](timing_routing.md)); cva6, a few endpoints behind its two nets
whose route closes a resistor loop: OpenSTA reduces a loop over its depth-first tree, Attar over a breadth-first one
(the same tree wherever there is no loop). Nangate45 has no `set_layer_rc -via`: its cut resistances are the LEF's
(`RESISTANCE`), which the zoo stores with the routing fact (`routing.cut_res`) and the timer uses by itself; the
table was made from the zoo alone (no `via_res=`, no `adjustment=`).

## 2. The global router (`d.global_route()`, `attar.route.global_route`)

Not a FastRoute port: OpenROAD's gcell grid and edge capacities (exact per layer, `attar.rules.routing.capacity`;
Nangate45 with its fastroute.tcl's per-layer adjustments, stored in the zoo as `routing.layer_adjustments`), OpenROAD's Prim-Dijkstra tree (alpha 0) per net over its
pins' gcells, every tree edge pattern-routed (Z / L, negotiated rip-up and reroute as gsta's ggr), then a chunked
layer assignment (runs of up to 10 gcells, lowest layer with room, negotiated) and via stacks at tree points and pins.
`tools/validate_router.py`: Attar's routes vs OpenROAD's on the same placement (the truth of section 1).
WL as grt reports it (each segment + one gcell); per-net figures over the nets OpenROAD routes beyond two gcells;
overflow: Attar's 3D total (OpenROAD's: H / V / total of its final congestion report); timing: Attar's timer on Attar's
routes vs OpenSTA on OpenROAD's routes (`WNS or`: Attar's timer on OpenROAD's routes, = OpenSTA by section 1).
Times: Attar on an RTX 3080 Ti (after a warm-up run), OpenROAD's `global_route` with 4 threads.

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

What this shows: wirelength 0.6-2.3% below OpenROAD's on every design (its trees are FLUTE's, Attar's Prim-Dijkstra;
its segments split where Attar's runs continue), per net Pearson 0.998-1.000 with half the nets exact; 16-41x faster
on jpeg, ethmac, cva6 and swerv, about as fast on the small designs, slower on gcd. Not equal: Attar leaves overflow OpenROAD does not (12-363 on the larger designs: no
maze stage; a chunk keeps one layer), uses 2-10% more vias, and its timing differs by a few ps per endpoint (WNS
within 10 ps on 12 of 14; nangate45 ibex 38 ps apart: its worst paths take other layers).

## 3. Ranking placements as OpenROAD's router does

`tools/rank_router.py`: per design 7 placements of the same netlist (ORFS's global placement, it with 10% of the cells
moved, Attar's `place()` at target density 0.60 / 0.75 (two seeds) / 0.90 / 0.95), each exported as DEF and routed by
OpenROAD (the settings above) and by Attar; 7 designs (asap7 aes, jpeg, dynamic_node, picorv32, riscv32i-mock-sram;
nangate45 ibex, picorv32), 147 pairs. A pair agrees when a predictor orders the two placements as OpenROAD's
wirelength does; pairs binned by how far apart OpenROAD puts them.

| OpenROAD GR wirelength difference | pairs | HPWL agrees | Attar's router agrees |
|---|---|---|---|
| under 1% | 22 | 72.7% | **90.9%** |
| 1-2% | 19 | 73.7% | **100%** |
| 2-5% | 49 | 87.8% | **100%** |
| 5% and more | 57 | 100% | **100%** |

Attar's wirelength sits 0.5-3.6% below OpenROAD's on every one of the 49 placements (the offset is steady, so the order holds). This
is the claim the README makes. (An earlier study, 2026-11-13, found gsta's ggr no better than HPWL at ranking ORFS's
global-route wirelength: there the truth was ORFS's flow continued through CTS and repair, a different netlist per
placement; here the netlist is the same and only the router differs.)
