# before_fp hand-off: Attar floorplans in ORFS

Each `before_fp/orfs_asap7` design is floorplanned by `attar.naive` (`d.floorplan(utilization=u)`: die, IO pins,
macros on their routing tracks), exported, and handed to ORFS with `tools/orfs_handoff.py <design> <run> <def>
--until 3_5_place_dp`. ORFS adds tapcells and the PDN, then runs global placement, resizing and detailed placement.
Checks, in ORFS: `check_placement`; max slew / capacitance / fanout violations (estimated parasitics); ERROR and
WARNING messages of every stage; whether Attar's macros and IO pins stayed put. The ORFS rows are ORFS's own
floorplan of the same run (`baseline`).

Utilization = std-cell area / (core - macro area); *loose* is 0.8x ORFS's, *tight* 1.15x.

| design | floorplan | utilization | die (um) | PDN | check_placement | DRV slew / cap / fanout | macros, pins moved | messages vs the baseline |
| --- | --- | ---: | --- | :---: | :---: | --- | --- | --- |
| aes | ORFS | 0.718 | 48.7 x 48.7 | ok | pass | 0 / 0 / 0 | | |
| aes | Attar loose | 0.574 | 53.8 x 54.4 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| aes | Attar tight | 0.825 | 45.6 x 46.0 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| ibex | ORFS | 0.422 | 76.7 x 76.7 | ok | pass | 0 / 0 / 0 | | |
| ibex | Attar loose | 0.337 | 85.2 x 85.6 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| ibex | Attar tight | 0.485 | 71.7 x 72.1 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| jpeg | ORFS | 0.708 | 101.7 x 101.7 | ok | pass | 0 / 0 / 0 | | |
| jpeg | Attar loose | 0.567 | 113.1 x 113.3 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| jpeg | Attar tight | 0.815 | 95.0 x 95.3 | ok | pass | 0 / 0 / 0 | 0, 0 | GPL-0302 (density near the utilization) |
| riscv32i | ORFS | 0.422 | 73.2 x 73.2 | ok | pass | 0 / 0 / 0 | | |
| riscv32i | Attar loose | 0.338 | 78.0 x 78.3 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| riscv32i | Attar tight | 0.486 | 70.4 x 70.7 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| tinyRocket | ORFS | 0.393 | 102.9 x 102.9 | ok | pass | 0 / 0 / 0 | | |
| tinyRocket | Attar loose | 0.314 | 114.2 x 114.5 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| tinyRocket | Attar tight | 0.451 | 96.3 x 96.7 | ok | pass | 0 / 0 / 0 | 0, 0 | same |
| swerv_wrapper | ORFS | 0.284 | 297.4 x 297.4 | ok | pass | 12 / 0 / 0 | | |
| swerv_wrapper | Attar loose | 0.227 | 319.4 x 319.8 | ok | pass | 1 / 0 / 0 | 0, 0 | PDN-0110, PDN-0195: vias dropped next to a macro edge |
| swerv_wrapper | Attar tight | 0.327 | 285.1 x 285.5 | ok | **fail** (DPL-0033) | | | ORFS's detailed placer fails in a 2.3 x 2 um pocket of rows boxed in by four macros |
| cva6 | ORFS | 0.674 | 151.2 x 151.2 | ok | pass | 0 / 0 / 0 | | PDN-0195 x74 |
| cva6 | Attar loose | 0.539 | 166.3 x 166.6 | ok | pass | 0 / 0 / 2 | 0, 0 | no PDN-0195 |
| cva6 | Attar tight | 0.775 | 142.6 x 142.8 | ok | not reached (GPL-0301 in 3_4) | | | the flow's own netlist growth (resizing, buffers, SWAP_ARITH_OPERATORS) takes the utilization to 100.7% |

The messages column leaves out what the hand-off itself removes: ORFS's macro placer (2_2) does not run.

**12 of 14 Attar floorplans** pass PDN, global and detailed placement and `check_placement`, with no more max-slew /
cap / fanout violations than ORFS's own floorplan except cva6 loose (2 fanout violations against 0). ORFS keeps every macro and IO pin where Attar put it. Both failures
are tight:

- **cva6:** ORFS grows the netlist by about a fifth before detailed placement, and incremental global placement
  stops at 100.7% utilization. A floorplan needs headroom for that growth.
- **swerv_wrapper:** Attar's macro legalizer left 4 short rows (2.3 um wide) between four staggered macros. ORFS puts
  buffers there that its detailed placer cannot legalize (12 site alignment problems, 6 overlaps). ORFS's macro
  placer avoids such channels (`RTLMP_MIN_CHANNEL_SIZE`: macros abut or leave at least 20 um); Attar's has no such
  rule yet.

**In the zoo:** aes, ibex, jpeg, riscv32i, tinyRocket. cva6 and swerv_wrapper stay out of `before_fp` (they are in
`after_fp`) until the floorplan engines handle these cases.

Reproduce: `tools/orfs_handoff.py <design> <design>_base <exported DEF | baseline> <tag> --until 3_5_place_dp
--make-args LEC_CHECK=0`; each run writes `orfs_runs/results/<design>/handoff_<tag>_3_5_place_dp.json`.

## nangate45 macro designs (`before_fp/orfs_nangate45`)

The macro-heavy ORFS nangate45 designs (the macro-placement benchmarks of MaskPlace, ChiPFormer, ReMap and their
successors), distilled at ORFS 2_1 and checked the same way (`tools/fp_handoff_sweep.py`). ORFS's floorplans are
sparse (19-30% utilization). The macro halo is each design's minimum channel (ORFS's `RTLMP_MIN_CHANNEL_SIZE`, zoo
`macro_channel`, the `d.floorplan()` default). With Attar's default 2.8 um halo, ORFS's PDN fails (PDN-0179) on
ariane133: the channels are too narrow for the power grid.

| design | floorplan | utilization | die (um) | halo (um) | PDN | check_placement | DRV slew / cap / fanout | macros, pins moved |
| --- | --- | ---: | --- | ---: | :---: | :---: | --- | --- |
| ariane133 | ORFS | 0.297 | | | ok | pass | 20 / 2 / 0 | |
| ariane133 | Attar loose | 0.237 | 1302.3 x 1303.9 | 16 | ok | pass | 20 / 5 / 0 | 0, 0 |
| ariane133 | Attar tight | 0.341 | 1145.1 x 1147.2 | 16 | **PDN-0179** | not reached | | |
| ariane136 | ORFS | 0.186 | | | ok | pass | 10 / 1 / 0 | |
| ariane136 | Attar loose | 0.148 | 1634.7 x 1638.6 | 24 | ok | pass | 10 / 0 / 0 | 0, 0 |
| ariane136 | Attar tight | 0.213 | 1422.7 x 1425.8 | 24 | ok | pass | 10 / 3 / 0 | 0, 0 |
| black_parrot | ORFS | 0.223 | | | ok | pass | 6 / 1 / 0 | |
| black_parrot | Attar loose | 0.178 | 1449.6 x 1451.2 | 30 | ok | pass | 6 / 0 / 0 | 0, 0 |
| black_parrot | Attar tight | 0.256 | 1253.7 x 1255.2 | 30 | ok | pass | 6 / 0 / 0 | 0, 0 |
| bp_be_top | ORFS | 0.248 | | | ok | pass | 0 / 1 / 0 | |
| bp_be_top | Attar loose | 0.198 | 810.6 x 813.4 | 20 | ok | pass | 0 / 0 / 0 | 0, 0 |
| bp_be_top | Attar tight | 0.285 | 711.8 x 714.0 | 20 | ok | pass | 0 / 4 / 0 | 0, 0 |
| bp_fe_top | ORFS | 0.202 | | | ok | pass | 0 / 2 / 0 | |
| bp_fe_top | Attar loose | 0.161 | 704.4 x 706.5 | 20 | ok | pass | 0 / 2 / 0 | 0, 0 |
| bp_fe_top | Attar tight | 0.231 | 628.4 x 630.9 | 20 | ok | pass | 0 / 25 / 0 | 0, 0 |
| bp_multi_top | ORFS | 0.191 | | | ok | pass | 0 / 1 / 0 | |
| bp_multi_top | Attar loose | 0.152 | 1192.5 x 1193.8 | 40 | ok | pass | 0 / 2 / 0 | 0, 0 |
| bp_multi_top | Attar tight | 0.219 | 1048.5 x 1049.6 | 40 | ok | pass | 0 / 0 / 0 | 0, 0 |
| swerv_wrapper | ORFS | 0.253 | | | ok | pass | 0 / 2 / 0 | |
| swerv_wrapper | Attar loose | 0.202 | 1117.3 x 1119.0 | 24 | ok | pass | 0 / 1 / 0 | 0, 0 |
| swerv_wrapper | Attar tight | 0.290 | 1010.5 x 1012.6 | 24 | ok | pass | 0 / 19 / 0 | 0, 0 |
| mempool_group | ORFS | 0.247 | | | ok | pass | 0 / 0 / 0 | |
| mempool_group | Attar loose | 0.197 | 1215.9 x 1219.0 | 20 | ok | pass | 0 / 1 / 0 | 0, 0 |
| mempool_group | Attar tight | 0.283 | 1032.7 x 1037.0 | 20 | ok | pass | 0 / 2 / 0 | 0, 0 |

**15 of 16 Attar floorplans** pass PDN, global and detailed placement and `check_placement`. ORFS keeps every macro
and IO pin in place. 7 of the 15 have no more max-slew / cap / fanout violations than ORFS's floorplan; the others
add max-capacitance violations (estimated parasitics, before ORFS's later repair). ariane133 on ORFS's own die with
the 16 um channel went through the whole flow: 0 DRC after routing (WNS -0.21 ns).

**ariane133, tight:** ORFS's PDN cannot repair two channels between rows of SRAM macros. Of the 16.1 um channel, the
PDN's macro grids take 2 um on each side, and a core metal7 stripe (30 um pitch) runs through the remaining 11.6 um.
That leaves 8.5 um, where no repair strap fits (`set_debug_level PDN Channel 2`). An equally narrow channel without a
stripe is repaired. Whether a channel is repairable depends on where the stripes fall, which Attar's macro legalizer
does not know.

**In the zoo:** all eight. ariane133 stays despite its tight failure (a limit of the naive macro legalizer): it is
the most used macro-placement benchmark.

Reproduce: `tools/fp_handoff_sweep.py nangate45/<design> ... --lanes 3 --cores 4`; floorplans and DEFs in
`orfs_runs/handoff_sweep/nangate45/<design>/`, results in
`orfs_runs/results/nangate45/<design>/handoff_<tag>_3_5_place_dp.json`.
