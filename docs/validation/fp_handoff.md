# before_fp hand-off: Attar floorplans in ORFS

Each `before_fp/orfs_asap7` design was floorplanned by `attar.naive` (`d.floorplan(utilization=u)`: a new die for the
utilization, IO pins, macros on their routing tracks), exported (`d.export`), and handed to ORFS with
`tools/orfs_handoff.py <design> <run> <def> --until 3_5_place_dp`. ORFS then made tapcells, the PDN, global placement,
resizing and detailed placement, and the result was checked in ORFS's own environment: `check_placement`, max slew /
capacitance / fanout violations (estimated parasitics), the ERROR and WARNING messages of every stage, and whether
Attar's macros and IO pins stayed put. `baseline` is ORFS's own floorplan of the same run, for comparison.

Utilization is std-cell area / (core - macro area); *loose* is 0.8x ORFS's, *tight* 1.15x.

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

**12 of 14 Attar floorplans** go through PDN, global and detailed placement, pass `check_placement`, and have as few
max-slew / cap / fanout violations as ORFS's own floorplan; ORFS keeps every macro and IO pin where Attar put it. The
two failures are both tight floorplans:

- **cva6, tight:** too tight for this flow. ORFS grows the netlist by about a fifth before detailed placement, and the
  incremental global placement stops at 100.7% utilization. A floorplan engine needs headroom for that growth.
- **swerv_wrapper, tight:** Attar's macro legalizer left a pocket of 4 short rows (2.3 um wide) between four
  staggered macros; ORFS's global placement puts buffers there and its detailed placer cannot legalize them
  (12 site alignment problems, 6 overlaps). ORFS's own macro placer avoids such channels (`RTLMP_MIN_CHANNEL_SIZE`:
  macros abut or leave at least 20 um); Attar's has no such rule yet.

**In the zoo:** the five designs whose floorplans all pass (aes, ibex, jpeg, riscv32i, tinyRocket). cva6 and
swerv_wrapper are left out of `before_fp` until the floorplan engines handle the two cases above (they stay in
`after_fp`).

Reproduce: `tools/orfs_handoff.py <design> <design>_base <exported DEF | baseline> <tag> --until 3_5_place_dp
--make-args LEC_CHECK=0`; each run writes `orfs_runs/results/<design>/handoff_<tag>_3_5_place_dp.json`.
