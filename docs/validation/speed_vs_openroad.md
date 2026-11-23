# Speed against OpenROAD kept warm

How long one timing update takes inside a placement loop: Attar against OpenROAD with the design already loaded in
the same session, so neither side pays for reading files. This is the cost a timing-driven placer pays on every
iteration. The comparison against a cold OpenSTA run (reads included) is in [timing_routing.md](timing_routing.md).

<p align="center"><img src="../images/speed_vs_openroad.png" alt="Seconds per timing update in global placement, OpenROAD vs Attar on the GPU: netcard_fast 7.2 vs 0.34 s, superblue18 13.0 vs 0.56 s, superblue1 19.6 vs 0.83 s, superblue7 33.6 vs 1.2 s; Attar is 21-27 times faster per GP update and 21-54 times faster per DP round" width="860"></p>

## Setup

- One machine: RTX 4090, 32 cores. Attar 1.0.6 (torch 2.5.1, 8 CPU threads); OpenROAD from the ORFS image
  `26Q4-368-g9ca2a9e76`, `-threads 8`. Each measurement runs alone.
- Designs: netcard_fast (ICCAD 2025, ASAP7, ideal clocks), superblue18, superblue1, superblue7 (ICCAD 2015, Late
  library, the contest wire RC). The start is the zoo's `after_fp` placement.
- Rounds, generated once by Attar and replayed on both sides:
  - **GP update**: every movable cell moves by up to 1 um in x and y, unlegalized (a timing update between global
    placement iterations).
  - **DP round**: 1% of the movable cells move by up to 1 um and are legalized (`naive.legalize(cells=mask)`), rounds
    chained (one round of timing-driven detailed placement).
- Each round, OpenROAD gets the changed instances through odb (`setOrient`, `setLocation`), then runs
  `estimate_parasitics -placement` and `sta::worst_slack -max` / `sta::total_negative_slack -max`. Attar gets
  `d.pos`, then runs `d.timing().summary()`. Handing the positions over is not timed on either side. Each process
  evaluates the start placement once before timing three rounds; the tables give the median.

## Results

| design | cells | GP update, OpenROAD | GP update, Attar GPU | speedup | DP round, OpenROAD | DP round, Attar GPU | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| netcard_fast | 298,384 | 7.25 s | 0.343 s | 21x | 5.19 s | 0.242 s | 21x |
| superblue18 | 744,947 | 13.0 s | 0.561 s | 23x | 11.5 s | 0.313 s | 37x |
| superblue1 | 1,159,346 | 19.6 s | 0.831 s | 24x | 16.2 s | 0.469 s | 35x |
| superblue7 | 1,865,884 | 33.6 s | 1.22 s | 27x | 30.6 s | 0.566 s | 54x |

On the CPU (8 threads) Attar takes 2.4-8.7 s per update, 2-4x faster than OpenROAD: the speed at this size comes from
the GPU. A DP round gains more than a GP update because Attar recomputes the parasitics of the moved nets only, while
`estimate_parasitics -placement` redoes every net.

Both sides agree: TNS within 0.13% on superblue (the same agreement as in [timing_routing.md](timing_routing.md)),
worst slack within 0.002 ps on netcard_fast, and OpenROAD's third round in the session matches a cold read of the
same placement bit for bit.

## Notes

- Moving the instances with `read_def -incremental` instead of odb leaves OpenROAD's timing different from a cold
  read of the same DEF (superblue18: TNS -2620824 vs -2623989), although the instance positions in odb are the same.
  The numbers above use odb, which matches the cold read.
- Global routing as the ruler (netcard_fast, before clock tree synthesis): FastRoute takes 1100-1460 s per call and
  its incremental mode 1620-2500 s, against 3.4 s (incremental 1.2 s) for Attar on the GPU. Most of FastRoute's time
  goes to the 84,026-pin ideal clock net; with `-skip_large_fanout_nets 1000` it takes 189 s. These numbers come
  from one design and are not in the figure.
- On small designs (2k-22k cells, ICCAD 2025 IWLS) the gap is 1-8x: OpenROAD is fast enough there, and its
  incremental global routing beats Attar's on the smallest designs.
