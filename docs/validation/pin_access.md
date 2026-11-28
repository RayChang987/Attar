# Pin access against routed runs

`Evaluator.pin_access` names the signal pins the router cannot reach past the PDN. This page checks it against the
DRC results of routed Innovus runs.

## Method

Data: the routed runs of pa_pdn on i3dPDK_NS3K, 17 designs, about 60 placement methods each, two PDN styles: M3
VDD/VSS stripe pairs (`stripe`) and M5 straps tied to the rails by via pillars (`pillar`). Router: Innovus 21.13,
unidirectional, M1 for pin access only. Each run's final DEF is written right after `routeDesign`, so it holds the
placement the router got and the PDN.

Per run, `tools/validate_pin_access.py` distills that DEF, computes `Evaluator.pin_access` per pin, and reads
`rpt/drc.rpt` (`verify_drc`). Each M1 or M2 marker goes to the nearest pin of its regular net within 0.5 um (LEF x4
units). A pin fails when a signal-to-PG marker lands on it.

The rule is geometry only, with no fitted constant. An access point is a track of the layer above (M2) crossing the
pin. The via lands a square of the layer's width on M1 and on M2. A landing closer than the layer's minimum spacing
to PG metal (Euclidean) is blocked. A pin is blocked when all its access points are.

```bash
python tools/validate_pin_access.py ~/scratch/pin_access --ns3k ~/i3dPDK_NS3K --jobs 8 --report docs/validation/pin_access.md
```

## Results

<!-- RESULTS -->
988 runs, 240,014,044 signal pins. Blocked pins against the pins with a signal-to-PG M1/M2 marker: TP 156,665, FP 0, FN 0 (precision 1.0000, recall 1.0000). OpenROAD dpl's PDN rule flags 66 cells in all. M1/M2 markers: 156,827 signal-to-PG, 215,892 signal-to-signal (not modelled). pin_access: median 1.75 s per run (CPU, 2 threads).

| design | PDN | runs | signal-to-PG M1/M2 markers | pins with one | blocked pins | TP | FP | FN | Kendall tau (blocked vs markers) | dpl rule: cells |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aes | pillar | 44 | 1,313 | 1,313 | 1,313 | 1,313 | 0 | 0 | 1.00 | 0 |
| aes | stripe | 51 | 2,144 | 2,144 | 2,144 | 2,144 | 0 | 0 | 1.00 | 0 |
| ariane133 | pillar | 29 | 14,375 | 14,375 | 14,375 | 14,375 | 0 | 0 | 1.00 | 0 |
| ariane133 | stripe | 29 | 22,717 | 22,705 | 22,705 | 22,705 | 0 | 0 | 1.00 | 7 |
| ariane136 | pillar | 21 | 11,813 | 11,813 | 11,813 | 11,813 | 0 | 0 | 1.00 | 0 |
| ariane136 | stripe | 21 | 18,909 | 18,909 | 18,909 | 18,909 | 0 | 0 | 1.00 | 0 |
| black_parrot | pillar | 27 | 4,233 | 4,231 | 4,231 | 4,231 | 0 | 0 | 1.00 | 1 |
| black_parrot | stripe | 29 | 5,915 | 5,915 | 5,915 | 5,915 | 0 | 0 | 1.00 | 0 |
| bp_be_top | pillar | 21 | 1,239 | 1,239 | 1,239 | 1,239 | 0 | 0 | 1.00 | 0 |
| bp_be_top | stripe | 21 | 1,786 | 1,786 | 1,786 | 1,786 | 0 | 0 | 1.00 | 0 |
| bp_fe_top | pillar | 39 | 761 | 761 | 761 | 761 | 0 | 0 | 1.00 | 0 |
| bp_fe_top | stripe | 44 | 1,056 | 1,056 | 1,056 | 1,056 | 0 | 0 | 1.00 | 0 |
| bp_multi_top | pillar | 27 | 3,683 | 3,683 | 3,683 | 3,683 | 0 | 0 | 1.00 | 0 |
| bp_multi_top | stripe | 29 | 5,347 | 5,346 | 5,346 | 5,346 | 0 | 0 | 1.00 | 0 |
| bp_quad | pillar | 1 | 1 | 1 | 1 | 1 | 0 | 0 | — | 0 |
| bp_quad | stripe | 1 | 143 | 143 | 143 | 143 | 0 | 0 | — | 0 |
| dynamic_node | pillar | 27 | 417 | 415 | 415 | 415 | 0 | 0 | 0.97 | 0 |
| dynamic_node | stripe | 26 | 926 | 805 | 805 | 805 | 0 | 0 | 0.87 | 46 |
| ethmac | pillar | 37 | 3,846 | 3,846 | 3,846 | 3,846 | 0 | 0 | 1.00 | 0 |
| ethmac | stripe | 25 | 6,319 | 6,319 | 6,319 | 6,319 | 0 | 0 | 1.00 | 0 |
| gcd | pillar | 19 | 15 | 15 | 15 | 15 | 0 | 0 | 1.00 | 0 |
| gcd | stripe | 17 | 29 | 29 | 29 | 29 | 0 | 0 | 1.00 | 0 |
| jpeg | pillar | 55 | 8,258 | 8,258 | 8,258 | 8,258 | 0 | 0 | 1.00 | 0 |
| jpeg | stripe | 48 | 14,040 | 14,040 | 14,040 | 14,040 | 0 | 0 | 1.00 | 0 |
| riscv32i | pillar | 21 | 389 | 389 | 389 | 389 | 0 | 0 | 1.00 | 0 |
| riscv32i | stripe | 22 | 540 | 540 | 540 | 540 | 0 | 0 | 1.00 | 0 |
| serv | pillar | 26 | 24 | 24 | 24 | 24 | 0 | 0 | 1.00 | 0 |
| serv | stripe | 26 | 39 | 39 | 39 | 39 | 0 | 0 | 1.00 | 0 |
| swerv | pillar | 31 | 4,175 | 4,175 | 4,175 | 4,175 | 0 | 0 | 1.00 | 0 |
| swerv | stripe | 33 | 7,267 | 7,267 | 7,267 | 7,267 | 0 | 0 | 1.00 | 0 |
| swerv_wrapper | pillar | 27 | 3,615 | 3,615 | 3,615 | 3,615 | 0 | 0 | 1.00 | 0 |
| swerv_wrapper | stripe | 29 | 6,127 | 6,103 | 6,103 | 6,103 | 0 | 0 | 0.98 | 12 |
| tinyRocket | pillar | 40 | 2,234 | 2,234 | 2,234 | 2,234 | 0 | 0 | 1.00 | 0 |
| tinyRocket | stripe | 45 | 3,132 | 3,132 | 3,132 | 3,132 | 0 | 0 | 1.00 | 0 |
<!-- /RESULTS -->

## What it shows

**OpenROAD dpl's PDN rule (`legality()['pdn_blocked']`) passes these pins.** The typical failing pin is a
single-access M1 pin (AOI21 A, XNOR2 A, XOR2 A: 0.064 um high, one M2 track) beside a PG via pillar. It is exactly one M1
spacing (0.032 um) from the pillar's M1 metal, which is legal, but its M2 landing is 0.040 um from the via's M2 pad,
under the M2 spacing (0.048 um). Taller pins at the same gap have other M2 tracks and never fail.

**The distance must be Euclidean.** A per-axis distance also flags pins diagonal to a pillar: 12,325 false positives
in the runs whose straps are off the site grid.

Markers can outnumber pins: 162 signal-to-PG markers lie farther than 0.5 um from every pin of their net, and some
pins carry two.

## Limits

- **Signal-to-signal markers are not modelled.** They are 58% of the M2 markers, mostly with the stripe PDN, so
  against all M1/M2 markers the per-design tau is lower.
- **Pins on a routing layer are left out.** The rule covers pins below the lowest signal routing layer (the design's
  `routing` range), where a via is the only way in. ISPD 2018/2019 route from metal1, so the rule does not apply to
  them.
- **No DRVs to compare on other PDKs.** ASAP7 and Nangate45 placements have no blocked pin, which agrees with the
  zero DRVs ORFS's detailed router leaves there; no failing pins were available to compare.
