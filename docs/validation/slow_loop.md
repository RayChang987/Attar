# The slow loop: ORFS (ASAP7, Nangate45)

*For Attar's own validation.* Users export a DEF (`d.export`) into their own flow; these tools check Attar against
[OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) on ASAP7 and Nangate45, from
the docker image `openroad/orfs:26Q4-368-g9ca2a9e76`. `attar._slow_loop` (not user API) holds the hand-off command
and its facts. All tools mount `orfs_runs/` at `/work` and never modify the original runs.

## Runs and distillation

A *run key* names a design: `aes` is ASAP7's aes, `nangate45/aes` Nangate45's. Any platform in the image works (gf12
is not in it: its PDK is proprietary).

```bash
tools/orfs_run_to_gp.sh <key>                # ORFS up to 3_3_place_gp: orfs_runs/<design>_base, orfs_runs/<platform>/<design>_base
tools/orfs_run_baseline.sh <design>          # the full flow (the baseline to compare against)
```

Each stage's DEF stays in `orfs_runs/rt/<key>/<stage>.def`. `attar.io.distill` turns 3_3 into
`zoo/after_fp/orfs_<platform>/<design>` (placement) and 2_1 into `zoo/before_fp/orfs_<platform>/<design>`
(floorplan).

**Designs ported to ASAP7.** `python tools/orfs_port_asap7.py` writes ASAP7 configs (`orfs_runs/designs/<design>/`)
for nangate45 designs ORFS has none for (ariane133/136, black_parrot, bp_be_top, bp_fe_top, bp_multi_top, bp_quad,
dynamic_node, swerv): unmodified RTL, ASAP7 views of the fakeram45 memories from FakeRAM2.0's asap7 backend, a
minimal SDC (clocks, IO delays of 20% of the period), and the same settings for all (CORE_UTILIZATION 50,
PLACE_DENSITY_LB_ADDON 0.2).

## Two injection points

**Placement (3_3).** `tools/orfs_roundtrip.py <design> <run> [identity|perturb|jitterN|def=<path>]` applies a
placement to `3_3_place_gp.odb` (`read_def -incremental`) in a copy of the run; ORFS then runs resize, detailed
placement, CTS, routing and reports. `--stage 3_2_place_iop` injects before global placement instead.

**Floorplan (2_1).** `tools/orfs_handoff.py <design> <run> <exported DEF> [tag]` applies Attar's die
(`initialize_floorplan`, `make_tracks`), IO pins (`place_pin`, FIXED) and macros (FIRM) to `2_1_floorplan.odb`, saved
as `2_2_floorplan_macro.odb` so that ORFS's macro placer is skipped. The tool checks in the final DEF that every
macro and pin stayed put. `--until 3_5_place_dp` stops after detailed placement and runs the checks of
[fp_handoff.md](fp_handoff.md); `baseline` in place of the DEF runs ORFS's own floorplan.

Both tools write a summary to `orfs_runs/results/<key>/` and delete their run copy unless `--keep`.

## Agreement checks

`tools/validate_orfs.py` compares, on every design in `after_fp/orfs_asap7` and `after_fp/orfs_nangate45`, Attar's
HPWL with OpenROAD's database computation on the same DEF, and Attar's legality checker with `check_placement`, plus
a broken-placement control. It writes `docs/validation/orfs_check.{md,json}`.

All 23 agree (15 ASAP7, 8 Nangate45, up to 1.05M cells): HPWL within 0.00002%, 23/23 legal verdicts, and 23/23 broken placements rejected by both.

`tools/orfs_roundtrip.py ... identity` writes the ORFS placement back unchanged: downstream metrics are
bit-identical on gcd, aes, ibex, jpeg and ethmac.

## Designs without an ORFS platform

For i3dPDK_NS3K, `tools/validate_openroad.py` takes Attar's DEF straight to OpenROAD (same image) with the design's
LEFs. It compares instance, net and IO counts and HPWL, runs `check_placement` and global routing, and checks that a
broken placement is rejected.

```bash
python tools/validate_openroad.py after_fp/ns3k/aes after_fp/ns3k/jpeg --lef-dir ~/i3dPDK_NS3K/innovus_enablement \
    --grt --layers M1-M9 --allow-one-site-gaps
```

`--allow-one-site-gaps` reads a dummy one-site spacer: OpenROAD forbids one-site gaps when the library has no
one-site master, and NS3K (no fillers; Innovus leaves such gaps) has none.

## Noise floor

ORFS is deterministic, so 1% of the cells are moved by one site (about 0.005% HPWL) and the flow is rerun. Spread
over a baseline plus 3 such runs (max minus min, relative to the baseline):

| design | cells | routed WL | GRT WL | WNS | TNS |
| --- | ---: | ---: | ---: | ---: | ---: |
| aes | 12k | 0.29% | 0.70% | 89% | 98% |
| ibex | 21k | 0.25% | 0.16% | 34% | 104% |
| jpeg | 58k | 0.26% | 0.27% | 9.8% | 7.6% |
| ethmac | 60k | 0.09% | 0.18% | 7.2% | 4.5% |

Routed wirelength is resolved to about 0.3% per run. WNS and TNS are dominated by repair_timing's chaotic behavior:
compare timing over several perturbed runs, never one.

## Macro legality: pins on their tracks

A macro whose pins sit off their routing tracks fails detailed routing: ORFS's own macro placement shifted by 24 dbu,
so that the macro pins leave the M4 tracks, fails; on-track it routes with 0 DRC. Macro legality is therefore "pins
on their routing tracks", not "on the standard-cell row grid". `Design.macro_grid` encodes this, and every macro
legalizer works on it.
