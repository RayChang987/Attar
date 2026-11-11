# The slow loop: ORFS (ASAP7, Nangate45)

Attar is checked against [OpenROAD-flow-scripts](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts)
on the ASAP7 and Nangate45 platforms, run from the docker image `openroad/orfs:26Q4-368-g9ca2a9e76`. All tools mount `orfs_runs/` at
`/work` and never modify the original runs.

## Runs and distillation

A design is named by its *run key*: `aes` is ASAP7's aes, `nangate45/aes` Nangate45's (any platform in the image
works the same way; gf12's is not in it, the PDK is proprietary).

```bash
tools/orfs_run_to_gp.sh <key>                # ORFS up to 3_3_place_gp: orfs_runs/<design>_base, orfs_runs/<platform>/<design>_base
tools/orfs_run_baseline.sh <design>          # the full flow (the baseline to compare against)
python tools/distill_orfs.py <key>           # 3_3 -> zoo/after_fp/orfs_<platform>/<design> (a placement problem)
python tools/distill_orfs.py <key> --stage 2_1_floorplan   # 2_1 -> zoo/before_fp/orfs_<platform>/<design> (a floorplan problem)
```

Each stage's DEF stays in `orfs_runs/rt/<key>/<stage>.def`: it is the source of the distillation. `--reuse` distills
an existing one again without docker.

**Designs ported to ASAP7.** `python tools/orfs_port_asap7.py` writes ASAP7 configs for nangate45 designs that ORFS
has no ASAP7 config of (ariane133/136, black_parrot, bp_be_top, bp_fe_top, bp_multi_top, bp_quad, dynamic_node,
swerv) to `orfs_runs/designs/<design>/`, where the run tools find them. The RTL is read unmodified from the image.
Their fakeram45 memories get ASAP7 views of the same name and pins from FakeRAM2.0's asap7 backend (the generator
behind ORFS's AUTO_MEMORIES), and each design a minimal SDC (its clocks, IO delays of 20% of the period; times in
ps). Every port uses the same floorplan settings (CORE_UTILIZATION 50, PLACE_DENSITY_LB_ADDON 0.2): nothing is tuned
per design.

## Two injection points

**Placement (3_3).** `tools/orfs_roundtrip.py <design> <run> [identity|perturb|jitterN|def=<path>]` stamps a
placement onto the 3_3 DEF (`--stage 3_2_place_iop` injects before ORFS's global placement instead), applies its COMPONENTS to `3_3_place_gp.odb` (`read_def -incremental`) in a copy
of the run, and lets ORFS finish: resize, detailed placement, CTS, global and detailed routing, reports.

**Floorplan (2_1).** `tools/orfs_handoff.py <design> <run> <exported DEF> [tag]` applies the die, rows, tracks, IO
pins and macros that Attar chose to `2_1_floorplan.odb`:

- If the die changed: `initialize_floorplan` and the platform's `make_tracks`.
- IO pins: `place_pin`, which leaves them FIXED.
- Macros: placed with status FIRM.

`--until 3_5_place_dp` stops there and checks the result in ORFS's own environment: `check_placement`, max slew /
capacitance / fanout violations, the ERROR and WARNING messages of every stage, and whether the macros and pins
stayed put. `baseline` in place of the DEF runs ORFS's own floorplan the same way, for comparison. The results on the
`before_fp` designs: [validation/fp_handoff.md](validation/fp_handoff.md).

The same database is saved as a newer `2_2_floorplan_macro.odb`, so make skips ORFS's macro placer. ORFS then does
tapcells, PDN, global placement, CTS and routing. Its `place_pins` and `-place_ios` find no pins to place, and the
tool checks the final DEF to confirm that every macro and pin stayed put.

Both tools write their summary to `orfs_runs/results/<key>/` and delete their run copy (full ORFS runs) unless
`--keep`. `--cores` sets make's NUM_CORES, `orfs_roundtrip.py --par` the runs at once, and `orfs_handoff.py
--make-args` passes extra make variables (use the same as the baseline).

## Agreement checks

`tools/validate_orfs.py` runs two checks on every design in `after_fp/orfs_asap7` and `after_fp/orfs_nangate45` (or the
zoo designs named) and writes `docs/validation/orfs_check.{md,json}` (the designs checked replace their rows):

- **HPWL:** Attar against OpenROAD's database computation on the same DEF.
- **Legality:** Attar's checker against `check_placement`, plus a broken-placement control.

All 51 agree (33 ASAP7, 18 Nangate45, up to 1.48M cells): HPWL within 0.00005%, 51/51 legal verdicts, and 51/51 broken placements rejected by both.

`tools/orfs_roundtrip.py ... identity` writes the ORFS placement back unchanged. Downstream metrics are
bit-identical on gcd, aes, ibex, jpeg and ethmac.

## Designs without an ORFS platform

i3dPDK_NS3K has no ORFS platform, so `tools/validate_openroad.py` takes Attar's DEF straight to OpenROAD (from the
same docker image) with the design's LEFs:

```bash
python tools/validate_openroad.py after_fp/ns3k/aes after_fp/ns3k/jpeg after_fp/stress/bp_quad --lef-dir ~/i3dPDK_NS3K/innovus_enablement \
    --grt --layers M1-M9 --allow-one-site-gaps
```

`--allow-one-site-gaps`: OpenROAD forbids a one-site gap between cells exactly when the library has no one-site-wide
master (`-disallow_one_site_gaps` is deprecated; the rule follows the library). NS3K has no fillers at all and its
own flow (Innovus) leaves such gaps, so the tool reads a dummy one-site spacer on the rows' site, which turns the
check off; nothing else of `check_placement` changes.

It reads the DEF, compares instance, net and IO counts, runs `check_placement` (violations per category), compares
OpenROAD's HPWL with Attar's, runs global routing, and checks that a placement broken on purpose is rejected.

## Noise floor

ORFS is deterministic, so a different seed does not measure noise. Instead, 1% of the cells are moved by one site
(about 0.005% HPWL) and the flow is rerun. Over a baseline plus 3 such runs (max minus min, relative to the
baseline):

| design | cells | routed WL | GRT WL | WNS | TNS |
| --- | ---: | ---: | ---: | ---: | ---: |
| aes | 12k | 0.29% | 0.70% | 89% | 98% |
| ibex | 21k | 0.25% | 0.16% | 34% | 104% |
| jpeg | 58k | 0.26% | 0.27% | 9.8% | 7.6% |
| ethmac | 60k | 0.09% | 0.18% | 7.2% | 4.5% |

Routed wirelength is resolved to about 0.3% per run. WNS and TNS are dominated by repair_timing's chaotic behavior,
so compare timing over several perturbed runs, never a single one.

## A lesson the slow loop taught

The first floorplan hand-offs failed in detailed routing. A controlled experiment found the cause: ORFS's own macro
placement, shifted 24 dbu so that the macro pins left the M4 tracks, failed in exactly the same way, while the same
placement on-track routed with 0 DRC. Macro legality is therefore "pins on their routing tracks", not "on the
standard-cell row grid". `Design.macro_grid` encodes that, and every macro legalizer works on it.
