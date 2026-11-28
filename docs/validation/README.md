# Validation

Each page compares Attar with OpenROAD / OpenSTA (the ORFS image `openroad/orfs:26Q4-368-g9ca2a9e76`) or with
routed Innovus runs, on the same netlist, placement and inputs.

| page | what is compared | headline result | reproduce |
| --- | --- | --- | --- |
| [timing_routing.md](timing_routing.md) | slacks, power, sizing, netlist edits and GR overflow against OpenSTA / OpenROAD | WNS error max 0.00% (ORFS asap7, nangate45), 0.01% (NS3K), 0.33% (ICCAD 2015 superblue); power error max 0.06% (ICCAD 2025); GROV ranking Pearson 0.962, Spearman 0.966 | `tools/validate_sta.py`, `validate_sta_ns3k.py`, `validate_sta_iccad2015.py`, `validate_iccad2025.py`, `validate_sizing.py`, `validate_gr.py`; then `python tools/make_validation_md.py` |
| [global_routing.md](global_routing.md) | parasitics on OpenROAD's global routes; Attar's router against OpenROAD's | wire C and R total error 0.000%; router wirelength 0.6-2.3% below OpenROAD's; on pairs under 1% apart, Attar's router ranks 90.9% like OpenROAD, HPWL 72.7% | `tools/validate_gr_parasitics.py`, `validate_router.py`, `rank_router.py` |
| [drv_power_cts.md](drv_power_cts.md) | max slew / cap / fanout, power, propagated clocks with CRPR, CTS against OpenSTA and TritonCTS | power error 0.00% on every design; ORFS clock trees, CRPR on: setup MAE at most 0.0291 ps | `tools/validate_drv.py`, `validate_power.py`, `validate_cts.py`, `validate_cts_engine.py`; then `python tools/make_drv_power_md.py` |
| [pin_access.md](pin_access.md) | `Evaluator.pin_access` against Innovus `verify_drc` markers | 988 runs: TP 156,665, FP 0, FN 0 | `python tools/validate_pin_access.py SCRATCH --ns3k ~/i3dPDK_NS3K --jobs 8 --report docs/validation/pin_access.md` |
| [fp_handoff.md](fp_handoff.md) | Attar floorplans run through ORFS to detailed placement | 12 of 14 (asap7) and 15 of 16 (nangate45 macro designs) pass `check_placement` | `tools/orfs_handoff.py ... --until 3_5_place_dp`; `tools/fp_handoff_sweep.py nangate45/<design> ...` |
| [orfs_check.md](orfs_check.md) | HPWL and legality against OpenROAD's database and `check_placement` | HPWL within 0.00002%, 23/23 legal verdicts, 23/23 broken placements rejected by both | `python tools/validate_orfs.py` |
| [slow_loop.md](slow_loop.md) | the ORFS hand-off tools; noise of the ORFS flow | identity round trip bit-identical on gcd, aes, ibex, jpeg and ethmac; routed wirelength resolved to about 0.3% per run | `tools/orfs_roundtrip.py <design> <run> identity` or `jitter<seed>` |
| [speed_vs_openroad.md](speed_vs_openroad.md) | time per timing update, against OpenROAD kept warm | 21-27x faster per GP update, 21-54x per DP round (RTX 4090) | not in `tools/` |
