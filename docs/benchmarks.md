# Benchmarks

The pip package ships readers, not data. The zoo of distilled designs lives on the Hugging Face Hub
(`RayChang987/attar-zoo`) and is downloaded on first use, or built locally from the raw benchmarks. Each design is a
folder `<suite>/<design>` with `core.safetensors` (integer arrays), `names.json.gz`, `tech_profile.safetensors` and
`manifest.json` (facts, provenance, counts, source hashes).

## Where Attar looks

The zoo is `zoo/` in a source checkout, else `~/.cache/attar/zoo`; every function that reads it takes `root=`.
There are no environment variables. A design is self-contained: loading, evaluating, placing (macros on their
routing tracks included) and plotting read only its folder. Manifests record each source file by name and sha256,
never by path. To get a DEF back, hand `Design.export` a DEF of the design (see [concepts](concepts.md#self-contained-designs-and-export)).

## Publishing the zoo

```bash
python tools/zoo_release.py                    # checks manifests, writes zoo/index.json and zoo/README.md (the card)
hf upload-large-folder RayChang987/attar-zoo zoo/ --repo-type dataset
```

## Building the zoo

| source | put it in | build |
| --- | --- | --- |
| ISPD 2005/2015, GSRC placement, ICCAD 2025 | `benchmarks/placement/<suite>/` | `python tools/distill_benchmarks.py <suite>` |
| ISPD 2018/2019/2024 routing benchmarks | `benchmarks/routing/<suite>/` | `python tools/distill_benchmarks.py <suite>` |
| MCNC (YAL), GSRC blocks, IBM-HB+ | `benchmarks/floorplan/...` | `python tools/distill_benchmarks.py mcnc gsrc_hard ...` |
| ORFS asap7 designs | `orfs_runs/<design>_base` (`tools/orfs_run_to_gp.sh`) | `python tools/distill_orfs.py <design>`; `--stage 2_1_floorplan` for the floorplan stage |

`python tools/distill_benchmarks.py --list` shows what each suite expects. The ORFS tools use the docker image
`openroad/orfs:26Q4-368-g9ca2a9e76`.

## What each reader provides

| reader | formats | facts |
| --- | --- | --- |
| `attar.distill` | LEF/DEF | netlist, geometry, pin offsets, die, IO pins and macros (when placed), rows, cell library, tracks, pin directions; Liberty files recorded (fidelity F1) |
| `attar.bookshelf` | Bookshelf (.aux) | netlist, geometry, pin offsets, pin directions (`I`/`O` in .nets), die (rows + fixed nodes), IO, macros, rows; no cell library |
| `attar.fpformats.distill_yal` | YAL (MCNC) | netlist, block geometry, pin offsets, die (parent outline), pads; blocks unplaced |
| `attar.fpformats.distill_blocks` | GSRC/UCSC .blocks | netlist; hard blocks: geometry, pins at block centers; soft blocks: area and aspect range only (no geometry) |

Rules every reader follows:

- **Only what the format has.** Fence regions, blockages, route guides, SDC and SPEF are counted or recorded by
  name (`not_stored`, `unread_sources` in the manifest), never half-claimed as a fact.
- **Placed non-core instances are floorplan.** Macros, pads, corners and endcaps that the source places are frozen.
- **Units are kept exact.** DEF in DBU; Bookshelf in the smallest of 1/2, 1/10, 1/100, 1/1000 unit that stores
  every value exactly (IBM-HB+'s sqrt(area) blocks are rounded to 1/1000 and the error is recorded).

## The zoo today

| suite | designs | cells | starts from | Attar result |
| --- | ---: | --- | --- | --- |
| ispd2005 | 8 | 211k-2.2M | cells unplaced | legal |
| ispd2015 | 20 | 29k-1.3M | cells unplaced | legal; fences not enforced |
| ispd2018, ispd2019 | 20 | 8k-900k | placed | re-placed legal; FILLCELLs set aside |
| ispd2024 | 5 | 120k-715k | placed | re-placed legal |
| iccad2025 | 13 | 2k-298k | placed | re-placed legal |
| gsrc_place | 4 | 0.8k-28k | cells unplaced | legal |
| ns3k | 16 | 0.3k-170k | cells unplaced | legal (`tools/distill_pa_pdn.py --ns3k <kit>`) |
| orfs_asap7 | 9 | 0.4k-125k | cells unplaced | legal; ORFS round trip |
| orfs_asap7_2_1, _3_2 | 9 | 9k-108k | IO, macros unplaced | floorplan + legal; ORFS hand-off |
| stress/bp_quad | 1 | 1.18M | placed | from scratch, legal |
| mcnc | 5 | 9-49 blocks | blocks unplaced | floorplan, no overlaps |
| gsrc_hard | 6 | 10-300 blocks | no die | die + floorplan, no overlaps |
| ispd2006_hb | 28 | 81-8.8k | hard + soft blocks | some overlaps remain |
| gsrc_soft, mcnc_soft | 11 | 9-300 blocks | soft blocks | no geometry |

## Data quirks worth knowing

- ICCAD 2025: 11 of 13 inputs have every cell in N orientation, also in FS rows, so they are rail-illegal as given.
- ISPD 2018/2019: the DEFs contain FILLCELLs (`unplace()` sets them aside); ispd19_test3 has placed pads.
- ISPD 2019 test7-10 run at about 96% utilization; GP raises its target density to the utilization.
- ORFS 3_3 DEFs have rows cut around macros; the gaps are obstacles (OpenROAD DPL-0004 otherwise).
- mcnc/xerox has no signal pads; IBM-HB+ blocks are often smaller than a row.
- Pin directions as the sources state them: superblue (ISPD 2015) has nets with two OUTPUT pins and nets with
  none; ISPD 2018/2019 have a few hundred nets driven by two flops; GSRC ibm05's I/O marks are inconsistent. The
  ORFS, NS3K and other ISPD 2015 designs have exactly one driver per net. MCNC and GSRC blocks state no directions.

## Adding a reader

A reader writes the IR with `attar.distill.save_arrays` and `save_names_manifest` (or `attar.fpformats._write`)
and lists in `manifest['facts']` exactly the facts its format provides. `tests/test_readers.py` shows what to check.
