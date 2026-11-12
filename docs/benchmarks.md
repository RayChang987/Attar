# Benchmarks

The pip package ships readers, not data. The zoo of distilled designs lives on the Hugging Face Hub
(`RayChang987/attar-zoo`) and is downloaded on first use, or built locally from the raw benchmarks. It has two modes,
by what the user solves:

- **`before_fp`, floorplan problems.** The netlist and the cell library are given; the die (utilization), the IO
  pins and the macros are the answer. A result with a cell library is written back as a DEF and handed to a flow
  that draws the power grid (ORFS: `tools/orfs_handoff.py`); Attar draws no power grid itself.
- **`after_fp`, placement problems.** The die, IO pins, macros and the power grid are given; the cells are the
  answer (timing optimization and global routing later).

Each design is a folder `<mode>/<suite>/<design>` with `core.safetensors` (integer arrays), `names.json.gz`, `tech_profile.safetensors` and
`manifest.json` (facts, provenance, counts, source hashes).

## Where Attar looks

The zoo is `zoo/` in a source checkout, else `~/.cache/attar/zoo`; every function that reads it takes `root=`.
There are no environment variables. A design is self-contained: loading, evaluating, placing (macros on their
routing tracks included) and plotting read only its folder. Manifests record each source file by name and sha256,
never by path. `d.export(out_def)` writes a DEF from the design's own frame (see [concepts](concepts.md#self-contained-designs-and-export)).

## Publishing the zoo

```bash
python tools/zoo_release.py                    # checks manifests, writes zoo/index.json and zoo/README.md (the card)
hf upload-large-folder RayChang987/attar-zoo zoo/ --repo-type dataset
```

## Building the zoo

The zoo is built by the maintainers and published on the Hub; the scripts that build it read local benchmark
downloads, ORFS runs and the NS3K kit, and are not part of the repository. Each manifest records what a design was
made from (source files by name and sha256, the run and its floorplan settings for NS3K). A design of your own
goes through the same readers:

```python
import attar
attar.distill('design.def', ['tech.lef', 'cells.lef'], 'my_zoo/after_fp/mine/design')   # LEF/DEF
d = attar.load('my_zoo/after_fp/mine/design', device='cuda')
```

| source | reader | in the zoo as |
| --- | --- | --- |
| ISPD 2005/2015, GSRC placement, ICCAD 2025 | Bookshelf, LEF/DEF | `after_fp/<suite>` |
| ISPD 2018/2019/2024 routing benchmarks | LEF/DEF | `after_fp/<suite>` |
| MCNC (YAL), GSRC blocks, IBM-HB+ | YAL, GSRC blocks, Bookshelf | `before_fp/<suite>` |
| ORFS designs (asap7, nangate45; nangate45 designs ported to ASAP7 by `tools/orfs_port_asap7.py`) | LEF/DEF of the run's 3_3 (2_1) stage | `after_fp/orfs_<platform>` (`before_fp/orfs_asap7`) |
| i3dPDK_NS3K (pa_pdn): Innovus floorplans (rows cut only at the macros, `PG_HALO_STRIP_CUT=macro`), cells and nets from the same design's Xplace input DEF | LEF/DEF | `after_fp/ns3k`; `after_fp/stress` (bp_quad, a routed DEF) |

The ORFS tools use the docker image `openroad/orfs:26Q4-368-g9ca2a9e76`.

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

**`after_fp`**: placement problems (floorplan and power grid given, the cells are the answer):

| suite | designs | cells | starts from | Attar result |
| --- | ---: | --- | --- | --- |
| ispd2005 | 8 | 211k-2.2M | cells unplaced | legal |
| ispd2015 | 20 | 29k-1.3M | cells unplaced | legal; fences not enforced |
| ispd2018, ispd2019 | 20 | 8k-900k | placed | re-placed legal; FILLCELLs set aside |
| ispd2024 | 5 | 120k-715k | placed | re-placed legal |
| iccad2025 | 13 | 2k-298k | placed | re-placed legal |
| gsrc_place | 4 | 0.8k-28k | cells unplaced | legal |
| ns3k | 16 | 0.3k-170k | cells unplaced | legal. OpenROAD `check_placement` passes and global routing has no overflow on 9 of the 15 new floorplans (the 6 designs without macros, ariane133, bp_multi_top, riscv32i); on ariane136, black_parrot, bp_be_top, bp_fe_top, swerv_wrapper and tinyRocket `check_placement` crashes (signal 11) on the floorplan DEF itself, with no cells placed |
| orfs_asap7 | 32 | 0.4k-1.05M | cells unplaced | legal, also by OpenROAD `check_placement`; 9 routed by ORFS with 0 DRC. ORFS's asap7 designs, 8 nangate45 designs ported to ASAP7 (ariane133/136, black_parrot, bp_*, swerv, dynamic_node) and jpeg_x20 (20 x jpeg, the million-cell scaling test) |
| orfs_nangate45 | 18 | 0.7k-209k | cells unplaced | legal, also by OpenROAD `check_placement` |
| stress | 1 | 1.18M | placed | bp_quad on NS3K: re-placed legal, also by OpenROAD `check_placement` |

**`before_fp`**: floorplan problems (the die, IO pins and macros are the answer):

| suite | designs | cells | starts from | Attar result |
| --- | ---: | --- | --- | --- |
| orfs_asap7 | 5 | 9k-52k | ORFS 2_1: die, rows, tracks initialized; IO pins, macros unplaced; no PDN | die for a looser and a tighter utilization than ORFS's, IO pins, macros; handed to ORFS, which draws the PDN: `check_placement` passes, no more DRV than ORFS's own floorplan ([validation](validation/fp_handoff.md)). cva6 and swerv_wrapper are left out: their tight floorplans fail in ORFS |
| mcnc | 5 | 9-49 blocks | blocks unplaced, pads placed | floorplan, no overlaps; no cell library: Attar is the end point |
| gsrc_hard | 6 | 10-300 blocks | no die | die + floorplan, no overlaps |
| ispd2006_hb | 28 | 81-8.8k | hard + soft blocks | some overlaps remain |
| gsrc_soft, mcnc_soft | 11 | 9-300 blocks | soft blocks | no geometry: block shapes are part of the answer, no engine yet |

## Data quirks worth knowing

- ICCAD 2025: 11 of 13 inputs have every cell in N orientation, also in FS rows, so they are rail-illegal as given.
- ISPD 2018/2019: the DEFs contain FILLCELLs (`unplace()` sets them aside); ispd19_test3 has placed pads.
- ISPD 2019 test7-10 run at about 96% utilization; GP raises its target density to the utilization.
- ORFS 3_3 DEFs have rows cut around macros; the gaps are obstacles (OpenROAD DPL-0004 otherwise).
- ORFS's own nangate45 bp_quad config stops in its PDN step (PDN-0179, channels between macros), so it is not in the
  zoo; bp_quad is there on NS3K (stress) and ported to ASAP7.
- mcnc/xerox has no signal pads; IBM-HB+ blocks are often smaller than a row.
- Pin directions as the sources state them: superblue (ISPD 2015) has nets with two OUTPUT pins and nets with
  none; ISPD 2018/2019 have a few hundred nets driven by two flops; GSRC ibm05's I/O marks are inconsistent. The
  ORFS, NS3K and other ISPD 2015 designs have exactly one driver per net. MCNC and GSRC blocks state no directions.
- i3dPDK_NS3K has no one-site filler, so OpenROAD's `check_placement` forbids one-site gaps between cells there; the
  NS3K reference flow (Innovus) does not (its own placements of gcd and aes have 57 and 2,937 such gaps), and
  Attar does not enforce it (`tools/validate_openroad.py --allow-one-site-gaps` checks NS3K without it). Its PDN via stacks from the M0 rails cross the M1 pin layer: Attar keeps signal pins
  clear of them (fact `pdn`).

## Adding a reader

A reader writes the IR with `attar.distill.save_arrays` and `save_names_manifest` (or `attar.fpformats._write`)
and lists in `manifest['facts']` exactly the facts its format provides. `tests/test_readers.py` shows what to check.
