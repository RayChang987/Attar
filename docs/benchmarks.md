# Benchmarks

The pip package ships readers, not data. The zoo of distilled designs lives on the Hugging Face Hub
(`RayChang987/attar-zoo`) and is downloaded on first use, or built locally from the raw benchmarks. It has two modes,
by what the user solves:

- **`before_fp`, floorplan problems.** The netlist and the cell library are given; the die (utilization), the IO
  pins and the macros are the answer. A result with a cell library is written back as a DEF for a flow that draws
  the power grid; Attar draws no power grid itself.
- **`after_fp`, placement problems.** The die, IO pins, macros and the power grid are given; the cells are the
  answer, then their sizes, buffers, clock trees and global routes.

Each design is a folder `<mode>/<suite>/<design>` with `core.safetensors` (integer arrays), `names.json.gz`, `tech_profile.safetensors` and
`manifest.json` (facts, provenance, counts, source hashes).

## Where Attar looks

The zoo is `zoo/` in a source checkout, else `~/.cache/attar/zoo`; every function that reads it takes `root=`.
There are no environment variables. A design is self-contained: loading, evaluating, placing (macros on their
routing tracks included) and plotting read only its folder. Manifests record each source file by name and sha256,
never by path. `d.export(out_def)` writes a DEF from the design's own frame (see [concepts](concepts.md#self-contained-designs-and-export)).

## Publishing the zoo

```bash
python tools/zoo_release.py      # checks manifests, packs zoo_hub/, writes index.json and README.md (the card) into both
hf upload-large-folder RayChang987/attar-zoo zoo_hub/ --repo-type dataset
```

On the hub the arrays and large JSON files are zstd-compressed (`<file>.zst`, level 19: 4.8 GB becomes 1.1 GB);
`attar.fetch` / `attar.load` decompress them on download, so a local zoo always holds the plain files. `zoo_hub/`
mirrors the zoo: unchanged files are not compressed again, and files the zoo no longer has are removed from it. The
upload does not delete files on the hub; remove stale ones there with `hf repo-files delete`.

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
| ICCAD 2015 problem C, superblue (the DREAMPlace 4.0 copy, `iccad2015.ot`) | LEF/DEF; the DEF has no NETS, the nets come from the flat Verilog (one instance per statement, named connections) | `after_fp/iccad2015` |
| MCNC (YAL), GSRC blocks, IBM-HB+ | YAL, GSRC blocks, Bookshelf | `before_fp/<suite>` |
| ORFS designs (asap7, nangate45; nangate45 designs ported to ASAP7 by `tools/orfs_port_asap7.py`) | LEF/DEF of the run's 3_3 (2_1) stage | `after_fp/orfs_<platform>` (`before_fp/orfs_<platform>`) |
| i3dPDK_NS3K (pa_pdn): Innovus floorplans (rows cut only at the macros, `PG_HALO_STRIP_CUT=macro`), cells and nets from the same design's Xplace input DEF | LEF/DEF | `after_fp/ns3k`, `after_fp/ns3k_{pl,st}_u{60,70,80}` |

The ORFS tools use the docker image `openroad/orfs:26Q4-368-g9ca2a9e76`.

## What each reader provides

| reader | formats | facts |
| --- | --- | --- |
| `attar.io.distill` | LEF/DEF | netlist, geometry, pin offsets, die, IO pins and macros (when placed), rows, cell library, tracks, pin directions; Liberty files recorded (fidelity F1) |
| `attar.io.bookshelf` | Bookshelf (.aux) | netlist, geometry, pin offsets, pin directions (`I`/`O` in .nets), die (rows + fixed nodes), IO, macros, rows; no cell library |
| `attar.io.fpformats.distill_yal` | YAL (MCNC) | netlist, block geometry, pin offsets, die (parent outline), pads; blocks unplaced |
| `attar.io.fpformats.distill_blocks` | GSRC/UCSC .blocks | netlist; hard blocks: geometry, pins at block centers; soft blocks: area and aspect range only (no geometry) |

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
| iccad2015 | 8 | 768k-1.93M | placed (Encounter; 1.6k-41k overlapping cells) | re-placed legal, 16-45 s per design, HPWL -11.6% to +3.2% against the given placement; timing attached, against OpenSTA: WNS within 0.33%, TNS within 0.13% ([validation](validation/timing_routing.md#iccad-2015-superblue)) |
| iccad2025 | 13 | 2k-298k | placed | re-placed legal |
| gsrc_place | 4 | 0.8k-28k | cells unplaced | legal |
| ns3k | 14 | 0.3k-170k | cells unplaced | legal. OpenROAD `check_placement` passes and global routing has no overflow on the 6 designs without macros, ariane133, bp_multi_top and riscv32i; on ariane136, black_parrot, swerv_wrapper and tinyRocket `check_placement` crashes (signal 11) on the floorplan DEF itself, with no cells placed |
| ns3k_{pl,st}_u{60,70,80} | 6 x 14 | 0.9k-1.18M | cells unplaced | Innovus strip-cut floorplans at 60/70/80% std-cell density over the rows, M5 pillar (pl) or M3 stripe (st) PDN. Attar: all 84 legal. OpenROAD `check_placement` + global routing (no overflow) pass on 62; it crashes on the floorplan DEF itself for black_parrot, bp_multi_top and bp_quad at every setting, swerv_wrapper at 70% and tinyRocket at 80% |
| orfs_asap7 | 15 | 0.4k-1.05M | cells unplaced | legal, also by OpenROAD `check_placement`; 6 routed by ORFS with 0 DRC. ORFS's asap7 designs, swerv and dynamic_node ported from nangate45, and jpeg_x20 (20 x jpeg, the million-cell scaling test) |
| orfs_nangate45 | 8 | 0.7k-109k | cells unplaced | legal, also by OpenROAD `check_placement` |

**ICCAD 2015 timing and published numbers.** Attar times superblue as OpenSTA does (table above). The
timing-driven placement papers on these designs (DREAMPlace 4.0 and the work built on it) report OpenTimer's numbers,
and the two timers disagree on a minority of endpoints. At one superblue1 placement, Attar's TNS was 3-4x OpenTimer's:
70% of the 145k endpoints were within 0.5 ns of each other, and the 11% that were 1.5 ns or more worse in Attar
carried most of its TNS. The likely cause, not fully confirmed: the `lcb*` local clock buffers are inverters, so under
OpenSTA's semantics their registers capture on the clock's falling edge, which leaves paths from input ports and to
output ports half a cycle; OpenTimer appears not to apply this. Compare methods on one timer. The ranking of two
published methods held under both (their TNS ratio was 1.48 under OpenTimer and 1.25 under Attar, over 8 designs),
but Attar's absolute numbers on these designs cannot be compared with published ones. Hold checks read each
design's Early library (manifest `library_min`), as OpenSTA's `read_liberty -min`; hold slacks match OpenSTA's within
0.11-0.21 ps on average per endpoint ([validation](validation/timing_routing.md#iccad-2015-superblue)). A
superblue run with global placement's CUDA graphs and the timer needs about 10 GB of GPU memory.

**`before_fp`**: floorplan problems (the die, IO pins and macros are the answer):

| suite | designs | cells | starts from | Attar result |
| --- | ---: | --- | --- | --- |
| orfs_asap7 | 5 | 9k-52k | ORFS 2_1: die, rows, tracks initialized; IO pins, macros unplaced; no PDN | die for a looser and a tighter utilization than ORFS's, IO pins, macros; handed to ORFS, which draws the PDN: `check_placement` passes, no more DRV than ORFS's own floorplan ([validation](validation/fp_handoff.md)). cva6 and swerv_wrapper are left out: their tight floorplans fail in ORFS |
| orfs_nangate45 | 8 | 34k-212k, 10-136 macros | ORFS 2_1, as orfs_asap7; the macro-placement benchmarks (ariane133/136, black_parrot, bp_be/fe/multi_top, swerv_wrapper, mempool_group); `macro_channel` = ORFS's `RTLMP_MIN_CHANNEL_SIZE` | handed to ORFS at 0.8x and 1.15x ORFS's utilization: 15 of 16 go through PDN and detailed placement with `check_placement` passing; ariane133 tight fails ORFS's PDN channel repair ([validation](validation/fp_handoff.md#nangate45-macro-designs-before_fporfs_nangate45)) |
| mcnc | 5 | 9-49 blocks | blocks unplaced, pads placed | floorplan, no overlaps; no cell library: Attar is the end point |
| gsrc_hard | 6 | 10-300 blocks | no die | die + floorplan, no overlaps |
| ispd2006_hb | 28 | 81-8.8k | hard + soft blocks | some overlaps remain |
| gsrc_soft, mcnc_soft | 11 | 9-300 blocks | soft blocks | no geometry: block shapes are part of the answer, no engine yet |

## Data quirks worth knowing

- ICCAD 2025: 11 of 13 inputs have every cell in N orientation, also in FS rows, so they are rail-illegal as given.
- ICCAD 2015: the given placements overlap (1.6k-41k cells), the DEF writes coordinates as decimals
  (`7214680.000000`), the LEFs define metal1-6 (superblue18 metal1-4) while the DEFs have TRACKS above them (OpenROAD's DEF
  parser stops on them), some IO pins sit just outside the die, and FIXED pinless blocks and INV_X1 cells that the Verilog never
  names serve as obstacles. Timing reads the Late library for setup and the Early one for hold (`library_min`).
  The Liberty has no `function` attributes, so there are no sizing classes (`d.resize`, buffering need them).
- superblue (ISPD 2015 and ICCAD 2015): the fixed macros overlap each other as given (2-359 pairs per design).
- ISPD 2018/2019: the DEFs contain FILLCELLs (`unplace()` sets them aside); ispd19_test3 has placed pads.
- ISPD 2019 test7-10 run at about 96% utilization; GP raises its target density to the utilization.
- ORFS 3_3 DEFs have rows cut around macros; the gaps are obstacles (OpenROAD DPL-0004 otherwise).
- ORFS's own nangate45 bp_quad config stops in its PDN step (PDN-0179, channels between macros), so it is not in the
  zoo; bp_quad is there on NS3K (ns3k_{pl,st}_u{60,70,80}).
- Designs whose flow floorplan (ORFS or Innovus) leaves the std cells under 50% of the row area are left out: too
  sparse to be a placement problem. Among them are most macro-heavy ORFS configs (BlackParrot, ariane, swerv_wrapper,
  mempool) and the NS3K bp_be_top and bp_fe_top, whose die the IO pins set. Raising their ORFS CORE_UTILIZATION to
  70-80% stops in ORFS's macro placer (MPL-0065 with the platform's 20 um channels).
- mcnc/xerox has no signal pads; IBM-HB+ blocks are often smaller than a row.
- Pin directions as the sources state them: superblue (ISPD 2015) has nets with two OUTPUT pins and nets with
  none; ISPD 2018/2019 have a few hundred nets driven by two flops; GSRC ibm05's I/O marks are inconsistent. The
  ORFS, NS3K and other ISPD 2015 designs have exactly one driver per net. MCNC and GSRC blocks state no directions.
- i3dPDK_NS3K has no one-site filler, so OpenROAD's `check_placement` forbids one-site gaps between cells there; the
  NS3K reference flow (Innovus) does not (its own placements of gcd and aes have 57 and 2,937 such gaps), and
  Attar does not enforce it (`tools/validate_openroad.py --allow-one-site-gaps` checks NS3K without it). Its PDN via stacks from the M0 rails cross the M1 pin layer: Attar keeps signal pins
  clear of them (fact `pdn`).

## Adding a reader

A reader writes the IR with `attar.io.distill.save_arrays` and `save_names_manifest` (or `attar.io.fpformats._write`)
and lists in `manifest['facts']` exactly the facts its format provides. `tests/test_readers.py` shows what to check.
