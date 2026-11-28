# Benchmarks

The pip package ships readers, not data. The zoo of distilled designs is on the Hugging Face Hub
(`RayChang987/attar-zoo`) and downloads on first use. Each design is a folder `<mode>/<suite>/<design>` with
`core.safetensors` (integer arrays), `names.json.gz`, `tech_profile.safetensors` and `manifest.json` (facts,
provenance, counts, source hashes). The modes are explained in [concepts](concepts.md#modes); a design is
self-contained ([concepts](concepts.md#self-contained-designs-and-export)).

The zoo is `zoo/` in a source checkout, else `~/.cache/attar/zoo`; every function that reads it takes `root=`.
Publishing it is in [releasing.md](releasing.md).

## Your own design

```python
import attar
attar.distill('design.def', ['tech.lef', 'cells.lef'], 'my_zoo/after_fp/mine/design')   # LEF/DEF
d = attar.load('my_zoo/after_fp/mine/design', device='cuda')
```

| reader | formats | facts |
| --- | --- | --- |
| `attar.io.distill` | LEF/DEF | netlist, geometry, pin offsets, die, IO pins and macros (when placed), rows, cell library, tracks, pin directions |
| `attar.io.bookshelf` | Bookshelf (.aux) | netlist, geometry, pin offsets, pin directions, die, IO, macros, rows; no cell library |
| `attar.io.fpformats.distill_yal` | YAL (MCNC) | netlist, block geometry, pin offsets, die, pads |
| `attar.io.fpformats.distill_blocks` | GSRC/UCSC .blocks | netlist; hard blocks with geometry; soft blocks with area and aspect range only |

Every reader stores only what its format has: fence regions, blockages, route guides, SDC and SPEF are counted or
named in the manifest (`not_stored`, `unread_sources`), never half-claimed. Placed macros, pads and endcaps are
frozen as floorplan. Units stay exact (DEF in DBU; Bookshelf in the smallest decimal unit that stores every value).
A new reader lists in `manifest['facts']` exactly the facts its format provides; `tests/test_readers.py` shows
what to check.

## The zoo today

The zoo was built by the maintainers from the raw benchmarks, ORFS runs (`openroad/orfs:26Q4-368-g9ca2a9e76`) and
the NS3K kit; each manifest records its sources by name and sha256.

**`after_fp`**: placement problems.

| suite | designs | cells | starts from | Attar result |
| --- | ---: | --- | --- | --- |
| ispd2005 | 8 | 211k-2.2M | cells unplaced | legal |
| ispd2015 | 20 | 29k-1.3M | cells unplaced | legal; fences not enforced |
| ispd2018, ispd2019 | 20 | 8k-900k | placed | re-placed legal; FILLCELLs set aside |
| ispd2024 | 5 | 120k-715k | placed | re-placed legal |
| iccad2015 | 8 | 768k-1.93M | placed (1.6k-41k overlapping cells) | re-placed legal, 16-45 s per design; timing within 0.33% WNS of OpenSTA ([validation](validation/timing_routing.md#iccad-2015-superblue)) |
| iccad2025 | 13 | 2k-298k | placed | re-placed legal |
| gsrc_place | 4 | 0.8k-28k | cells unplaced | legal |
| ns3k | 14 | 0.3k-170k | cells unplaced | legal; OpenROAD `check_placement` crashes on 4 floorplan DEFs |
| ns3k_{pl,st}_u{60,70,80} | 6 x 14 | 0.9k-1.18M | cells unplaced | Innovus floorplans at 60/70/80% density, M5 pillar or M3 stripe PDN; all 84 legal |
| orfs_asap7 | 15 | 0.4k-1.05M | cells unplaced | legal, also by `check_placement`; includes jpeg_x20 (1.05M cells) |
| orfs_nangate45 | 8 | 0.7k-109k | cells unplaced | legal, also by `check_placement` |

**`before_fp`**: floorplan problems.

| suite | designs | cells | starts from | Attar result |
| --- | ---: | --- | --- | --- |
| orfs_asap7 | 5 | 9k-52k | ORFS 2_1, no PDN | handed to ORFS: `check_placement` passes, no more DRVs than ORFS's floorplan ([validation](validation/fp_handoff.md)) |
| orfs_nangate45 | 8 | 34k-212k, 10-136 macros | ORFS 2_1 (macro-placement benchmarks) | 15 of 16 go through ORFS's PDN and detailed placement ([validation](validation/fp_handoff.md#nangate45-macro-designs-before_fporfs_nangate45)) |
| mcnc | 5 | 9-49 blocks | blocks unplaced, pads placed | no overlaps |
| gsrc_hard | 6 | 10-300 blocks | no die | die + floorplan, no overlaps |
| ispd2006_hb | 28 | 81-8.8k | hard + soft blocks | some overlaps remain |
| gsrc_soft, mcnc_soft | 11 | 9-300 blocks | soft blocks | no engine yet (block shapes are the answer) |

## ICCAD 2015 timing and published numbers

Attar times superblue as OpenSTA does. The papers on these designs report OpenTimer, and the two timers disagree on
a minority of endpoints: at one superblue1 placement Attar's TNS was 3-4x OpenTimer's. A likely cause: the `lcb*`
clock buffers invert, so under OpenSTA their registers capture on the falling edge. Compare methods on one timer;
the ranking of two published methods held under both (TNS ratio 1.48 under OpenTimer, 1.25 under Attar). Hold
uses each design's Early library (`library_min`). A superblue run needs about 10 GB of GPU memory.

## Data quirks

- ICCAD 2025: 11 of 13 inputs have every cell in N orientation, also in FS rows (rail-illegal as given).
- ICCAD 2015: overlapping given placements; decimal DEF coordinates; TRACKS above the LEF's layers; some IO pins
  just outside the die; no Liberty `function`, so no sizing.
- superblue: fixed macros overlap each other as given (2-359 pairs per design).
- ISPD 2018/2019: FILLCELLs in the DEFs (set aside); test7-10 at about 96% utilization.
- ORFS 3_3 DEFs cut rows around macros; the gaps are obstacles.
- Flow floorplans under 50% std-cell density are left out (too sparse); this drops most macro-heavy ORFS configs.
- Pin directions as the sources state them: superblue has nets with two drivers or none; ISPD 2018/2019 a few
  hundred nets with two.
- i3dPDK_NS3K has no one-site filler, so OpenROAD forbids one-site gaps; Innovus and Attar do not
  (`tools/validate_openroad.py --allow-one-site-gaps`).
