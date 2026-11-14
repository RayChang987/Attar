"""Make the zoo ready to upload to the Hugging Face Hub: zoo/index.json (what attar.zoo(remote=True) reads) and
zoo/README.md (the dataset card), then check that no manifest holds a path (sources are recorded by name + sha256).

  python3 tools/zoo_release.py [--zoo DIR]     # writes the two files into the zoo (default: zoo/)
  hf upload-large-folder RayChang987/attar-zoo zoo/ --repo-type dataset     # then upload (docs/benchmarks.md)
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from attar import paths  # noqa: E402

# <mode>/<suite> -> (what it is, where the raw files come from, terms)
SUITES = {
    'after_fp/ispd2005': ('ISPD 2005 placement contest (adaptec1-4, bigblue1-4), with a DREAMPlace legalized placement as '
                 'the dreamplace_lg snapshot', 'ISPD 2005 contest (Bookshelf)', 'academic benchmark'),
    'before_fp/ispd2006_hb': ('IBM-HB+ (ibm01-18) and Calypto hierarchical floorplanning benchmarks',
                    'vlsicad.eecs.umich.edu/BK/ISPD06bench (Bookshelf, *_capo.aux)', 'academic benchmark'),
    'after_fp/ispd2015': ('ISPD 2015 blockage-aware detailed-routing-driven placement contest (20 mgc_* designs)',
                 'ISPD 2015 contest (LEF/DEF)', 'academic benchmark'),
    'after_fp/ispd2018': ('ISPD 2018 initial detailed routing contest, test1-10', 'ISPD 2018 contest (LEF/DEF)', 'academic benchmark'),
    'after_fp/ispd2019': ('ISPD 2019 initial detailed routing contest, test1-10 (test4 from the ICCAD 2019 problem C copy)',
                 'ISPD 2019 contest (LEF/DEF)', 'academic benchmark'),
    'after_fp/ispd2024': ('ISPD 2024 GPU/ML-enhanced global routing contest, the smaller designs (ariane133_51/68, nvdla, '
                 'mempool_tile, bsg_chip) on Nangate45', 'ISPD 2024 contest (LEF/DEF)', 'academic benchmark'),
    'after_fp/iccad2025': ('ICCAD 2025 CAD Contest problem C (incremental placement on ASAP7), public and hidden cases',
                  'ICCAD 2025 contest (LEF/DEF)', 'academic benchmark'),
    'after_fp/gsrc_place': ('GSRC Bookshelf placement examples (ibm05, p1Unit)', 'vlsicad.ucsd.edu GSRC Bookshelf',
                   'academic benchmark'),
    'before_fp/gsrc_hard': ('GSRC floorplanning n10-n300, hard blocks', 'vlsicad.eecs.umich.edu/BK/GSRCbench', 'academic benchmark'),
    'before_fp/gsrc_soft': ('GSRC floorplanning n10-n300, soft blocks (at their .blocks shape)',
                  'vlsicad.eecs.umich.edu/BK/GSRCbench', 'academic benchmark'),
    'before_fp/mcnc': ('MCNC floorplanning apte, xerox, hp, ami33, ami49 (YAL)', 'MCNC benchmarks (YAL)', 'academic benchmark'),
    'before_fp/mcnc_soft': ('MCNC floorplanning, soft-block variants', 'MCNC benchmarks (blocks)', 'academic benchmark'),
    'after_fp/orfs_asap7': ('OpenROAD-flow-scripts designs on ASAP7 at global placement (3_3), with ORFS\'s placement as '
                   'the reference snapshot; ORFS\'s asap7 designs, plus nangate45 designs ported to ASAP7 with the '
                   'RTL unmodified (tools/orfs_port_asap7.py: FakeRAM2.0 asap7 views of their fakeram45 memories)',
                   'ORFS runs (docker image in each manifest)', 'ASAP7: BSD-3-Clause; designs: their ORFS licenses'),
    'after_fp/orfs_nangate45': ('OpenROAD-flow-scripts designs on Nangate45 (FreePDK45) at global placement (3_3), with '
                       'ORFS\'s placement as the reference snapshot', 'ORFS runs (docker image in each manifest)',
                       'Nangate45 (ORFS platform): Apache-2.0; designs: their ORFS licenses'),
    'before_fp/orfs_asap7': ('ORFS designs on ASAP7 at the floorplan stage (2_1): the die, IO pins and macros are the '
                             'answer; d.export writes the floorplan as a DEF for a flow that draws the PDN and goes on', 'ORFS runs', 'ASAP7: BSD-3-Clause; designs: their ORFS licenses'),
    'after_fp/ns3k': ('Open designs placed on the i3dPDK_NS3K v1.1 generic GAAFET kit (Innovus enablement)',
             'i3dPDK_NS3K (github.com/i3dknu1/i3dPDK_NS3K)', 'CC BY-NC-SA 4.0 (i3dPDK_NS3K)'),
    **{'after_fp/ns3k_%s_u%d' % (p, u): (
        'The same open designs on i3dPDK_NS3K at %d%% std-cell density over the rows (Innovus strip-cut floorplan, %s PDN)'
        % (u, {'pl': 'M5 pillar', 'st': 'M3 stripe'}[p]),
        'i3dPDK_NS3K (github.com/i3dknu1/i3dPDK_NS3K)', 'CC BY-NC-SA 4.0 (i3dPDK_NS3K)')
       for u in (60, 70, 80) for p in ('pl', 'st')},
    'after_fp/stress': ('bp_quad on i3dPDK_NS3K: a 1.2M-cell routed DEF, for scaling tests',
               'i3dPDK_NS3K (github.com/i3dknu1/i3dPDK_NS3K)', 'CC BY-NC-SA 4.0 (i3dPDK_NS3K)'),
}

CARD = """---
license: other
license_name: mixed-academic-benchmarks
pretty_name: Attar zoo
tags:
- eda
- physical-design
- placement
- floorplanning
- vlsi
size_categories:
- n<1K
---

# Attar zoo

{n} chip designs from {k} public physical-design benchmark suites, distilled into the
[Attar](https://github.com/RayChang987/Attar) IR: netlist, cell and macro geometry, pin offsets, die, rows and IO pins
as safetensors arrays, ready to load onto a GPU in a second or two.

```python
import attar                                   # pip install attar
d = attar.load('after_fp/ispd2005/adaptec1')  # downloads this design on first use
attar.zoo(remote=True)                         # every design here
attar.fetch('after_fp/ispd2015')               # a whole suite
```

## Layout

Two modes. `before_fp/`: floorplan problems, the die (utilization), IO pins and macros are the answer.
`after_fp/`: placement problems, the floorplan and power grid are given and the cells are the answer.
`<mode>/<suite>/<design>/` holds `core.safetensors` (the arrays), `names.json.gz` (cell, net, master and layer names),
`manifest.json` (source, facts, counts) and, for LEF/DEF sources, `tech_profile.safetensors`. `index.json` lists
every design with its size and facts.

## Self-contained

Loading, evaluating (HPWL, density, legality), placing with `attar.naive` (macros on their routing tracks included)
and plotting need only this dataset. Manifests name each source file with its sha256, never a path. A LEF/DEF design
also keeps its source DEF's frame (`frame.def.gz`: PDN, IO pins, fixed cells), so `d.export('out.def')` writes a
complete DEF with no raw file at hand. Bookshelf, YAL and GSRC designs have no cell library behind them, so they do
not export. `d.status()` shows what a design can do.

## Timing and cell libraries

{n_timing} designs (the ORFS, NS3K and ICCAD 2025 suites) carry timing: Liberty arcs, pin capacitances, the SDC's
clocks and I/O delays, and wire RC, distilled, so `d.timing()` / `d.timing_report()` run with no .lib or .sdc at hand.
Their cells come from a shared cell library in `libraries/<name>/` (every cell of the platform corner: all drive
strengths and threshold-voltage flavors, Liberty tables, LEF geometry, equivalence classes, buffers and inverters, for
sizing and buffering); cells only one design has (its SRAMs) are in its own `library.safetensors`.
`attar.load` / `attar.fetch` download the libraries a design references.

| library | cells | equivalence classes | designs | size |
| --- | ---: | ---: | ---: | ---: |
{lib_rows}

## Suites

| suite | designs | largest (cells) | size | contents | source | terms |
| --- | ---: | ---: | ---: | --- | --- | --- |
{rows}

## Terms

The designs are derived from the benchmarks listed above and remain under their original terms. The contest and
academic suites are published by their organizers for research use; cite the contest or benchmark when you report
results on it. The `ns3k` and `stress` suites are derived from
[i3dPDK_NS3K](https://github.com/i3dknu1/i3dPDK_NS3K) and are distributed under CC BY-NC-SA 4.0; cite
Taigon Song, "i3dPDK_NS3K_v1.1", I3DVLSI Laboratory, Kyungpook National University, 2026.
"""


def main(root=None):
    zoo = paths.zoo_dir(root)
    designs, per = {}, {}
    bad = []
    for m in sorted(glob.glob(os.path.join(zoo, '*', '*', '*', 'manifest.json'))):
        folder = os.path.dirname(m)
        mode, s, name = folder.split(os.sep)[-3:]
        if mode not in paths.MODES:
            continue
        suite = '%s/%s' % (mode, s)
        with open(m) as f:
            man = json.load(f)
        if any('/' in v for v in _strings(man.get('source', {}))) or 'run' in man.get('orfs', {}):
            bad.append(m)
        size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(folder, '*')))
        c = man.get('counts', {})
        designs['%s/%s' % (suite, name)] = {
            'design': man.get('design'), 'format': man['source'].get('format'), 'fidelity': man.get('fidelity'),
            'cells': c.get('cells', 0), 'movable': c.get('movable', 0), 'macros': c.get('macros', 0),
            'terminals': c.get('terminals', 0), 'nets': c.get('nets', 0), 'pins': c.get('pins', 0),
            'facts': sorted(man.get('facts', {})), 'bytes': size}
        p = per.setdefault(suite, [0, 0, 0])
        p[0] += 1; p[1] = max(p[1], c.get('cells', 0)); p[2] += size
    if bad:
        raise SystemExit('paths in %d manifests (attar-ir < 0.5: distill them again): %s' % (len(bad), bad[0]))
    missing = sorted(set(per) - set(SUITES))
    if missing:
        raise SystemExit('describe these suites in SUITES first: %s' % ', '.join(missing))
    # shared cell libraries (attar.library bundles) the designs reference
    libraries = {}
    for lj in sorted(glob.glob(os.path.join(zoo, 'libraries', '*', 'library.json'))):
        folder = os.path.dirname(lj)
        with open(lj) as f:
            meta = json.load(f)
        users = sorted(k for k, v in designs.items() if 'library' in v['facts']
                       and json.load(open(os.path.join(zoo, k, 'manifest.json'))).get('library', {}).get('bundle')
                       == os.path.basename(folder))
        libraries[os.path.basename(folder)] = {
            'cells': len(meta['cells']), 'classes': len(meta.get('classes', [])), 'designs': len(users),
            'sources': [r['name'] for r in meta['sources'].get('lib', [])],
            'bytes': sum(os.path.getsize(os.path.join(folder, x)) for x in os.listdir(folder))}
    with open(os.path.join(zoo, 'index.json'), 'w') as f:
        json.dump({'schema': 'attar-zoo-index/1', 'designs': designs, 'libraries': libraries}, f, indent=1)
    rows = '\n'.join('| `%s` | %d | %s | %s | %s | %s | %s |' % (s, n, '{:,}'.format(cmax), _mb(b), *SUITES[s])
                     for s, (n, cmax, b) in sorted(per.items()))
    lib_rows = '\n'.join('| `%s` | %d | %d | %d | %s |' % (k, v['cells'], v['classes'], v['designs'], _mb(v['bytes']))
                         for k, v in sorted(libraries.items()))
    n_timing = sum('timing' in v['facts'] for v in designs.values())
    with open(os.path.join(zoo, 'README.md'), 'w') as f:
        f.write(CARD.format(n=len(designs), k=len(per), rows=rows, lib_rows=lib_rows, n_timing=n_timing))
    total = sum(v['bytes'] for v in designs.values())
    print('%d designs in %d suites, %s; wrote index.json and README.md in %s' % (len(designs), len(per), _mb(total), zoo))


def _mb(b):
    return '%.1f GB' % (b / 1e9) if b >= 1e9 else '%.0f MB' % (b / 1e6) if b >= 1e6 else '%.0f kB' % (b / 1e3)


def _strings(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _strings(v)
    elif isinstance(o, str):
        yield o


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--zoo', help='the zoo folder (default: zoo/ in the checkout)')
    main(ap.parse_args().zoo)
