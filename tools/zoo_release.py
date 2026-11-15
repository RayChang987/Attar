"""Make the zoo ready to upload to the Hugging Face Hub: check that no manifest holds a path (sources are recorded by
name + sha256), pack the upload folder (zoo_hub/ beside the zoo: the arrays and large JSON files zstd-compressed as
<file>.zst, attar.hub decompresses them on download; everything else hard-linked) and write index.json (what
attar.zoo(remote=True) reads) and README.md (the dataset card) into both.

  python3 tools/zoo_release.py [--zoo DIR] [--hub DIR]     # default: zoo/ and zoo_hub/ in the checkout
  hf upload-large-folder RayChang987/attar-zoo zoo_hub/ --repo-type dataset     # then upload (docs/benchmarks.md)

Packing is incremental: a file is compressed again only when the zoo copy is newer. Files of the upload folder the
zoo no longer has are deleted (the hub keeps them until they are deleted there too).
"""
import argparse
import glob
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

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

```bash
pip install git+https://github.com/RayChang987/Attar
```

```python
import attar
d = attar.load('after_fp/ispd2005/adaptec1')  # downloads this design on first use
attar.zoo(remote=True)                         # every design here
attar.fetch('after_fp/ispd2015')               # a whole suite
```

## Layout

Two modes. `before_fp/`: floorplan problems, the die (utilization), IO pins and macros are the answer.
`after_fp/`: placement problems, the floorplan and power grid are given and the cells are the answer.
`<mode>/<suite>/<design>/` holds `core.safetensors` (the arrays), `names.json.gz` (cell, net, master and layer names),
`manifest.json` (source, facts, counts) and, for LEF/DEF sources, `tech_profile.safetensors`. `index.json` lists
every design with its size and facts. The arrays and large JSON files are stored zstd-compressed (`*.zst`);
`attar.load` / `attar.fetch` decompress them on download (`pip install zstandard`), so a local zoo holds the plain
files. Sizes below are unpacked, with the download in parentheses.

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

| library | cells | equivalence classes | designs | size (download) |
| --- | ---: | ---: | ---: | ---: |
{lib_rows}

## Suites

| suite | designs | largest (cells) | size (download) | contents | source | terms |
| --- | ---: | ---: | ---: | --- | --- | --- |
{rows}

## Terms

The designs are derived from the benchmarks listed above and remain under their original terms. The contest and
academic suites are published by their organizers for research use; cite the contest or benchmark when you report
results on it. The `ns3k` suites are derived from
[i3dPDK_NS3K](https://github.com/i3dknu1/i3dPDK_NS3K) and are distributed under CC BY-NC-SA 4.0; cite
Taigon Song, "i3dPDK_NS3K_v1.1", I3DVLSI Laboratory, Kyungpook National University, 2026.
"""


ZSTD_LEVEL = 19
PLAIN = ('index.json', 'README.md', 'manifest.json')   # stay readable on the hub; manifest.json also marks a design complete


def compressed(name):
    """Whether a zoo file goes to the hub as <name>.zst: arrays and JSON (manifests, index and card excepted)."""
    return name not in PLAIN and name.endswith(('.safetensors', '.json'))


def pack(zoo, hub, folders):
    """Mirror the files of `folders` (relative to the zoo) into the upload folder; -> {zoo relpath: hub bytes}."""
    import zstandard
    jobs, out = [], {}
    for folder in folders:
        for name in sorted(os.listdir(os.path.join(zoo, folder))):
            rel = os.path.join(folder, name)
            if os.path.isfile(os.path.join(zoo, rel)) and not name.endswith('.part'):
                jobs.append(rel)

    def one(rel):
        src = os.path.join(zoo, rel)
        dst = os.path.join(hub, rel + ('.zst' if compressed(os.path.basename(rel)) else ''))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if compressed(os.path.basename(rel)):
            if not (os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src)):
                cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=4)
                with open(src, 'rb') as fi, open(dst + '.part', 'wb') as fo:
                    cctx.copy_stream(fi, fo, size=os.path.getsize(src))
                os.replace(dst + '.part', dst)
        elif not (os.path.exists(dst) and os.path.samefile(src, dst)):
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
            except OSError:                    # another file system
                shutil.copy2(src, dst)
        return rel, dst

    with ThreadPoolExecutor(max(1, (os.cpu_count() or 4) // 4)) as ex:
        for rel, dst in ex.map(one, jobs):
            out[rel] = os.path.getsize(dst)
    keep = {os.path.join(hub, rel + ('.zst' if compressed(os.path.basename(rel)) else '')) for rel in out}
    keep |= {os.path.join(hub, x) for x in ('index.json', 'README.md')}
    for dirpath, dirnames, filenames in os.walk(hub, topdown=False):
        if any(part.startswith('.') for part in os.path.relpath(dirpath, hub).split(os.sep)):
            continue                           # .cache/huggingface: hf upload-large-folder's resume state
        for f in filenames:
            p = os.path.join(dirpath, f)
            if p not in keep:
                os.remove(p)
        if dirpath != hub and not os.listdir(dirpath):
            os.rmdir(dirpath)
    return out


def main(root=None, hub=None):
    zoo = paths.zoo_dir(root)
    hub = hub or os.path.normpath(zoo) + '_hub'
    if os.path.abspath(hub).startswith(os.path.abspath(zoo) + os.sep):
        raise SystemExit('the upload folder must not be inside the zoo: %s' % hub)
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
        designs['%s/%s' % (suite, name)] = {'folder': os.path.relpath(folder, zoo),
            'design': man.get('design'), 'format': man['source'].get('format'), 'fidelity': man.get('fidelity'),
            'cells': c.get('cells', 0), 'movable': c.get('movable', 0), 'macros': c.get('macros', 0),
            'terminals': c.get('terminals', 0), 'nets': c.get('nets', 0), 'pins': c.get('pins', 0),
            'facts': sorted(man.get('facts', {})), 'bytes': size}
    if bad:
        raise SystemExit('paths in %d manifests (attar-ir < 0.5: distill them again): %s' % (len(bad), bad[0]))
    missing = sorted({k.rpartition('/')[0] for k in designs} - set(SUITES))
    if missing:
        raise SystemExit('describe these suites in SUITES first: %s' % ', '.join(missing))
    # shared cell libraries (attar.timing.library bundles) the designs reference
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
            'bytes': sum(os.path.getsize(os.path.join(folder, x)) for x in os.listdir(folder)),
            'folder': os.path.relpath(folder, zoo)}
    sizes = pack(zoo, hub, [v['folder'] for v in designs.values()] + [v['folder'] for v in libraries.values()])
    for v in list(designs.values()) + list(libraries.values()):
        folder = v.pop('folder')
        v['download'] = sum(b for rel, b in sizes.items() if os.path.dirname(rel) == folder)
    for k, v in designs.items():
        p = per.setdefault(k.rpartition('/')[0], [0, 0, 0, 0])
        p[0] += 1; p[1] = max(p[1], v['cells']); p[2] += v['bytes']; p[3] += v['download']
    rows = '\n'.join('| `%s` | %d | %s | %s (%s) | %s | %s | %s |' % (s, n, '{:,}'.format(cmax), _mb(b), _mb(dl), *SUITES[s])
                     for s, (n, cmax, b, dl) in sorted(per.items()))
    lib_rows = '\n'.join('| `%s` | %d | %d | %d | %s (%s) |' % (k, v['cells'], v['classes'], v['designs'], _mb(v['bytes']),
                                                              _mb(v['download'])) for k, v in sorted(libraries.items()))
    n_timing = sum('timing' in v['facts'] for v in designs.values())
    card = CARD.format(n=len(designs), k=len(per), rows=rows, lib_rows=lib_rows, n_timing=n_timing)
    for d in (zoo, hub):
        with open(os.path.join(d, 'index.json'), 'w') as f:
            json.dump({'schema': 'attar-zoo-index/1', 'designs': designs, 'libraries': libraries}, f, indent=1)
        with open(os.path.join(d, 'README.md'), 'w') as f:
            f.write(card)
    total = sum(v['bytes'] for v in list(designs.values()) + list(libraries.values()))
    dl = sum(v['download'] for v in list(designs.values()) + list(libraries.values()))
    print('%d designs in %d suites, %s (download %s); wrote index.json and README.md in %s, packed %s'
          % (len(designs), len(per), _mb(total), _mb(dl), zoo, hub))


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
    ap.add_argument('--hub', help='the upload folder (default: zoo_hub/ beside the zoo)')
    a = ap.parse_args()
    main(a.zoo, a.hub)
