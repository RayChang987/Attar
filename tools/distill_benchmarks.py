#!/usr/bin/env python3
"""Distill the downloaded benchmarks (benchmarks/, see benchmarks/BENCHMARKS.md) into zoo/<mode>/<suite>/<design>:
after_fp for placement problems (die, IO pins, macros given), before_fp for floorplan problems (blocks unplaced).

usage: distill_benchmarks.py [suite ...] [--only design,...]     (no suite: all)  |  --list
Each suite says where its files are and which reader takes them; every reader hands over only the facts its
format has (see the manifest of each design and d.status())."""
import glob
import os
import re
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
from attar.bookshelf import distill_bookshelf  # noqa: E402
from attar.paths import zoo_dir  # noqa: E402
from attar.distill import distill, source_ref  # noqa: E402
from attar.fpformats import distill_blocks, distill_yal  # noqa: E402

B = os.path.join(ROOT, 'benchmarks')
A7 = os.path.join(B, 'placement', 'iccad2025_c', 'ASAP7')


def ispd2015():
    for d in sorted(glob.glob(os.path.join(B, 'placement', 'ispd2015', 'mgc_*'))):
        extra = {}
        c = os.path.join(d, 'placement.constraints')
        if os.path.exists(c):
            m = re.search(r'maximum_utilization\s*=\s*([\d.]+)\s*%', open(c).read())
            if m:
                extra['target_density'] = float(m.group(1)) / 100        # the contest's density limit
                extra['target_density_source'] = 'placement.constraints maximum_utilization'
        yield os.path.basename(d), 'lefdef', dict(def_path=os.path.join(d, 'floorplan.def'),
                                                  lef_paths=[os.path.join(d, 'tech.lef'), os.path.join(d, 'cells.lef')], extra=extra)


def ispd_route(year):
    base = os.path.join(B, 'routing', 'ispd20%d' % year)
    for d in sorted(glob.glob(os.path.join(base, 'ispd%d_test*' % year)), key=lambda p: int(p.rsplit('test', 1)[1])):
        n = os.path.basename(d)
        df = os.path.join(d, n + '.input.def')
        lef = os.path.join(d, n + '.input.lef')
        extra = {}
        if not os.path.exists(df) or not os.access(df, os.R_OK):          # ISPD19 test4: the ICCAD19 copy
            alt = os.path.join(B, 'routing', 'iccad2019_gr', n)
            df, lef = os.path.join(alt, n + '.input.def'), os.path.join(alt, n + '.input.lef')
            extra['note'] = 'DEF/LEF from the ICCAD 2019 problem C copy (the ISPD original is missing here)'
        guide = glob.glob(os.path.join(d, '*.guide'))
        if guide:
            extra['unread_sources'] = {'route_guide': guide[0]}
        yield n.split('_')[1], 'lefdef', dict(def_path=df, lef_paths=[lef], extra=extra)


def ispd2024():
    d = os.path.join(B, 'routing', 'ispd2024_gr', 'nangate45')
    for f in sorted(glob.glob(os.path.join(d, 'def', '*.def'))):
        n = os.path.splitext(os.path.basename(f))[0]
        unread = {k: os.path.join(d, 'Simple_inputs', n + '.' + k) for k in ('cap', 'net')
                  if os.path.exists(os.path.join(d, 'Simple_inputs', n + '.' + k))}
        yield n, 'lefdef', dict(def_path=f, lef_paths=[os.path.join(d, 'lef', 'Nangate.lef')], extra={'unread_sources': unread})


def iccad2025():
    libs = sorted(glob.glob(os.path.join(A7, 'LIB', '*.lib')))
    base_lefs = [os.path.join(A7, 'techlef', 'asap7_tech_1x_201209.lef')] + sorted(glob.glob(os.path.join(A7, 'LEF', '*.lef')))
    for sub in ('ICCAD25_testcases', 'hidden'):
        for d in sorted(glob.glob(os.path.join(B, 'placement', 'iccad2025_c', sub, '*'))):
            n = os.path.basename(d)
            if not os.path.exists(os.path.join(d, n + '.def')):
                continue
            unread = {k: os.path.join(d, n + '.' + k) for k in ('v', 'sdc', 'spef') if os.path.exists(os.path.join(d, n + '.' + k))}
            yield n, 'lefdef', dict(def_path=os.path.join(d, n + '.def'), lef_paths=base_lefs + [os.path.join(d, n + '.lef')],
                                    lib_paths=libs, extra={'unread_sources': unread, 'split': 'hidden' if sub == 'hidden' else 'public'})


def ispd2005():
    for aux in sorted(glob.glob(os.path.join(B, 'placement', 'ispd2005', '*', '*.aux'))):
        n = os.path.splitext(os.path.basename(aux))[0]
        snaps = {os.path.basename(f)[len(n) + 1:-3].replace('.', '_'): f       # adaptec1.dreamplace.lg.pl -> dreamplace_lg
                 for f in glob.glob(os.path.join(os.path.dirname(aux), n + '.*.pl'))}
        yield n, 'bookshelf', dict(aux=aux, snapshots=snaps)


def gsrc_place():
    for aux in sorted(glob.glob(os.path.join(B, 'placement', 'gsrc_bookshelf_place', '*.aux'))):
        yield os.path.splitext(os.path.basename(aux))[0], 'bookshelf', dict(aux=aux)


def ispd2006_hb():
    for aux in sorted(glob.glob(os.path.join(B, 'floorplan', 'ispd06_hb', '*', '*', '*_capo.aux'))):
        yield os.path.basename(aux)[:-len('_capo.aux')], 'bookshelf', dict(aux=aux)


def mcnc():
    for f in sorted(glob.glob(os.path.join(B, 'floorplan', 'mcnc', '*.yal.txt'))):
        yield os.path.basename(f)[:-len('.yal.txt')], 'yal', dict(path=f)


def blocks(sub):
    def gen():
        for f in sorted(glob.glob(os.path.join(B, 'floorplan', *sub.split('/'), '*.blocks'))):
            if os.path.exists(f[:-len('.blocks')] + '.nets'):
                yield os.path.basename(f)[:-len('.blocks')], 'blocks', dict(blocks_path=f)
    return gen


SUITES = {
    'ispd2005': ispd2005, 'ispd2015': ispd2015, 'ispd2018': lambda: ispd_route(18), 'ispd2019': lambda: ispd_route(19),
    'ispd2024': ispd2024, 'iccad2025': iccad2025,
    'gsrc_place': gsrc_place, 'ispd2006_hb': ispd2006_hb,
    'mcnc': mcnc, 'gsrc_hard': blocks('gsrc/hard'), 'gsrc_soft': blocks('gsrc/soft'), 'mcnc_soft': blocks('mcnc/soft_modified'),
}
# floorplan problems: the blocks (macros) are unplaced, and the die too where the format has none
BEFORE_FP = {'ispd2006_hb', 'mcnc', 'gsrc_hard', 'gsrc_soft', 'mcnc_soft'}


def mode(suite):
    return 'before_fp' if suite in BEFORE_FP else 'after_fp'



def run(suite, name, kind, kw):
    out = os.path.join(zoo_dir(), mode(suite), suite, name)
    extra = dict(kw.pop('extra', {}) or {}, suite=suite)
    if 'unread_sources' in extra:                 # by name + sha256, like every source
        extra['unread_sources'] = {k: source_ref(v) for k, v in extra['unread_sources'].items()}
    if kind == 'lefdef':
        m = distill(kw['def_path'], kw['lef_paths'], out, lib_paths=kw.get('lib_paths', ()), extra=extra)
    elif kind == 'bookshelf':
        m = distill_bookshelf(kw['aux'], out, snapshots=kw.get('snapshots'), extra=extra)
    elif kind == 'yal':
        m = distill_yal(kw['path'], out, extra=extra)
    else:
        m = distill_blocks(kw['blocks_path'], out, extra=extra)
    return m


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--list' in sys.argv:
        for s, g in SUITES.items():
            print('%-12s %s' % (s, ' '.join(n for n, _, _ in g())))
        sys.exit(0)
    only = None
    if '--only' in sys.argv:
        only = set(sys.argv[sys.argv.index('--only') + 1].split(','))
        args = [a for a in args if a != sys.argv[sys.argv.index('--only') + 1]]
    for suite in args or SUITES:
        for name, kind, kw in SUITES[suite]():
            if only and name not in only:
                continue
            t = time.time()
            try:
                m = run(suite, name, kind, kw)
                c = m['counts']
                print('%-11s %-22s %8d cells %6d macros %6d IO %8d nets  facts %-60s %s %.1fs' % (
                    suite, name, c['cells'], c['macros'], c['terminals'], c['nets'], ','.join(sorted(m['facts'])),
                    m.get('not_stored') or '', time.time() - t), flush=True)
            except Exception as e:  # noqa: BLE001
                print('%-11s %-22s FAILED %r' % (suite, name, e), flush=True)
