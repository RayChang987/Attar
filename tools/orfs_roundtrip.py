#!/usr/bin/env python3
"""ASAP7 + ORFS round trip (spec 7.2 step 3) on a finished ORFS run.

1. In the ORFS container: 3_3_place_gp.odb -> DEF (the distillation source: netlist after port buffering, IOs placed).
2. On the host: distill it, change the placement (or not), write it back into a copy of that DEF.
3. In the ORFS container: apply the DEF to 3_3_place_gp.odb (read_def -incremental) in a copy of the run,
   then let ORFS redo 3_4 resize .. 6 finish, and collect the metrics.

usage: orfs_roundtrip.py <design> <finished WORK_HOME under orfs_runs/> [mode ...] [--stage 3_3_place_gp]
                         [--cores 4] [--par 4] [--keep] [--until 5_1_grt] [--make-args "LEC_CHECK=0"]
modes: identity (write back the unchanged placement), perturb (shift 10% of the movable cells by +-3 sites),
jitter<seed> (1% of the movable cells by +-1 site), def=<path> (a DEF written by attar's Design.export).
--stage: where to distill from and inject into (3_3_place_gp, or 3_2_place_iop: before ORFS's global placement).
The stage's DEF goes to orfs_runs/rt/<design>/<stage>.def (the template Design.export stamps onto).
Summary: orfs_runs/results/<design>/roundtrip.summary.json; --keep keeps the run copies under orfs_runs/rt."""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
import attar  # noqa: E402

IMAGE = 'openroad/orfs:26Q4-368-g9ca2a9e76'
RUNS = os.path.join(ROOT, 'orfs_runs')          # mounted at /work
PLAT = os.path.join(RUNS, 'platform_asap7')
LEFS = [PLAT + '/lef/asap7_tech_1x_201209.lef', PLAT + '/lef/asap7sc7p5t_28_R_1x_220121a.lef']
LIBS = sorted(glob.glob(PLAT + '/lib/NLDM/asap7sc7p5t_*_RVT_FF_nldm_*.lib*'))


def inputs(base):
    """LEF and Liberty files of an ORFS run (from <base>.vars written by run_to_gp.sh), as host paths.
    Platform files map to the copy in platform_asap7; anything else is copied out of the image once."""
    vf = base.rstrip('/') + '.vars'
    if not os.path.exists(vf):
        return LEFS, LIBS
    v = {}
    for line in open(vf):
        if ':' in line:
            k, val = line.split(':', 1); v[k.strip()] = val.split()
    def host(p):
        pre = '/OpenROAD-flow-scripts/flow/platforms/asap7/'
        if p.startswith(pre):
            h = os.path.join(PLAT, p[len(pre):])
        else:
            h = os.path.join(RUNS, 'inputs', p.lstrip('/'))
        if not os.path.exists(h):
            os.makedirs(os.path.dirname(h), exist_ok=True)
            orfs('cp %s %s' % (p, w(h)))
        return h
    lefs = [host(p) for k in ('TECH_LEF', 'SC_LEF', 'ADDITIONAL_LEFS') for p in v.get(k, [])]
    libs = [host(p) for k in ('LIB_FILES',) for p in v.get(k, [])]
    # AUTO_MEMORIES designs generate their memory macro views during the flow
    mem = sorted(glob.glob(os.path.join(base, 'results/asap7/*/base/memories/*')))
    lefs += [f for f in mem if f.endswith('.lef')]
    libs += [f for f in mem if f.endswith('.lib') and not f.endswith('_pre_layout.lib')]
    return lefs, libs


STAGES = ('3_3_place_gp', '3_2_place_iop', '2_1_floorplan')


def suite(stage):
    return 'orfs_asap7' if stage == '3_3_place_gp' else 'orfs_asap7_' + stage[:3]


def design_config(d):
    """The ORFS DESIGN_CONFIG (container path): orfs_runs/designs/<d>/config.mk for a design of our own (e.g. the
    jpeg_x20 scaling test), else ORFS's designs/asap7/<d>/config.mk."""
    if os.path.exists(os.path.join(RUNS, 'designs', d, 'config.mk')):
        return '/work/designs/%s/config.mk' % d
    return './designs/asap7/%s/config.mk' % d


def routing_resources(d):
    """What ORFS's router may use for this design: {'layers': [lowest, highest] signal layer, 'adjustment': the share
    of every layer's capacity global routing holds back} (ORFS MIN/MAX_ROUTING_LAYER, ROUTING_LAYER_ADJUSTMENT)."""
    names = ('MIN_ROUTING_LAYER', 'MAX_ROUTING_LAYER', 'ROUTING_LAYER_ADJUSTMENT')
    out = orfs('make --no-print-directory DESIGN_CONFIG=%s %s' % (design_config(d), ' '.join('print-' + n for n in names)))
    v = {k.strip(): val.strip() for k, val in (line.split(':', 1) for line in out.splitlines() if ':' in line)}
    return {'layers': [v['MIN_ROUTING_LAYER'].strip(), v['MAX_ROUTING_LAYER'].strip()],
            'adjustment': float(v['ROUTING_LAYER_ADJUSTMENT']), 'source': 'orfs'}


def place_density(base, d):
    """The target density ORFS's global placement used for this design (its 3_3 log, GPL-0023: PLACE_DENSITY or
    the lower bound plus PLACE_DENSITY_LB_ADDON), or None."""
    log = os.path.join(base, 'logs/asap7', d, 'base', '3_3_place_gp.log')
    if not os.path.exists(log):
        return None
    v = [line.split()[-1] for line in open(log, errors='replace') if 'GPL-0023' in line]
    return float(v[-1]) if v else None


def rt_dir(d):
    """orfs_runs/rt/<design>: the stage DEFs (distillation sources and export templates) and the run copies."""
    rt = os.path.join(RUNS, 'rt', d)
    os.makedirs(rt, exist_ok=True)
    return rt


def export_and_distill(d, base, rt, stage, reuse=False):
    """<stage>.odb -> DEF -> zoo/<suite>/<d> (movable cells unplaced, the stage's own placement kept as 'reference').
    3_3_place_gp: after ORFS global placement (its netlist includes the buffers ORFS's GP inserted).
    3_2_place_iop: before it (cells at the 3_1 skip-IO GP positions; IOs placed only if GPL_PLACE_IOS=0).
    2_1_floorplan: the floorplan (IO pins and macros unplaced). reuse: distill an existing <stage>.def again."""
    res = os.path.join(base, 'results/asap7', d, 'base')
    src = os.path.join(rt, stage + '.def')
    if not (reuse and os.path.exists(src)):
        openroad('read_db %s\nwrite_def %s' % (w(res + '/%s.odb' % stage), w(src)), d + '_export')
    lefs, libs = inputs(base)
    ir = os.path.join(ROOT, 'zoo', suite(stage), d)
    ref = {'3_3_place_gp': 'orfs 3_3_place_gp (global_place)',
           '3_2_place_iop': 'orfs 3_1_place_gp_skip_io (global_place without IOs)'}.get(stage, 'orfs ' + stage)
    cfg = design_config(d)
    m = attar.distill(src, lefs, ir, lib_paths=libs, reference=ref, facts={'slow_loop': 'orfs:asap7'},
                      extra={'suite': 'orfs/asap7', 'orfs_image': IMAGE, 'orfs_stage': stage, 'routing': routing_resources(d),
                             # the placement target density of the flow: Design.place's default for this design
                             'target_density': place_density(base, d), 'target_density_source': 'orfs 3_3 global_place (GPL-0023)',
                             'orfs': {'image': IMAGE, 'design': d, 'design_config': cfg, 'stage': stage,
                                      'platform': 'asap7'}})
    return ir, m, lefs


def w(p):
    """host path under orfs_runs -> container path."""
    return '/work/' + os.path.relpath(p, RUNS)


def orfs(cmd):
    r = subprocess.run(['docker', 'run', '--rm', '-u', '%d:%d' % (os.getuid(), os.getgid()), '-v', RUNS + ':/work', IMAGE,
                        'bash', '-c', 'cd /OpenROAD-flow-scripts/flow && ' + cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    return r.stdout


def openroad(tcl, tag):
    path = os.path.join(RUNS, 'tmp_%s.tcl' % tag)
    with open(path, 'w') as f:
        f.write(tcl + '\nexit\n')
    try:
        return orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash %s' % w(path))
    finally:
        os.remove(path)


def components_only(src, dst):
    """Header + COMPONENTS: read_def -incremental would otherwise add the PINS (and ROWS) a second time."""
    keep = True
    with open(src) as fi, open(dst, 'w') as fo:
        for line in fi:
            k = line.split()[0] if line.strip() else ''
            if k in ('VERSION', 'DIVIDERCHAR', 'BUSBITCHARS', 'DESIGN', 'UNITS'):
                fo.write(line)
            elif k == 'COMPONENTS':
                keep = True; fo.write(line)
            elif k == 'END' and line.split()[1:2] == ['COMPONENTS']:
                fo.write(line); keep = False
            elif keep and line.startswith(' '):
                fo.write(line)
            else:
                keep = False
        fo.write('END DESIGN\n')


def metrics(work, d):
    L = os.path.join(work, 'logs/asap7', d, 'base')
    j = lambda s: json.load(open(os.path.join(L, s + '.json'))) if os.path.exists(os.path.join(L, s + '.json')) else {}
    dp, grt, rt, fin = j('3_5_place_dp'), j('5_1_grt'), j('5_2_route'), j('6_report')
    return {'dp_hpwl_est': dp.get('detailedplace__dpl__route__wirelength__estimated'),
            'dpl_disp_total': dp.get('detailedplace__dpl__instance__displacement__total'),
            'grt_wl': grt.get('globalroute__global_route__wirelength'),
            'routed_wl': rt.get('detailedroute__route__wirelength'), 'drc': rt.get('detailedroute__route__drc_errors'),
            'wns': fin.get('finish__timing__setup__ws'), 'tns': fin.get('finish__timing__setup__tns')}


def gpl_final_hpwl(work, d):
    for line in open(os.path.join(work, 'logs/asap7', d, 'base', '3_3_place_gp.log')):
        if 'Final HPWL' in line:
            return float(line.split()[-1])


def move(x, frac, sites, seed):
    """Shift a random fraction of the movable cells by up to +-sites sites (x only)."""
    g = torch.Generator().manual_seed(seed)
    mv = torch.nonzero(x.movable).flatten().cpu()
    pick = mv[torch.randperm(len(mv), generator=g)[:max(1, int(len(mv) * frac))]].to(x.device)
    site = float(x.raw['r_site_w'][0]) / x.dbu
    step = torch.randint(-sites, sites + 1, (len(pick),), generator=g)
    step[step == 0] = 1
    x.pos[pick, 0] += step.to(x.device).float() * site


def prepare(d, base, rt, ir, mode, dev, stage):
    """Write the mode's placement back, copy the run and inject it into 3_3_place_gp.odb."""
    if mode.startswith('def='):  # a DEF written by attar's export (Design.export)
        out_def = os.path.abspath(mode[4:])
        if not out_def.startswith(RUNS + os.sep):  # must be visible in the container
            shutil.copy(out_def, os.path.join(rt, os.path.basename(out_def))); out_def = os.path.join(rt, os.path.basename(out_def))
        attar.distill(out_def, inputs(base)[0], os.path.join(rt, 'ir_' + _tag(mode)))
        hp = attar.load(os.path.join(rt, 'ir_' + _tag(mode)), dev).hpwl().item()
        return _inject(d, base, rt, _tag(mode), out_def, hp, stage)
    x = attar.load(ir, dev).restore('reference')
    if mode == 'perturb':
        move(x, 0.10, 3, 0)
    elif mode.startswith('jitter'):
        move(x, 0.01, 1, int(mode[6:]))
    elif mode != 'identity':
        raise ValueError(mode)
    hp = x.hpwl().item()
    out_def = os.path.join(rt, 'attar_%s_%s.def' % (stage[:3], mode))
    x.export(out_def)
    return _inject(d, base, rt, mode, out_def, hp, stage)


def _tag(mode):
    return os.path.splitext(os.path.basename(mode[4:]))[0] if mode.startswith('def=') else mode


def _inject(d, base, rt, mode, out_def, hp, stage):
    """Apply out_def's COMPONENTS to <stage>.odb of a copy of the run and save it as 3_3_place_gp.odb, so ORFS
    continues with resize (3_4). From 3_2 this replaces ORFS's whole global_place step, netlist edits included."""
    work = os.path.join(rt, mode if stage == '3_3_place_gp' else '%s_%s' % (stage[:3], mode))
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(base, work, symlinks=True)
    r = os.path.join(work, 'results/asap7', d, 'base')
    comp_def = os.path.join(rt, mode + '.components.def')
    components_only(out_def, comp_def)
    openroad('read_db %s\nread_def -incremental %s\nwrite_db %s' % (w(r + '/%s.odb' % stage), w(comp_def), w(r + '/3_3_place_gp.odb')),
             d + '_' + mode)
    for sub in ('results', 'logs', 'reports'):
        for f in glob.glob(os.path.join(work, sub, 'asap7', d, 'base', '*')):
            b = os.path.basename(f)
            stale_gp = stage != '3_3_place_gp' and b.startswith('3_3') and not b.endswith('.odb')  # ORFS GP's logs
            if b[:1] in '3456' and not b.startswith(('3_1', '3_2', '3_3')) or b.startswith(('route', 'final', 'cts', 'drt', 'congestion')) or stale_gp:
                os.remove(f) if os.path.isfile(f) else shutil.rmtree(f)
    # a run from a given netlist (SYNTH_NETLIST_FILES, e.g. jpeg_x20) has no 1_1_yosys_canonicalize.rtlil: make would
    # redo synthesis and the whole flow after it (the injection lost); an empty one older than 1_2_yosys.v stops that
    rtlil, y2 = os.path.join(r, '1_1_yosys_canonicalize.rtlil'), os.path.join(r, '1_2_yosys.v')
    if not os.path.exists(rtlil) and os.path.exists(y2):
        open(rtlil, 'w').close()
        t0 = os.stat(y2).st_mtime_ns - 10**9
        os.utime(rtlil, ns=(t0, t0))
    # make treats 3_3 as up to date: 3_2_place_iop.odb (not kept by ORFS) is recreated older than it
    p32 = r + '/3_2_place_iop.odb'
    if not os.path.exists(p32):
        shutil.copy(r + '/3_1_place_gp_skip_io.odb', p32)
    # whole nanoseconds: float seconds can round 3_2 below 3_1 and make then reruns 3_2 and 3_3 (lost injection)
    t1 = os.stat(r + '/3_1_place_gp_skip_io.odb').st_mtime_ns + 10**9
    os.utime(p32, ns=(t1, t1))
    t3 = time.time_ns(); os.utime(r + '/3_3_place_gp.odb', ns=(t3, t3))
    return work, hp, t3


def finish(d, work, cores, target='', make_args=''):
    t = time.time()
    try:
        orfs('make NUM_CORES=%d DESIGN_CONFIG=%s WORK_HOME=%s %s %s > %s 2>&1'
             % (cores, design_config(d), w(work), make_args, target, w(os.path.join(work, 'make.log'))))
        ok = True
    except RuntimeError:
        ok = False
    return ok, round(time.time() - t)


if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('design'); ap.add_argument('run', help='finished WORK_HOME under orfs_runs/')
    ap.add_argument('modes', nargs='*', default=['identity', 'perturb'])
    ap.add_argument('--stage', default='3_3_place_gp', choices=STAGES[:2])
    ap.add_argument('--cores', type=int, default=4, help='NUM_CORES per ORFS run')
    ap.add_argument('--par', type=int, default=4, help='ORFS runs at once')
    ap.add_argument('--keep', action='store_true', help='keep the run copies (each a full ORFS run, up to GBs)')
    ap.add_argument('--until', default='', help='a make target to stop at (e.g. 5_1_grt); default: the whole flow')
    ap.add_argument('--make-args', default='', help='extra make variables, e.g. LEC_CHECK=0 (use the same as the baseline)')
    a = ap.parse_args()
    d, base, modes, STAGE = a.design, os.path.join(RUNS, a.run), a.modes, a.stage
    rt = rt_dir(d)

    # 1. distillation source
    t = time.time()
    ir, m, LEFS = export_and_distill(d, base, rt, STAGE)
    print('distilled %s: %d cells (%d movable, %d macros), %d nets in %.1fs' % (d, m['counts']['cells'], m['counts']['movable'],
          m['counts']['macros'], m['counts']['nets'], time.time() - t), flush=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_m = metrics(base, d)
    x0 = attar.load(ir, dev)
    print(x0.summary(), flush=True)
    base_m['hpwl_attar'] = x0.restore('reference').hpwl().item() if STAGE == '3_3_place_gp' else None
    if STAGE == '3_3_place_gp':
        print('HPWL of the ORFS GP result: attar %.2f um, OpenROAD gpl %.2f um' % (base_m['hpwl_attar'], gpl_final_hpwl(base, d)), flush=True)
    rows = [('orfs (baseline)', base_m)]

    prep = {mode: prepare(d, base, rt, ir, mode, dev, STAGE) for mode in modes}
    with ThreadPoolExecutor(a.par) as P:
        done = dict(zip(modes, P.map(lambda mo: finish(d, prep[mo][0], a.cores, a.until, a.make_args), modes)))
    for mode in modes:
        work, hp, now = prep[mode]; ok, sec = done[mode]; mode = _tag(mode)
        # make must not redo global placement: the injected 3_3_place_gp.odb has to be untouched
        rerun = os.stat(os.path.join(work, 'results/asap7', d, 'base', '3_3_place_gp.odb')).st_mtime_ns != now
        ok = ok and not rerun  # make redid global placement: the injected placement was overwritten
        mm = metrics(work, d) if ok else {}; mm['hpwl_attar'] = hp; mm['flow_s'] = sec; mm['ok'] = ok
        print('%s: %s, 3_4..finish in %ds (global placement redone by make: %s)' % (mode, 'ok' if ok else 'FAILED', sec, rerun))
        rows.append((mode, mm))

    keys = ['hpwl_attar', 'dp_hpwl_est', 'grt_wl', 'routed_wl', 'drc', 'wns', 'tns']
    print('\n%-16s' % 'run' + ''.join('%13s' % k for k in keys))
    for name, mm in rows:
        print('%-16s' % name + ''.join('%13s' % ('%.2f' % mm[k] if isinstance(mm.get(k), float) else mm.get(k)) for k in keys))
    res = os.path.join(RUNS, 'results', d); os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, 'roundtrip.summary.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    if not a.keep:                                     # each mode's run copy is a full ORFS run (up to GBs)
        for mode in modes:
            shutil.rmtree(prep[mode][0], ignore_errors=True)
