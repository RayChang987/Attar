#!/usr/bin/env python3
"""Port ORFS designs that have no asap7 config (nangate45's ariane, black_parrot, ...) to asap7, RTL unmodified.

Per design, orfs_runs/designs/<design>/ gets a config.mk (PLATFORM asap7, the source's Verilog read from the image),
a constraint.sdc (asap7 times are in ps) and, for the designs with fakeram45 memories, asap7 views of those memories:
FakeRAM2.0's asap7 backend (the generator ORFS's AUTO_MEMORIES uses) makes a single-port RAM of the same depth and
width that keeps the fakeram45 macro's name and pins (clk, ce_in, we_in, addr_in, wd_in, w_mask_in, rd_out), so the
RTL that instantiates fakeram45_<D>x<W> links to it as is. The memories are abstract (size, pins, timing defaults),
as fakeram45 is.

usage: orfs_port_asap7.py [design ...] [--util 50]   (default: all of PORTS); then tools/orfs_run_to_gp.sh <design>
--util: ORFS's CORE_UTILIZATION (std-cell + macro area over the core area), the same for the designs given."""
import json
import math
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from orfs_roundtrip import IMAGE, RUNS  # noqa: E402

SRC = '$(DESIGN_HOME)/src'
NG45 = '$(DESIGN_HOME)/nangate45'

# design: source (as in ORFS's nangate45 config), clocks (ports), clock period (ps), fakeram45 memories, flow settings
PORTS = {
    'ariane133': dict(name='ariane', files=[SRC + '/ariane133/ariane.sv2v.v', NG45 + '/ariane133/macros.v'],
                      clocks=['clk_i'], period=1000, mems=['256x16'], hier=1),
    'ariane136': dict(name='ariane', files=[SRC + '/ariane136/ariane.sv2v.v', NG45 + '/ariane136/macros.v'],
                      clocks=['clk_i'], period=1000, mems=['256x16'], hier=1),
    'black_parrot': dict(name='black_parrot', files=[SRC + '/black_parrot/pickled.v', NG45 + '/black_parrot/macros.v'],
                         clocks=['clk_i'], period=1000, mems=['512x64', '256x95', '64x7', '64x15', '64x96'], hier=1),
    'bp_be_top': dict(name='bp_be_top', files=[SRC + '/bp_be_top/pickled.v', NG45 + '/bp_be_top/macros.v'],
                      clocks=['clk_i'], period=800, mems=['512x64', '64x15', '64x96'], hier=1, keep=3000),
    'bp_fe_top': dict(name='bp_fe_top', files=[SRC + '/bp_fe_top/pickled.v', NG45 + '/bp_fe_top/macros.v'],
                      clocks=['clk_i'], period=800, mems=['512x64', '64x7', '64x96'], hier=1),
    'bp_multi_top': dict(name='bp_multi_top', files=[SRC + '/bp_multi_top/pickled.v', NG45 + '/bp_multi_top/macros.v'],
                         clocks=['clk_i'], period=1000, mems=['512x64', '256x96', '32x64', '64x7', '64x15', '64x96'], hier=1),
    'bp_quad': dict(name='bsg_chip', files=[SRC + '/bp_quad/bsg_chip_block.sv2v.v', NG45 + '/bp_quad/macros.v'],
                    clocks=['p_clk_A_i', 'p_clk_B_i', 'p_clk_C_i', 'p_bsg_tag_clk_i', 'p_ci_clk_i', 'p_co_clk_i',
                            'p_ci2_tkn_i', 'p_co2_tkn_i'], period=1000,
                    mems=['256x48', '32x32', '64x124', '512x64', '64x62', '128x116'], hier=1),
    'dynamic_node': dict(name='dynamic_node_top_wrap', files=[SRC + '/dynamic_node/dynamic_node.pickle.v'],
                         clocks=['clk'], period=500, mems=[], hier=0),
    'swerv': dict(name='swerv', files=[SRC + '/swerv/swerv_wrapper.sv2v.v'], clocks=['clk'], period=1000, mems=[], hier=0),
}
UTIL, LB_ADDON = 50, 0.20          # the same for every port: the flow's own floorplan, nothing hand-tuned per design


def memory(depth, width):
    """A memories.json entry (ORFS AUTO_MEMORIES schema) for a fakeram45-compatible single-port RAM."""
    a = max(1, math.ceil(math.log2(depth)))
    pin = lambda n, d, w, f: {'name': n, 'direction': d, 'width': w, 'port_id': 'RW0', 'function': f}
    return {'name': 'fakeram45_%dx%d' % (depth, width), 'rows': depth, 'bits': width, 'addr_w': a, 'rw_ports': 1,
            'mask_lanes': width, 'idiomatic': True, 'port_convention': 'fakeram45',
            'reason': 'asap7 view of fakeram45_%dx%d: same name and pins, bit write mask' % (depth, width),
            'pins': [pin('clk', 'input', 1, 'clk'), pin('ce_in', 'input', 1, 'en'), pin('we_in', 'input', 1, 'wmode'),
                     pin('addr_in', 'input', a, 'addr'), pin('wd_in', 'input', width, 'data_in'),
                     pin('w_mask_in', 'input', width, 'mask'), pin('rd_out', 'output', width, 'data_out')]}


def make_memories(mems):
    """asap7 views of the fakeram45 memories under orfs_runs/designs/_fakeram45_asap7 (made once, shared)."""
    out = os.path.join(RUNS, 'designs', '_fakeram45_asap7')
    os.makedirs(out, exist_ok=True)
    todo = [m for m in mems if not os.path.exists(os.path.join(out, 'fakeram45_%s.lef' % m))]
    if todo:
        spec = os.path.join(out, 'memories.json')
        json.dump({'version': 1, 'memories': [memory(*map(int, m.split('x'))) for m in todo]}, open(spec, 'w'), indent=1)
        subprocess.run(['docker', 'run', '--rm', '-u', '%d:%d' % (os.getuid(), os.getgid()), '-v', RUNS + ':/work', IMAGE,
                        'bash', '-c', 'cd /OpenROAD-flow-scripts/tools/FakeRAM2.0 && python3 run.py '
                        '/work/designs/_fakeram45_asap7/memories.json --orfs_asap7_backend --output_dir /tmp/m && '
                        'cp /tmp/m/*.lef /tmp/m/*.lib /work/designs/_fakeram45_asap7/ && rm -f /work/designs/_fakeram45_asap7/*_pre_layout.lib'],
                       check=True)
        os.remove(spec)
    return ['/work/designs/_fakeram45_asap7/fakeram45_%s' % m for m in mems]


def sdc(clocks, period):
    io = 0.2 * period
    L = ['# asap7 (ps). A placement problem: the clocks, and IO delays of 20% of the period on the first one.',
         'set clk_period %g' % period]
    for c in clocks:
        L.append('create_clock -name %s -period $clk_period [get_ports %s]' % (c, c))
    L += ['set clk_ports [get_ports {%s}]' % ' '.join(clocks),
          'set_input_delay %g -clock %s [delete_from_list [all_inputs] $clk_ports]' % (io, clocks[0]),
          'set_output_delay %g -clock %s [all_outputs]' % (io, clocks[0])]
    if len(clocks) > 1:
        L.append('set_clock_groups -asynchronous ' + ' '.join('-group {%s}' % c for c in clocks))
    return '\n'.join(L) + '\n'


def port(d, util=UTIL):
    p = PORTS[d]
    out = os.path.join(RUNS, 'designs', d)
    os.makedirs(out, exist_ok=True)
    mem = make_memories(p['mems']) if p['mems'] else []
    with open(os.path.join(out, 'constraint.sdc'), 'w') as f:
        f.write(sdc(p['clocks'], p['period']))
    L = ['# %s on asap7, ported from ORFS designs/nangate45/%s (tools/orfs_port_asap7.py): the RTL unmodified;' % (d, d),
         '# fakeram45_* are asap7 views of the same name and pins (FakeRAM2.0, orfs_asap7 backend).',
         'export PLATFORM = asap7', 'export DESIGN_NAME = %s' % p['name'], 'export DESIGN_NICKNAME = %s' % d,
         'export VERILOG_FILES = %s' % ' '.join(p['files']),
         'export SDC_FILE = /work/designs/%s/constraint.sdc' % d,
         'export CORE_UTILIZATION = %d' % util, 'export PLACE_DENSITY_LB_ADDON = %.2f' % LB_ADDON,
         'export SYNTH_HIERARCHICAL = %d' % p['hier']]
    if p['hier']:
        L.append('export SYNTH_MINIMUM_KEEP_SIZE = %d' % p.get('keep', 10000))
    if mem:
        L += ['export ADDITIONAL_LEFS = %s' % ' '.join(m + '.lef' for m in mem),
              'export ADDITIONAL_LIBS = %s' % ' '.join(m + '.lib' for m in mem),
              '# kepler-formal (LEC) cannot load the memory macros', 'export LEC_CHECK = 0']
    with open(os.path.join(out, 'config.mk'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('%-14s -> %s (%d memories, CORE_UTILIZATION %d)' % (d, os.path.relpath(out, ROOT), len(mem), util))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='*')
    ap.add_argument('--util', type=int, default=UTIL)
    a = ap.parse_args()
    for d in a.designs or PORTS:
        port(d, a.util)
