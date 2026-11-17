#!/usr/bin/env python3
"""OpenROAD's global routes, and its timing on them, for a zoo design at its reference placement: the truth Attar's
global-routing parasitics and global router are checked against.

On the ORFS run's 3_3_place_gp.odb (the placement the zoo design was distilled from) OpenROAD (the ORFS image):
  - reads the design's Liberty files, its SDC (2_floorplan.sdc) and the platform's setRC.tcl,
  - routes with the flow's settings (routing layers from the manifest, the platform's fastroute.tcl,
    global_route -congestion_iterations 30 -allow_congestion) but without pin_access: pins connect from their
    shapes (GlobalRouter::findOnGridPositions without access points), the information the IR has,
  - writes the routes (write_global_route_segments), the pins' on-grid positions (write_pin_locations), the
    parasitics (estimate_parasitics -global_routing -spef_file) and every endpoint's setup and hold slack,
  - dumps every master's pin shapes (master frame, in odb order) and every IO pin's shapes: grt's pin model
    (attar.route.pins) reads them.
Cached in orfs_runs/gr_par/<run key>/. usage: gr_truth.py <design | platform/design> ... [--redo]"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'tools'))
from orfs_roundtrip import PLATFORMS, RUNS, base_dir, flow_dir, orfs, split, w  # noqa: E402

OUT = os.path.join(RUNS, 'gr_par')
CLK_LAYER = {'asap7': 'M4', 'nangate45': 'metal4'}          # the platforms' MIN_CLK_ROUTING_LAYER

DUMP_TCL = r'''
proc dump_ep {path} {
  set f [open $path w]
  foreach p [all_registers -data_pins] {
    puts $f "[get_full_name $p] [get_property $p slack_max] [get_property $p slack_min]"
  }
  foreach p [all_outputs] {
    puts $f "[get_full_name $p] [get_property $p slack_max] [get_property $p slack_min]"
  }
  close $f
}
proc dump_shapes {path} {
  set f [open $path w]
  set block [ord::get_db_block]
  set used [dict create]
  foreach inst [$block getInsts] { dict set used [[$inst getMaster] getName] 1 }
  foreach lib [[ord::get_db] getLibs] {
    foreach m [$lib getMasters] {
      if {![dict exists $used [$m getName]]} { continue }
      puts $f "M [$m getName] [$m getType] [$m getWidth] [$m getHeight] [$m isBlock] [$m isPad]"
      foreach mt [$m getMTerms] {
        foreach mp [$mt getMPins] {
          foreach b [$mp getGeometry] {
            set L [$b getTechLayer]
            if {$L == "NULL"} { continue }
            puts $f "S [$mt getName] [$L getName] [$L getType] [$L getRoutingLevel] [$b xMin] [$b yMin] [$b xMax] [$b yMax]"
          }
        }
      }
    }
  }
  foreach bt [$block getBTerms] {
    foreach bp [$bt getBPins] {
      if {[lsearch -exact {PLACED FIRM LOCKED COVER} [$bp getPlacementStatus]] < 0} { continue }
      foreach b [$bp getBoxes] {
        set L [$b getTechLayer]
        puts $f "P [$bt getName] [$L getName] [$L getType] [$L getRoutingLevel] [$b xMin] [$b yMin] [$b xMax] [$b yMax]"
      }
    }
  }
  set die [$block getDieArea]
  puts $f "D [$die xMin] [$die yMin] [$die xMax] [$die yMax]"
  close $f
}
'''


def layer_adjustments(d, plat):
    """{layer: adjustment} as the platform's fastroute.tcl sets them (set_global_routing_layer_adjustment per layer
    range, with the flow's MIN / MAX_ROUTING_LAYER and ROUTING_LAYER_ADJUSTMENT): what OpenROAD routed with."""
    from validate_sta import platform_file
    r = d.manifest['routing']
    names = d.names['layers']
    env = {'MIN_ROUTING_LAYER': r['layers'][0], 'MAX_ROUTING_LAYER': r['layers'][1],
           'ROUTING_LAYER_ADJUSTMENT': str(r['adjustment'])}
    text = open(platform_file(plat, 'fastroute.tcl')).read().replace('\\\n', ' ')
    out = {}
    for line in text.splitlines():
        t = line.split()
        if len(t) == 3 and t[0] == 'set_global_routing_layer_adjustment':
            rng, v = t[1], t[2]
            for k, x in env.items():
                rng = rng.replace('$::env(%s)' % k, x); v = v.replace('$::env(%s)' % k, x)
            lo, hi = (rng.split('-') + [rng])[:2]
            for i in range(names.index(lo), names.index(hi) + 1):
                out[names[i]] = float(v)
    return out or float(r['adjustment'])


def same_adjustments(routing, adj):
    """Does the zoo's routing entry (one 'adjustment', or 'layer_adjustments' per layer) hold back what `adj` (from
    layer_adjustments) says, layer by layer?"""
    if not isinstance(adj, dict):
        return not routing.get('layer_adjustments') and float(routing['adjustment']) == float(adj)
    per = routing.get('layer_adjustments')
    if per:
        return {k: float(v) for k, v in per.items()} == adj
    return all(float(v) == float(routing['adjustment']) for v in adj.values())


def truth(key, redo=False):
    """Run (or read the cache of) OpenROAD's routing and timing for a run key. Returns the output folder."""
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import attar
    from validate_sta import lib_paths, platform_file, sdc_path
    plat, design = split(key)
    out = os.path.join(OUT, key)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'truth.json')) and not redo:
        return out
    d = attar.load(os.path.join(attar.paths.zoo_dir(), 'after_fp', 'orfs_' + plat, design), 'cpu')
    libs = lib_paths(d, plat, design)
    sdc = sdc_path(design, plat)
    rc = platform_file(plat, 'setRC.tcl')
    r = d.manifest['routing']
    odb = os.path.join(flow_dir(base_dir(key)), '3_3_place_gp.odb')
    o = lambda f: w(os.path.join(out, f))  # noqa: E731
    tcl = [DUMP_TCL] + ['read_liberty %s' % w(l) for l in libs] + [
        'read_db %s' % w(odb), 'read_sdc %s' % w(sdc), 'source %s' % w(rc),
        'set ::env(MIN_ROUTING_LAYER) %s' % r['layers'][0], 'set ::env(MAX_ROUTING_LAYER) %s' % r['layers'][1],
        'set ::env(MIN_CLK_ROUTING_LAYER) %s' % CLK_LAYER[plat],
        'set ::env(ROUTING_LAYER_ADJUSTMENT) %g' % r['adjustment'],
        'source %s%s/fastroute.tcl' % (PLATFORMS, plat),
        'set t0 [clock milliseconds]',
        'global_route -congestion_iterations 30 -allow_congestion -verbose',
        'puts "GR_DONE [expr {([clock milliseconds] - $t0) / 1000.0}]"',
        'write_global_route_segments %s' % o('routes.segs'),
        'grt::write_pin_locations %s' % o('pins.loc'),
        'set t1 [clock milliseconds]',
        'estimate_parasitics -global_routing -spef_file %s' % o('gr.spef'),
        'set wns [sta::worst_slack -max]', 'set tns [sta::total_negative_slack -max]',
        'puts "STA_DONE [expr {([clock milliseconds] - $t1) / 1000.0}] $wns $tns"',
        'dump_ep %s' % o('endpoints.tsv'), 'dump_shapes %s' % o('shapes.txt')]
    path = os.path.join(RUNS, 'tmp_grtruth_%s.tcl' % key.replace('/', '_'))
    with open(path, 'w') as f:
        f.write('\n'.join(tcl) + '\nexit\n')
    try:
        log = orfs('/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -no_init -no_splash -threads 4 %s' % w(path))
    finally:
        os.remove(path)
    open(os.path.join(out, 'openroad.log'), 'w').write(log)
    g = re.findall(r'GR_DONE ([\d.]+)', log)
    s = re.findall(r'STA_DONE (\S+) (\S+) (\S+)', log)
    if not g or not s:
        raise RuntimeError(log[-3000:])
    wl = re.findall(r'Total wirelength:\s*([\d.]+)\s*um', log)
    ov = re.findall(r'Total\s+\d+\s+\d+\s+[\d.]+%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)', log)
    from attar.io import liberty
    tu = liberty.read(libs[0])['time_unit_ps']
    meta = {'gr_s': float(g[-1]), 'sta_s': float(s[-1][0]), 'wns_ps': float(s[-1][1]) * tu, 'tns_ps': float(s[-1][2]) * tu,
            'gr_wl_um': float(wl[-1]) if wl else None, 'overflow': [int(v) for v in ov[-1]] if ov else None,
            'time_unit_ps': tu, 'routing': r, 'libs': [os.path.basename(l) for l in libs]}
    json.dump(meta, open(os.path.join(out, 'truth.json'), 'w'), indent=1)
    return out


def endpoints(out):
    """{endpoint name: (setup ps, hold ps)} of a truth folder."""
    tu = json.load(open(os.path.join(out, 'truth.json')))['time_unit_ps']
    ep = {}
    for l in open(os.path.join(out, 'endpoints.tsv')):
        t = l.split()
        if len(t) == 3:
            try:
                ep[t[0]] = (float(t[1]) * tu, float(t[2]) * tu)
            except ValueError:
                pass
    return ep


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('designs', nargs='+')
    ap.add_argument('--redo', action='store_true')
    a = ap.parse_args()
    for k in a.designs:
        o = truth(k, a.redo)
        print(k, json.dumps({x: v for x, v in json.load(open(os.path.join(o, 'truth.json'))).items() if x != 'routing'}))
