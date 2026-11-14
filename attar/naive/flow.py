"""Naive flows: the engines in the order a design needs them, each step's answer becoming a fact.

floorplan  make_die (if a utilization is given), then whatever is missing: IO pins in name order, macros, a quick
           placement, IO pins again next to the cells they connect to; the cells end unplaced (floorplan.py).
place      global_place (gp.py) + legalize (lg.py), at the source's own target density.
"""
import time

from .floorplan import make_die, place_io, place_macros


def floorplan(d, utilization=None, aspect_ratio=1.0, io_pitch=None, halo=None, packing=False, packing_args=None,
              verbose=True):
    """Make the floorplan of d: a new die if `utilization` is given, then place what is missing (IO pins, macros).
    IO pins first go in name order; after a quick placement of the cells (the mixed-size GP of the macro placer, or a
    coarse GP when there are no macros to place) they move next to the cells they connect to. The cells end
    unplaced: the floorplan is the question for attar.naive.place.
    packing=True: macros by B*-tree packing instead of GP + LP legalization (block-level designs; see place_macros)."""
    from .gp import global_place
    t0 = time.time()
    res = {}
    if utilization is not None:
        res['die'] = make_die(d, utilization, aspect_ratio, verbose)
    need_io, need_mac = 'floorplan.io' not in d.facts, 'floorplan.macros' not in d.facts
    if not (need_io or need_mac):
        if verbose:
            print('  floorplan: IO pins and macros are placed already (%s, %s)' % (d.facts['floorplan.io'], d.facts['floorplan.macros']))
        return res
    if need_io:
        place_io(d, mode='order', pitch=io_pitch, verbose=False)
    gp_done = False
    if need_mac:
        res['macros'] = place_macros(d, halo=halo, packing=packing, packing_args=packing_args, verbose=verbose)
        gp_done = res['macros']['macros'] > 0
    if need_io:
        if not gp_done:
            global_place(d, stop_overflow=0.2, verbose=False, snapshot='_floorplan')
            d.snapshots.pop('_floorplan', None)
        res['io'] = place_io(d, mode='nets', pitch=io_pitch, verbose=verbose)
    d.unplace()
    res['seconds'] = time.time() - t0
    if verbose:
        print('  floorplan done in %.2fs: %s' % (res['seconds'], ', '.join('%s<%s>' % (k, d.facts[k]) for k in
                                                                          ('floorplan.die', 'floorplan.io', 'floorplan.macros') if k in d.facts)))
    return res


def place(d, target_density=None, verbose=True, gp_args=None, lg_args=None, macro_halo=0.0):
    """Place the standard cells: attar.naive.global_place (cells spread to at most target_density: default the
    source's own limit when it has one: ISPD 2015 maximum_utilization, the density ORFS's own global placement used;
    else 0.9), then attar.naive.legalize.
    Returns HPWL, legality, time."""
    if target_density is None:
        target_density = d.manifest.get('target_density') or 0.9
    from ..evaluator import Evaluator
    from .gp import global_place
    from .lg import legalize
    t0 = time.time()
    if not bool(d.movable.any()):
        d.require('global_place')
        d.pos = d.pos                                  # nothing to place: the floorplan is the placement
        if verbose:
            print('  place: no movable cells (the floorplan places everything)')
        return {'hpwl': d.hpwl().item(), 'legal': None, 'seconds': time.time() - t0}
    gp = global_place(d, target_density=target_density, verbose=False, macro_halo=macro_halo, **(gp_args or {}))
    lg = legalize(d, verbose=False, macro_halo=macro_halo, **(lg_args or {}))
    ev = Evaluator(d)
    leg = ev.legality()
    res = {'hpwl_gp': gp['hpwl'], 'hpwl': ev.hpwl().item(), 'legal': leg['legal'], 'gp_iterations': gp['iterations'],
           'target_density': target_density,
           'gp_seconds': gp['seconds'], 'lg_seconds': lg['seconds'], 'mean_disp': lg['mean_disp'], 'seconds': time.time() - t0}
    if verbose:
        print('  place: GP %d it (%.2fs), LG (%.2fs), HPWL %.4g (GP %.4g), legal %s, %.2fs'
              % (gp['iterations'], gp['seconds'], lg['seconds'], res['hpwl'], gp['hpwl'], res['legal'], res['seconds']))
    return res
