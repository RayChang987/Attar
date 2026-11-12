"""Design: a distilled design on the GPU, as a state (spec 5.1).

Problem facts (netlist, geometry, floorplan, rows, ...) come from the IR and may be missing; the solution state
(positions, orientations) is changed by algorithms and can be saved as named snapshots. What a Design can do follows
from the facts it has: capabilities(), missing(capability), require(capability).
Coordinates in microns (float32); `scale` = 1 / dbu."""
import gzip
import json
import os

import numpy as np
import torch

from . import paths, st
from .deffile import ORIENTS
from .distill import ASIDE, FIXED, MACRO, PLACED, TERMINAL, PHYSICAL  # noqa: F401
from .facts import CAPABILITIES, FACTS, MissingFacts, infer_facts, upgrade_facts


def _orient_offset(o, px, py, w, h):
    """Pin point (px, py) relative to the master lower-left in N -> relative to the placed lower-left, orientation o.
    All arguments are tensors; o holds orientation ids in deffile.ORIENTS order."""
    cand = torch.stack([
        torch.stack([px, py], -1),          # N
        torch.stack([w - px, h - py], -1),  # S
        torch.stack([h - py, px], -1),      # W  (R90)
        torch.stack([py, w - px], -1),      # E  (R270)
        torch.stack([w - px, py], -1),      # FN (MY)
        torch.stack([px, h - py], -1),      # FS (MX)
        torch.stack([h - py, w - px], -1),  # FW (MY R90)
        torch.stack([py, px], -1),          # FE (MX R90)
    ], 0)
    return cand[o.long(), torch.arange(len(o), device=o.device)]


class Design:
    def __init__(self, path, device='cpu'):
        self.path = path
        self.device = torch.device(device)
        a, _ = st.load(os.path.join(path, 'core.safetensors'))
        with open(os.path.join(path, 'manifest.json')) as f:
            self.manifest = json.load(f)
        self._names = None
        self.dbu = self.manifest['dbu']; self.scale = 1.0 / self.dbu
        T = lambda x, dt=None: torch.as_tensor(x, device=self.device, dtype=dt)
        s = self.scale
        self.raw = a
        self.facts = upgrade_facts(self.manifest.get('facts') or infer_facts(self.manifest, a), self.manifest)
        self.flags = T(a['c_flags'].astype(np.int32))
        self.orient = T(a['c_orient'].astype(np.int64))
        self._pos = T(a['c_xy'].astype(np.float64) * s, torch.float32)        # lower-left, um
        m = a['c_master'].astype(np.int64)
        msize = a['m_size'].astype(np.float64) * s
        size = np.zeros((len(m), 2)); size[m >= 0] = msize[m[m >= 0]]
        self.size_n = T(size, torch.float32)                                      # master (N) size
        self.movable = (self.flags & (FIXED | TERMINAL | ASIDE)) == 0
        self.placed = (self.flags & PLACED) != 0                 # objects (cells and IO pins) whose pos is meaningful
        mv = (a['c_flags'] & (FIXED | TERMINAL)) == 0
        self._placement = bool(((a['c_flags'][mv] & PLACED) != 0).all())  # solution state: all movable cells placed
        self.snapshots = {}
        if 'ref_xy' in a:
            self.snapshots['reference'] = (a['ref_xy'], a['ref_orient'])
        for k in a:  # other named placements kept at distillation, e.g. snap_dreamplace_lg_xy
            if k.startswith('snap_') and k.endswith('_xy'):
                self.snapshots[k[5:-3]] = (a[k], a[k[:-3] + '_orient'])
        # pins: master pin center (N, relative to master lower-left) -> oriented, relative to the cell center
        p_cell = a['p_cell'].astype(np.int64); p_mpin = a['p_mpin'].astype(np.int64)
        ctr = np.zeros((len(p_cell), 2))
        ok = p_mpin >= 0
        if 'pin_offsets' in self.facts:
            box = a['m_pin_box'].astype(np.float64) * s
            ctr[ok] = np.stack([(box[p_mpin[ok], 0] + box[p_mpin[ok], 2]) / 2, (box[p_mpin[ok], 1] + box[p_mpin[ok], 3]) / 2], -1)
        else:  # no pin offsets: pin_off stays at the cell center, and capabilities needing offsets are unavailable
            ok = np.zeros(len(p_cell), bool) | (p_mpin >= 0)
            ctr[ok] = size[p_cell[ok]] / 2
        self.p_cell = T(p_cell)
        # pin directions seen from the net (distill.D_IN sink, D_OUT driver, D_INOUT, D_UNKNOWN): 'pin_direction'
        self.pin_dir = T(a['p_dir'].astype(np.int64)) if 'p_dir' in a else torch.full((len(p_cell),), 3, dtype=torch.int64, device=self.device)
        self._p_ctr_n = T(ctr, torch.float32); self._p_term = T(~ok)
        self.n_ptr = T(a['n_ptr'].astype(np.int64))
        deg = self.n_ptr[1:] - self.n_ptr[:-1]
        self.p_net = torch.repeat_interleave(torch.arange(len(deg), device=self.device), deg)
        self.n_weight = T(a['n_weight'], torch.float32)
        self.n_mask = deg >= 2
        self.n_flags = T(a['n_flags'].astype(np.int32))
        self._load_dies(a, T, s)
        self.update_orient()

    def _load_dies(self, a, T, s):
        """Dies (D >= 1; a 2D design has one, its DEF DIEAREA), rows, IO pins. Boxes are (x0, y0, x1, y1) in um."""
        N = len(self.flags)
        self.cell_die = T(a['c_die'].astype(np.int64)) if 'c_die' in a else torch.zeros(N, dtype=torch.int64, device=self.device)
        self.io = None; self.io_dir = None
        if 'floorplan.die' not in self.facts:
            self.die_box = self.core_box = None
            self.row_xy = self.row_h = self.row_site_w = self.row_sites = self.row_die = self.row_box = self.row_orient = None
            return
        die = a['die_box'] if 'die_box' in a else a['die'].reshape(1, 4)       # schema 0.1 stored one 'die'
        self.die_box = T(die.astype(np.float64) * s, torch.float32)              # (D, 4)
        self._load_io(a, T)
        if 'rows' not in self.facts:
            self.row_xy = self.row_h = self.row_site_w = self.row_sites = self.row_die = self.row_box = self.row_orient = None
            self.core_box = self.die_box.clone()
            return
        # rows
        self.row_xy = T(a['r_xy'].astype(np.float64) * s, torch.float32)        # (R, 2) origin
        self.row_h = T(a['r_h'].astype(np.float64) * s, torch.float32)
        self.row_site_w = T(a['r_site_w'].astype(np.float64) * s, torch.float32)
        self.row_sites = T(a['r_sites'].astype(np.int64))
        self.row_die = T(a['r_die'].astype(np.int64)) if 'r_die' in a else torch.zeros(len(self.row_h), dtype=torch.int64, device=self.device)
        self.row_orient = T(a['r_orient'].astype(np.int64)) if 'r_orient' in a else None   # None: not known
        self.row_box = torch.stack([self.row_xy[:, 0], self.row_xy[:, 1],
                                    self.row_xy[:, 0] + self.row_sites * self.row_site_w, self.row_xy[:, 1] + self.row_h], 1)
        # core = bounding box of each die's rows (the die box when a die has no rows)
        D = len(self.die_box)
        lo = torch.full((D, 2), float('inf'), device=self.device).scatter_reduce(0, self.row_die[:, None].expand(-1, 2), self.row_box[:, :2], 'amin')
        hi = torch.full((D, 2), -float('inf'), device=self.device).scatter_reduce(0, self.row_die[:, None].expand(-1, 2), self.row_box[:, 2:], 'amax')
        core = torch.cat([lo, hi], 1)
        self.core_box = torch.where(torch.isfinite(core), core, self.die_box)

    def _load_io(self, a, T):
        """IO pins: zero-size fixed terminal cells; io indexes them in the cell arrays."""
        if 'io_cell' in a:
            self.io = T(a['io_cell'].astype(np.int64)); self.io_dir = T(a['io_dir'].astype(np.int64))
        else:
            self.io = torch.nonzero((self.flags & TERMINAL) != 0).flatten()
            self.io_dir = torch.full((len(self.io),), 2, dtype=torch.int64, device=self.device)

    # -- facts and capabilities ---------------------------------------------------------------------------------
    @property
    def present(self):
        """Facts the design has now (problem facts plus the 'placement' solution state)."""
        f = set(self.facts)
        if self._placement:
            f.add('placement')
        return f

    def has(self, fact):
        if fact == 'core':  # schema 0.1 name
            return {'netlist', 'cell_geometry'} <= set(self.facts)
        if fact not in FACTS:
            raise KeyError('unknown fact %r (known: %s)' % (fact, ', '.join(FACTS)))
        return fact in self.present

    def capabilities(self):
        have = self.present
        return sorted(c for c, req in CAPABILITIES.items() if req <= have)

    def missing(self, capability):
        return sorted(CAPABILITIES[capability] - self.present)

    def require(self, capability, ignore=()):
        miss = set(CAPABILITIES[capability]) - self.present - set(ignore)
        if miss:
            raise MissingFacts(self.manifest.get('design', self.path), capability, miss)

    def status(self, show=True):
        """Checklist of facts (✅ / ❌), solution state, capabilities, and what a hand-off to ORFS still needs."""
        from .status import status
        s = status(self)
        if show:
            print(s)
        return s

    def tier(self):
        """A: can be validated in ORFS; B: DEF write-back only; C: Attar is the end point."""
        from .status import tier
        return tier(self)

    def handoff_command(self, out_def, run_dir=None):
        """Command that hands an exported DEF to an ORFS run of this design (run_dir: its WORK_HOME) and finishes the
        flow there. A floorplan goes to ORFS only when it is legal (floorplan_legal)."""
        from .status import handoff_command
        fp = self.manifest.get('orfs', {}).get('stage') == '2_1_floorplan'
        self.require('orfs_floorplan_handoff' if fp else 'orfs_handoff')
        if fp:
            ok, issues = self.floorplan_legal()
            if not ok:
                raise ValueError('%s: the floorplan is not legal (%s); ORFS would not take it'
                                 % (self.manifest.get('design'), '; '.join(issues)))
        return handoff_command(self, out_def, run_dir)

    def floorplan_check(self, pos=None):
        """What is wrong with the floorplan, from the positions now (pos: these instead): -> (problems, a bool mask
        over the cells of the macros outside their region). Nothing is stored: the answer follows the positions.
        A macro's region is the core (the rows' outline) when an Attar engine placed the macros, the die when the
        source did: a flow may put macros where there are no rows (the IR keeps no DEF core area)."""
        problems = []
        N = len(self.size)
        outside = torch.zeros(N, dtype=torch.bool, device=self.device)
        if 'floorplan.die' not in self.facts:
            return ['no die'], outside
        p = self.pos if pos is None else pos
        mac = self.macros
        if len(mac):
            if 'floorplan.macros' not in self.facts and pos is None:
                problems.append('macros not placed')
            else:
                ours = str(self.facts.get('floorplan.macros', 'naive')).startswith('naive') or pos is not None
                cb = (self.core_box if ours else self.die_box)[self.cell_die[mac]]
                q, s = p[mac], self.size[mac]
                o = ((q < cb[:, :2] - 1e-3) | (q + s > cb[:, 2:] + 1e-3)).any(1)
                outside[mac] = o
                if bool(o.any()):
                    k = int(o.sum()); problems.append('%d macro%s outside the %s' % (k, 's' * (k > 1), 'core' if ours else 'die'))
                from .naive.floorplan import _overlapping_pairs
                ov = 0
                for die in torch.unique(self.cell_die[mac]).tolist():
                    m = self.cell_die[mac] == die
                    ov += _overlapping_pairs(q[m], s[m])
                if ov:
                    problems.append('%d overlapping macro pair%s' % (ov, 's' * (ov > 1)))
        if self.io is not None and len(self.io):
            if 'floorplan.io' not in self.facts and pos is None:
                problems.append('IO pins not placed')
            else:
                db = self.die_box[self.cell_die[self.io]]
                q = p[self.io]
                k = int(((q < db[:, :2] - 1e-3) | (q > db[:, 2:] + 1e-3)).any(1).sum())
                if k:
                    problems.append('%d IO pin%s outside the die' % (k, 's' * (k > 1)))
        return problems, outside

    def floorplan_legal(self, pos=None):
        """(legal, problems): macros inside their core and apart, IO pins inside the die, all of them placed. Checked
        from the positions each time it is called (not a stored state)."""
        problems, _ = self.floorplan_check(pos)
        return not problems, problems

    def plot(self, path=None, **kw):
        """Draw die, core, rows, cells, macros and IO pins (attar.viz.plot); saved to path if given."""
        from .viz import plot
        return plot(self, path, **kw)

    def __repr__(self):
        c = self.manifest.get('counts', {})
        return '<Design %s: %d cells, %d nets, tier %s, can %s>' % (
            self.manifest.get('design'), c.get('cells', 0), c.get('nets', 0), self.tier(), ', '.join(self.capabilities()))

    # -- solution state -----------------------------------------------------------------------------------------
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        """Assigning positions makes them the placement (all cells count as placed)."""
        self._pos = value
        self.placed = torch.ones_like(self.placed)
        self._placement = True

    # -- floorplan state (answers of the IO and macro placers become facts) -------------------------------------
    def _floorplan_changed(self, why):
        """An engine changed the floorplan (IO pins, macros, die). Export writes all of it onto the DEF. The slow loop takes a changed floorplan only from the ORFS floorplan stage (2_1: tapcells, PDN and
        everything after are redone); a later stage already built them around the old floorplan, so 'slow_loop'
        goes there."""
        if 'slow_loop' in self.facts and self.manifest.get('orfs', {}).get('stage') != '2_1_floorplan':
            self.facts.pop('slow_loop')
            self.dropped = getattr(self, 'dropped', {})
            self.dropped['slow_loop'] = why + ' after ORFS %s' % self.manifest.get('orfs', {}).get('stage')

    @property
    def macros(self):
        return torch.nonzero((self.flags & MACRO) != 0).flatten()

    def set_io(self, xy, provenance):
        """IO pin locations (T, 2) in io order; makes floorplan.io a fact."""
        p = self._pos.clone(); p[self.io] = xy.to(p.dtype)
        self._pos = p
        self.placed[self.io] = True
        self.flags[self.io] |= PLACED
        self.facts['floorplan.io'] = provenance
        self._floorplan_changed('IO pins placed by %s' % provenance)

    def release_macros(self):
        """Make the macros movable and unplaced (for a macro placer): floorplan.macros is no longer a fact."""
        m = self.macros
        self.flags[m] &= ~FIXED
        self.flags[m] &= ~PLACED
        self.movable = (self.flags & (FIXED | TERMINAL | ASIDE)) == 0
        self.placed[m] = False
        self._placement = False
        self.facts.pop('floorplan.macros', None)
        return self

    def fix_macros(self, provenance):
        """Freeze the macros where they are: floorplan.macros becomes a fact."""
        m = self.macros
        self.flags[m] |= FIXED | PLACED
        self.movable = (self.flags & (FIXED | TERMINAL | ASIDE)) == 0
        self.placed[m] = True
        self._placement = bool(self.placed[self.movable].all()) if bool(self.movable.any()) else True
        self.facts['floorplan.macros'] = provenance
        self._floorplan_changed('macros placed by %s' % provenance)
        return self

    def set_floorplan(self, die_box, rows, provenance):
        """A new die outline (D, 4) and rows {xy, h, site_w, sites, orient[, die]}: floorplan.die and rows become
        facts with this provenance. Everything placed in the old outline is stale: IO pins, macros and fixed cells
        (tap cells etc.) become unplaced and movable again, the placement and the snapshots go."""
        dev = self.device
        f32 = lambda x: torch.as_tensor(x, device=dev, dtype=torch.float32)
        self.die_box = f32(die_box).reshape(-1, 4)
        if self.io is None:
            self._load_io(self.raw, lambda x, dt=None: torch.as_tensor(x, device=dev, dtype=dt))
        if rows is None:
            self.row_xy = self.row_h = self.row_site_w = self.row_sites = self.row_die = self.row_box = self.row_orient = None
            self.core_box = self.die_box.clone()
            self.facts.pop('rows', None)
        else:
            self.row_xy, self.row_h, self.row_site_w = f32(rows['xy']), f32(rows['h']), f32(rows['site_w'])
            self.row_sites = torch.as_tensor(rows['sites'], device=dev, dtype=torch.int64)
            self.row_orient = torch.as_tensor(rows['orient'], device=dev, dtype=torch.int64)
            self.row_die = torch.as_tensor(rows['die'], device=dev, dtype=torch.int64) if 'die' in rows else \
                torch.zeros(len(self.row_h), dtype=torch.int64, device=dev)
            self.row_box = torch.stack([self.row_xy[:, 0], self.row_xy[:, 1],
                                        self.row_xy[:, 0] + self.row_sites * self.row_site_w, self.row_xy[:, 1] + self.row_h], 1)
            D = len(self.die_box)
            idx = self.row_die[:, None].expand(-1, 2)
            lo = torch.full((D, 2), float('inf'), device=dev).scatter_reduce(0, idx, self.row_box[:, :2], 'amin')
            hi = torch.full((D, 2), -float('inf'), device=dev).scatter_reduce(0, idx, self.row_box[:, 2:], 'amax')
            core = torch.cat([lo, hi], 1)
            self.core_box = torch.where(torch.isfinite(core), core, self.die_box)
            self.facts['rows'] = provenance
        self.facts['floorplan.die'] = provenance
        # stale: IO pins, macros, fixed cells, placement, snapshots
        released = int(((self.flags & FIXED) != 0).logical_and((self.flags & (MACRO | TERMINAL)) == 0).sum())
        self.flags &= ~FIXED & ~PLACED
        self.flags[self.io] |= FIXED
        self.movable = (self.flags & (FIXED | TERMINAL | ASIDE)) == 0
        self.placed = torch.zeros_like(self.placed)
        self._placement = False
        self.snapshots = {}
        self.facts.pop('floorplan.io', None)
        self.facts.pop('floorplan.macros', None)
        self._floorplan_changed('die and rows made by %s' % provenance)
        return released

    def unplace(self, set_aside_fill=True):
        """Forget the movable cells' positions: placement is no longer a fact (the floorplan facts stay).
        set_aside_fill: movable cells without connections (fill, spacer: e.g. the FILLCELLs of the ISPD 2018 / 2019
        routing benchmarks) are not part of the placement question; they are set aside (flag ASIDE): not placed,
        no obstacle, no utilization, written back UNPLACED. Returns the number set aside."""
        aside = 0
        if set_aside_fill:
            fill = self.movable & ((self.flags & PHYSICAL) != 0)
            aside = int(fill.sum())
            if aside:
                self.flags[fill] |= ASIDE
                self.placed[fill] = False
                self.movable = (self.flags & (FIXED | TERMINAL | ASIDE)) == 0
        self.placed[self.movable] = False
        self._placement = not bool(self.movable.any())     # nothing movable: the floorplan is the placement
        return aside

    # -- shortcuts to the Naive engines ---------------------------------------------------------------------------
    def floorplan(self, utilization=None, aspect_ratio=1.0, verbose=True, **kw):
        """Make the floorplan: a new die for the given utilization (std-cell area / (core - macro area)) and aspect
        ratio (height / width), if given; then place whatever is missing: IO pins, macros. Shortcut to
        attar.naive.floorplan; the answers become facts (floorplan.die, rows, floorplan.io, floorplan.macros)."""
        from .naive import floorplan
        return floorplan(self, utilization=utilization, aspect_ratio=aspect_ratio, verbose=verbose, **kw)

    def place(self, verbose=True, **kw):
        """Place the standard cells: global placement, then legalization (attar.naive.place)."""
        from .naive import place
        return place(self, verbose=verbose, **kw)

    def snapshot(self, name):
        """Save the current solution state (positions, orientations) under a name."""
        self.snapshots[name] = (self._pos.detach().clone(), self.orient.clone(), self._placement)
        return self

    def restore(self, name):
        """Restore a saved solution state; 'reference' is the placement kept at distillation (e.g. the ORFS GP)."""
        if name not in self.snapshots:
            raise KeyError('%s has no snapshot %r (have: %s)' % (self.manifest['design'], name, ', '.join(self.snapshots) or 'none'))
        snap = self.snapshots[name]
        if isinstance(snap[0], np.ndarray):  # stored at distillation, in DBU
            self.pos = torch.as_tensor(snap[0].astype(np.float64) * self.scale, device=self.device, dtype=torch.float32)
            self.orient = torch.as_tensor(snap[1].astype(np.int64), device=self.device)
        else:
            self.pos = snap[0].clone(); self.orient = snap[1].clone()
            if not snap[2]:
                self._placement = False
        self.update_orient()
        return self

    # -- dies, rows, IO ---------------------------------------------------------------------------------------
    IO_DIRS = ('INPUT', 'OUTPUT', 'INOUT', 'FEEDTHRU')

    @property
    def num_dies(self):
        return len(self.die_box)

    @property
    def die_w(self):
        return self.die_box[:, 2] - self.die_box[:, 0]          # (D,)

    @property
    def die_h(self):
        return self.die_box[:, 3] - self.die_box[:, 1]

    @property
    def die_area(self):
        return self.die_w * self.die_h

    @property
    def io_pos(self):
        return self.pos[self.io]                                 # (T, 2); IO pins have zero size

    @property
    def units(self):
        """'um' for sources with a technology (LEF/DEF), else the format's own unit."""
        return 'um' if self.manifest.get('source', {}).get('format') == 'lefdef' else 'units'

    def utilization(self):
        """Standard-cell area / (core area - macro area), per die (the ORFS definition).
        Not row area: libraries with multi-height sites define overlapping row sets over the same area."""
        self.require('plot')  # die outline + cell geometry
        cell = (self.flags & (MACRO | TERMINAL | ASIDE)) == 0
        mac = ((self.flags & MACRO) != 0) & self.placed
        area = self.size_n.prod(1)
        D = self.num_dies
        # fixed objects count with the part of them inside the core (routing benchmarks place macros outside the
        # rows, pads sit in the IO ring); movable cells with all of their area
        cb = self.core_box[self.cell_die]
        lo = torch.maximum(self.pos, cb[:, :2]); hi = torch.minimum(self.pos + self.size, cb[:, 2:])
        inside = (hi - lo).clamp(min=0).prod(1)
        std = torch.where(self.movable | ~self.placed, area, inside) * cell.float()
        num = torch.zeros(D, device=self.device).index_add_(0, self.cell_die, std)
        unplaced_mac = ((self.flags & MACRO) != 0) & ~self.placed          # not placed yet: their whole area
        mac_area = torch.zeros(D, device=self.device).index_add_(0, self.cell_die, torch.where(mac, inside, torch.zeros_like(inside))
                                                                 + area * unplaced_mac.float())
        core = (self.core_box[:, 2] - self.core_box[:, 0]) * (self.core_box[:, 3] - self.core_box[:, 1])
        return num / (core - mac_area)

    def summary(self):
        c = self.manifest['counts']
        n_io = len(self.io) if self.io is not None else c.get('terminals', 0)
        lines = ['%s: %d cells (%d movable, %d macros), %d nets, %d IO pins, %s'
                 % (self.manifest['design'], c['cells'], c['movable'], c['macros'], c['nets'], n_io,
                    '%d die(s)' % self.num_dies if self.die_box is not None else 'no die outline')]
        if self.die_box is not None:
            util = self.utilization()
            for k in range(self.num_dies):
                die, core = self.die_box[k].tolist(), self.core_box[k].tolist()
                rows = 'no rows'
                if self.row_h is not None:
                    hs = sorted({round(h, 4) for h in self.row_h[self.row_die == k].tolist()})
                    rows = '%d rows (height %s um)' % (int((self.row_die == k).sum()), '/'.join('%g' % h for h in hs))
                lines.append('  die %d: %.2f x %.2f um, core (%.2f, %.2f)-(%.2f, %.2f), %s, util %.1f%%'
                             % (k, die[2] - die[0], die[3] - die[1], *core, rows, 100 * util[k]))
        unplaced = int((~self.placed).sum())
        if unplaced:
            hint = " (d.restore('reference') restores the kept placement)" if 'reference' in self.snapshots else ''
            lines.append('  %d cells unplaced%s' % (unplaced, hint))
        if self.io_dir is not None and len(self.io_dir):
            n = torch.bincount(self.io_dir, minlength=4).tolist()
            lines.append('  IO: ' + ', '.join('%d %s' % (n[i], self.IO_DIRS[i].lower()) for i in range(4) if n[i]))
        lines.append('  facts: ' + ', '.join('%s<%s>' % (k, v) for k, v in sorted(self.facts.items())))
        lines.append('  can do: ' + ', '.join(self.capabilities()))
        return '\n'.join(lines)

    # -- geometry ---------------------------------------------------------------------------------------------
    def update_orient(self):
        """Recompute the effective cell size and the pin offsets (relative to the cell center) after orient changes."""
        w, h = self.size_n[:, 0], self.size_n[:, 1]
        rot = (self.orient == 2) | (self.orient == 3) | (self.orient == 6) | (self.orient == 7)
        self.size = torch.where(rot[:, None], self.size_n.flip(1), self.size_n)
        c = self.p_cell
        off = _orient_offset(self.orient[c], self._p_ctr_n[:, 0], self._p_ctr_n[:, 1], w[c], h[c])
        self.pin_off = torch.where(self._p_term[:, None], torch.zeros_like(off), off - self.size[c] / 2)

    def pin_pos(self, pos=None):
        pos = self.pos if pos is None else pos
        return pos[self.p_cell] + self.size[self.p_cell] / 2 + self.pin_off

    def hpwl(self, pos=None, per_net=False, exclude_clock=False):
        """Pin-based HPWL in microns (a 0-d tensor; no host sync). pos: evaluate these positions instead."""
        self.require('hpwl', ignore=('placement',) if pos is not None else ())
        p = self.pin_pos(pos)
        E = len(self.n_weight)
        hi = torch.full((E, 2), -float('inf'), device=self.device).scatter_reduce(0, self.p_net[:, None].expand(-1, 2), p, 'amax')
        lo = torch.full((E, 2), float('inf'), device=self.device).scatter_reduce(0, self.p_net[:, None].expand(-1, 2), p, 'amin')
        w = (hi - lo).sum(1) * self.n_weight
        mask = self.n_mask & ((self.n_flags & 1) == 0) if exclude_clock else self.n_mask
        w = torch.where(mask, w, torch.zeros_like(w))
        return w if per_net else w.sum()

    def load_reference(self):
        """Same as restore('reference')."""
        return self.restore('reference')

    def wa_wirelength(self, pos=None, gamma=1.0):
        """Weighted-average (WA) wirelength, a smooth HPWL approximation (microns); differentiable in pos.
        gamma: smoothing length in microns (smaller = closer to HPWL)."""
        self.require('wirelength_model')
        if pos is None:
            self.require('hpwl')
        p = self.pin_pos(pos)
        E = len(self.n_weight); idx = self.p_net
        with torch.no_grad():
            hi = torch.full((E, 2), -float('inf'), device=self.device).scatter_reduce(0, idx[:, None].expand(-1, 2), p, 'amax')
            lo = torch.full((E, 2), float('inf'), device=self.device).scatter_reduce(0, idx[:, None].expand(-1, 2), p, 'amin')
        a = torch.exp((p - hi[idx]) / gamma); b = torch.exp((lo[idx] - p) / gamma)
        z = lambda v: torch.zeros((E, 2), device=self.device, dtype=p.dtype).index_add_(0, idx, v)
        wa = z(p * a) / z(a) - z(p * b) / z(b)
        w = wa.sum(1) * self.n_weight
        return torch.where(self.n_mask, w, torch.zeros_like(w)).sum()

    # -- names ----------------------------------------------------------------------------------------------------
    @property
    def names(self):
        if self._names is None:
            with gzip.open(os.path.join(self.path, 'names.json.gz'), 'rt') as f:
                self._names = json.load(f)
        return self._names

    # -- write-back -------------------------------------------------------------------------------------------
    def pin_layers(self):
        """(horizontal, vertical) layer for IO pins: ORFS's IO_PLACER_H/V when known (asap7: M4, M5), else the first
        horizontal layer above the third and the first vertical one above it, from the tech profile."""
        if self.manifest.get('orfs', {}).get('platform') == 'asap7':
            return 'M4', 'M5'
        from . import st
        tp, _ = st.load(os.path.join(self.path, 'tech_profile.safetensors'))
        names = self.names['layers']; dr = tp['layer_dir']
        h = next(i for i in range(3, len(names)) if dr[i] == 0)
        v = next(i for i in range(h + 1, len(names)) if dr[i] == 1)
        return names[h], names[v]

    def export(self, out_def):
        """Write this design as a DEF: the source DEF's frame (fact 'def_frame': everything of the source but the components
          Attar may move and the signal nets, kept verbatim: header, die, rows, tracks, IO pins with their shapes,
          PDN, blockages, fixed macros, ...) with the design written into it from the IR: every component Attar owns
          (PLACED where placed, FIXED where an engine fixed it, UNPLACED otherwise or when set aside) and every
          signal net (connectivity; any wiring of the source is dropped: it is stale once cells move). What an
          engine changed of the floorplan (IO pins, die, rows, macros) is stamped onto the frame first.
        Needs the facts netlist, cell_library and def_frame (LEF/DEF sources). Returns counts of what was written."""
        import gzip
        import tempfile
        from .deffile import comp_statement, net_statements, write_def, write_framed
        self.require('export')
        with gzip.open(os.path.join(self.path, 'frame.def.gz'), 'rt') as f:
            frame = f.read()
        dbu = self.dbu
        xy = torch.round(self.pos * dbu).to(torch.int64).cpu().numpy()
        o = self.orient.cpu().numpy(); fl = self.flags.cpu().numpy(); placed = self.placed.cpu().numpy()
        a = self.raw; names = self.names; cells = names['cells']
        n_comp = int(len(cells) - len(self.io)) if self.io is not None else len(cells)
        raw_fl = a['c_flags']
        owned = [i for i in range(n_comp) if not (raw_fl[i] & FIXED)]
        # what an engine changed of the floorplan goes onto the frame first (IO pins, die, rows, fixed macros)
        site = next((l.split()[2] for l in frame.splitlines() if l.startswith('ROW ')), 'core')
        comps, pins, die, rows = self._floorplan_changes(dbu, self.tracks(), site)
        owned_names = {cells[i] for i in owned}
        comps = {k: v for k, v in comps.items() if k not in owned_names}   # owned macros are written below anyway
        done = {'pins': 0, 'die': 0, 'rows': 0, 'tracks': 0}
        if comps or pins or die is not None or rows is not None:
            with tempfile.TemporaryDirectory() as t:
                with open(os.path.join(t, 'in.def'), 'w') as fo:
                    fo.write(frame)
                done = write_def(os.path.join(t, 'in.def'), os.path.join(t, 'out.def'), comps=comps, pins=pins, die=die, rows=rows)
                with open(os.path.join(t, 'out.def')) as fi:
                    frame = fi.read()
        ca, na = names.get('cell_attrs') or {'default': '', 'except': {}}, names.get('net_attrs') or {'default': '', 'except': {}}
        masters = names['masters']; cm = a['c_master']
        lines = []
        for i in owned:
            if fl[i] & ASIDE or not placed[i]:
                st = 'UNPLACED'
            else:
                st = 'FIXED' if fl[i] & FIXED else 'PLACED'
            lines.append(comp_statement(cells[i], masters[cm[i]], st, int(xy[i, 0]), int(xy[i, 1]), ORIENTS[o[i]],
                                        ca['except'].get(str(i), ca['default'])))
        mpin_flat = [p for ms in names['master_pins'] for p in ms]
        is_term = (raw_fl & TERMINAL) != 0
        nets = net_statements(names['nets'], a['n_ptr'], a['p_cell'], a['p_mpin'], cells, is_term, mpin_flat,
                              lambda k: na['except'].get(str(k), na['default']))
        n = write_framed(frame, out_def, lines, nets)
        done.update({'components': sum(1 for l in lines if '+ UNPLACED' not in l), 'nets': n['NETS']})
        return done

    def _floorplan_changes(self, dbu, tracks, site):
        """What an engine changed of the floorplan, as write_def arguments: macros {name: (x, y, orient, 'FIXED')}, IO
        pins with their shapes, die and rows (None when unchanged)."""
        xy = torch.round(self.pos * dbu).to(torch.int64).cpu().numpy()
        o = self.orient.cpu().numpy(); fl = self.flags.cpu().numpy(); cells = self.names['cells']
        comps = {}
        if str(self.facts.get('floorplan.macros', '')).startswith('naive'):     # only what an engine changed
            comps = {cells[i]: (int(xy[i, 0]), int(xy[i, 1]), ORIENTS[o[i]], 'FIXED') for i in np.nonzero((fl & MACRO) != 0)[0]}
        pins = {}
        if str(self.facts.get('floorplan.io', '')).startswith('naive') and self.io is not None and len(self.io):
            pins = self._pin_shapes(xy, dbu, tracks)
        die = rows = None
        if str(self.facts.get('floorplan.die', '')).startswith('naive'):
            die = [int(round(v * dbu)) for v in self.die_box[0].tolist()]
            r_xy = torch.round(self.row_xy * dbu).long().tolist(); sw = torch.round(self.row_site_w * dbu).long().tolist()
            rows = [('ROW_%d' % k, site, r_xy[k][0], r_xy[k][1], ORIENTS[int(self.row_orient[k])], int(self.row_sites[k]), sw[k])
                    for k in range(len(r_xy))]
        return comps, pins, die, rows

    def tracks(self):
        """{(axis 'X' or 'Y', layer name): (start, step)} in DBU: the routing tracks stored in the IR (empty when the
        source had none, or for an IR older than attar-ir 0.5)."""
        a = self.raw
        if 'tr_axis' not in a or not len(a['tr_axis']):
            return {}
        names = self.names['layers']
        return {('Y' if a['tr_axis'][k] else 'X', names[a['tr_layer'][k]]): (int(a['tr_start'][k]), int(a['tr_step'][k]))
                for k in range(len(a['tr_axis'])) if a['tr_layer'][k] >= 0}

    def macro_grid(self, mac):
        """Where macros may go so that their pins can be routed: per axis a step and, per macro, an offset, so the
        legal lower-left positions are offset + k * step (microns). With the 'tracks' fact, the signal pins of a
        macro must sit on the preferred-direction tracks of their layer (y for horizontal layers, x for vertical
        ones), as ORFS's macro placer leaves them: off-track macro pins made ORFS detailed routing fail (riscv32i:
        ORFS's own macro placement moved 24 dbu off the M4 tracks gave 12 unroutable nets). The residue most of a
        macro's pins agree on wins (ties: the smallest). Without it (and on an axis no pin constrains) the site /
        row grid from the core origin, or the database grid without rows. Everything comes from the IR. Orientation
        N only (as Naive places macros). Sets self.macro_grid_source ('tracks' or 'rows').
        Returns (step (2,), offsets (len(mac), 2)) as tensors."""
        dev = self.device
        n = len(mac)
        core = self.core_box[0].tolist()
        sw = float(self.row_site_w.min()) if self.row_site_w is not None else 1.0 / self.dbu
        hb = float(self.row_h.min()) if self.row_h is not None else 1.0 / self.dbu
        step = [sw, hb]
        off = np.zeros((n, 2))
        off[:, 0] = core[0] % sw; off[:, 1] = core[1] % hb
        tracks = self.tracks() if 'tracks' in self.facts else {}
        self.macro_grid_source = 'tracks' if tracks else 'rows'
        if not tracks or n == 0 or 'm_pin_layer' not in self.raw:
            return torch.tensor(step, device=dev), torch.as_tensor(off, device=dev, dtype=torch.float32)
        a = self.raw
        tp, _ = st.load(os.path.join(self.path, 'tech_profile.safetensors'))
        names, ldir = self.names['layers'], tp['layer_dir']           # 0: horizontal (tracks along y)
        ptr, box, use, lay, cm = a['m_pin_ptr'], a['m_pin_box'], a['m_pin_use'], a['m_pin_layer'], a['c_master']
        from .distill import U_GROUND, U_POWER
        votes = [{}, {}]                                  # axis -> {(step, residue): macros}
        per = []
        for i in mac.tolist():
            v = [{}, {}]
            m = int(cm[i])
            for k in (range(int(ptr[m]), int(ptr[m + 1])) if m >= 0 else ()):
                L = int(lay[k])
                if use[k] in (U_POWER, U_GROUND) or L < 0:
                    continue
                ax = 1 if ldir[L] == 0 else 0
                start, st_ = tracks.get(('Y' if ax else 'X', names[L]), (None, None))
                if st_ is None:
                    continue
                c = (int(box[k, 0]) + int(box[k, 2])) // 2 if ax == 0 else (int(box[k, 1]) + int(box[k, 3])) // 2
                key = (st_, (start - c) % st_)
                v[ax][key] = v[ax].get(key, 0) + 1
            per.append(v)
            for ax in (0, 1):
                if v[ax]:
                    kk = max(v[ax].items(), key=lambda kv: (kv[1], -kv[0][1]))[0][0]
                    votes[ax][kk] = votes[ax].get(kk, 0) + 1
        for ax in (0, 1):
            if not votes[ax]:
                continue
            st_ = max(votes[ax].items(), key=lambda kv: (kv[1], -kv[0]))[0]       # the step most macros' pins want
            step[ax] = st_ / self.dbu
            for j, v in enumerate(per):
                cand = {k: c for k, c in v[ax].items() if k[0] == st_}
                off[j, ax] = (max(cand.items(), key=lambda kv: (kv[1], -kv[0][1]))[0][1] if cand else 0) / self.dbu
        return torch.tensor(step, device=dev), torch.as_tensor(off, device=dev, dtype=torch.float32)

    def _pin_shapes(self, xy, dbu, tracks):
        """IO pins as DEF pins, as ORFS's place_pins makes them: on the nearest die edge, a shape on the vertical
        layer (bottom, top) or horizontal layer (left, right), half of it inside the die, centered on a track of
        that layer (tracks: {(axis, layer): (start, step)} in DBU), at least 2 tracks from the next pin on the same
        edge."""
        tp, _ = st.load(os.path.join(self.path, 'tech_profile.safetensors'))
        names = self.names['layers']
        h, v = self.pin_layers()
        width = {n: int(tp['layer_width'][i]) for i, n in enumerate(names)}
        die = [int(round(t * dbu)) for t in self.die_box[0].tolist()]
        cells = self.names['cells']
        sides = {0: [], 1: [], 2: [], 3: []}
        for i in self.io.tolist():
            x, y = int(xy[i, 0]), int(xy[i, 1])
            sides[int(np.argmin([y - die[1], die[2] - x, die[3] - y, x - die[0]]))].append((i, x, y))
        out = {}
        for side, pins in sides.items():
            layer = v if side in (0, 2) else h
            axis = 'X' if side in (0, 2) else 'Y'
            start, step = tracks.get((axis, layer), (0, 1))
            lo, hi = (die[0], die[2]) if axis == 'X' else (die[1], die[3])
            wd = max(2, width.get(layer, 20)); wd -= wd % 2
            ln = 2 * ((7 * wd // 2) // 2)
            along = sorted(((x if axis == 'X' else y), i) for i, x, y in pins)
            k_lo = -(-(lo + step - start) // step); k_hi = (hi - step - start) // step   # tracks inside the edge
            last = None
            for u, i in along:
                k = min(max(round((u - start) / step), k_lo), k_hi)
                if last is not None and k < last + 2:
                    k = last + 2                 # ORFS place_pins' default minimum distance: 2 tracks
                last = k
                c = start + k * step
                if side == 0:   p = (c, die[1] + ln // 2, (-wd // 2, -ln // 2, wd // 2, ln // 2))
                elif side == 2: p = (c, die[3] - ln // 2, (-wd // 2, -ln // 2, wd // 2, ln // 2))
                elif side == 1: p = (die[2] - ln // 2, c, (-ln // 2, -wd // 2, ln // 2, wd // 2))
                else:           p = (die[0] + ln // 2, c, (-ln // 2, -wd // 2, ln // 2, wd // 2))
                out[cells[i]] = (p[0], p[1], 'N', layer, p[2])
        return out


def load(name, device='cpu', download=True, root=None):
    """Load a distilled design: a folder path, or a zoo name <mode>/<suite>/<design> such as 'after_fp/orfs_asap7/aes'
    (paths.MODES: before_fp, a floorplan problem; after_fp, a placement problem) looked up in the zoo (`root`,
    default paths.zoo_dir()). A zoo name that is not there is downloaded from the Hugging Face Hub first
    (attar.hub), unless download=False."""
    if os.path.isdir(name):
        path = name
    else:
        path = os.path.join(paths.zoo_dir(root), name)
        if download and not os.path.exists(os.path.join(path, 'manifest.json')) and not os.path.isabs(name):
            from .hub import fetch
            fetch(name, root=root)
    if not os.path.exists(os.path.join(path, 'manifest.json')):
        raise FileNotFoundError('no distilled design %r (looked in %s%s)' % (
            name, path, ' and on the Hugging Face Hub' if download and not os.path.isdir(name) else ''))
    return Design(path, device)


def zoo(suite=None, remote=False, root=None):
    """Names <mode>/<suite>/<design> of the distilled designs: in the local zoo (`root`, default paths.zoo_dir()), or
    (remote=True) on the Hugging Face Hub. `suite` filters by a prefix: a mode ('after_fp') or a mode and suite
    ('after_fp/ispd2005')."""
    keep = lambda n: suite is None or n == suite.strip('/') or n.startswith(suite.strip('/') + '/')
    if remote:
        from .hub import index
        return sorted(n for n in index() if keep(n))
    root = paths.zoo_dir(root)
    out = []
    for mode in paths.MODES:
        mdir = os.path.join(root, mode)
        for s in sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []:
            sdir = os.path.join(mdir, s)
            if s.startswith('.') or not os.path.isdir(sdir):
                continue
            out += [n for n in ('%s/%s/%s' % (mode, s, d) for d in sorted(os.listdir(sdir))
                                if os.path.exists(os.path.join(sdir, d, 'manifest.json'))) if keep(n)]
    return out
