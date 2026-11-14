"""Design: a distilled design on the GPU, as a state (spec 5.1).

Problem facts (netlist, geometry, floorplan, rows, ...) come from the IR and may be missing; the solution state
(positions, orientations) is changed by algorithms and can be saved as named snapshots. What a Design can do follows
from the facts it has: capabilities(), missing(capability), require(capability).
Coordinates in microns (float32); `scale` = 1 / dbu.

The Design holds the state and answers questions about it; the larger jobs live in their own modules and are reached
through its methods: status.py (status, summary), export.py (DEF write-back), sizing.py (resize), sta.py
(timing), routing.py (GROV), viz.py (plot), naive (engines)."""
import difflib
import gzip
import json
import os

import numpy as np
import torch

from . import export as _export, metrics as M, paths, sizing as _sizing, st
from .distill import ASIDE, FIXED, MACRO, PLACED, TERMINAL, PHYSICAL  # noqa: F401
from .facts import BLOCKED, CAPABILITIES, FACTS, MissingFacts, infer_facts, upgrade_facts
from .kernels import overlapping_pairs


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


def _unknown(kind, name, known):
    """A KeyError for an unknown fact / capability name, with the closest known names."""
    near = difflib.get_close_matches(name, list(known), n=3)
    return KeyError('unknown %s %r%s (known: %s)' % (kind, name, ', did you mean %s?' % ' or '.join(map(repr, near))
                                                     if near else '', ', '.join(known)))


def _similar_designs(name, names, n=3):
    """Zoo names close to `name`: by the design part first (in the same suite first), then by the whole name."""
    head, _, base = name.rpartition('/')
    r = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()  # noqa: E731
    cand = [m for m in names if r(base, m.rpartition('/')[2]) >= 0.75] or \
        difflib.get_close_matches(name, names, n=n, cutoff=0.8)
    cand.sort(key=lambda m: (m.rpartition('/')[0] != head, -r(base, m.rpartition('/')[2]), -r(name, m)))
    return cand[:n]


class Design:
    IO_DIRS = ('INPUT', 'OUTPUT', 'INOUT', 'FEEDTHRU')

    def __init__(self, path, device='cpu'):
        self.path = path
        self.device = torch.device(device)
        a, _ = st.load(os.path.join(path, 'core.safetensors'))
        with open(os.path.join(path, 'manifest.json')) as f:
            self.manifest = json.load(f)
        self._names = None
        self._timer = None                    # attar.sta.Timer, built by the first timing()
        self._cell_id = None                  # instance name -> index, built by the first lookup by name
        self.dropped = {}                     # facts an engine made stale: {fact: why}
        self.dbu = self.manifest['dbu']; self.scale = 1.0 / self.dbu
        T = lambda x, dt=None: torch.as_tensor(x, device=self.device, dtype=dt)  # noqa: E731
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
        self._refresh_movable()
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

    # -- loading: dies, rows, IO pins ---------------------------------------------------------------------------------
    def _load_dies(self, a, T, s):
        """Dies (D >= 1; a 2D design has one, its DEF DIEAREA), rows, IO pins. Boxes are (x0, y0, x1, y1) in um."""
        N = len(self.flags)
        self.cell_die = T(a['c_die'].astype(np.int64)) if 'c_die' in a else torch.zeros(N, dtype=torch.int64, device=self.device)
        self.io = None; self.io_dir = None
        if 'floorplan.die' not in self.facts:
            self.die_box = self.core_box = None
            self._no_rows()
            return
        die = a['die_box'] if 'die_box' in a else a['die'].reshape(1, 4)       # schema 0.1 stored one 'die'
        self.die_box = T(die.astype(np.float64) * s, torch.float32)              # (D, 4)
        self._load_io(a, T)
        if 'rows' not in self.facts:
            self._no_rows()
            return
        f32 = lambda k: T(a[k].astype(np.float64) * s, torch.float32)          # noqa: E731
        n_rows = len(a['r_h'])
        self._set_rows(f32('r_xy'), f32('r_h'), f32('r_site_w'), T(a['r_sites'].astype(np.int64)),
                       T(a['r_orient'].astype(np.int64)) if 'r_orient' in a else None,       # None: not known
                       T(a['r_die'].astype(np.int64)) if 'r_die' in a else
                       torch.zeros(n_rows, dtype=torch.int64, device=self.device))

    def _load_io(self, a, T):
        """IO pins: zero-size fixed terminal cells; io indexes them in the cell arrays."""
        if 'io_cell' in a:
            self.io = T(a['io_cell'].astype(np.int64)); self.io_dir = T(a['io_dir'].astype(np.int64))
        else:
            self.io = torch.nonzero((self.flags & TERMINAL) != 0).flatten()
            self.io_dir = torch.full((len(self.io),), 2, dtype=torch.int64, device=self.device)

    def _no_rows(self):
        """No rows: the row arrays are None and the core is the die."""
        self.row_xy = self.row_h = self.row_site_w = self.row_sites = self.row_die = self.row_box = self.row_orient = None
        if self.die_box is not None:
            self.core_box = self.die_box.clone()

    def _set_rows(self, xy, h, site_w, sites, orient, die):
        """Rows (tensors: origin (R, 2), height, site width, sites, orientation or None, die), their boxes, and the core
        of each die: the bounding box of its rows (the die box when a die has no rows)."""
        self.row_xy, self.row_h, self.row_site_w, self.row_sites = xy, h, site_w, sites
        self.row_orient, self.row_die = orient, die
        self.row_box = torch.stack([xy[:, 0], xy[:, 1], xy[:, 0] + sites * site_w, xy[:, 1] + h], 1)
        D = len(self.die_box)
        idx = die[:, None].expand(-1, 2)
        lo = torch.full((D, 2), float('inf'), device=self.device).scatter_reduce(0, idx, self.row_box[:, :2], 'amin')
        hi = torch.full((D, 2), -float('inf'), device=self.device).scatter_reduce(0, idx, self.row_box[:, 2:], 'amax')
        core = torch.cat([lo, hi], 1)
        self.core_box = torch.where(torch.isfinite(core), core, self.die_box)

    def _refresh_movable(self):
        """movable follows the flags: neither fixed, an IO pin, nor set aside."""
        self.movable = (self.flags & (FIXED | TERMINAL | ASIDE)) == 0

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
            raise _unknown('fact', fact, FACTS)
        return fact in self.present

    @property
    def mode(self):
        """'after_fp' (the floorplan is given: only the cells are to place), 'before_fp', or None (not from the zoo)."""
        parts = os.path.normpath(os.path.abspath(self.path)).split(os.sep)   # <zoo>/<mode>/<suite>/<design>
        return parts[-3] if len(parts) > 3 and parts[-3] in paths.MODES else None

    def blocked(self, capability):
        """Why the design cannot do a capability whatever facts it gets (BLOCKED), or None."""
        rule = BLOCKED.get(capability)
        if rule is None:
            return None
        hit, why = rule(self)
        return why if hit else None

    def capabilities(self):
        have = self.present
        return sorted(c for c, req in CAPABILITIES.items() if req <= have and not self.blocked(c))

    def missing(self, capability):
        if capability not in CAPABILITIES:
            raise _unknown('capability', capability, CAPABILITIES)
        return sorted(CAPABILITIES[capability] - self.present)

    def require(self, capability, ignore=()):
        if capability not in CAPABILITIES:
            raise _unknown('capability', capability, CAPABILITIES)
        why = self.blocked(capability)
        if why:
            raise MissingFacts(self.manifest.get('design', self.path), capability, set(), blocked=why)
        miss = set(CAPABILITIES[capability]) - self.present - set(ignore)
        if miss:
            raise MissingFacts(self.manifest.get('design', self.path), capability, miss)

    def status(self, show=True):
        """Checklist of facts (✅ / ❌), solution state, capabilities, and what is left before d.export.
        Printed when show (in color on a terminal); returned as text either way."""
        from .status import status, styled
        s = status(self)
        if show:
            print(styled(s))
        return s

    def summary(self):
        """A few lines: counts, dies with their core, rows and utilization, unplaced cells, IO pins, facts."""
        from .status import summary
        return summary(self)

    def __repr__(self):
        c = self.manifest.get('counts', {})
        return '<Design %s: %d cells, %d nets, can %s>' % (
            self.manifest.get('design'), c.get('cells', 0), c.get('nets', 0), ', '.join(self.capabilities()))

    # -- floorplan checks ---------------------------------------------------------------------------------------------
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
                ov = 0
                for die in torch.unique(self.cell_die[mac]).tolist():
                    m = self.cell_die[mac] == die
                    ov += overlapping_pairs(q[m], s[m])
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

    def snapshot(self, name):
        """Save the current solution state (positions, orientations) under a name."""
        self.snapshots[name] = (self._pos.detach().clone(), self.orient.clone(), self._placement)
        return self

    def restore(self, name):
        """Restore a saved solution state; 'reference' is the placement kept at distillation (e.g. the ORFS GP)."""
        if name not in self.snapshots:
            near = difflib.get_close_matches(name, list(self.snapshots), n=1)
            raise KeyError('%s has no snapshot %r (have: %s)%s' % (
                self.manifest['design'], name, ', '.join(self.snapshots) or 'none',
                '; did you mean %r?' % near[0] if near else ''))
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

    def load_reference(self):
        """Same as restore('reference')."""
        return self.restore('reference')

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
                self._refresh_movable()
        self.placed[self.movable] = False
        self._placement = not bool(self.movable.any())     # nothing movable: the floorplan is the placement
        return aside

    # -- floorplan state (answers of the IO and macro placers become facts) -------------------------------------
    def _floorplan_changed(self, why):
        """An engine changed the floorplan (IO pins, macros, die). Export writes all of it onto the DEF. Attar's own
        validation (attar._slow_loop) takes a changed floorplan back to ORFS only from its floorplan stage (2_1:
        tapcells, PDN and everything after are redone); a later stage already built them around the old floorplan, so
        the internal fact 'slow_loop' goes there."""
        if 'slow_loop' in self.facts and self.manifest.get('orfs', {}).get('stage') != '2_1_floorplan':
            self.facts.pop('slow_loop')
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
        self._refresh_movable()
        self.placed[m] = False
        self._placement = False
        self.facts.pop('floorplan.macros', None)
        return self

    def fix_macros(self, provenance):
        """Freeze the macros where they are: floorplan.macros becomes a fact."""
        m = self.macros
        self.flags[m] |= FIXED | PLACED
        self._refresh_movable()
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
        f32 = lambda x: torch.as_tensor(x, device=dev, dtype=torch.float32)  # noqa: E731
        i64 = lambda x: torch.as_tensor(x, device=dev, dtype=torch.int64)    # noqa: E731
        self.die_box = f32(die_box).reshape(-1, 4)
        if self.io is None:
            self._load_io(self.raw, lambda x, dt=None: torch.as_tensor(x, device=dev, dtype=dt))
        if rows is None:
            self._no_rows()
            self.facts.pop('rows', None)
        else:
            self._set_rows(f32(rows['xy']), f32(rows['h']), f32(rows['site_w']), i64(rows['sites']), i64(rows['orient']),
                           i64(rows['die']) if 'die' in rows else torch.zeros(len(rows['h']), dtype=torch.int64, device=dev))
            self.facts['rows'] = provenance
        self.facts['floorplan.die'] = provenance
        # stale: IO pins, macros, fixed cells, placement, snapshots
        released = int(((self.flags & FIXED) != 0).logical_and((self.flags & (MACRO | TERMINAL)) == 0).sum())
        self.flags &= ~FIXED & ~PLACED
        self.flags[self.io] |= FIXED
        self._refresh_movable()
        self.placed = torch.zeros_like(self.placed)
        self._placement = False
        self.snapshots = {}
        self.facts.pop('floorplan.io', None)
        self.facts.pop('floorplan.macros', None)
        self._floorplan_changed('die and rows made by %s' % provenance)
        return released

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

    # -- geometry -----------------------------------------------------------------------------------------------------
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

    # -- wirelength ---------------------------------------------------------------------------------------------------
    def hpwl(self, pos=None, per_net=False, exclude_clock=False):
        """Pin-based HPWL in microns (a 0-d tensor; no host sync). pos: evaluate these positions instead."""
        self.require('hpwl', ignore=('placement',) if pos is not None else ())
        lo, hi = M.net_bbox(self.pin_pos(pos), self.p_net, len(self.n_weight))
        w = (hi - lo).sum(1) * self.n_weight
        mask = self.n_mask & ((self.n_flags & 1) == 0) if exclude_clock else self.n_mask
        w = torch.where(mask, w, torch.zeros_like(w))
        return w if per_net else w.sum()

    def wa_wirelength(self, pos=None, gamma=1.0):
        """Weighted-average (WA) wirelength, a smooth HPWL approximation (microns); differentiable in pos.
        gamma: smoothing length in microns (smaller = closer to HPWL)."""
        self.require('wirelength_model')
        if pos is None:
            self.require('hpwl')
        p = self.pin_pos(pos)
        E = len(self.n_weight); idx = self.p_net
        with torch.no_grad():
            lo, hi = M.net_bbox(p, idx, E)
        a = torch.exp((p - hi[idx]) / gamma); b = torch.exp((lo[idx] - p) / gamma)
        z = lambda v: torch.zeros((E, 2), device=self.device, dtype=p.dtype).index_add_(0, idx, v)  # noqa: E731
        wa = z(p * a) / z(a) - z(p * b) / z(b)
        w = wa.sum(1) * self.n_weight
        return torch.where(self.n_mask, w, torch.zeros_like(w)).sum()

    # -- timing and routing congestion --------------------------------------------------------------------------------
    def timing(self, pos=None):
        """Static timing at pos (default: the current placement): an attar.sta.TimingResult with per-endpoint setup /
        hold slack, WNS / TNS. Needs the 'timing' fact (attar.timing.attach). The timer is built once per design and
        kept; on CUDA a re-time is a CUDA-graph replay."""
        self.require('sta')
        from .sta import Timer
        if self._timer is None:
            self._timer = Timer(self)
        return self._timer.run(pos)

    def timing_report(self, pos=None, endpoints=10, path=True, file=None):
        """Print (and return) a timing summary: WNS, TNS, violations, hold, the worst endpoints and the worst path
        (attar.sta.TimingResult.report)."""
        return self.timing(pos).report(endpoints=endpoints, path=path, file=file)

    def gr_overflow(self, pos=None, adjustment=None, maps=False):
        """Global-routing overflow (GROV) at pos (default: the current placement): RUDY demand per gcell edge and
        direction against OpenROAD's gcell grid and edge capacities (tracks minus what pins, OBS and the PDN block,
        times 1 - the layer adjustment). Returns {'grov_%': share of gcell edges whose demand exceeds capacity,
        'grov_h_%', 'grov_v_%', 'overflow_edges', 'edges', 'total_overflow' (tracks), 'max_util', ...}.
        adjustment overrides the design's layer adjustment; maps=True adds the demand and capacity maps."""
        self.require('gr_overflow', ignore=('placement',) if pos is not None else ())
        from .routing import overflow
        return overflow(self, pos, adjustment=adjustment, maps=maps)

    # -- sizing (attar/sizing.py) -------------------------------------------------------------------------------------
    def library(self):
        """The design's cell library chain (attar.timing.chain): its local library (macros), then the shared bundle."""
        return _sizing.library(self)

    def equivalents(self, cell, dont_use=False):
        """The masters cell (index or name) can be resized to: its master's equivalence class in the library
        (drive strengths, threshold-voltage flavors), without the library's dont_use cells unless dont_use=True."""
        return _sizing.equivalents(self, cell, dont_use)

    def resize(self, cells, masters, check=True):
        """Swap cells (indices or names) to other masters (one name, or one per cell): sizing and threshold-voltage
        swaps. check: each new master must be in its cell's equivalence class (d.equivalents). Masters the design
        does not have yet come from its cell library. The cells keep their position and orientation (as OpenROAD's
        replace_cell); size and pin offsets follow the new master. The timer is rebuilt on the next timing().
        Returns the previous master names."""
        return _sizing.resize(self, cells, masters, check)

    # -- names ----------------------------------------------------------------------------------------------------
    @property
    def names(self):
        if self._names is None:
            with gzip.open(os.path.join(self.path, 'names.json.gz'), 'rt') as f:
                self._names = json.load(f)
        return self._names

    # -- output: plot, DEF write-back ---------------------------------------------------------------------------
    def plot(self, path=None, **kw):
        """Draw die, core, rows, cells, macros and IO pins (attar.viz.plot); saved to path if given.
        tiles='gr_overflow' | 'pin_access' | 'density' draws that tile map in place of the cells."""
        from .viz import plot
        return plot(self, path, **kw)

    def export(self, out_def):
        """Write this design as a DEF: the source DEF's frame (fact 'def_frame': everything of the source but the
          components Attar may move and the signal nets, kept verbatim: header, die, rows, tracks, IO pins with their
          shapes, PDN, blockages, fixed macros, ...) with the design written into it from the IR: every component
          Attar owns (PLACED where placed, FIXED where an engine fixed it, UNPLACED otherwise or when set aside) and
          every signal net (connectivity; any wiring of the source is dropped: it is stale once cells move). What an
          engine changed of the floorplan (IO pins, die, rows, macros) is stamped onto the frame first.
        Needs the facts netlist, cell_library and def_frame (LEF/DEF sources). Returns counts of what was written."""
        return _export.export(self, out_def)

    def pin_layers(self):
        """(horizontal, vertical) layer for IO pins: ORFS's IO_PLACER_H/V when known (asap7: M4, M5), else the first
        horizontal layer above the third and the first vertical one above it, from the tech profile."""
        return _export.pin_layers(self)


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
        near = [] if os.path.isabs(name) else _similar_designs(name, zoo(root=root))
        raise FileNotFoundError('no distilled design %r (looked in %s%s)%s' % (
            name, path, ' and on the Hugging Face Hub' if download and not os.path.isdir(name) else '',
            '; did you mean %s?' % ' or '.join(map(repr, near)) if near else ''))
    return Design(path, device)


def zoo(suite=None, remote=False, root=None):
    """Names <mode>/<suite>/<design> of the distilled designs: in the local zoo (`root`, default paths.zoo_dir()), or
    (remote=True) on the Hugging Face Hub. `suite` filters by a prefix: a mode ('after_fp') or a mode and suite
    ('after_fp/ispd2005')."""
    keep = lambda n: suite is None or n == suite.strip('/') or n.startswith(suite.strip('/') + '/')  # noqa: E731
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
