"""Evaluator: measures a Design (spec 5.1 facts, spec 7 proxies).

The Evaluator is the second main object next to Design. Each metric declares the facts it needs; what can be measured
follows from what the design has, as for capabilities. Implemented:
  hpwl, star_wirelength, steiner_wirelength (RSMT estimate, see attar/metrics.py), rudy (congestion map),
  density_overflow (ePlace overflow), legality (rows, sites, overlaps; multi-row aware).
Listed for later: global-route wirelength and overflow, timing, pin access, thermal.

Speed: one HPWL call is ~0.15 ms up to ~1M pins (launch-bound). Exact speed-ups:
  fast=True     CUDA graph for hpwl (0.22 -> 0.05 ms on 43k pins)
  compile=True  torch.compile for hpwl (0.34 -> 0.19 ms on 0.9M pins, 1.5 -> 0.9 ms on 4.3M)
  hpwl_batch    many placements at once: (B, N, 2) -> (B,)
"""
import torch

from . import metrics as M
from .distill import ASIDE, MACRO, TERMINAL
from .facts import MissingFacts
from .kernels import Grid, default_bins

WL = {'netlist', 'cell_geometry', 'pin_offsets', 'placement'}
# metric -> (required facts, implemented)
METRICS = {
    'hpwl': (WL, True),
    'star_wirelength': (WL, True),
    'steiner_wirelength': (WL, True),
    'rudy': (WL | {'floorplan.die'}, True),
    'density_overflow': ({'cell_geometry', 'placement', 'floorplan.die'}, True),
    'legality': ({'cell_geometry', 'placement', 'rows'}, True),
    'gr_wirelength': (WL | {'routing'}, False),
    'gr_overflow': (WL | {'routing'}, False),
    'timing': ({'netlist', 'timing'}, False),
    'pin_access': (WL | {'routing'}, False),
    'thermal': ({'cell_geometry', 'placement', 'floorplan.die', 'timing'}, False),
}


class Evaluator:
    def __init__(self, design, fast=False, compile=False):
        """fast: CUDA-graph the HPWL kernels (CUDA only; positions are copied into a static buffer each call).
        compile: torch.compile the HPWL function (needs Triton)."""
        self.d = design
        self.dev = design.device
        self.fast = fast and self.dev.type == 'cuda'
        self.compile = compile
        self._key = None
        self._graph = None
        self._rows = None
        self._prepare()

    # -- what can be measured ---------------------------------------------------------------------------------------
    def available(self):
        have = self.d.present
        return sorted(m for m, (req, impl) in METRICS.items() if impl and req <= have)

    def missing(self, metric):
        req, impl = METRICS[metric]
        miss = sorted(req - self.d.present)
        return miss + ([] if impl else ['(not implemented yet)'])

    def _require(self, metric, pos):
        req, impl = METRICS[metric]
        if not impl:
            raise NotImplementedError('metric %r is listed but not implemented yet' % metric)
        miss = req - self.d.present - ({'placement'} if pos is not None else set())
        if miss:
            raise MissingFacts(self.d.manifest.get('design'), metric, miss)

    # -- shared state ---------------------------------------------------------------------------------------------------
    def _prepare(self):
        """Pin offsets from the cell lower-left and the net index; redone when orientations change."""
        d = self.d
        key = (id(d.pin_off), id(d.size))
        if key == self._key:
            return
        self._key = key
        self.cell = d.p_cell
        self.off = (d.size[d.p_cell] / 2 + d.pin_off).contiguous()
        self.ptr = d.n_ptr
        self.deg = d.n_ptr[1:] - d.n_ptr[:-1]
        self.net = d.p_net
        self.E = len(self.deg)
        self.wmask = torch.where(self.deg >= 2, d.n_weight, torch.zeros_like(d.n_weight))
        self.idx4 = d.p_net[:, None].expand(-1, 4).contiguous()
        self.init = torch.full((self.E, 4), -float('inf'), device=self.dev)
        self._fn = self._hpwl_nets
        if self.compile:
            self._fn = torch.compile(self._hpwl_nets, mode='max-autotune-no-cudagraphs')
        self._graph = None

    def _pins(self, pos):
        return (self.d.pos if pos is None else pos)[self.cell] + self.off

    # -- HPWL ---------------------------------------------------------------------------------------------------------
    def _hpwl_nets(self, pos):
        """Per-net HPWL in one scatter: max over pins of (x, y, -x, -y) gives max x - min x + max y - min y."""
        p = pos[self.cell] + self.off
        m = self.init.scatter_reduce(0, self.idx4, torch.cat([p, -p], 1), 'amax', include_self=True)
        return torch.where(torch.isfinite(m), m, torch.zeros_like(m)).sum(1) * self.wmask

    def _graphed(self, pos):
        dev = pos.device                            # capture and replay on the design's GPU
        with torch.cuda.device(dev):
            if self._graph is None:
                self._static = pos.detach().clone()
                s = torch.cuda.Stream(dev); s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s):
                    for _ in range(3):
                        self._fn(self._static).sum()
                torch.cuda.current_stream(dev).wait_stream(s)
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph, stream=s):   # not torch's shared default capture stream
                    self._static_out = self._fn(self._static).sum()
            self._static.copy_(pos)
            self._graph.replay()
            return self._static_out.clone()

    def hpwl(self, pos=None):
        """Total weighted HPWL (0-d tensor, no host sync). pos: evaluate these positions instead of the design's."""
        self._require('hpwl', pos)
        self._prepare()
        p = self.d.pos if pos is None else pos
        if self.fast and not p.requires_grad and p.shape == self.d.pos.shape:
            return self._graphed(p)
        return self._fn(p).sum()

    def hpwl_per_net(self, pos=None):
        self._require('hpwl', pos)
        self._prepare()
        return self._hpwl_nets(self.d.pos if pos is None else pos)

    def hpwl_batch(self, pos):
        """HPWL of B placements at once: pos (B, N, 2) -> (B,). Memory grows with B x pins."""
        self._require('hpwl', pos)
        self._prepare()
        B = pos.shape[0]
        p = pos[:, self.cell] + self.off
        m = self.init[None].expand(B, -1, -1).scatter_reduce(1, self.idx4[None].expand(B, -1, -1), torch.cat([p, -p], 2),
                                                             'amax', include_self=True)
        return (torch.where(torch.isfinite(m), m, torch.zeros_like(m)).sum(2) * self.wmask).sum(1)

    # -- other wirelength models ------------------------------------------------------------------------------------
    def star_wirelength(self, pos=None, per_net=False):
        self._require('star_wirelength', pos); self._prepare()
        s = M.star(self._pins(pos), self.net, self.deg, self.E) * self.wmask
        return s if per_net else s.sum()

    def steiner_wirelength(self, pos=None, per_net=False):
        """RSMT estimate: exact for degree <= 3, else the shortest of single-trunk, RMST and star trees."""
        self._require('steiner_wirelength', pos); self._prepare()
        p = self._pins(pos)
        lo, hi = M.net_bbox(p, self.net, self.E)
        s = M.steiner(p, self.net, self.deg, self.ptr, self.E, lo, hi) * self.wmask
        return s if per_net else s.sum()

    # -- maps ---------------------------------------------------------------------------------------------------------
    def grid(self, bins=None, box=None):
        """A bin grid over the core (default) with ~sqrt(#cells) bins per side."""
        b = self.d.core_box[0] if box is None else box
        n = bins or default_bins(len(self.d.flags))
        nx, ny = (n, n) if isinstance(n, int) else n
        return Grid.over_box(b.tolist(), nx, ny, self.dev)

    def rudy(self, pos=None, bins=None):
        """RUDY congestion map (wire length per unit area, 1/um) and its statistics. Each net spreads
        HPWL / bbox area uniformly over its bounding box (width/height at least one bin)."""
        self._require('rudy', pos); self._prepare()
        g = self.grid(bins, self.d.die_box[0])
        p = self._pins(pos)
        lo, hi = M.net_bbox(p, self.net, self.E)
        ok = self.deg >= 2
        lo, hi, w = lo[ok], hi[ok], self.wmask[ok]
        c = (lo + hi) / 2
        sz = torch.maximum(hi - lo, torch.tensor([g.bw, g.bh], device=self.dev))
        lo, hi = c - sz / 2, c + sz / 2
        dens = w * (1 / sz[:, 0] + 1 / sz[:, 1])
        m = g.accumulate(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1], dens) / g.bin_area
        v = m.flatten().sort(descending=True).values
        top = lambda f: v[:max(1, int(len(v) * f))].mean().item()
        return {'map': m, 'max': v[0].item(), 'mean': v.mean().item(), 'top0.5%': top(0.005), 'top1%': top(0.01),
                'top2%': top(0.02), 'top5%': top(0.05), 'bins': (g.nx, g.ny)}

    def density_overflow(self, pos=None, bins=None, target=1.0):
        """ePlace overflow: sum over bins of the movable area above target x free area, over the movable area."""
        self._require('density_overflow', pos)
        d = self.d
        g = self.grid(bins)
        p = d.pos if pos is None else pos
        mv = d.movable
        fx = ~mv & ((d.flags & (TERMINAL | ASIDE)) == 0)
        mov = g.accumulate(p[mv, 0], p[mv, 1], p[mv, 0] + d.size[mv, 0], p[mv, 1] + d.size[mv, 1],
                           torch.ones(int(mv.sum()), device=self.dev))
        fix = g.accumulate(p[fx, 0], p[fx, 1], p[fx, 0] + d.size[fx, 0], p[fx, 1] + d.size[fx, 1],
                           torch.ones(int(fx.sum()), device=self.dev)).clamp(max=g.bin_area)
        over = (mov - target * (g.bin_area - fix)).clamp(min=0).sum()
        return (over / d.size[mv].prod(1).sum()).item()

    # -- legality -----------------------------------------------------------------------------------------------------
    def _base_rows(self):
        """Base rows: the smallest row height; one entry per distinct row y (multi-height sites share the grid)."""
        if self._rows is None:
            d = self.d
            hb = d.row_h.min()
            base = d.row_h <= hb + 1e-6
            ys, inv = torch.unique(d.row_xy[base, 1], return_inverse=True)
            x0 = torch.full((len(ys),), float('inf'), device=self.dev).scatter_reduce(0, inv, d.row_box[base, 0], 'amin')
            x1 = torch.full((len(ys),), -float('inf'), device=self.dev).scatter_reduce(0, inv, d.row_box[base, 2], 'amax')
            sw = torch.zeros(len(ys), device=self.dev).scatter_reduce(0, inv, d.row_site_w[base], 'amax', include_self=False)
            self._rows = (hb.item(), ys, x0, x1, sw)
        return self._rows

    def legality(self, pos=None, tol=1e-3):
        """Placement legality of the movable standard cells: on a row, on the site grid, inside the rows, no overlap
        with any other cell or fixed object. Multi-row cells occupy k = height / row height consecutive rows."""
        self._require('legality', pos)
        d = self.d
        p = d.pos if pos is None else pos
        hb, ys, rx0, rx1, sw = self._base_rows()
        R = len(ys)
        std = (d.flags & (MACRO | TERMINAL | ASIDE)) == 0
        mv = d.movable & std
        k = (d.size[:, 1] / hb).round().clamp(min=1).long()
        # row of the bottom edge
        r = torch.searchsorted(ys, p[:, 1] - tol).clamp(max=R - 1)
        on_row = (ys[r] - p[:, 1]).abs() < tol
        inside = on_row & (r + k <= R) & (p[:, 0] >= rx0[r] - tol) & (p[:, 0] + d.size[:, 0] <= rx1[r] + tol)
        inside0 = inside
        site = ((p[:, 0] - rx0[r]) / sw[r])
        on_site = (site - site.round()).abs() < tol / sw[r].clamp(min=1e-9)
        # overlap: every placed object appears in each base row it spans; sort by (row, x); compare with the running
        # maximum right edge of the earlier objects in the same row
        obj = (~((d.flags & (TERMINAL | ASIDE)) != 0)) & (d.placed if pos is None else torch.ones_like(d.placed))
        y0 = p[:, 1]; y1 = p[:, 1] + d.size[:, 1]
        from .rails import rows_spanned
        rlo, rhi = rows_spanned(ys, hb, y0, y1)                           # every row the object overlaps
        rlo = rlo.clamp(max=R)
        nrow = (rhi - rlo).clamp(min=0) * obj
        ids = torch.repeat_interleave(torch.arange(len(p), device=self.dev), nrow)
        rows = rlo[ids] + (torch.arange(len(ids), device=self.dev) - torch.repeat_interleave(torch.cumsum(nrow, 0) - nrow, nrow))
        xl = p[ids, 0]; xr = xl + d.size[ids, 0]
        span = float((rx1.max() - rx0.min()).item() + d.size[:, 0].max().item() + 1)
        order = torch.argsort(rows.double() * 4 * span + (xl - rx0.min()).double())
        rows, xl, xr, ids = rows[order], xl[order], xr[order], ids[order]
        run = torch.cummax(rows.double() * 4 * span + xr.double(), 0).values - rows.double() * 4 * span
        prev = torch.cat([torch.full((1,), -float('inf'), device=self.dev, dtype=run.dtype), run[:-1]])
        same = torch.cat([torch.zeros(1, dtype=torch.bool, device=self.dev), rows[1:] == rows[:-1]])
        hit = same & (xl.double() < prev - tol)
        bad = torch.zeros(len(p), dtype=torch.bool, device=self.dev)
        bad[ids[hit]] = True
        # the earlier object a hit overlaps is also involved: mark items whose right edge passes the next start
        nxt = torch.cat([xl[1:].double(), torch.full((1,), float('inf'), device=self.dev, dtype=run.dtype)])
        same_n = torch.cat([rows[1:] == rows[:-1], torch.zeros(1, dtype=torch.bool, device=self.dev)])
        bad[ids[same_n & (xr.double() > nxt + tol)]] = True
        # inside the rows also means not in a gap between the segments of a row it spans (rows cut around macros)
        from .rails import row_gaps
        g_r, g_x0, g_x1 = row_gaps(d, ys)
        in_gap = torch.zeros(len(p), dtype=torch.bool, device=self.dev)
        if len(g_r):
            gkey = g_r.double() * 4 * span + (g_x0 - rx0.min()).double()
            j = torch.searchsorted(gkey, rows.double() * 4 * span + (xr - rx0.min()).double() - tol) - 1
            jc = j.clamp(min=0)
            hitg = (j >= 0) & (g_r[jc] == rows) & (g_x1[jc].double() > xl.double() + tol)
            in_gap[ids[hitg]] = True
        # rail parity (OpenROAD dpl powerCompatible): the cell's rails, flipped by its orientation, against the rows
        from . import rails as RL
        RI = RL.info(d, (ys, RL.base_rows(d)[5]))
        wrong_rail = torch.zeros(len(p), dtype=torch.bool, device=self.dev)
        if RI['known']:
            flip = torch.zeros(len(p), dtype=torch.bool, device=self.dev)
            for o in RL.FLIPS_Y:
                flip |= d.orient == o
            eb = torch.where(flip, RI['cell_top'], RI['cell_bot']); et = torch.where(flip, RI['cell_bot'], RI['cell_top'])
            rt_idx = (r + k - 1).clamp(max=R - 1)
            ok = RL._eq(eb, RI['row_bot'][r]) & RL._eq(et, RI['row_top'][rt_idx])
            wrong_rail = inside & ~ok
        inside = inside0 & ~in_gap
        n_mv = int(mv.sum())
        res = {'cells': n_mv, 'wrong_rail': int((mv & wrong_rail).sum()),
               'off_row': int((mv & ~on_row).sum()), 'outside_rows': int((mv & on_row & ~inside).sum()),
               'off_site': int((mv & inside & ~on_site).sum()), 'overlapping': int((mv & bad).sum())}
        # PDN: signal pins clear of the PG via metal on the pin layers (fact 'pdn', as OpenROAD's check_placement)
        if 'pdn' in self.d.facts:
            from . import pdn
            res['pdn_blocked'] = int((pdn.conflicts(self.d, pos=p) & mv).sum())
        res['legal'] = res['off_row'] == res['outside_rows'] == res['off_site'] == res['overlapping'] == res['wrong_rail'] \
            == res.get('pdn_blocked', 0) == 0
        return res

    # -- report -------------------------------------------------------------------------------------------------------
    def evaluate(self, pos=None, metrics=('hpwl',)):
        """A report: metric values plus what they rest on (fidelity, provenance of the facts used)."""
        out = {}
        for m in metrics:
            self._require(m, pos)
            if m == 'hpwl':
                out[m] = self.hpwl(pos).item()
            elif m == 'star_wirelength':
                out[m] = self.star_wirelength(pos).item()
            elif m == 'steiner_wirelength':
                out[m] = self.steiner_wirelength(pos).item()
            elif m == 'rudy':
                out[m] = {k: v for k, v in self.rudy(pos).items() if k != 'map'}
            elif m == 'density_overflow':
                out[m] = self.density_overflow(pos)
            elif m == 'legality':
                out[m] = self.legality(pos)
        man = self.d.manifest
        out['units'] = 'Bookshelf units' if man.get('units', '').startswith('Bookshelf') else 'um'
        out['fidelity'] = man.get('fidelity')
        out['facts'] = {f: self.d.facts[f] for f in sorted(set().union(*(METRICS[m][0] for m in metrics)) - {'placement'})
                        if f in self.d.facts}
        return out

    def status(self):
        lines = ['Evaluator for %s' % self.d.manifest.get('design')]
        for m, (req, impl) in METRICS.items():
            miss = self.missing(m)
            ok = impl and not [x for x in miss if not x.startswith('(')]
            lines.append('  %s %-20s %s' % ('✅' if ok else '❌', m, '' if ok else 'needs ' + ', '.join(miss)))
        return '\n'.join(lines)
