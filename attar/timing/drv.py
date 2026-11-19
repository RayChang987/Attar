"""Electrical design rules (DRV): max slew, max capacitance and max fanout, as OpenSTA's report_check_types
-max_slew -max_capacitance -max_fanout (search/CheckSlews.cc, CheckCapacitances.cc, CheckFanouts.cc), on the timer's
last run (attar.timing.sta): its slews and its estimated parasitics.

  slew         every pin but the ideal clock network's: limit - slew, the worse of rise and fall (ties: rise). Limit:
               the tightest of the SDC's design limit (set_max_transition [current_design]), the port's (ports), the
               Liberty pin's max_transition (else its library's default_max_transition, outputs only), and the SDC
               clock limits (-data_path) of the clocks whose paths reach the pin.
  capacitance  driver pins (cell outputs, input ports), not constant (tie cells: the timer's constant propagation),
               not in the ideal clock network: limit - load, the load being GraphDelayCalc::loadCap, the larger over
               rise / fall of the net's pi model total (pins and wire) when it is at least the pins' capacitance, else
               the pins'. Limit: the SDC's design / port limit, the Liberty pin's max_capacitance (else
               default_max_capacitance, outputs).
  fanout       the same drivers: limit - fanout, fanout the sum over the net's loads but output ports of their Liberty
               fanout_load (else the library's default_fanout_load; neither: 0), 1 for a load without a Liberty port.
               Limit: set_max_fanout on the design / port, the Liberty pin's max_fanout (else default_max_fanout).
A pin without a limit is not checked (slack inf). Units: ps, fF.
"""
import numpy as np
import torch

from ..io.distill import D_INOUT, D_OUT

INF = float('inf')


class SegSum:
    """A deterministic segmented sum over a static index (CUDA's index_add_ adds in no fixed order): values (n, ...)
    -> (m, ...), segment i the sum of the values whose index is i, always in the same order (a stable sort, then
    torch.segment_reduce)."""

    def __init__(self, index, m, device=None):
        """device: where to keep it (default: the index's), built where the index is (a large one on the CPU: the
        sort's working memory stays off the GPU)."""
        o = torch.argsort(index, stable=True)
        self.len = torch.bincount(index[o].long(), minlength=m)
        self.o = o.int() if len(o) < 2 ** 31 else o          # (the order as int32: half the memory, same gather)
        if device is not None:
            self.o, self.len = self.o.to(device), self.len.to(device)
        self.m = m

    def __call__(self, v):
        if not len(self.o):
            return torch.zeros((self.m,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
        return torch.segment_reduce(v[self.o], 'sum', lengths=self.len, axis=0)


def _static(T):
    """The placement-independent part, once per timer: per pin its limits and check masks, per driver its fanout."""
    if getattr(T, '_drv', None) is not None:
        return T._drv
    d, dev = T.d, T.dev
    from . import ir
    t, _, _ = ir.arrays(d)
    a = d.raw
    P = T.P
    p_mpin = a['p_mpin'].astype(np.int64)
    p_cell = T.p_cell_np
    p_net = d.p_net.cpu().numpy()
    E = len(a['n_ptr']) - 1
    pdir = a['p_dir'].astype(np.int64) if 'p_dir' in a else np.full(P, 3)
    lim = (T.info.get('limits') or {})
    io_cell = a['io_cell'] if 'io_cell' in a else np.zeros(0, np.int64)
    is_port = np.isin(p_cell, io_cell) & (p_mpin < 0)
    ok = p_mpin >= 0
    get = lambda k: t[k][p_mpin.clip(0)] if k in t else np.full(P, np.nan, np.float32)  # noqa: E731
    has_lib = np.where(ok, get('mp_lib') > 0, False) if 'mp_lib' in t else np.zeros(P, bool)

    def resolve(kind, lib_vals):
        """Per pin the tightest limit: design, port (SDC), Liberty (resolved defaults); NaN: none."""
        L = lim.get(kind) or {}
        out = np.where(ok & has_lib, lib_vals, np.nan).astype(np.float64)
        if L.get('design') is not None:
            out = np.fmin(out, L['design'])
        for c, v in (L.get('ports') or {}).items():
            m = is_port & (p_cell == int(c))
            out[m] = np.fmin(out[m], v)
        return out
    slew_lim = resolve('max_transition', get('mp_maxtran'))
    cap_lim = resolve('max_capacitance', get('mp_caplim'))
    fo_lim = resolve('max_fanout', get('mp_maxfo'))
    clock = T.iclk                                         # the ideal clock network (OpenSTA: isIdealClock)
    const = (T.const_net[p_net] >= 0) if hasattr(T, 'const_net') else np.zeros(P, bool)
    drv = (pdir == D_OUT) | (pdir == D_INOUT)
    # fanout: per net, the loads' fanout_load (output ports excluded), 1 for loads without a Liberty port
    load = (~drv | (pdir == D_INOUT)) & ~is_port
    fl = np.where(has_lib, np.nan_to_num(get('mp_fanload'), nan=0.0), 1.0)
    net_fo = np.zeros(E); np.add.at(net_fo, p_net[load], fl[load]) if load.any() else None
    T_ = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)  # noqa: E731
    # SDC clock limits on data paths: (clock index, limit)
    cl = (lim.get('max_transition') or {}).get('clocks') or {}
    cidx = {k: i for i, k in enumerate(T.clock_names)}
    clk_lims = [(cidx[c], v['data']) for c, v in cl.items() if c in cidx and v.get('data') is not None]
    T._drv = {'net_sum': SegSum(d.p_net, E),
              'slew_lim': T_(np.where(clock, np.nan, slew_lim)), 'cap_lim': T_(cap_lim), 'fo_lim': T_(fo_lim),
              'cap_chk': T_(drv & ~const & ~clock & np.isfinite(cap_lim), torch.bool),
              'fo_chk': T_(drv & ~const & ~clock & np.isfinite(fo_lim), torch.bool),
              'fanout': T_(net_fo[p_net]), 'clk_lims': clk_lims, 'clock': T_(clock, torch.bool),
              'p_net': d.p_net}
    return T._drv


def check(result):
    """The DRV checks of a TimingResult's run: a DrvResult."""
    T = result.timer
    S = _static(T)
    smax = T.smax                                           # (P, 2) max analysis slews of the last run
    lim = S['slew_lim']
    if S['clk_lims']:                                       # clock data-path limits of the clocks reaching the pin
        reach = torch.isfinite(T.amax).any(1)               # (P, NT) tags with an arrival
        for ci, v in S['clk_lims']:
            hit = reach[:, 2 * ci] | reach[:, 2 * ci + 1]
            lim = torch.where(hit & ~S['clock'], torch.fmin(lim, torch.full_like(lim, v)), lim)
    sl = lim[:, None] - smax
    slew_rf = (sl[:, 1] < sl[:, 0]).long()                  # ties: rise (OpenSTA, for regression stability)
    slew_slack = torch.where(torch.isfinite(lim), sl.gather(1, slew_rf[:, None])[:, 0], torch.full_like(lim, INF))
    slew = smax.gather(1, slew_rf[:, None])[:, 0]
    # load capacitance: per net, max over rise / fall of max(pi total, pin caps) (GraphDelayCalc::loadCap)
    pins = S['net_sum'](T.pin_cap)
    if T.pi is not None:
        pit = T.pi[:, :2, 0] + T.pi[:, :2, 2]              # rise / fall max: near + far C
        net_cap = torch.where(pit >= pins, pit, pins).amax(1)
    else:
        net_cap = pins.amax(1)
    cap = net_cap[S['p_net']]
    cap_slack = torch.where(S['cap_chk'], S['cap_lim'] - cap, torch.full_like(cap, INF))
    fo_slack = torch.where(S['fo_chk'], S['fo_lim'] - S['fanout'], torch.full_like(cap, INF))
    return DrvResult(result, slew, lim, slew_slack, cap, cap_slack, S['fanout'], fo_slack)


class DrvResult:
    """Per pin (P,): slew (the checked transition's), slew limit and slack; load capacitance, its slack (drivers);
    fanout, its slack (drivers). inf where a pin is not checked. Summaries: counts of violations and the worst
    slacks, as OpenSTA's max_slew_violation_count / max_slew_check_slack."""

    def __init__(self, result, slew, slew_limit, slew_slack, cap, cap_slack, fanout, fanout_slack):
        self.result = result
        self.slew, self.slew_limit, self.slew_slack = slew, slew_limit, slew_slack
        self.cap, self.cap_slack = cap, cap_slack
        self.fanout, self.fanout_slack = fanout, fanout_slack
        T = result.timer
        S = T._drv
        self.cap_limit, self.fanout_limit = S['cap_lim'], S['fo_lim']

        def summ(s):
            f = torch.isfinite(s)
            return (int(f.sum()), int((s[f] < 0).sum()), float(s[f].min()) if bool(f.any()) else INF)
        self.slew_checked, self.slew_violations, self.worst_slew_slack = summ(slew_slack)
        self.cap_checked, self.cap_violations, self.worst_cap_slack = summ(cap_slack)
        self.fanout_checked, self.fanout_violations, self.worst_fanout_slack = summ(fanout_slack)

    def summary(self):
        return {'max_slew_violations': self.slew_violations, 'worst_slew_slack': self.worst_slew_slack,
                'max_capacitance_violations': self.cap_violations, 'worst_cap_slack': self.worst_cap_slack,
                'max_fanout_violations': self.fanout_violations, 'worst_fanout_slack': self.worst_fanout_slack,
                'slew_pins': self.slew_checked, 'cap_pins': self.cap_checked, 'fanout_pins': self.fanout_checked}

    def violators(self, kind='slew'):
        """[(pin name, value, limit, slack)] of the pins violating one rule ('slew', 'capacitance', 'fanout'), worst
        first (report_check_types -violators)."""
        v, lim, s = {'slew': (self.slew, self.slew_limit, self.slew_slack),
                     'capacitance': (self.cap, self.cap_limit, self.cap_slack),
                     'fanout': (self.fanout, self.fanout_limit, self.fanout_slack)}[kind]
        idx = torch.nonzero(s < 0).flatten()
        idx = idx[torch.argsort(s[idx])]
        nm = self.result._pin_name
        return [(nm(int(p)), float(v[p]), float(lim[p]), float(s[p])) for p in idx.tolist()]

    def report(self, file=None, max_count=5):
        """Print (and return) the summary and the worst violators of each rule."""
        import sys
        out = file or sys.stdout
        sm = self.summary()
        print('Design rules (max slew / capacitance / fanout, OpenSTA report_check_types):', file=out)
        for kind, n, w, c, unit in (('slew', self.slew_violations, self.worst_slew_slack, self.slew_checked, 'ps'),
                                    ('capacitance', self.cap_violations, self.worst_cap_slack, self.cap_checked, 'fF'),
                                    ('fanout', self.fanout_violations, self.worst_fanout_slack, self.fanout_checked, '')):
            print('  max %-11s %6d violations of %d pins checked (worst slack %s)'
                  % (kind, n, c, '-' if w == INF else '%.2f %s' % (w, unit)), file=out)
            for name, v, lim, s in self.violators(kind)[:max_count]:
                print('    %-50s %10.2f limit %8.2f slack %9.2f' % (name, v, lim, s), file=out)
        return sm
