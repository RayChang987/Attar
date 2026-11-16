"""Power analysis, as OpenROAD's report_power (OpenSTA power/Power.cc, the revision OpenROAD-flow-scripts ships), on
the timer's last run (attar.timing.sta): its slews (max analysis) and its estimated parasitics (load capacitance).

Activity (density: transitions per ps, duty: probability of 1), as Power::ensureActivities:
  seeds        the roots of the timing graph (input ports but clock sources; tie cell outputs, loads of undriven
               nets) and the outputs of cells without a function or sequential (macros) take the default input
               activity: density 0.1 / the fastest clock's period, duty 0.5. Clock network pins have their clock's,
               2 / period and (fall - rise) / period; constant pins (tie cells, the timer's constant propagation)
               density 0 and their value as duty.
  propagation  breadth first by levels. A load takes its driver's activity; a driver its Liberty function's: duty the
               probability that it is 1 (inputs independent); density the probability that it differs between two
               consecutive periods of the fastest clock launching its inputs, each input a two-state process with its
               duty and density (Power::evalBddChangeDensity), over that period; with a clock input, the sum over the
               inputs of their density times the probability of the Boolean difference. Exact, as OpenSTA's BDDs,
               here from per-master truth tables. A pin is visited only when a fanin changed by more than 1% (or its
               origin did), as PropActivityVisitor; densities are capped at 1 / the pin's slew, a driver's at one
               transition per launching clock period (two on clock nets). Registers: up to 50 further passes, each
               seeding the registers whose inputs changed, a flop's output density min(data density, min(duty,
               1 - duty) x clock density).
Power per instance (Power::findInternalPower / findSwitchingPower / findLeakagePower):
  internal     output pins: per internal_power group (related pin, pg pin), its energy table at the related pin's
               slew (by its arcs' unateness) and the pin's load, averaged over rise and fall, weighted by the related
               pin's density times its duty (the probability of the output function's Boolean difference with respect
               to it and the group's `when`; else the `when`'s; else 0.5) over the sum of those of the groups of the
               same pg pin, times the output's density; input pins: per group its table at the pin's slew times its
               `when`'s probability (1 without) times the pin's density.
  switching    output pins: 1/2 load x V^2 x density, V the related power pin's voltage (voltage_map), else the
               library's default operating conditions'.
  leakage      per primary power pg pin: a leakage_power group whose `when` is 1 under the constants, else the groups
               weighted by their `when`'s probability plus cell_leakage_power times the rest, else the unconditional
               group; cell_leakage_power without groups.
Groups as report_power: macro (is_macro_cell, memories), pad, clock (every output in the clock network), sequential,
combinational. Units: W. Not modeled: clock gating cells (gated clock activity), user activities
(set_power_activity, VCD, SAIF), next_state functions of the register's own state (Power::evalRegHoldActivity).
"""
import numpy as np
import torch

from ..io.distill import D_INOUT, D_OUT
from .ir import S_NEG, T_CLEAR, T_COMB, T_FALL_EDGE, T_HOLD_F, T_HOLD_R, T_PRESET, T_RISE_EDGE, T_SETUP_F, T_SETUP_R, \
    T_TRISTATE

INF = float('inf')
TOL = 0.01                       # PropActivityVisitor::change_tolerance_
MAX_PASSES = 50                  # Power::max_activity_passes_
O_UNKNOWN, O_INPUT, O_PROP, O_CLOCK, O_CONST = 0, 1, 2, 3, 4
F64 = torch.float64
ACT = torch.float32              # activities, as OpenSTA keeps them (float): a register loop that has not converged
#                                  after its 50 passes ends where float rounding takes it
R_PIN, R_SEQ, R_NONE, R_PAD = 0, 1, 2, 3   # an expression variable: an instance pin, a register state, absent,
#                                            padding (constant 0: merged groups of a level, see Power.__init__)


def _truth(vs, f):
    """A function's truth table over its variables vs (row r: variable i is bit i of r)."""
    return np.array([f({v: (r >> i) & 1 for i, v in enumerate(vs)}) for r in range(1 << len(vs))], np.float64)


class _Exprs:
    """Boolean expressions over instance pins and register states, grouped by their number of variables k: per
    entry its variables (kind, index) and the id of its truth table (2^k rows, one bank per k: an expression's table
    is stored once however many instances use it). Evaluates, inputs independent: the probability that an entry is 1
    (duty), the change density (Power::evalBddChangeDensity) and the Boolean difference density sum."""

    def __init__(self, dev):
        self.dev = dev
        self.rows = {}

    def _g(self, k):
        return self.rows.setdefault(k, {'kind': [], 'idx': [], 'tid': [], 'n': 0, 'bank': [], 'bank_id': {}})

    def add_batch(self, kind, idx, key, table):
        """n entries with variables kind / idx (n, k) and one truth table (key: its identity). Returns (k, rows)."""
        k = kind.shape[1]
        g = self._g(k)
        if key not in g['bank_id']:
            g['bank_id'][key] = len(g['bank']); g['bank'].append(np.asarray(table, np.float64))
        n = kind.shape[0]
        g['kind'].append(kind); g['idx'].append(idx); g['tid'].append(np.full(n, g['bank_id'][key], np.int64))
        rows = np.arange(g['n'], g['n'] + n)
        g['n'] += n
        return k, rows

    def build(self):
        T_ = lambda x, dt: torch.as_tensor(np.asarray(x), device=self.dev, dtype=dt)  # noqa: E731
        self.groups = {}
        for k, g in self.rows.items():
            R = 1 << k
            kk = max(k, 1)
            if k == 0:                                   # constant functions (tie cells)
                kind = np.full((g['n'], 1), R_NONE, np.int64); idx = np.zeros((g['n'], 1), np.int64)
            else:
                kind = np.concatenate(g['kind']).reshape(-1, k) if g['n'] else np.zeros((0, k), np.int64)
                idx = np.concatenate(g['idx']).reshape(-1, k) if g['n'] else np.zeros((0, k), np.int64)
            bits = ((np.arange(R)[:, None] >> np.arange(kk)[None]) & 1).astype(np.float64)
            self.groups[k] = {'kind': T_(kind, torch.long), 'idx': T_(idx, torch.long),
                              'tid': T_(np.concatenate(g['tid']) if g['n'] else np.zeros(0, np.int64), torch.long),
                              'bank': T_(np.stack(g['bank']) if g['bank'] else np.zeros((0, R)), ACT),
                              'bits': T_(bits, ACT), 'n': g['n'], 'flip_base': torch.arange(R, device=self.dev)}
        self.rows = None

    def table(self, k, rows):
        g = self.groups[k]
        return g['bank'][g['tid'][rows]]

    def _vars(self, k, rows, act):
        """Per entry and variable: duty, density, launch period, is a clock pin (the variable's activity)."""
        g = self.groups[k]
        k = g.get('K', k)
        kind, idx = g['kind'][rows], g['idx'][rows]
        pin_u, pin_d, seq_u, seq_d, per, clk = act
        P, S = len(pin_u), len(seq_u)
        ip, iq = idx.clamp(max=P - 1), idx.clamp(max=max(S - 1, 0))
        z = torch.zeros_like(idx, dtype=ACT)
        sq = lambda x: x[iq] if S else z  # noqa: E731
        u = torch.where(kind == R_PIN, pin_u[ip], torch.where(kind == R_SEQ, sq(seq_u), z))
        d = torch.where(kind == R_PIN, pin_d[ip], torch.where(kind == R_SEQ, sq(seq_d), z))
        pr = torch.where(kind == R_PIN, per[ip], torch.full_like(z, INF))
        ck = (kind == R_PIN) & clk[ip]
        return kind[:, :k], u[:, :k], d[:, :k], pr[:, :k], ck[:, :k]

    @staticmethod
    def _weights(bits, p, k):
        """(n, 2^k) probabilities of the truth table's rows."""
        w = torch.ones(p.shape[0], bits.shape[0], dtype=ACT, device=p.device)
        for j in range(k):
            w = w * torch.where(bits[None, :, j] > 0.5, p[:, j:j + 1], 1 - p[:, j:j + 1])
        return w

    def duty(self, k, rows, act):
        g = self.groups[k]
        tab = self.table(k, rows)
        k = g.get('K', k)
        if k == 0:
            return tab[:, 0]
        _, u, _, _, _ = self._vars(k, rows, act)
        return (self._weights(g['bits'], u, k) * tab).sum(1)

    def activity(self, k, rows, act, min_period, unit):
        """(duty, density) of the entries: Power::evalActivity's BDD path (evalBddDuty, evalBddActivity)."""
        g = self.groups[k]
        tab = self.table(k, rows)
        kind, u, d, per, ck = self._vars(k, rows, act)
        k = g.get('K', k)
        if k == 0:
            return tab[:, 0], torch.zeros_like(tab[:, 0])
        n, R = tab.shape
        w = self._weights(g['bits'], u, k)
        duty = (w * tab).sum(1)
        # change density: P(f(previous period) != f(current)) over the period of the fastest launching clock
        period = torch.where(kind == R_PIN, per, torch.full_like(per, INF)).amin(1)
        period = torch.where(torch.isfinite(period), period, torch.full_like(period, min_period))
        period = torch.where(period > 0, period, torch.full_like(period, unit))
        uc = u.clamp(0, 1)
        half = torch.minimum(d * period[:, None], 2 * torch.minimum(uc, 1 - uc)) / 2          # (n, k)
        # Z = (Kronecker of Q_j) (1 - T), Q_j = [[1 - u - h, h], [h, u - h]]; change = 2 T . Z
        z = (1 - tab).reshape((n,) + (2,) * k)                      # dims: bit k-1 .. bit 0 (row r: bit j of r)
        for j in range(k):
            ax = k - j                                              # tensor axis of variable j
            q00 = (1 - uc[:, j] - half[:, j]); q01 = half[:, j]; q11 = uc[:, j] - half[:, j]
            z0, z1 = z.select(ax, 0), z.select(ax, 1)
            sh = (n,) + (1,) * (k - 1)
            n0 = q00.view(sh) * z0 + q01.view(sh) * z1
            n1 = q01.view(sh) * z0 + q11.view(sh) * z1
            z = torch.stack([n0, n1], ax)
        change = 2 * (tab * z.reshape(n, R)).sum(1)
        ok = ~((kind == R_NONE) | ck).any(1)                        # a missing or clock input: the sum below
        dens_change = change / period
        # Boolean difference sum over the pin variables (register states have no pin: not counted); computed for
        # every entry (no host sync: CUDA-graph capturable)
        dsum = torch.zeros(n, dtype=ACT, device=self.dev)
        r = g['flip_base']
        for j in range(k):
            flip = tab[:, r ^ (1 << j)]
            pdiff = (w * (tab - flip).abs()).sum(1)
            dsum = dsum + torch.where(kind[:, j] == R_PIN, d[:, j] * pdiff, torch.zeros_like(pdiff))
        return duty, torch.where(ok, dens_change, dsum)


class Power:
    """Built once per timer (the netlist's activity graph, function tables, power groups); run(result) evaluates."""

    def __init__(self, T):
        from . import ir
        from .library import KIND_MACRO, KIND_PAD, design_power
        from ..io.liberty import parse_function
        self.T = T
        d, dev = T.d, T.dev
        self.dev = dev
        Tt = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)  # noqa: E731
        a = d.raw
        P = T.P
        p_cell = T.p_cell_np
        p_mpin = a['p_mpin'].astype(np.int64)
        p_net = d.p_net.cpu().numpy()
        pdir = a['p_dir'].astype(np.int64) if 'p_dir' in a else np.full(P, 3)
        c_master = a['c_master'].astype(np.int64)
        m_ptr = a['m_pin_ptr'].astype(np.int64)
        names = d.names
        mpn = [x for ps in names['master_pins'] for x in ps]
        chain = ir.chain(d.path, d.manifest)
        used = set(np.unique(c_master[c_master >= 0]).tolist())
        pw, meta = design_power(chain, names['masters'], names['master_pins'], m_ptr, used)
        self.pw = pw
        t, functions, _ = ir.arrays(d)
        functions = {int(k): v for k, v in (functions or {}).items()}
        seqs = {}                                      # master -> its sequentials (output = vars[0], inv = vars[1])
        for m in used:
            mn = names['masters'][m]
            for L in chain:
                if mn in L.cid and L.a['lc_file'][L.cid[mn]] >= 0:
                    s = L.meta.get('sequentials', {}).get(mn)
                    if s and s.get('seq'):
                        seqs[m] = s['seq']
                    break
        keys = np.where(p_mpin >= 0, p_cell * len(mpn) + p_mpin, -1)
        o = np.argsort(keys); sk = keys[o]

        def pin_of(cell, mp):
            k = np.asarray(cell) * len(mpn) + np.asarray(mp)
            i = np.minimum(np.searchsorted(sk, k), len(sk) - 1)
            return np.where(sk[i] == k, o[i], -1)
        is_drv = (pdir == D_OUT) | (pdir == D_INOUT)
        is_load = pdir != D_OUT
        drv_of_net = T.net_driver
        clock = T.clk_of >= 0
        src = np.zeros(P, bool)                        # clock sources (Sdc::isLeafPinClock)
        for c in T.clocks:
            for cell in c['sources']:
                src |= (p_cell == cell) & (p_mpin < 0)
        has_lib = np.zeros(len(c_master) + 1, bool)
        has_lib[:-1] = np.where(c_master >= 0, pw['m_lib'][c_master.clip(0)] > 0, False)
        mp_lib = np.where(p_mpin >= 0, t['mp_lib'][p_mpin.clip(0)] > 0, False) if 'mp_lib' in t else (p_mpin >= 0)
        mp_lib &= np.where(p_mpin >= 0, has_lib[p_cell], False)
        seq_cell = np.zeros(len(c_master), bool)
        latch_cell = np.zeros(len(c_master), bool)
        for m, sq in seqs.items():
            seq_cell[c_master == m] = True
            if any(q['kind'].startswith('latch') for q in sq):
                latch_cell[c_master == m] = True

        # -- the activity graph: wires (driver -> loads) and instance arcs (ActivitySrchPred) ----------------------
        lib_id, fr, to, atype, _ = T._arcs_all
        acell = p_cell[fr]
        check = np.isin(atype, (T_SETUP_R, T_HOLD_R, T_SETUP_F, T_HOLD_F))
        edge = np.isin(atype, (T_RISE_EDGE, T_FALL_EDGE))
        comb = np.isin(atype, (T_COMB, T_TRISTATE))
        # not traversed: timing checks, a flop's clock-to-output (regClkToQ), a latch's data-to-output (latchDtoQ)
        act_arc = ~check & ~(edge & ~latch_cell[acell]) & ~(comb & latch_cell[acell])
        wl = np.nonzero(is_load & (drv_of_net[p_net] >= 0) & (drv_of_net[p_net] != np.arange(P)))[0]
        w_fr, w_to = drv_of_net[p_net[wl]], wl
        # roots (Levelize::isRoot): no edge in but timing checks, latch data-to-output, preset / clear
        lev_arc = ~check & ~(comb & latch_cell[acell]) & ~np.isin(atype, (T_PRESET, T_CLEAR))
        indeg = np.bincount(np.concatenate([w_to, to[lev_arc]]), minlength=P)
        self.root = np.nonzero((indeg == 0) & ~src)[0]
        g_fr = np.concatenate([w_fr, fr[act_arc]]); g_to = np.concatenate([w_to, to[act_arc]])
        lev, _ = T._levelize(P, g_fr, g_to)
        self.n_levels = int(lev.max(initial=0)) + 1
        oo = np.argsort(g_fr, kind='stable')
        self.succ_ptr = np.searchsorted(g_fr[oo], np.arange(P + 1))
        self.succ = g_to[oo]
        self.enq_ok = np.where(p_mpin >= 0, has_lib[p_cell], False)    # instances with a Liberty cell enqueue

        # -- expressions (built per master and function for all its instances at once) -----------------------------
        X = _Exprs(dev)
        C = len(c_master)
        cells_of = {}
        for m in used:
            cells_of[m] = np.nonzero(c_master == m)[0]
        slot_of = {}                                    # register state name -> (C,) its slot per instance (-1)
        nslot = 0
        for m, sq in seqs.items():
            cells = cells_of[m]
            for q in sq:
                for v in q['vars'][:2]:
                    arr = slot_of.setdefault(v, np.full(C, -1, np.int64))
                    new_ = cells[arr[cells] < 0]
                    arr[new_] = nslot + np.arange(len(new_)); nslot += len(new_)
        self.S = nslot
        fcache, mpcache = {}, {}

        def compile_expr(expr):
            if expr not in fcache:
                vs, f = parse_function(expr)
                fcache[expr] = (vs, f, _truth(vs, f))
            return fcache[expr]

        def loc_of(m):
            if m not in mpcache:
                mpcache[m] = {pn: int(m_ptr[m]) + i for i, pn in enumerate(names['master_pins'][m])}
            return mpcache[m]

        def refs(cells, m, vs):
            """(kind, idx) (n, k) of the variables vs on instances cells of master m."""
            loc = loc_of(m)
            kind = np.full((len(cells), len(vs)), R_NONE, np.int64); idx = np.zeros_like(kind)
            for j, v in enumerate(vs):
                if v in loc:
                    pp = pin_of(cells, loc[v])
                    kind[:, j] = np.where(pp >= 0, R_PIN, R_NONE); idx[:, j] = np.maximum(pp, 0)
                elif v in slot_of:
                    sl = slot_of[v][cells]
                    kind[:, j] = np.where(sl >= 0, R_SEQ, R_NONE); idx[:, j] = np.maximum(sl, 0)
            return kind, idx

        def add(cells, m, vs, key, tab):
            kind, idx = refs(cells, m, vs)
            return X.add_batch(kind, idx, key, tab)
        # per driver pin: an expression, or a register state (a function that is one internal state)
        fn_kind = np.zeros(P, np.int64); fn_k = np.zeros(P, np.int64); fn_row = np.zeros(P, np.int64)
        fn_seq = np.zeros(P, np.int64)
        for m, fs in functions.items():
            cells = cells_of.get(m, np.zeros(0, np.int64))
            loc = loc_of(m)
            for pn, expr in fs.items():
                if pn not in loc or not len(cells):
                    continue
                vs, _, tab = compile_expr(expr)
                pins = pin_of(cells, loc[pn]); ok = pins >= 0
                cs, pins = cells[ok], pins[ok]
                if not len(cs):
                    continue
                if len(vs) == 1 and vs[0] in slot_of and expr.strip().strip('()').strip() == vs[0] \
                        and (slot_of[vs[0]][cs] >= 0).all():
                    fn_kind[pins] = 2; fn_seq[pins] = slot_of[vs[0]][cs]
                    continue
                k, rows_ = add(cs, m, vs, expr, tab)
                fn_kind[pins] = 1; fn_k[pins] = k; fn_row[pins] = rows_
        outs = is_drv & mp_lib
        dir_out = np.where(p_mpin >= 0, np.isin(t['mp_dir'][p_mpin.clip(0)], (1, 2)), False)
        funless = outs & dir_out & (fn_kind == 0) & ~np.where(p_mpin >= 0, seq_cell[p_cell], False) & ~clock
        self.funless = np.nonzero(funless)[0]
        # registers (one per instance and sequential): data and clock expressions, output slots
        rg = {k: [] for k in ('cell', 'dk', 'dr', 'ck', 'cr', 'so', 'si', 'isreg', 'clkinv')}
        sp_pin, sp_cell = [], []                       # outputs whose function names a register state
        for m, sq in seqs.items():
            cells = cells_of[m]
            if not len(cells):
                continue
            loc = loc_of(m)
            for pn, expr in functions.get(m, {}).items():
                if pn in loc and any(v in compile_expr(expr)[0] for q in sq for v in q['vars'][:2]):
                    pins = pin_of(cells, loc[pn]); ok = pins >= 0
                    sp_pin.append(pins[ok]); sp_cell.append(cells[ok])
            for q in sq:
                if not q.get('data') or not q.get('clock'):
                    continue
                vs = q['vars']
                dv, _, dt = compile_expr(q['data'])
                cv, _, ct = compile_expr(q['clock'])
                cs_ = q['clock'].replace(' ', '')
                clk_inv = len(cv) == 1 and cs_ in ('!' + cv[0], '!(%s)' % cv[0], cv[0] + "'")
                dk_, dr_ = add(cells, m, dv, q['data'], dt)
                ck_, cr_ = add(cells, m, cv, q['clock'], ct)
                n_ = len(cells)
                rg['cell'].append(cells); rg['dk'].append(np.full(n_, dk_)); rg['dr'].append(dr_)
                rg['ck'].append(np.full(n_, ck_)); rg['cr'].append(cr_)
                rg['so'].append(slot_of[vs[0]][cells])
                rg['si'].append(slot_of[vs[1]][cells] if len(vs) > 1 else np.full(n_, -1))
                rg['isreg'].append(np.full(n_, q['kind'].startswith('ff'))); rg['clkinv'].append(np.full(n_, clk_inv))
        rg = {k: (np.concatenate(v) if v else np.zeros(0, np.int64)) for k, v in rg.items()}
        self.reg_cell = rg['cell'].astype(np.int64)
        self.n_reg = len(self.reg_cell)
        self.reg_h = {1: (rg['dk'].astype(np.int64), rg['dr'].astype(np.int64)),
                      2: (rg['ck'].astype(np.int64), rg['cr'].astype(np.int64))}
        self.reg_so = Tt(rg['so'] if self.n_reg else [0]); self.reg_si = Tt(rg['si'] if self.n_reg else [0])
        self.reg_isreg = Tt(rg['isreg'] if self.n_reg else [True], torch.bool)
        self.reg_clkinv = Tt(rg['clkinv'] if self.n_reg else [False], torch.bool)
        # (state pin, register) pairs: a register enqueues the outputs of its instance that name a state
        sp_pin = np.concatenate(sp_pin) if sp_pin else np.zeros(0, np.int64)
        sp_cell = np.concatenate(sp_cell) if sp_cell else np.zeros(0, np.int64)
        ro = np.argsort(self.reg_cell, kind='stable'); rc_sorted = self.reg_cell[ro]
        lo_ = np.searchsorted(rc_sorted, sp_cell); hi_ = np.searchsorted(rc_sorted, sp_cell, side='right')
        rep = hi_ - lo_
        self.sp_pin = np.repeat(sp_pin, rep)
        self.sp_reg = ro[np.repeat(lo_ - np.cumsum(rep) + rep, rep) + np.arange(rep.sum())] if rep.sum() else \
            np.zeros(0, np.int64)

        # -- internal power rows -----------------------------------------------------------------------------------
        ip_when = {int(k): v for k, v in meta['ip_when'].items()}
        lk_when = {int(k): v for k, v in meta['leak_when'].items()}
        first_sense = {}
        am, af, at_ = t['a_master'], t['a_from'], t['a_to']
        for i in range(len(am) - 1, -1, -1):
            first_sense[(int(am[i]), int(af[i]), int(at_[i]))] = int(t['a_sense'][i])
        mfun = {}
        for m, fs in functions.items():
            loc = loc_of(m)
            for pn, expr in fs.items():
                if pn in loc:
                    mfun[(m, loc[pn])] = expr
        diff_cache = {}

        def diff_when(fun, x, w):
            """The truth table of (the Boolean difference of fun with respect to x) and w, over their variables."""
            key = (fun, x, w)
            if key not in diff_cache:
                vf, ff, _ = compile_expr(fun)
                vw, fw, _ = compile_expr(w) if w else ([], None, None)
                vs = list(vf) + [v for v in vw if v not in vf]

                def g(env):
                    dv = ff(dict(env, **{x: 1})) ^ ff(dict(env, **{x: 0}))
                    return dv & fw(env) if fw is not None else dv
                diff_cache[key] = (vs, _truth(vs, g))
            return diff_cache[key]
        ip_master, ip_pin, ip_rel, ip_pg = pw['ip_master'], pw['ip_pin'], pw['ip_rel'], pw['ip_pg']
        rows = {k: [] for k in ('inst', 'grp', 'pin', 'rel', 'out', 'dk', 'ek', 'er', 'inv', 'pg')}
        # dk: duty kind: 0 constant 0.5, 1 an expression (ek, er), 3 constant 1, 4 constant 0

        def put(cells, g, pins, rel, side, dk, ek, er, inv, pg):
            n_ = len(cells)
            for key, v in (('inst', cells), ('grp', np.full(n_, g)), ('pin', pins), ('rel', rel),
                           ('out', np.full(n_, side)), ('dk', dk), ('ek', ek), ('er', er), ('inv', np.full(n_, inv)),
                           ('pg', np.full(n_, pg))):
                rows[key].append(np.asarray(v))
        for g in range(len(ip_master)):
            m = int(ip_master[g])
            cells = cells_of.get(m)
            if cells is None or not len(cells):
                continue
            to_mp, rel_mp = int(ip_pin[g]), int(ip_rel[g])
            to_pins = pin_of(cells, to_mp); ok = to_pins >= 0
            cells, to_pins = cells[ok], to_pins[ok]
            if not len(cells):
                continue
            n_ = len(cells)
            rel_pins = pin_of(cells, rel_mp) if rel_mp >= 0 else np.full(n_, -1)
            w = ip_when.get(g)
            fun = mfun.get((m, to_mp))
            none_k, none_r = np.full(n_, -1), np.zeros(n_, np.int64)
            if t['mp_dir'][to_mp] in (1, 2):                 # output: findInputDuty
                if rel_mp < 0:
                    dk, ek, er = np.full(n_, 4), none_k, none_r
                elif fun is not None and mpn[rel_mp] in compile_expr(fun)[0]:
                    vs, tab = diff_when(fun, mpn[rel_mp], w)
                    k_, er = add(cells, m, vs, ('diff', fun, mpn[rel_mp], w), tab)
                    dk, ek = np.where(rel_pins >= 0, 1, 4), np.full(n_, k_)
                elif w:
                    vs, _, tab = compile_expr(w)
                    k_, er = add(cells, m, vs, w, tab)
                    dk, ek = np.where(rel_pins >= 0, 1, 4), np.full(n_, k_)
                else:
                    dk, ek, er = np.where(rel_pins >= 0, 0, 4), none_k, none_r
                inv = first_sense.get((m, rel_mp, to_mp)) == S_NEG if rel_mp >= 0 else False
                put(cells, g, to_pins, rel_pins, 1, dk, ek, er, inv, int(ip_pg[g]))
            if t['mp_dir'][to_mp] in (0, 2):                 # input: P(when)
                if w:
                    vs, _, tab = compile_expr(w)
                    k_, er = add(cells, m, vs, w, tab)
                    dk, ek = np.full(n_, 1), np.full(n_, k_)
                else:
                    dk, ek, er = np.full(n_, 3), none_k, none_r
                put(cells, g, to_pins, to_pins, 0, dk, ek, er, False, int(ip_pg[g]))
        rows = {k: (np.concatenate(v) if v else np.zeros(0, np.int64)) for k, v in rows.items()}
        # leakage groups of primary power pg pins
        lk_master, lk_val, lk_pg, lk_prim = pw['lk_master'], pw['lk_val'], pw['lk_pg'], pw['lk_primary']
        lrows = {k: [] for k in ('inst', 'val', 'pg', 'cond', 'ek', 'er')}
        for g in np.nonzero(lk_prim)[0]:
            m = int(lk_master[g]); w = lk_when.get(int(g))
            cells = cells_of.get(m)
            if cells is None or not len(cells):
                continue
            n_ = len(cells)
            if w:
                vs, _, tab = compile_expr(w)
                k_, er = add(cells, m, vs, w, tab)
                ek, cond = np.full(n_, k_), np.ones(n_, bool)
            else:
                ek, er, cond = np.full(n_, -1), np.zeros(n_, np.int64), np.zeros(n_, bool)
            for key, v in (('inst', cells), ('val', np.full(n_, float(lk_val[g]))), ('pg', np.full(n_, int(lk_pg[g]))),
                           ('cond', cond), ('ek', ek), ('er', er)):
                lrows[key].append(v)
        lrows = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in lrows.items()}
        X.build()
        self.X = X
        self.fn_kind, self.fn_seq = Tt(fn_kind), Tt(fn_seq)
        # levels: loads (copy their driver), register state outputs, expression drivers by group
        order = np.argsort(lev, kind='stable')
        bnd = np.searchsorted(lev[order], np.arange(self.n_levels + 1))
        self.levels = []
        for L in range(self.n_levels):
            pins = order[bnd[L]:bnd[L + 1]]
            ld = pins[is_load[pins] & (drv_of_net[p_net[pins]] >= 0) & (drv_of_net[p_net[pins]] != pins)]
            dv = pins[is_drv[pins] & (fn_kind[pins] > 0)]
            st_ = dv[fn_kind[dv] == 2]
            fp = dv[fn_kind[dv] == 1]
            fun = []
            if len(fp):                        # one group per level: every function padded to the level's widest
                K = int(fn_k[fp].max())
                kinds, idxs, tids, bank, bid = [], [], [], [], {}
                for k in np.unique(fn_k[fp]):
                    k = int(k)
                    sel = fp[fn_k[fp] == k]
                    g = X.groups[k]
                    rr = torch.as_tensor(fn_row[sel], device=dev)
                    kd = np.full((len(sel), max(K, 1)), R_PAD, np.int64); ix = np.zeros_like(kd)
                    if k:
                        kd[:, :k] = g['kind'][rr][:, :k].cpu().numpy(); ix[:, :k] = g['idx'][rr][:, :k].cpu().numpy()
                    tid = g['tid'][rr].cpu().numpy()
                    for t in np.unique(tid):
                        if (k, int(t)) not in bid:
                            bid[(k, int(t))] = len(bank)
                            bank.append(np.tile(g['bank'][int(t)].cpu().numpy(), 1 << (K - k)))
                    kinds.append(kd); idxs.append(ix); tids.append(np.array([bid[(k, int(t))] for t in tid]))
                    fun_pins = sel if not fun else np.concatenate([fun[0], sel])
                    fun = [fun_pins]
                key = ('level', L)
                R = 1 << K
                X.groups[key] = {'kind': Tt(np.concatenate(kinds)), 'idx': Tt(np.concatenate(idxs)),
                                 'tid': Tt(np.concatenate(tids)), 'bank': Tt(np.stack(bank), ACT),
                                 'bits': Tt(((np.arange(R)[:, None] >> np.arange(max(K, 1))[None]) & 1).astype(np.float64), ACT),
                                 'n': len(fun[0]), 'K': K, 'flip_base': torch.arange(R, device=dev)}
                v = fun[0]
                fun = [(key, v, Tt(v), torch.arange(len(v), device=dev))]
            self.levels.append({'load': (ld, Tt(ld)), 'load_drv': Tt(drv_of_net[p_net[ld]]), 'state': (st_, Tt(st_)),
                                'fun': fun})
        self.is_seq_load = is_load & (p_mpin >= 0) & np.where(p_mpin >= 0, seq_cell[p_cell], False)
        self.is_drv_t = Tt(is_drv, torch.bool)
        self.is_clock = Tt(clock, torch.bool)
        self.is_iclock = Tt(T.iclk, torch.bool)               # ideal clock pins (getSlew: the clock's transition)
        # Power::findClk on a clock network pin: the fastest clock of the paths through it, its own and those of the
        # input delays on its clock's source ports (their data arrivals run through the clock network too)
        cidx = {k: i for i, k in enumerate(T.clock_names)}
        per_ = [c['period'] for c in T.clocks]
        eff = list(range(len(T.clocks)))
        for ci, c in enumerate(T.clocks):
            for e in T.info.get('input_delay', []):
                cj = cidx.get(e.get('clock'))
                if e.get('cell') in c['sources'] and cj is not None and per_[cj] < per_[eff[ci]]:
                    eff[ci] = cj
        self.clk_of = Tt(np.asarray(eff, np.int64)[np.maximum(T.clk_of, 0)] if len(eff) else np.zeros(P, np.int64))
        self.p_cell_t = Tt(p_cell)
        self.p_net = d.p_net
        self.r = {k: Tt(rows[k], torch.bool if k == 'inv' else torch.long)
                  for k in ('inst', 'grp', 'pin', 'rel', 'out', 'dk', 'inv', 'pg')}
        self.l_ = {'inst': Tt(lrows['inst'].astype(np.int64)), 'val': Tt(lrows['val'], F64),
                   'pg': Tt(lrows['pg'].astype(np.int64)), 'cond': Tt(lrows['cond'].astype(bool), torch.bool)}
        # expression handles of the internal power rows, then the leakage rows: grouped by k once
        hk = np.concatenate([rows['ek'], lrows['ek'].astype(np.int64)])
        hr = np.concatenate([rows['er'], lrows['er'].astype(np.int64)])
        self.n_ip = len(rows['ek'])
        self.n_h = len(hk)
        self._hk = [(int(k), Tt(np.nonzero(hk == k)[0]), Tt(hr[hk == k])) for k in np.unique(hk) if k >= 0]
        self.ip_tab = (Tt(pw['ip_ix1'], torch.float32), Tt(pw['ip_ix2'], torch.float32),
                       Tt(pw['ip_val'], torch.float32), Tt(pw['ip_n']))
        # per instance: report_power group, cell leakage
        self.C = C
        m_kind, m_cleak = pw['m_kind'], pw['m_cleak']
        lib_cell = has_lib[:-1]
        nonclk_out = np.zeros(C, bool)
        oc = outs & dir_out & ~clock
        nonclk_out[p_cell[oc]] = True
        kind = np.full(C, -1, np.int64)                  # 0 seq, 1 comb, 2 clock, 3 macro, 4 pad; -1 not counted
        mk = np.where(c_master >= 0, m_kind[c_master.clip(0)], 0)
        kind[lib_cell] = 1
        kind[lib_cell & seq_cell] = 0
        kind[lib_cell & ~nonclk_out] = 2
        kind[lib_cell & ((mk & KIND_PAD) > 0)] = 4
        kind[lib_cell & ((mk & KIND_MACRO) > 0)] = 3
        self.inst_kind = Tt(kind)
        self.lib_cell = lib_cell
        self.lib_cell_t = Tt(lib_cell, torch.bool)
        self.cleak = Tt(np.where(c_master >= 0, m_cleak[c_master.clip(0)], np.nan), F64)
        has_lk = np.zeros(C, bool)
        if len(lrows['inst']):
            has_lk[lrows['inst'].astype(np.int64)] = True
        self.has_lk = Tt(has_lk, torch.bool)
        mv = np.where(p_mpin >= 0, pw['mp_volt'][p_mpin.clip(0)], np.nan)
        dv_ = np.where(c_master >= 0, pw['m_volt'][c_master.clip(0)], 0.0)[p_cell]
        self.sw_pins = Tt(np.nonzero(outs & dir_out)[0])
        self.volt = Tt(np.where(np.isfinite(mv), mv, dv_), F64)
        self.const_pin = Tt(T.const_net[p_net])        # constants (Sim): per pin -1, 0, 1
        per = [c['period'] for c in T.clocks if c.get('period')]
        self.min_period = min(per) if per else 0.0
        self.in_density = 0.1 / (self.min_period if self.min_period > 0 else 1.0)
        wf = [c['waveform'] for c in T.clocks]
        self.clk_period = Tt([c['period'] for c in T.clocks] or [INF], ACT)
        self.clk_duty = Tt([(w[1] - w[0]) / c['period'] for w, c in zip(wf, T.clocks)] or [0.0], ACT)
        self.debug = 0

    # ------------------------------------------------------------------------------------------------------------
    def _launch_period(self):
        """Power::launchClkPeriod per pin: the fastest clock of the paths through it (its tags with an arrival; a
        clock network pin's clock), else the fastest clock in the design (0 without clocks)."""
        T = self.T
        C = len(T.clocks)
        per = torch.full((T.P,), INF, dtype=ACT, device=self.dev)
        if C:
            ok = torch.isfinite(T.amax[:, :, :2 * C]).any(1) | torch.isfinite(T.amin[:, :, :2 * C]).any(1)  # (P, 2C)
            cp = self.clk_period.repeat_interleave(2)[None].expand_as(ok)
            per = torch.where(ok, cp, torch.full_like(cp, INF)).amin(1)
            per = torch.where(self.is_clock, self.clk_period[self.clk_of], per)
        return torch.where(torch.isfinite(per), per, torch.full_like(per, self.min_period))

    def _find(self, dens, duty, orig):
        """findActivity: constants (0, value); clock network pins without a propagated activity, their clock's."""
        clk = self.is_clock & (orig == O_UNKNOWN)
        fd = torch.where(clk, 2.0 / self.clk_period[self.clk_of], dens)
        fu = torch.where(clk, self.clk_duty[self.clk_of], duty)
        cst = self.const_pin >= 0
        return torch.where(cst, torch.zeros_like(fd), fd), torch.where(cst, self.const_pin.to(fu.dtype), fu)

    def _dense_setup(self):
        """Static tensors for the dense (GPU-resident) propagation: per level the successor edges of its pins, the
        registers' data / clock entries by group, the state pins of each register."""
        if hasattr(self, '_dn'):
            return self._dn
        dev, P = self.dev, self.T.P
        Tt = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)  # noqa: E731
        lev = np.zeros(P, np.int64)
        for L, Ld in enumerate(self.levels):
            for key in ('load', 'state'):
                lev[Ld[key][0]] = L
            for _, pn, _, _ in Ld['fun']:
                lev[pn] = L
        src = np.repeat(np.arange(P), np.diff(self.succ_ptr))
        keep = self.enq_ok[src]
        src, dst = src[keep], self.succ[keep]
        o = np.argsort(lev[src], kind='stable'); src, dst = src[o], dst[o]
        b = np.searchsorted(lev[src], np.arange(self.n_levels + 1))
        edges = [(Tt(src[b[L]:b[L + 1]]), Tt(dst[b[L]:b[L + 1]])) for L in range(self.n_levels)]
        regs = {}
        for which in (1, 2):
            ks, rs = self.reg_h[which]
            regs[which] = [(int(k), Tt(np.nonzero(ks == k)[0]), Tt(rs[ks == k])) for k in np.unique(ks)]
        seeds = np.concatenate([self.root, self.funless])          # seeds enqueue their fanout unconditionally
        lo, hi = self.succ_ptr[seeds], self.succ_ptr[seeds + 1]
        rep = hi - lo
        seed_dst = self.succ[np.repeat(lo - np.cumsum(rep) + rep, rep) + np.arange(rep.sum())] if rep.sum() else \
            np.zeros(0, np.int64)
        self._dn = {'edges': edges, 'regs': regs, 'sp_pin': Tt(self.sp_pin if len(self.sp_pin) else [0]),
                    'sp_reg': Tt(self.sp_reg if len(self.sp_reg) else [0]), 'n_sp': len(self.sp_pin),
                    'isl': Tt(self.is_seq_load, torch.bool), 'reg_cell': Tt(self.reg_cell),
                    'root': Tt(self.root), 'funless': Tt(self.funless), 'seed_dst': Tt(seed_dst)}
        return self._dn

    def activities(self, slew_min):
        """Per pin (density per ps, duty, origin) and per register state (density, duty): Power::ensureActivities.
        slew_min: (P, 2) the min analysis slews (the density cap). The breadth-first visits of OpenSTA are emulated
        level by level with an enqueued mask on the GPU (a pin at a level is evaluated only when enqueued, its fanout
        enqueued only when it changed by more than 1%). A sweep over the levels and a register seeding step are each
        one CUDA graph on the GPU (persistent buffers); one host sync per register pass."""
        D = self._dense_setup()
        B = self._buffers()
        avg = slew_min.to(ACT).mean(1)
        B['cap'].copy_(torch.where(avg > 0, 1.0 / avg.clamp(min=1e-30), torch.full_like(avg, INF)))
        lp = self._launch_period()
        self._lp = lp
        B['lp'].copy_(lp)
        pc = torch.where(lp > 0, torch.where(self.is_clock, 2.0, 1.0).to(ACT) / lp.clamp(min=1e-30),
                         torch.full_like(lp, INF))
        B['prop_cap'].copy_(torch.where(self.is_drv_t, pc, torch.full_like(pc, INF)))
        for k in ('dens', 'duty', 'orig', 'sq_d', 'sq_u', 'sq_o', 'enq', 'regs'):
            B[k].zero_()
        # seeds: roots and function-less outputs take the input activity, their fanout enqueued
        for seeds in (D['root'], D['funless']):
            B['dens'][seeds] = self.in_density; B['duty'][seeds] = 0.5; B['orig'][seeds] = O_INPUT
        B['enq'][D['seed_dst']] = 1
        sweep, seed = self._graphs()
        sweep()
        n_pass = 1
        R = self.n_reg
        while n_pass <= MAX_PASSES and R and bool(B['regs'][D['reg_cell']].any()):
            seed()
            sweep()
            n_pass += 1
        self.n_passes = n_pass - 1
        self.converged = not (R and bool(B['regs'][D['reg_cell']].any()))
        return (B['dens'].clone(), B['duty'].clone(), B['orig'].clone(), B['sq_d'].clone(), B['sq_u'].clone())

    def _buffers(self):
        if not hasattr(self, '_B'):
            dev, P, S = self.dev, self.T.P, self.S
            f = lambda n: torch.zeros(n, dtype=ACT, device=dev)  # noqa: E731
            self._B = {'dens': f(P), 'duty': f(P), 'orig': torch.zeros(P, dtype=torch.long, device=dev),
                       'sq_d': f(S), 'sq_u': f(S), 'sq_o': torch.zeros(S, dtype=torch.long, device=dev),
                       'enq': torch.zeros(P, dtype=torch.uint8, device=dev),
                       'regs': torch.zeros(self.C, dtype=torch.uint8, device=dev),
                       'changed': torch.zeros(P, dtype=torch.bool, device=dev),
                       'cap': f(P), 'prop_cap': f(P), 'lp': f(P)}
        return self._B

    def _graphs(self):
        """(sweep, seed) callables: CUDA-graph replays on the GPU, the eager functions elsewhere."""
        if hasattr(self, '_G'):
            return self._G
        B, D, X = self._B, self._dn, self.X
        dev = self.dev
        dens, duty, orig, sq_d, sq_u, sq_o = (B[k] for k in ('dens', 'duty', 'orig', 'sq_d', 'sq_u', 'sq_o'))
        enq, regs, changed, cap, prop_cap, lp = (B[k] for k in ('enq', 'regs', 'changed', 'cap', 'prop_cap', 'lp'))
        p_cell, isl = self.p_cell_t, D['isl']
        R = self.n_reg

        def pct(v, prev):
            return torch.where(prev == 0, (v != 0).to(v.dtype), (v - prev).abs() / prev.abs().clamp(min=1e-300))

        def set_check(pins, nd, nu, no):
            m = enq[pins] > 0
            nd = torch.minimum(nd, cap[pins])
            nd = torch.where(no == O_PROP, torch.minimum(nd, prop_cap[pins]), nd)
            ch = m & ((pct(nd, dens[pins]) > TOL) | (pct(nu, duty[pins]) > TOL) | (no != orig[pins]))
            dens[pins] = torch.where(m, nd, dens[pins]); duty[pins] = torch.where(m, nu, duty[pins])
            orig[pins] = torch.where(m, no, orig[pins])
            return ch

        def act():
            fd, fu = self._find(dens, duty, orig)
            return (fu, fd, sq_u, sq_d, lp, self.is_clock)

        def step(L, Ld):
            """One level of the breadth-first visit."""
            ld = Ld['load'][1]
            if len(ld):
                dr = Ld['load_drv']
                ch = set_check(ld, dens[dr], duty[dr], torch.full_like(ld, O_PROP))
                changed[ld] = ch
                regs.scatter_reduce_(0, p_cell[ld], (ch & isl[ld]).to(torch.uint8), 'amax')
            stt = Ld['state'][1]
            if len(stt):
                s_ = self.fn_seq[stt]
                changed[stt] = set_check(stt, sq_d[s_], sq_u[s_], sq_o[s_])
            if Ld['fun']:
                a_ = act()
                for k, _, pins, rows_ in Ld['fun']:
                    u, dd = X.activity(k, rows_, a_, self.min_period, 1.0)
                    changed[pins] = set_check(pins, dd, u, torch.full_like(pins, O_PROP))
            src, dst = D['edges'][L]
            if len(src):
                enq.scatter_reduce_(0, dst, changed[src].to(torch.uint8), 'amax')

        def segment(lo, hi, first, last):
            def f():
                if first:
                    changed.zero_()
                for L in range(lo, hi):
                    step(L, self.levels[L])
                if last:
                    enq.zero_()
            return f

        def sweep():
            segment(0, len(self.levels), True, True)()

        def seed():
            """seedRegOutputActivities for the registers whose inputs changed (regs), their state pins enqueued."""
            sel = regs[D['reg_cell']] > 0
            regs.zero_()
            a_ = act()
            ins = []
            for which in (1, 2):                          # data, clock: duty and density of every register
                u_ = torch.zeros(R, dtype=ACT, device=dev); d_ = torch.zeros_like(u_)
                for k, jj, rr in D['regs'][which]:
                    uu, dd = X.activity(k, rr, a_, self.min_period, 1.0)
                    u_[jj] = uu; d_[jj] = dd
                ins.append((u_, d_))
            (u, dn), (cu, cd) = ins
            latch = torch.where(dn > cd / 2, torch.where(self.reg_clkinv, dn * (1 - cu), dn * cu), dn)
            od = torch.where(self.reg_isreg, torch.minimum(dn, torch.minimum(u, 1 - u) * cd), latch)
            for slots, inv in ((self.reg_so, False), (self.reg_si, True)):
                ok = sel & (slots >= 0)
                s_ = slots.clamp(min=0)
                nu = (1 - u) if inv else u
                sq_d.scatter_(0, s_, torch.where(ok, od, sq_d[s_]))
                sq_u.scatter_(0, s_, torch.where(ok, nu, sq_u[s_]))
                sq_o.scatter_(0, s_, torch.where(ok, torch.full_like(s_, O_PROP), sq_o[s_]))
            if D['n_sp']:
                enq.scatter_reduce_(0, D['sp_pin'], sel[D['sp_reg']].to(torch.uint8), 'amax')
        if dev.type != 'cuda' if hasattr(dev, 'type') else not str(dev).startswith('cuda'):
            self._G = (sweep, seed if R else (lambda: None))
            return self._G
        # capture: warm up on a side stream with copies of the state, then record each function once
        saved = {k: v.clone() for k, v in B.items()}
        dev = torch.device(dev)
        dev = torch.device('cuda', dev.index if dev.index is not None else torch.cuda.current_device())
        # warm-up on a side stream, level by level, measuring each level's working memory: a capture keeps the working
        # memory of everything it records, so the levels are captured in segments (one graph each, sharing one pool,
        # replayed in order) whose summed working memory fits; too little room: run eagerly (an out-of-memory error
        # inside a capture leaves the CUDA graph state unusable)
        torch.cuda.synchronize(dev)
        need = []
        st_ = torch.cuda.Stream(device=dev); st_.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(st_):
            changed.zero_()
            for L, Ld in enumerate(self.levels):
                torch.cuda.synchronize(dev)
                base = torch.cuda.memory_allocated(dev)
                torch.cuda.reset_peak_memory_stats(dev)
                step(L, Ld)
                torch.cuda.synchronize(dev)
                need.append(torch.cuda.max_memory_allocated(dev) - base + (2 << 20))
            enq.zero_()
            base = torch.cuda.memory_allocated(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            if R:
                seed()
            torch.cuda.synchronize(dev)
            need_seed = torch.cuda.max_memory_allocated(dev) - base + (2 << 20)
        torch.cuda.current_stream(dev).wait_stream(st_)
        torch.cuda.synchronize(dev)
        free, _ = torch.cuda.mem_get_info(dev)
        room = free + torch.cuda.memory_reserved(dev) - torch.cuda.memory_allocated(dev)
        budget = room // 4
        if max(need + [need_seed]) > budget:
            self._G = (sweep, seed if R else (lambda: None))
            for k, v in saved.items():
                B[k].copy_(v)
            return self._G
        segs, lo, acc = [], 0, 0
        for L, b in enumerate(need):
            if (acc + b > budget or L - lo >= 8) and L > lo:      # (and few levels: instantiating a graph of many
                                                                   # thousand kernels needs driver memory too)
                segs.append((lo, L)); lo, acc = L, 0
            acc += b
        segs.append((lo, len(need)))
        pool = torch.cuda.graph_pool_handle()
        graphs = []
        fns = [segment(lo, hi, i == 0, i == len(segs) - 1) for i, (lo, hi) in enumerate(segs)]
        # an instantiated graph holds driver memory too (about its kernel count): capture while there is room for
        # another of the largest seen, run the remaining segments eagerly
        cost = 0
        for fn in ([seed] if R else []) + fns:
            free0, _ = torch.cuda.mem_get_info(dev)
            if free0 < 3 * cost + (2 << 30):           # (leave room for the evaluation that follows)
                graphs.append(fn)
                continue
            for k, v in saved.items():
                B[k].copy_(v)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                fn()
            graphs.append(g.replay)
            cost = max(cost, free0 - torch.cuda.mem_get_info(dev)[0])
        for k, v in saved.items():                        # capture runs nothing: restore anyway
            B[k].copy_(v)
        seed_fn = graphs.pop(0) if R else (lambda: None)
        sw_fns = graphs

        def sweep_graph():
            for f in sw_fns:
                f()
        self.n_segments = len(fns)
        self._G = (sweep_graph, seed_fn)
        return self._G

    def run(self, result):
        """Per instance internal, switching and leakage power (W) at the timer's last run: a PowerResult."""
        T, dev = self.T, self.dev
        from .drv import _static
        from .sta import nldm
        S = _static(T)
        dens, duty, orig, sq_d, sq_u = self.activities(T.smin)
        fd, fu = self._find(dens, duty, orig)
        a_ = (fu, fd, sq_u, sq_d, self._lp, self.is_clock)
        # slews (max analysis; ideal clock pins: the clock's transition) and load capacitance
        smax = T.smax.float()
        ctr = T.ctr.float()[self.clk_of] if len(T.clocks) else torch.zeros_like(smax)
        slew = torch.where(self.is_iclock[:, None], ctr, smax)
        pins_c = S['net_sum'](T.pin_cap)
        if T.pi is not None:
            pit = T.pi[:, :2, 0] + T.pi[:, :2, 2]
            net_cap = torch.where(pit >= pins_c, pit, pins_c).amax(1)
        else:
            net_cap = pins_c.amax(1)
        load = net_cap.float()[self.p_net]
        C = self.C
        ev = self._eval_handles(a_)
        nr = self.n_ip
        # -- internal power ------------------------------------------------------------------------------------------
        r = self.r
        n = len(r['inst'])
        internal = torch.zeros(C, dtype=F64, device=dev)
        if n:
            dk = r['dk']
            one, half_, zero = (torch.full((n,), v, dtype=F64, device=dev) for v in (1.0, 0.5, 0.0))
            gd = torch.where(dk == 3, one, torch.where(dk == 0, half_, zero))
            gd = torch.where(dk == 1, ev[:nr], gd)
            pin, rel, out, g = r['pin'], r['rel'], r['out'] == 1, r['grp']
            ix1, ix2, val, nn = self.ip_tab
            energy = torch.zeros(n, dtype=F64, device=dev)
            relc = rel.clamp(min=0)
            CH = 1 << 20                                    # rows per chunk: the gathered tables stay small
            for a0 in range(0, n, CH):
                sl_ = slice(a0, min(n, a0 + CH))
                gg, oo, rr, pp = g[sl_], out[sl_], rel[sl_], pin[sl_]
                for trf in (0, 1):
                    frf = torch.where(r['inv'][sl_], torch.full_like(gg, 1 - trf), torch.full_like(gg, trf))
                    s_rel = slew[relc[sl_], frf]
                    s_in = torch.where(oo, torch.where(rr >= 0, s_rel, torch.zeros_like(s_rel)), slew[pp, trf])
                    cl = torch.where(oo | self.is_drv_t[pp], load[pp], torch.zeros_like(s_in))
                    e = nldm(ix1[gg, trf], ix2[gg, trf], val[gg, trf], nn[gg, trf].long(), s_in, cl).double()
                    energy[sl_] += torch.where(nn[gg, trf, 0] > 0, e, torch.zeros_like(e))
            energy = energy / 2
            fdr = torch.where(rel >= 0, fd[relc], torch.zeros_like(energy))
            contrib = torch.where(out & (rel >= 0), fdr * gd, torch.zeros_like(energy))
            ds = self._seg('pinpg')(contrib)[self._inv_pinpg]
            weight = torch.where((ds != 0) & (rel >= 0), fdr * gd / torch.where(ds != 0, ds, torch.ones_like(ds)),
                                 torch.zeros_like(ds))
            pwr = torch.where(out, weight * energy * fd[pin], energy * gd * fd[pin])
            internal = self._seg('inst')(pwr)
        # -- switching -----------------------------------------------------------------------------------------------
        sp = self.sw_pins
        sw = self._seg('sw')(0.5 * load[sp].double() * self.volt[sp] ** 2 * fd[sp])
        # -- leakage -------------------------------------------------------------------------------------------------
        leak = self._leakage(ev[nr:])
        res = PowerResult(self, internal * 1e-3, sw * 1e-3, leak * 1e-9)     # fJ / ps = mW; nW
        res.activity = (fd, fu, orig)
        return res

    def _seg(self, name):
        """Deterministic segmented sums (attar.timing.drv.SegSum) over the static indices of the power rows."""
        if not hasattr(self, '_segs'):
            from .drv import SegSum
            dev, C = self.dev, self.C
            r, l_ = self.r, self.l_
            self._segs = {'inst': SegSum(r['inst'], C), 'sw': SegSum(self.p_cell_t[self.sw_pins], C)}
            key = r['pin'] * 64 + (r['pg'] + 1)
            uk, inv = torch.unique(key, return_inverse=True)
            self._segs['pinpg'] = SegSum(inv, len(uk)); self._inv_pinpg = inv
            lk = l_['inst'] * 64 + (l_['pg'] + 1)
            uk, inv = torch.unique(lk, return_inverse=True)
            self._K_leak = len(uk); self._inv_leak = inv
            self._segs['leak'] = SegSum(inv, len(uk))
            self._inst_leak = torch.zeros(len(uk), dtype=torch.long, device=dev).scatter_(0, inv, l_['inst'])
            self._segs['leak_inst'] = SegSum(self._inst_leak, C)
        return self._segs[name]

    def _eval_handles(self, a_):
        """Duty of every expression handle: the internal power rows', then the leakage rows'."""
        out = torch.zeros(self.n_h, dtype=F64, device=self.dev)
        for k, ii, rr in self._hk:
            out[ii] = self.X.duty(k, rr, a_).to(out.dtype)
        return out

    def _leakage(self, duty):
        """Power::findLeakagePower per instance (nW); duty: each leakage row's `when` probability."""
        dev, C = self.dev, self.C
        l_ = self.l_
        out = torch.where(self.has_lk, torch.zeros(C, dtype=F64, device=dev), torch.nan_to_num(self.cleak, nan=0.0))
        n = len(l_['inst'])
        if n:
            cond = l_['cond']
            duty = torch.where(cond, duty, torch.zeros_like(duty))
            true_ = cond & self._const_true()
            ss, inv, K = self._seg('leak'), self._inv_leak, self._K_leak
            c2 = cond & ~true_
            cond_l = ss(torch.where(c2, l_['val'] * duty, torch.zeros_like(duty)))
            cond_ds = ss(torch.where(c2 & (l_['val'] > 0), duty, torch.zeros_like(duty)))
            has_c = ss(c2.double()) > 0
            has_t = ss(true_.double()) > 0
            idx = torch.arange(n, device=dev)
            last_t = torch.full((K,), -1, dtype=torch.long, device=dev).scatter_reduce_(
                0, inv, torch.where(true_, idx, torch.full_like(idx, -1)), 'amax')
            last_u = torch.full((K,), -1, dtype=torch.long, device=dev).scatter_reduce_(
                0, inv, torch.where(~cond, idx, torch.full_like(idx, -1)), 'amax')
            inst_k = self._inst_leak
            cl = self.cleak[inst_k]
            v = torch.where(has_t, l_['val'][last_t.clamp(min=0)],
                            torch.where(has_c, cond_l + torch.where(torch.isfinite(cl), cl * (1 - cond_ds),
                                                                    torch.zeros_like(cl)),
                                        torch.where(last_u >= 0, l_['val'][last_u.clamp(min=0)], torch.zeros_like(cl))))
            out = out + self._seg('leak_inst')(v)
        return torch.where(self.lib_cell_t, out, torch.zeros_like(out))

    def _const_true(self):
        """Per leakage row: its `when` is 1 under the constants (tie cells), as Sim::evalExpr."""
        if hasattr(self, '_ct'):
            return self._ct
        cst = self.const_pin
        fu = torch.where(cst >= 0, cst.to(ACT), torch.full_like(cst, 0, dtype=ACT) + 0.5)
        sq = torch.full((self.S,), 0.5, dtype=ACT, device=self.dev)
        z = torch.zeros_like(fu)
        a_ = (fu, z, sq, torch.zeros_like(sq), z, self.is_clock)
        v = self._eval_handles(a_)[self.n_ip:]
        self._ct = self.l_['cond'] & (v > 1 - 1e-12)
        return self._ct


class PowerResult:
    """Per instance (C,) internal, switching and leakage power in W (0 for instances without a Liberty cell), and
    totals per report_power group."""
    GROUPS = ('Sequential', 'Combinational', 'Clock', 'Macro', 'Pad')

    def __init__(self, power, internal, switching, leakage):
        self.power = power
        self.internal, self.switching, self.leakage = internal, switching, leakage
        self.total_per_instance = internal + switching + leakage
        k = power.inst_kind
        self.groups = {}
        for i, name in enumerate(self.GROUPS):
            m = k == i
            self.groups[name] = (float(internal[m].sum()), float(switching[m].sum()), float(leakage[m].sum()))
        m = k >= 0
        self.total = (float(internal[m].sum()), float(switching[m].sum()), float(leakage[m].sum()))

    def summary(self):
        i, s, l_ = self.total
        out = {'internal': i, 'switching': s, 'leakage': l_, 'total': i + s + l_}
        for g, (a, b, c) in self.groups.items():
            out[g.lower()] = a + b + c
        return out

    def report(self, file=None):
        """Print (and return) the report_power table (W)."""
        import sys
        out = file or sys.stdout
        print('%-14s %10s %10s %10s %10s' % ('Group', 'Internal', 'Switching', 'Leakage', 'Total'), file=out)
        tot = sum(self.total) or 1.0
        for g, (a, b, c) in list(self.groups.items()) + [('Total', self.total)]:
            print('%-14s %10.3e %10.3e %10.3e %10.3e %5.1f%%' % (g, a, b, c, a + b + c, 100 * (a + b + c) / tot),
                  file=out)
        return self.summary()

    def instances(self):
        """{instance name: (internal, switching, leakage, total)} for the instances with a Liberty cell (W)."""
        d = self.power.T.d
        names = d.names['cells']
        I, S, L = (x.cpu().numpy() for x in (self.internal, self.switching, self.leakage))
        return {names[c].replace('\\', ''): (float(I[c]), float(S[c]), float(L[c]), float(I[c] + S[c] + L[c]))
                for c in np.nonzero(self.power.lib_cell)[0]}
