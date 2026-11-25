"""
window.py
=========
The static stage: the capacity window (Q_n, Q_p, x100, y100) of each cell from its OWN budget tests,
with the uniqueness check that goes with it.

    V_ocv(q) = U_p(y100 + q / Q_p) - U_n(x100 - q / Q_n),     q = charge below the top of charge [A.h]

Data ("fit_on" in the study config, any combination; "check_on" blocks are scored, never fitted):
    rest         end-of-rest voltages of the dataset's rest test (HPPC, GITT): equilibrium points
    pseudo_ocv   mean of the discharge and charge branches of the low-rate OCV test on a common q grid -
                 cancels the low-rate overpotential and most of the hysteresis
    discharge, charge   one branch alone, with a constant overpotential offset as a nuisance parameter
Each block is weighted equally (its residuals divided by sqrt(n)), so 20 rest points count as much as
400 curve points. The RMSE reported is sqrt(mean over blocks of each block's mean square) [mV].

Q_n ("Q_n" in the config): "free", a number [A.h], or "literature" (the parameter set's geometric
value, physics.truth_groups). OCV data identify the anode's lithium Q_n x100 but bound Q_n only from
below - on both the Mohtat and the LG M50 cells the RMSE stays within 0.5 mV over 0.4-0.8 A.h of Q_n.
So whatever the adopted choice, every cell is ALSO fitted with Q_n free from many starts and profiled:

    multi-start   `starts` random starts of a bounded least squares -> the distinct minima
    profile       Q_n fixed on a grid, the other three refitted -> RMSE(Q_n)
    reported      the free optimum, the Q_n valley within +0.5 mV of it, the number of minima within
                  1 mV, and whether the adopted Q_n lies inside the valley

OCP correction ("ocp_correction", optional): one piecewise-linear dU on fixed knots of ONE electrode,
shared by all the cells of the stage (so no cell gets a curve of its own), bounded by bound_mV and
smoothed (weight on its second difference), fitted jointly with every cell's window. Stored with the
study and used by the dynamic model afterwards. Knots: "knot_positions", or "knots" equally spaced
over "range" (the stoichiometries the electrode actually spans). On the LG M50 cells 33 knots over
x in [0, 0.92] with a 25 mV bound bring the C/10 pseudo-OCV from 8.5 to 2.9 mV with ONE correction
for two cells - but the GITT rest points do not follow (silicon hysteresis): keep "check_on": ["rest"]
to show it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from srcs import physics
from srcs.ocp import OCPSet

log = logging.getLogger(__name__)

WINDOW = ('Q_n', 'Q_p', 'x100', 'y100')
BOUNDS = {'Q_n': (3.5, 10.0), 'Q_p': (3.5, 12.0), 'x100': (0.4, 1.0), 'y100': (0.0, 0.6)}
OFFSET_BOUND = 0.1                        # [V] constant overpotential of a single branch
XSCALE = {'Q_n': 1.0, 'Q_p': 1.0, 'x100': 0.1, 'y100': 0.05, 'off': 0.01}


@dataclass
class Block:
    kind: str
    q: np.ndarray
    v: np.ndarray
    offset: bool = False


# ------------------------------------------------------------------ data
def blocks(ds, cell: str, kinds, v_range=None, n_curve=400):
    """The data blocks of one cell. v_range [V] trims the OCV-test curves (end-of-discharge spike,
    start-of-charge relaxation, top-of-charge tail); the rest points are kept whole."""
    out, br = [], None
    for k in kinds:
        if k == 'rest':
            q, v = ds.rest_points(cell)
            out.append(Block('rest', q, v))
            continue
        br = br if br is not None else ds.ocv_branches(cell)
        if k == 'pseudo_ocv':
            missing = {'discharge', 'charge'} - set(br)
            if missing:
                raise ValueError('%s: %s has no %s branch - use "rest" or a single branch'
                                 % (cell, ds.window_data['ocv_test'], missing.pop()))
            (qd, vd), (qc, vc) = br['discharge'], br['charge']
            lo, hi = max(qd.min(), qc.min()), min(qd.max(), qc.max())
            g = np.linspace(lo, hi, n_curve)
            od, oc = np.argsort(qd), np.argsort(qc)
            vdg, vcg = np.interp(g, qd[od], vd[od]), np.interp(g, qc[oc], vc[oc])
            keep = np.ones(len(g), bool) if v_range is None else (vdg >= v_range[0]) & (vcg <= v_range[1])
            out.append(Block('pseudo_ocv', g[keep], 0.5 * (vdg + vcg)[keep]))
        elif k in ('discharge', 'charge'):
            if k not in br:
                raise ValueError('%s: %s has no %s branch' % (cell, ds.window_data['ocv_test'], k))
            q, v = br[k]
            o = np.argsort(q)
            g = np.linspace(q.min(), q.max(), n_curve)
            vg = np.interp(g, q[o], v[o])
            keep = np.ones(len(g), bool) if v_range is None else (vg >= v_range[0]) & (vg <= v_range[1])
            out.append(Block(k, g[keep], vg[keep], offset=True))
        else:
            raise ValueError('unknown window data %r' % k)
    return out


# ------------------------------------------------------------------ model and residuals
def ocv(w, q, ocp: OCPSet):
    Q_n, Q_p, x100, y100 = w[:4]
    return ocp.U_p(y100 + q / Q_p) - ocp.U_n(x100 - q / Q_n)


def rmse_mV(weighted_residual, n_blocks):
    return float(1e3 * np.sqrt(np.sum(weighted_residual ** 2) / n_blocks))


class CellFit:
    """Parameter vector of one cell: [Q_n (unless fixed), Q_p, x100, y100, one offset per branch block]."""

    def __init__(self, bl, Qn_fixed=None):
        self.bl, self.Qn_fixed = bl, Qn_fixed
        self.n_off = sum(b.offset for b in bl)
        keys = WINDOW[1:] if Qn_fixed is not None else WINDOW
        self.keys = list(keys) + ['off'] * self.n_off
        self.lo = np.array([BOUNDS[k][0] for k in keys] + [-OFFSET_BOUND] * self.n_off)
        self.hi = np.array([BOUNDS[k][1] for k in keys] + [OFFSET_BOUND] * self.n_off)
        self.scale = np.array([XSCALE[k] for k in self.keys])

    def window(self, p):
        return np.r_[self.Qn_fixed, p[:3]] if self.Qn_fixed is not None else np.asarray(p[:4], float)

    def offsets(self, p):
        return np.asarray(p[len(self.keys) - self.n_off:], float)

    def residual(self, p, ocp):
        w, off, j, out = self.window(p), self.offsets(p), 0, []
        for b in self.bl:
            m = ocv(w, b.q, ocp)
            if b.offset:
                m = m + off[j]
                j += 1
            out.append((m - b.v) / np.sqrt(len(b.q)))
        return np.concatenate(out)

    def x0(self, w, off=None):
        head = list(w[1:4]) if self.Qn_fixed is not None else list(w[:4])
        tail = list(off)[:self.n_off] if off is not None and len(off) >= self.n_off else [0.0] * self.n_off
        return np.clip(np.asarray(head + tail, float), self.lo, self.hi)

    def solve(self, p0, ocp):
        r = least_squares(self.residual, np.clip(p0, self.lo, self.hi), args=(ocp,), bounds=(self.lo, self.hi),
                          x_scale=self.scale, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        return r.x, rmse_mV(r.fun, len(self.bl))


def block_rmse(bl, w, ocp, off=()):
    out, j = {}, 0
    for b in bl:
        m = ocv(w, b.q, ocp)
        if b.offset:
            m = m + off[j]
            j += 1
        out[b.kind] = float(1e3 * np.sqrt(np.mean((m - b.v) ** 2)))
    return out


def random_window(rng, n):
    return np.column_stack([rng.uniform(*BOUNDS[k], n) for k in WINDOW])


# ------------------------------------------------------------------ uniqueness
def multistart(bl, ocp, rng, n_starts):
    """Free-Q_n fits from random starts -> (best p, best rmse, distinct minima [(Q_n, x100, rmse, hits)])."""
    cf = CellFit(bl)
    fits = [cf.solve(cf.x0(w), ocp) for w in random_window(rng, n_starts)]
    P = np.array([f[0] for f in fits])
    S = np.array([f[1] for f in fits])
    order, keep = np.argsort(S), []
    for i in order:
        if all(abs(P[i, 0] - P[j, 0]) > 0.05 for j in keep):
            keep.append(i)
    mins = [(float(P[i, 0]), float(P[i, 2]), float(S[i]), int((np.abs(P[:, 0] - P[i, 0]) <= 0.05).sum()))
            for i in keep]
    return P[order[0]], float(S[order[0]]), mins


def profile(bl, ocp, grid, best, rng, restarts=4):
    """RMSE with Q_n fixed on `grid`, warm-started from the free optimum (x100 rescaled so that Q_n x100
    is kept) plus a few random restarts per node. -> [(Q_n, rmse)]."""
    out = []
    for Qn in grid:
        cf = CellFit(bl, Qn_fixed=float(Qn))
        w0 = np.array(best[:4], float)
        w0[2] = best[0] * best[2] / Qn
        starts = [cf.x0(w0, best[4:])] + [cf.x0(w) for w in random_window(rng, restarts)]
        out.append((float(Qn), min(cf.solve(p, ocp)[1] for p in starts)))
    return out


def uniqueness(best, best_rmse, mins, prof, Qn_adopted):
    g = np.array(prof)
    near = g[g[:, 1] <= best_rmse + 0.5, 0]
    lo, hi = (float(near.min()), float(near.max())) if len(near) else (np.nan, np.nan)
    return dict(free_Q_n=float(best[0]), free_x100=float(best[2]), free_rmse_mV=best_rmse,
                Qn_valley_lo=lo, Qn_valley_hi=hi, Qn_valley_width=hi - lo,
                n_minima_1mV=int(sum(m[2] <= best_rmse + 1.0 for m in mins)),
                competing_Qn=[m[0] for m in mins if m[2] <= best_rmse + 1.0],
                adopted_in_valley=bool(lo - 1e-9 <= Qn_adopted <= hi + 1e-9) if np.isfinite(lo) else None)


# ------------------------------------------------------------------ the stage
def Qn_value(spec, pm):
    """The adopted Q_n: None (free), the literature geometric value, or a number [A.h]."""
    if spec in (None, 'free'):
        return None
    if spec == 'literature':
        return float(physics.truth_groups(pm)['Q_n'])
    return float(spec)


def fit_cells(ds, cells, wcfg, ocp: OCPSet, seed=0):
    """Window of every cell + its uniqueness report -> (windows {cell: {...}}, profiles {cell: [(Q_n, rmse)]})."""
    rng = np.random.default_rng(seed)
    kinds, checks = list(wcfg['fit_on']), list(wcfg.get('check_on', []))
    v_range = wcfg.get('ocv_V_range')
    Qn_fix = Qn_value(wcfg.get('Q_n', 'free'), ds.pm)
    lo, hi, n = wcfg.get('profile', [4.0, 8.0, 81])
    grid = np.linspace(lo, hi, int(n))
    windows, profiles = {}, {}
    for cell in cells:
        bl = blocks(ds, cell, kinds, v_range)
        best, best_rmse, mins = multistart(bl, ocp, rng, int(wcfg.get('starts', 200)))
        prof = profile(bl, ocp, grid, best, rng)
        cf = CellFit(bl, Qn_fixed=Qn_fix)
        w0 = np.array(best[:4], float)
        if Qn_fix is not None:
            w0[2] = best[0] * best[2] / Qn_fix
        p, rm = cf.solve(cf.x0(w0, best[4:]), ocp)
        w = cf.window(p)
        res = dict(zip(WINDOW, map(float, w)), rmse_mV=rm, Q_n_mode=str(wcfg.get('Q_n', 'free')),
                   li_n_Ah=float(w[0] * w[2]), offsets_mV=[1e3 * float(o) for o in cf.offsets(p)],
                   rmse_blocks_mV=block_rmse(bl, w, ocp, cf.offsets(p)), n_points={b.kind: len(b.q) for b in bl},
                   **uniqueness(best, best_rmse, mins, prof, float(w[0])))
        if checks:
            res['check_rmse_mV'] = block_rmse(blocks(ds, cell, checks, v_range), w, ocp)
        windows[cell], profiles[cell] = res, prof
        log.info('window %s: Q_n %.3f Q_p %.3f x100 %.4f y100 %.4f | %.2f mV %s | free Q_n %.3f, valley %.2f-%.2f, '
                 '%d minima within 1 mV%s', cell, *w, rm, res['rmse_blocks_mV'], res['free_Q_n'],
                 res['Qn_valley_lo'], res['Qn_valley_hi'], res['n_minima_1mV'],
                 '' if res['adopted_in_valley'] in (True, None) else '  ** adopted Q_n OUTSIDE the valley **')
    return windows, profiles


def fit_correction(ds, cells, wcfg, ocp: OCPSet, windows):
    """One OCP correction shared by `cells`, fitted jointly with their windows (warm-started from them)
    -> (corrected OCPSet, refitted windows, report)."""
    cc = wcfg['ocp_correction']
    el = cc.get('electrode', 'n')
    if 'knot_positions' in cc:
        knots = np.asarray(cc['knot_positions'], float)
    else:                                  # equally spaced over the stoichiometry range the electrode uses
        knots = np.linspace(*cc.get('range', [0.0, 1.0]), int(cc.get('knots', 33)))
    K = len(knots)
    bound, lam = 1e-3 * float(cc.get('bound_mV', 15.0)), float(cc.get('smooth', 1.0))
    Qn_fix = Qn_value(wcfg.get('Q_n', 'free'), ds.pm)
    kinds, v_range = list(wcfg['fit_on']), wcfg.get('ocv_V_range')
    fits = [(c, CellFit(blocks(ds, c, kinds, v_range), Qn_fixed=Qn_fix)) for c in cells]
    starts = np.cumsum([0] + [len(cf.keys) for _, cf in fits])

    def split(z):
        return [z[a:b] for a, b in zip(starts[:-1], starts[1:])], z[starts[-1]:]

    def res(z):
        ps, dU = split(z)
        o = ocp.corrected(el, knots, dU)
        r = [cf.residual(p, o) / np.sqrt(len(fits)) for p, (_, cf) in zip(ps, fits)]
        return np.concatenate(r + [np.sqrt(lam) * np.diff(dU, 2)])

    p0 = [cf.x0([windows[c][k] for k in WINDOW], [1e-3 * o for o in windows[c]['offsets_mV']]) for c, cf in fits]
    z0 = np.concatenate(p0 + [np.zeros(K)])
    lo = np.concatenate([cf.lo for _, cf in fits] + [np.full(K, -bound)])
    hi = np.concatenate([cf.hi for _, cf in fits] + [np.full(K, bound)])
    xs = np.concatenate([cf.scale for _, cf in fits] + [np.full(K, 1e-3)])
    r = least_squares(res, z0, bounds=(lo, hi), x_scale=xs, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    ps, dU = split(r.x)
    o = ocp.corrected(el, knots, dU)
    out = {}
    for (c, cf), p in zip(fits, ps):
        w = cf.window(p)
        new = dict(windows[c])
        new.update(dict(zip(WINDOW, map(float, w))), li_n_Ah=float(w[0] * w[2]),
                   offsets_mV=[1e3 * float(x) for x in cf.offsets(p)],
                   rmse_mV=rmse_mV(cf.residual(p, o), len(cf.bl)), rmse_uncorrected_mV=windows[c]['rmse_mV'],
                   rmse_blocks_mV=block_rmse(cf.bl, w, o, cf.offsets(p)))
        if wcfg.get('check_on'):
            new['check_rmse_mV'] = block_rmse(blocks(ds, c, wcfg['check_on'], v_range), w, o)
        out[c] = new
        log.info('window %s with OCP correction: %.2f mV (was %.2f)', c, new['rmse_mV'], new['rmse_uncorrected_mV'])
    report = dict(o.describe()['correction'], bound_mV=1e3 * bound, smooth=lam, fitted_on=list(cells),
                  n_knots_at_bound=int(np.sum(np.abs(dU) >= 0.999 * bound)))
    return o, out, report
