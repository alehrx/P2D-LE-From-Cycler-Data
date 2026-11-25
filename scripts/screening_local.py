"""
screening_local.py
==================
Step 0 of the identification around a pre-calibrated point: which candidate groups the calibration
tests can see within the uncertainty that remains after a short pre-calibration at 25 C, and which of them the
tests can tell apart. A box drawn over the whole literature range makes every group "sensitive" (a Boeing 747 on
a bridge tests the bridge, not whether the mass of a car matters); the box here is the one the data leave open.

    python scripts/screening_local.py <design study> <pre-calibration study> [--r 40] [--k 2.5] [--wmin 0.2]

1. Centre: the groups of the pre-calibration (its dynamics.json); the groups it did not fit (the activation
   energies, gamma0) take the parameter set's reference values (study.thermal) and the middle of their bounds.
2. Fisher at the centre on the tests of the design study (all its calibration cells and temperatures):
   S_j = RMS dV/d ln p_j [mV] (forward differences of --dlnp), each test weighing equally; correlations from
   (S^T S)^-1; collinearity index of Brun et al.; sigma_j = relative standard deviation with the noise sigma of the
   pre-calibration (its cost).
3. Local box: ln p_j within +-max(k sigma_j, ln(1 + wmin)) of the centre (linear groups with lb = 0: +-max(k sigma_j,
   wmin) times the middle of their bounds), clipped to the design study's bounds.
4. Morris (r trajectories, p levels) in that box: median RMS voltage change of each group's move, per test.
5. Fisher at r points drawn uniformly in the box: distribution of S_j, of |rho| and of the collinearity index, for
   all candidates and for the design study's fitted groups.
Candidates: the design study's fitted groups plus the stack groups (L_e, l_s, eps_s, eps_p, b).
Writes runs/screening/<design>_local/screening_local.json and .log.
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from srcs import identification as ident, study as S        # noqa: E402
from sensitivity import Box                                   # noqa: E402
from screening import TYPICAL, simulate, trajectories       # noqa: E402
from screening_model import collinearity, correlation       # noqa: E402

log = logging.getLogger('screening_local')
EXTRA = {'L_e': TYPICAL['L_e'], 'l_s': TYPICAL['l_s'], 'eps_s': TYPICAL['eps_s'], 'eps_p': TYPICAL['eps_p'],
         'brugg': (1.5, 2.5)}


def gram(em, tests, groups, P, D, dlnp, sgn):
    """Gram matrices of P points whose candidates are laid out as (point, then its D steps), S per unit ln p [mV]:
    (P, D, D) with the tests equally weighted (each per sample, then averaged), and (P, D, D) summed over all the
    scored samples (the Fisher matrix times sigma^2)."""
    G = np.zeros((P, D, D))
    Graw = np.zeros((P, D, D))
    ok = np.ones(P, bool)
    for t in tests:
        V = simulate(em, t, groups)
        m = t['mask']
        VL = V[:, m].astype(np.float64).reshape(P, D + 1, -1)
        J = 1e3 * (VL[:, 1:] - VL[:, :1]) / (sgn[:, :, None] * dlnp)
        ok &= np.isfinite(J).all(axis=(1, 2))
        J = np.where(np.isfinite(J), J, 0.0)
        g = np.einsum('pin,pjn->pij', J, J)
        G += g / m.sum()
        Graw += g
    return G / len(tests), Graw, ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('design')
    ap.add_argument('pre')
    ap.add_argument('--r', type=int, default=40)
    ap.add_argument('--levels', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dlnp', type=float, default=0.05)
    ap.add_argument('--k', type=float, default=2.5, help='box half-width in standard deviations')
    ap.add_argument('--wmin', type=float, default=0.2, help='minimum box half-width (relative)')
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, 'runs', 'screening', a.design + '_local')
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(out_dir, 'screening_local.log'))])
    st = S.load(os.path.join(ROOT, 'configs', a.design + '.json'))
    d = st.cfg['dynamics']
    em = st.em()
    fitted = list(d['names'])
    names = fitted + [n for n in EXTRA if n not in fitted]
    lb = np.array(list(d['lb']) + [EXTRA[n][0] for n in names[len(fitted):]], float)
    ub = np.array(list(d['ub']) + [EXTRA[n][1] for n in names[len(fitted):]], float)
    D = len(names)
    fixed = {k: v for k, v in {**st.thermal(d.get('thermal', 'reference')), **st.fixed_groups(d.get('fixed'))}.items()
             if k not in names}
    pre = json.load(open(os.path.join(ROOT, 'runs', 'studies', a.pre, 'dynamics.json')))
    ref = {**st.thermal('reference'), **pre['groups']}
    centre = np.array([ref[n] if n in ref and n != 'dUdT0' else 0.5 * (lo + hi) for n, lo, hi in zip(names, lb, ub)], float)
    centre = np.clip(centre, lb, ub)
    sigma_noise = float(pre['cost_mV'])
    tests = ident.prepare_tests(st.ds, st.expand('dynamics', d['tests'], 'dynamics'), st.windows(), em.ocp)
    logL = lb > 0
    mid = 0.5 * (lb + ub)
    # a linear group with symmetric bounds (dUdT0 in +-a) has mid 0: scale it by the half-range instead
    mid = np.where(np.abs(mid) > 1e-12 * np.maximum(np.abs(lb), np.abs(ub)), mid, 0.5 * (ub - lb))
    # u: ln p for the log groups, p / mid for the linear ones
    to_u = lambda X: np.where(logL, np.log(np.where(logL, X, 1.0)), X / mid)
    to_p = lambda U: np.where(logL, np.exp(U), U * mid)
    u_lo, u_hi = to_u(lb), to_u(ub)

    def candidates(Ub):
        """(P, D) points in u -> (P (D + 1), D) physical candidates (each point, then its D forward steps)."""
        sgn = np.where(Ub + a.dlnp <= u_hi, 1.0, -1.0)
        L = np.repeat(Ub[:, None, :], D + 1, 1)
        for j in range(D):
            L[:, j + 1, j] += sgn[:, j] * a.dlnp
        return to_p(L.reshape(-1, D)), sgn

    def as_groups(X):
        return {**{k: np.full(len(X), v) for k, v in fixed.items()}, **{n: X[:, i] for i, n in enumerate(names)}}

    t0 = time.time()
    log.info('local screening of %s around %s (cost %.2f mV): %d candidates, %d tests %s', a.design, a.pre, sigma_noise,
             D, len(tests), [t['label'] for t in tests])
    # 2. Fisher at the centre
    u0 = to_u(centre)[None, :]
    X0, s0 = candidates(u0)
    G0, Graw0, ok0 = gram(em, tests, as_groups(X0), 1, D, a.dlnp, s0)
    G0, Graw0 = G0[0], Graw0[0]
    S0 = np.sqrt(np.diag(G0))
    R0 = correlation(G0)
    sig = sigma_noise * np.sqrt(np.abs(np.diag(np.linalg.pinv(Graw0))))   # sd of ln p: F = sum_samples J^T J / sigma^2
    fi = [names.index(n) for n in fitted]
    log.info('centre: collinearity all %.1f, fitted %.1f', collinearity(G0), collinearity(G0[np.ix_(fi, fi)]))
    # 3. local box
    half = np.maximum(a.k * sig, np.where(logL, np.log(1.0 + a.wmin), a.wmin))      # linear groups: wmin x the middle
    blo = np.maximum(u0[0] - half, u_lo)
    bhi = np.minimum(u0[0] + half, u_hi)
    box_phys = np.array([to_p(blo), to_p(bhi)])
    for i, n in enumerate(names):
        log.info('%-10s centre %10.4g | S %7.2f mV | sigma(ln p) %.3f | box %10.4g .. %-10.4g | most correlated %s', n,
                 centre[i], S0[i], sig[i], box_phys[0, i], box_phys[1, i],
                 ', '.join('%s %+.2f' % (names[k], R0[i, k]) for k in [k for k in np.argsort(-np.abs(R0[i])) if k != i][:3]))
    # 4. Morris in the box (on the unit cube of the box)
    rng = np.random.default_rng(a.seed)
    U, order = trajectories(a.r, D, a.levels, rng)
    XM = to_p(blo + U.reshape(-1, D) * (bhi - blo))
    # 5. Fisher at r points of the box
    UL = blo + rng.uniform(0.0, 1.0, (a.r, D)) * (bhi - blo)
    XL, sL = candidates(UL)
    nM = len(XM)
    groups = as_groups(np.concatenate([XM, XL]))
    morris = {}
    G = np.zeros((a.r, D, D))
    ok = np.ones(a.r, bool)
    for t in tests:
        V = simulate(em, t, groups)
        m = t['mask']
        VM = V[:nM, m].astype(np.float64).reshape(a.r, D + 1, -1)
        rms = np.sqrt(np.mean((1e3 * np.diff(VM, axis=1)) ** 2, axis=2))
        E = np.full((a.r, D), np.nan)
        for k in range(a.r):
            E[k, order[k]] = rms[k]
        morris[t['label']] = np.nanmedian(np.where(np.isfinite(E), E, np.nan), 0)
        VL = V[nM:, m].astype(np.float64).reshape(a.r, D + 1, -1)
        J = 1e3 * (VL[:, 1:] - VL[:, :1]) / (sL[:, :, None] * a.dlnp)
        ok &= np.isfinite(J).all(axis=(1, 2))
        G += np.einsum('pin,pjn->pij', np.where(np.isfinite(J), J, 0.0), np.where(np.isfinite(J), J, 0.0)) / m.sum()
        log.info('%-22s %5.1f C | Morris median effect in the box [mV]: %s | %.0f s', t['label'], float(np.mean(t['T']) - 273.15),
                 '  '.join('%s %.1f' % (names[i], morris[t['label']][i]) for i in np.argsort(-np.nan_to_num(morris[t['label']]))[:6]),
                 time.time() - t0)
    G /= len(tests)
    best = np.nanmax(np.array(list(morris.values())), 0)
    pts = np.flatnonzero(ok)
    Sj = np.sqrt(np.array([np.diag(G[k]) for k in pts]))
    R = np.array([correlation(G[k]) for k in pts])
    ci_all = np.array([collinearity(G[k]) for k in pts])
    ci_fit = np.array([collinearity(G[k][np.ix_(fi, fi)]) for k in pts])
    q = lambda x: [float(np.percentile(x, 10)), float(np.median(x)), float(np.percentile(x, 90))]
    Am, Rm = np.median(np.abs(R), 0), np.median(R, 0)
    res = dict(design=a.design, pre=a.pre, names=names, fitted=fitted, centre=dict(zip(names, centre.tolist())),
               sigma_noise_mV=sigma_noise, k=a.k, wmin=a.wmin, dlnp=a.dlnp, r=a.r, levels=a.levels, seed=a.seed,
               box=dict(zip(names, box_phys.T.tolist())),
               centre_S_mV=dict(zip(names, S0.tolist())), centre_corr=R0.tolist(), centre_sigma_lnp=dict(zip(names, sig.tolist())),
               centre_collinearity=dict(all=collinearity(G0), fitted=collinearity(G0[np.ix_(fi, fi)])),
               morris_median_mV={k: dict(zip(names, v.tolist())) for k, v in morris.items()},
               morris_max_median_mV=dict(zip(names, best.tolist())), points_ok=int(len(pts)),
               S_local_mV={n: q(Sj[:, i]) for i, n in enumerate(names)}, abscorr_median=Am.tolist(), corr_median=Rm.tolist(),
               collinearity_all=q(ci_all), collinearity_fitted=q(ci_fit))
    json.dump(res, open(os.path.join(out_dir, 'screening_local.json'), 'w'), indent=1)
    log.info('%d of %d local points finite', len(pts), a.r)
    log.info('%-10s %10s | %24s | most correlated in the box (median |rho|, sign)', 'group', 'Morris', 'local S [mV] 10/50/90 %')
    for i in np.argsort(-best):
        j = [k for k in np.argsort(-Am[i]) if k != i][:3]
        log.info('%-10s %8.2f mV | %6.1f %6.1f %6.1f | %s', names[i], best[i], *np.percentile(Sj[:, i], [10, 50, 90]),
                 ', '.join('%s %s%.2f' % (names[k], '+' if Rm[i, k] >= 0 else '-', Am[i, k]) for k in j))
    log.info('collinearity index in the box, all %d: %.1f / %.1f / %.1f (10/50/90 %%)', D, *q(ci_all))
    log.info('collinearity index in the box, the %d fitted: %.1f / %.1f / %.1f (10/50/90 %%)', len(fi), *q(ci_fit))
    log.info('done in %.2f h', (time.time() - t0) / 3600)


if __name__ == '__main__':
    main()
