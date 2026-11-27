"""
screening_model.py
==================
Helper of screening_local.py (collinearity, correlation). Standalone: step 2 on the model alone, which candidate groups the calibration tests can
see, and which of them the tests can tell apart, before any fit. No measured voltage and no fitted value enters:
the tests give only the current, the temperature and the scored samples, step 1 gives the electrode windows.

    python scripts/screening_model.py [fitB] [--r 40] [--levels 6] [--h 0.01]

1. Global screening (Morris elementary effects, scripts/screening.py): r trajectories on a p-level grid of the
   candidates' box (log scale where lb > 0). Effect of a group on a test: the median over the trajectories of the
   RMS voltage change of its move [mV]; its screening effect is the largest median over the tests.
   The simulation keeps the measured current past the cutoffs, where a cycler would stop, so a move is scored only
   on the samples that both its candidates run within --vclip (at least 20 % of the test, else no effect).
2. Local sensitivities and correlations at r points around the literature set of the cell (st.pm converted to
   groups, its reference activation energies, f_De = 1; R_c and gamma0, zero or absent in the set, at the middle of
   their box), each group times a log-uniform factor in [1/--factor, --factor], on the samples that the point and all
   its steps run within --vclip (a point with less than 20 % of a test inside is left out):
   forward differences of step --dlnp in ln p give the sensitivity matrix S of each test (columns per unit ln p,
   or, for the groups with lb = 0, per unit relative change of the middle of their box). The tests weigh equally (each normalised by its number
   of scored samples), as in the cost. At each point: S_j (RMS mV per unit ln p), the correlations of the groups
   (from the inverse of S^T S) and the collinearity index of Brun et al. (S with unit-norm columns), for all the
   candidates and for the fitted ones. Reported as median and 10-90 % over the points.

Candidates: the study's fitted groups plus the stack groups that its fit keeps fixed (L_e, l_s, eps_s, eps_p, b),
boxed by the study's bounds and screening.TYPICAL (b 1.5-2.5). Every other group keeps the study's fixed value.
Writes runs/screening/<study>_model/screening_model.json and .log.
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

log = logging.getLogger('screening_model')
EXTRA = {'L_e': TYPICAL['L_e'], 'l_s': TYPICAL['l_s'], 'eps_s': TYPICAL['eps_s'], 'eps_p': TYPICAL['eps_p'],
         'brugg': (1.5, 2.5)}


def collinearity(G):
    """Brun et al.: 1 / sqrt(lambda_min) of the Gram matrix of the unit-norm sensitivity columns."""
    d = np.sqrt(np.diag(G))
    Gn = G / np.outer(d, d)
    return float(1.0 / np.sqrt(max(np.linalg.eigvalsh(Gn)[0], 1e-300)))


def correlation(G):
    C = np.linalg.pinv(G)
    s = np.sqrt(np.abs(np.diag(C)))
    return C / np.outer(s, s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('study', nargs='?', default='fitB')
    ap.add_argument('--r', type=int, default=40)
    ap.add_argument('--levels', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--dlnp', type=float, default=0.05,
                    help='forward-difference step in ln p (relative change of the middle of the box for lb = 0)')
    ap.add_argument('--factor', type=float, default=2.0,
                    help='local points: the literature set of the cell times a log-uniform factor in [1/f, f]')
    ap.add_argument('--vclip', type=float, nargs=2, default=[2.5, 4.4],
                    help='Morris: the simulated voltage is clipped to this range, as a cycler stops at its cutoffs [V]')
    ap.add_argument('--vrange', type=float, nargs=2, default=[2.0, 4.6],
                    help='a local point whose voltage leaves this range on a scored sample has collapsed [V]')
    a = ap.parse_args()
    out_dir = os.path.join(ROOT, 'runs', 'screening', a.study + '_model')
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(out_dir, 'screening_model.log'))])
    st = S.load(os.path.join(ROOT, 'configs', a.study + '.json'))
    d = st.cfg['dynamics']
    em = st.em()
    fitted = list(d['names'])
    names = fitted + [n for n in EXTRA if n not in fitted]
    lb = list(d['lb']) + [EXTRA[n][0] for n in names[len(fitted):]]
    ub = list(d['ub']) + [EXTRA[n][1] for n in names[len(fitted):]]
    box = Box(lb, ub)
    D = len(names)
    fixed = {k: v for k, v in {**st.thermal(d.get('thermal', 'reference')), **st.fixed_groups(d.get('fixed'))}.items()
             if k not in names}
    tests = ident.prepare_tests(st.ds, st.expand('dynamics', d['tests'], 'dynamics'), st.windows(), em.ocp)
    rng = np.random.default_rng(a.seed)
    U, order = trajectories(a.r, D, a.levels, rng)                     # (r, D + 1, D)
    # the nominal point: the literature set of the cell (st.pm, converted to groups), its reference activation
    # energies, f_De = 1; the groups the set leaves at zero or does not have (R_c, gamma0) at the middle of their box
    lit = {**st.literature(), **st.thermal('reference'), 'De_f': 1.0}
    nom = np.array([lit.get(n) if (lit.get(n) is not None and lb_ > 0) else None for n, lb_ in zip(names, lb)], object)
    u_nom = np.array([float(box.unit(np.array([[float(v) if v is not None else 1.0] * D]))[0, i]) if v is not None
                      else 0.5 for i, v in enumerate(nom)])
    u_nom = np.clip(u_nom, 0.0, 1.0)
    span = np.where(box.log, np.log(a.factor) / (box.hi - box.lo), 0.25)
    mid = 0.5 * (box.lo + box.hi)
    dlnp_du = np.where(box.log, box.hi - box.lo, (box.hi - box.lo) / np.maximum(np.abs(mid), 1e-30))
    hu = a.dlnp / dlnp_du                                             # (D,) the step in the unit cube
    base = np.clip(u_nom + span * rng.uniform(-1.0, 1.0, (a.r, D)), 0.0, 1.0)    # (r, D): the local points
    sgn = np.where(base + hu <= 1.0, 1.0, -1.0)                       # step inward at the upper face
    L = np.repeat(base[:, None, :], D + 1, 1)                          # (r, D + 1, D): the point, then D steps
    for j in range(D):
        L[:, j + 1, j] += sgn[:, j] * hu[j]
    Xall = box.phys(np.concatenate([U.reshape(-1, D), L.reshape(-1, D)]))
    groups = {**{k: np.full(len(Xall), v) for k, v in fixed.items()}, **{n: Xall[:, i] for i, n in enumerate(names)}}
    nM = a.r * (D + 1)
    log.info('model-only screening of %s: %d candidates %s | r = %d, %d levels, step %g in ln p | %d simulations x %d tests',
             a.study, D, names, a.r, a.levels, a.dlnp, len(Xall), len(tests))
    t0 = time.time()
    G = np.zeros((a.r, D, D))
    ok = np.ones(a.r, bool)
    morris, collapsed_M, collapsed_L = {}, {}, {}
    for t in tests:
        V = simulate(em, t, groups)
        m = t['mask']
        bad = ~np.isfinite(V[:, m]).all(1) | (np.nan_to_num(V[:, m], nan=0.0) < a.vrange[0]).any(1)             | (np.nan_to_num(V[:, m], nan=9.0) > a.vrange[1]).any(1)                     # collapsed candidates
        Vm = V[:, m].astype(np.float64)
        inside = np.isfinite(Vm) & (Vm >= a.vclip[0]) & (Vm <= a.vclip[1])          # (P, n) within the cutoffs
        VM = Vm[:nM].reshape(a.r, D + 1, -1)
        IM = inside[:nM].reshape(a.r, D + 1, -1)
        bM = bad[:nM].reshape(a.r, D + 1)
        both = IM[:, 1:] & IM[:, :-1]                                 # samples both candidates of a move can run
        dV = np.where(both, 1e3 * np.diff(np.nan_to_num(VM), axis=1), 0.0)
        cnt = both.sum(2)
        rms = np.where(cnt >= max(5, 0.2 * m.sum()), np.sqrt(dV ** 2).sum(2) * 0 + np.sqrt((dV ** 2).sum(2) / np.maximum(cnt, 1)), np.nan)
        collapsed_M[t['label']] = float(bM.mean())
        E = np.full((a.r, D), np.nan)
        for k in range(a.r):
            E[k, order[k]] = rms[k]
        E = np.where(np.isfinite(E), E, np.nan)
        morris[t['label']] = np.nanmedian(E, 0)
        collapsed_L[t['label']] = float(bad[nM:].reshape(a.r, D + 1)[:, 0].mean())
        VL = np.nan_to_num(Vm[nM:]).reshape(a.r, D + 1, -1)
        IL = inside[nM:].reshape(a.r, D + 1, -1).all(1)               # (r, n): the point and all its steps inside
        J = 1e3 * (VL[:, 1:] - VL[:, :1]) / (sgn[:, :, None] * a.dlnp)   # (r, D, n) mV per unit ln p
        J = np.where(IL[:, None, :], J, 0.0)
        good = np.isfinite(J).all(axis=(1, 2)) & (IL.sum(1) >= 0.2 * m.sum())
        ok &= good
        J = np.where(np.isfinite(J), J, 0.0)
        G += np.einsum('rin,rjn->rij', J, J) / np.maximum(IL.sum(1), 1)[:, None, None]
        log.info('%-22s %5.1f C | past the cutoffs: Morris %.0f %%, local %.0f %% | Morris median effect [mV]: %s | %.0f s',
                 t['label'], float(np.mean(t['T']) - 273.15), 100 * collapsed_M[t['label']], 100 * collapsed_L[t['label']],
                 '  '.join('%s %.1f' % (names[i], morris[t['label']][i]) for i in np.argsort(-np.nan_to_num(morris[t['label']]))[:6]),
                 time.time() - t0)
    G /= len(tests)
    best = np.nanmax(np.array(list(morris.values())), 0)
    pts = np.flatnonzero(ok)
    Sj = np.sqrt(np.array([np.diag(G[k]) for k in pts]))              # (P, D) RMS mV per unit ln p
    R = np.array([correlation(G[k]) for k in pts])
    fi = [names.index(n) for n in fitted]
    ci_all = np.array([collinearity(G[k]) for k in pts])
    ci_fit = np.array([collinearity(G[k][np.ix_(fi, fi)]) for k in pts])
    q = lambda x: [float(np.percentile(x, 10)), float(np.median(x)), float(np.percentile(x, 90))]
    res = dict(study=a.study, names=names, fitted=fitted, lb=lb, ub=ub, r=a.r, levels=a.levels, dlnp=a.dlnp, seed=a.seed,
               points_ok=int(len(pts)), factor=a.factor, vclip=a.vclip, vrange=a.vrange, collapsed_fraction_morris=collapsed_M, collapsed_fraction_local=collapsed_L,
               nominal=dict(zip(names, box.phys(u_nom[None, :])[0].tolist())), morris_median_mV={k: dict(zip(names, v.tolist())) for k, v in morris.items()},
               morris_max_median_mV=dict(zip(names, best.tolist())),
               S_local_mV={n: q(Sj[:, i]) for i, n in enumerate(names)},
               corr_median=np.median(R, 0).tolist(), abscorr_median=np.median(np.abs(R), 0).tolist(),
               abscorr_p90=np.percentile(np.abs(R), 90, axis=0).tolist(),
               collinearity_all=q(ci_all), collinearity_fitted=q(ci_fit))
    json.dump(res, open(os.path.join(out_dir, 'screening_model.json'), 'w'), indent=1)
    log.info('%d of %d local points finite', len(pts), a.r)
    log.info('%-10s %10s | %24s | most correlated (median |rho|, sign of the median rho)', 'group', 'Morris', 'local S [mV] 10/50/90 %')
    Am = np.median(np.abs(R), 0)
    Rm = np.median(R, 0)
    for i in np.argsort(-best):
        j = [k for k in np.argsort(-Am[i]) if k != i][:3]
        log.info('%-10s %8.2f mV | %6.1f %6.1f %6.1f | %s', names[i], best[i], *np.percentile(Sj[:, i], [10, 50, 90]),
                 ', '.join('%s %s%.2f' % (names[k], '+' if Rm[i, k] >= 0 else '-', Am[i, k]) for k in j))
    log.info('collinearity index, all %d candidates: %.1f / %.1f / %.1f (10/50/90 %%)', D, *q(ci_all))
    log.info('collinearity index, the %d fitted: %.1f / %.1f / %.1f (10/50/90 %%)', len(fi), *q(ci_fit))
    log.info('done in %.2f h', (time.time() - t0) / 3600)


if __name__ == '__main__':
    main()
