"""
sensitivity.py
==============
Helper of screening_local.py (Box: the unit cube of the optimiser). Standalone: sensitivity and identifiability
of a fitted study's groups on its own dynamics tests (not used for the results of the paper):

    python scripts/sensitivity.py fitB                                       # local + Sobol + profiles
    python scripts/sensitivity.py fitB --names GROUPS --skip profiles

  local     sensitivities dV/dln p at the optimum (central differences, one batch): Fisher lower bounds,
            correlations, collinearity index, on all tests and on each test kind (srcs/identifiability.analyse)
  sobol     first-order and total Sobol indices of log10 J over the study's search box (Saltelli design on a
            scrambled Sobol sequence in the unit cube of the optimiser, log scale for positive bounds; Jansen
            estimators, bootstrap 95 % intervals), for J on all tests and on each test kind
  profiles  each group fixed on a grid over its box, the others re-fitted by CMA-ES warm-started at the
            optimum; all grid points of a group run as threads whose batches are merged into one GPU call

--names: the parameters analysed (default: every fitted one; GROUPS = the 14 groups of BatchMS). Every
other parameter stays at the optimum of dynamics.json. Test kinds: HPPC, VAL at 25 C, VAL cold, VAL hot.
Writes runs/studies/<study>/sensitivity.json (+ sobol_samples.npz, profiles.csv).
"""
import argparse
import json
import logging
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
from scipy.stats import qmc

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
from srcs import identifiability, identification as ident, study as S        # noqa: E402
from srcs.ms_model import BatchMS                                              # noqa: E402
from srcs.optimizers import optimize                                           # noqa: E402

log = logging.getLogger('sensitivity')


def kind(t):
    """HPPC | VAL 25 C | VAL cold | VAL hot, from the test's label and mean temperature."""
    if 'HPPC' in t['label']:
        return 'HPPC'
    T = float(np.mean(t['T'])) - 273.15
    return 'VAL cold' if T < 10 else ('VAL hot' if T > 37 else 'VAL 25 C')


class Box:
    """The optimiser's unit cube: log scale where lb > 0, linear otherwise (srcs/optimizers._Counted)."""

    def __init__(self, lb, ub):
        lb, ub = np.asarray(lb, float), np.asarray(ub, float)
        self.log = lb > 0
        self.lo = np.where(self.log, np.log(np.where(self.log, lb, 1.0)), lb)
        self.hi = np.where(self.log, np.log(np.where(self.log, ub, 1.0)), ub)

    def phys(self, U):
        X = self.lo + np.clip(U, 0.0, 1.0) * (self.hi - self.lo)
        return np.where(self.log, np.exp(X), X)

    def unit(self, X):
        X = np.where(self.log, np.log(np.where(self.log, X, 1.0)), X)
        return (X - self.lo) / (self.hi - self.lo)


class PerTest:
    """(P, n_tests) RMSE [mV] of a (P, d) array of the parameters `names`, the rest from `fixed`."""

    def __init__(self, em, tests, names, fixed):
        self.em, self.tests, self.names, self.fixed = em, tests, list(names), dict(fixed)
        self.chunk = {t['label']: ident.auto_chunk(len(t['I']), em.device) for t in tests}

    def __call__(self, X):
        import torch
        X = np.atleast_2d(X)
        out = np.zeros((len(X), len(self.tests)))
        for k, t in enumerate(self.tests):
            Vt = torch.as_tensor(t['V'], dtype=torch.float32, device=self.em.device)[None, :]
            m = torch.as_tensor(t['mask'], device=self.em.device)
            c = self.chunk[t['label']]
            for a in range(0, len(X), c):
                g, q = ident.test_groups(t, {**self.fixed, **{n: X[a:a + c, i] for i, n in enumerate(self.names)}},
                                         self.em.ocp)
                V = self.em.simulate(g, t['I'], t['dt'], T=t['T'], q_start=q)
                out[a:a + c, k] = 1e3 * torch.sqrt(torch.mean((V - Vt)[:, m] ** 2, dim=1)).cpu().numpy()
        return np.where(np.isfinite(out), out, np.inf)


def local(em, tests, names, x, fixed):
    res = {'all': identifiability.analyse(em, tests, names, x, fixed)}
    for k in sorted({kind(t) for t in tests}):
        sub = [t for t in tests if kind(t) == k]
        res[k] = identifiability.analyse(em, sub, names, x, fixed, sigma_V=1e-3 * res['all']['sigma_mV'])
        log.info('local %-9s sensitivity [mV] %s', k, {n: '%.2f' % v for n, v in res[k]['sensitivity_mV'].items()})
    log.info('local all: collinearity %.1f, max |corr| %.2f', res['all']['collinearity_index'],
             res['all']['max_abs_correlation'])
    return res


def sobol(f_tests, kinds, box, d, N, seed, out_npz, n_boot=200):
    """Saltelli design: A, B (N, d) and AB_j = A with column j from B; Jansen estimators on log10 J."""
    rng = np.random.default_rng(seed)
    base = qmc.Sobol(2 * d, scramble=True, seed=seed).random(N)
    A, B = base[:, :d], base[:, d:]
    U = np.vstack([A, B] + [np.where(np.arange(d) == j, B, A) for j in range(d)])
    t0 = time.time()
    F = np.vstack([f_tests(box.phys(U[a:a + 4096])) for a in range(0, len(U), 4096)])     # (N (d + 2), n_tests)
    log.info('sobol: %d evaluations in %.1f min', len(U), (time.time() - t0) / 60)
    np.savez_compressed(out_npz, U=U, F=F)
    groups = {'all': np.ones(len(kinds), bool), **{k: np.array([q == k for q in kinds]) for k in sorted(set(kinds))}}
    res = {}
    for gname, sel in groups.items():
        J = np.log10(np.clip(F[:, sel].mean(1), 1e-3, None))
        J[~np.isfinite(J)] = np.nanmax(J[np.isfinite(J)])
        fA, fB = J[:N], J[N:2 * N]
        fAB = J[2 * N:].reshape(d, N)

        def est(idx):
            a, b, ab = fA[idx], fB[idx], fAB[:, idx]
            var = np.var(np.r_[a, b])
            S1 = np.mean(b * (ab - a), axis=1) / var
            ST = 0.5 * np.mean((a - ab) ** 2, axis=1) / var
            return S1, ST
        S1, ST = est(np.arange(N))
        boot = [est(rng.integers(0, N, N)) for _ in range(n_boot)]
        lo1, hi1 = np.percentile([b[0] for b in boot], [2.5, 97.5], axis=0)
        loT, hiT = np.percentile([b[1] for b in boot], [2.5, 97.5], axis=0)
        res[gname] = dict(S1=S1.tolist(), ST=ST.tolist(), S1_ci=[lo1.tolist(), hi1.tolist()], ST_ci=[loT.tolist(), hiT.tolist()])
    return res


def profiles(cost_full, box, names, x_opt, grid, budget, pop, sigma0, seed):
    """Every group on `grid` points over its box, the others re-fitted (CMA-ES warm-started at the optimum)."""
    d, u_opt, rows = len(names), box.unit(np.asarray(x_opt, float)), []
    for j, n in enumerate(names):
        vals_u = np.linspace(0.0, 1.0, grid)
        lock, waiting, results, active = threading.Condition(), {}, {}, [grid]

        def flush():
            ids = list(waiting)
            f = cost_full(np.vstack([waiting[i] for i in ids]))
            a = 0
            for i in ids:
                results[i] = f[a:a + len(waiting[i])]
                a += len(waiting[i])
            waiting.clear()
            lock.notify_all()

        def request(k, Xr, v):
            X = np.insert(np.atleast_2d(Xr), j, v, axis=1)
            with lock:
                waiting[k] = X
                if len(waiting) == active[0]:
                    flush()
                while k not in results:
                    lock.wait()
                return results.pop(k)

        out = [None] * grid
        keep = [i for i in range(d) if i != j]
        lb_r, ub_r = box.phys(np.zeros(d))[keep], box.phys(np.ones(d))[keep]

        def run(k):
            v = float(box.phys(np.where(np.arange(d) == j, vals_u[k], u_opt))[j])
            try:
                out[k] = (v, optimize('cmaes', lambda Xr: request(k, Xr, v), lb_r, ub_r, budget, seed=seed + k,
                                      pop=pop, sigma0=sigma0, x0=u_opt[keep]))
            finally:
                with lock:
                    active[0] -= 1
                    if active[0] and len(waiting) == active[0]:
                        flush()

        t0 = time.time()
        th = [threading.Thread(target=run, args=(k,), daemon=True) for k in range(grid)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for v, r in out:
            rows.append(dict(name=n, value=v, cost_mV=r['f'], **{'x_' + m: float(r['x'][i]) for i, m in
                                                                 enumerate([names[q] for q in keep])}))
        log.info('profile %-9s %s  (%.1f min)', n, ' '.join('%.3g:%.2f' % (v, r['f']) for v, r in out),
                 (time.time() - t0) / 60)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('study')
    ap.add_argument('--names', nargs='*', default=None, help='parameters analysed; GROUPS = the 14 groups')
    ap.add_argument('--skip', nargs='*', default=[], choices=['local', 'sobol', 'profiles'])
    ap.add_argument('--sobol-n', type=int, default=1024)
    ap.add_argument('--grid', type=int, default=9)
    ap.add_argument('--budget', type=int, default=1000)
    ap.add_argument('--pop', type=int, default=50)
    ap.add_argument('--sigma0', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    st = S.load(os.path.join(ROOT, 'configs', a.study + '.json'))
    os.makedirs(st.path(''), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(st.path('sensitivity.log'))])
    em = st.em()
    dyn = st.load('dynamics.json')
    names = dyn['fitted'] if not a.names else (list(BatchMS.GROUPS) if a.names == ['GROUPS'] else a.names)
    idx = [dyn['fitted'].index(n) for n in names]
    lb, ub = np.array(dyn['lb'])[idx], np.array(dyn['ub'])[idx]
    x = np.array([dyn['groups'][n] for n in names])
    fixed = {k: v for k, v in dyn['groups'].items() if k not in names}
    specs = st.expand('dynamics', st.cfg['dynamics']['tests'], 'dynamics')
    tests = ident.prepare_tests(st.ds, specs, st.windows(), em.ocp)
    kinds = [kind(t) for t in tests]
    log.info('sensitivity %s: %d parameters %s | tests %s', a.study, len(names), names,
             ['%s (%s)' % (t['label'], k) for t, k in zip(tests, kinds)])
    path = st.path('sensitivity.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update(study=a.study, names=names, x=x.tolist(), lb=lb.tolist(), ub=ub.tolist(), tests=[t['label'] for t in tests],
               kinds=kinds)
    f_tests = PerTest(em, tests, names, fixed)
    box = Box(lb, ub)
    if 'local' not in a.skip:
        res['local'] = local(em, tests, names, x, fixed)
        ident.save(path, res)
    if 'sobol' not in a.skip:
        res['sobol'] = dict(N=a.sobol_n, output='log10 J', **sobol(f_tests, kinds, box, len(names), a.sobol_n, a.seed,
                                                                    st.path('sobol_samples.npz')))
        for g, r in res['sobol'].items():
            if isinstance(r, dict):
                log.info('sobol %-9s ST %s', g, {n: '%.2f' % v for n, v in zip(names, r['ST'])})
        ident.save(path, res)
    if 'profiles' not in a.skip:
        df = profiles(lambda X: f_tests(X).mean(1), box, names, x, a.grid, a.budget, a.pop, a.sigma0, a.seed)
        df.to_csv(st.path('profiles.csv'), index=False)
        res['profiles'] = dict(grid=a.grid, budget=a.budget, pop=a.pop, sigma0=a.sigma0, optimum_mV=dyn['cost_mV'])
        ident.save(path, res)
    log.info('done: %s', path)


if __name__ == '__main__':
    main()
