"""
fit_lm.py
=========
Dynamics stage of a study by batched multi-start Levenberg-Marquardt (srcs/optimizers.lm_multistart) on the
least-squares form of its cost (identification.ResidualCost), instead of the stage's global optimiser.
Meant for a smooth model: the finite-volume particles (model "particles": "fv").

    python scripts/fit_lm.py configs/fitB.json [--fresh]
    python scripts/fit_lm.py configs/fitB.json --starts 4 --iters 2      # quick check

The config's "dynamics" block gives the tests (with knee_weight / knee_V for the sample weights), names, lb,
ub and fixed, plus "lm": {starts, h, max_iter, seed}. Writes dynamics.json in the
study folder in the stage's format (groups, fitted, lb, ub, cost_mV = the Cost at the optimum, rmse_per_test
unweighted, identifiability) plus the starts, and lm.log.

After every LM iteration the search state is saved to lm_state.npz in the study folder. A rerun of the same
fit (same groups, bounds, starts, seed, iterations and tests) continues from it instead of starting again, so
that a crash costs at most one iteration; --fresh ignores it. The file is removed when the fit has finished.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
from srcs import identifiability, identification as ident, study as S      # noqa: E402
from srcs.optimizers import lm_multistart                                  # noqa: E402

log = logging.getLogger('fit_lm')
R_GAS = 8.314462618


def start_box(c, d, n, seed, theta_ref_C):
    """The n starts as a scrambled Sobol set in a box around a pre-calibration (lm.center):
    {"study": <pre-calibration>, "width": w}. In the unit cube of the fit (log between the bounds for a positive
    lower bound, linear otherwise) a group the pre-calibration fitted is drawn in [u_c - w/2, u_c + w/2]
    (clipped to [0, 1]), u_c its pre-calibrated value; the others (the temperature terms) over their whole range.
    The pre-calibration is at 25 C: when the fit takes the diffusion times at theta_ref_C, each start's centre is
    theta(25 C) moved to theta_ref_C with that start's own activation energy (Ea_Dn, Ea_Dp)."""
    import json
    from scipy.stats import qmc
    pre = json.load(open(os.path.join(ROOT, 'runs', 'studies', c['study'], 'dynamics.json')))
    g = {k: pre['groups'][k] for k in pre['fitted']}     # only what it fitted: its other groups are defaults
    w = float(c.get('width', 0.2))
    names, lb, ub = d['names'], np.asarray(d['lb'], float), np.asarray(d['ub'], float)
    lg = lb > 0
    to_u = lambda x, j: (np.log(x / lb[j]) / np.log(ub[j] / lb[j])) if lg[j] else (x - lb[j]) / (ub[j] - lb[j])   # noqa: E731
    to_x = lambda u, j: (lb[j] * (ub[j] / lb[j]) ** u) if lg[j] else lb[j] + u * (ub[j] - lb[j])             # noqa: E731
    S = qmc.Sobol(len(names), scramble=True, seed=seed).random(n)
    X = np.empty_like(S)
    order = sorted(range(len(names)), key=lambda j: names[j].startswith('theta_'))   # Ea before theta
    for j in order:
        k = names[j]
        if k not in g:
            X[:, j] = to_x(S[:, j], j)
            continue
        xc = np.full(n, float(g[k]))
        if k.startswith('theta_') and theta_ref_C is not None:
            ea = names.index('Ea_D' + k[-1])
            xc = xc * np.exp(X[:, ea] / R_GAS * (1.0 / (float(theta_ref_C) + 273.15) - 1.0 / 298.15))
        uc = np.clip(to_u(xc, j), 0.0, 1.0)
        X[:, j] = to_x(np.clip(uc + w * (S[:, j] - 0.5), 0.0, 1.0), j)
    log.info('starts: %d Sobol in a box of width %.2f (unit cube) around %s for %s; full range for %s', n, w,
             c['study'], [k for k in names if k in g], [k for k in names if k not in g])
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config')
    ap.add_argument('--starts', type=int)
    ap.add_argument('--iters', type=int)
    ap.add_argument('--fresh', action='store_true', help='ignore a saved search state (lm_state.npz)')
    a = ap.parse_args()
    st = S.load(a.config)
    os.makedirs(st.path(''), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(st.path('lm.log'))])
    d = st.cfg['dynamics']
    o = dict(st.cfg.get('lm', {}))
    starts = a.starts or int(o.get('starts', 32))
    iters = a.iters or int(o.get('max_iter', 40))
    em = st.em()
    specs = st.expand('dynamics', d['tests'], 'dynamics')
    fixed = {**st.thermal(d.get('thermal', 'reference')), **st.fixed_groups(d.get('fixed'))}
    missing = [k for k in em.dynamic if k not in d['names'] and k not in fixed]
    if missing:
        raise SystemExit('dynamics: groups %s are neither fitted nor fixed' % missing)
    tests = ident.prepare_tests(st.ds, specs, st.windows(), em.ocp)
    res_cost = ident.ResidualCost(em, tests, d['names'], fixed)
    cost = ident.Cost(em, tests, d['names'], fixed)
    log.info('fit_lm %s: %d tests %s | %d parameters | %d starts, %d iterations, h %g | knee-weighted samples %d',
             st.name, len(tests), [t['label'] for t in tests], len(d['names']), starts, iters, float(o.get('h', 1e-3)),
             int(sum((t['sample_w'] > 1).sum() for t in tests)))
    t0 = time.time()
    x0 = None
    if o.get('warm_from'):             # {"study": name, "n": k}: that study's k best starts as the first starts
        import json
        prev = json.load(open(os.path.join(ROOT, 'runs', 'studies', o['warm_from']['study'], 'dynamics.json')))
        best = sorted(prev['starts'], key=lambda z: z['sum_r2'])[:int(o['warm_from'].get('n', 8))]
        dflt = o['warm_from'].get('defaults', {})              # parameters that study did not fit
        x0 = np.array([[z['groups'].get(n, dflt.get(n)) for n in d['names']] for z in best], dtype=float)
        log.info('warm starts: the %d best of %s', len(x0), o['warm_from']['study'])
    if o.get('center'):                # Sobol starts in a box around a pre-calibration (start_box)
        x0 = start_box(o['center'], d, starts, int(o.get('seed', 0)), st.cfg.get('model', {}).get('theta_ref_C'))
    sig = repr((list(d['names']), [float(v) for v in d['lb']], [float(v) for v in d['ub']], starts,
                int(o.get('seed', 0)), iters, float(o.get('h', 1e-3)), [t['label'] for t in tests],
                o.get('warm_from'), o.get('prune')) + ((o['select'],) if o.get('select') else ())
           + ((o['center'],) if o.get('center') else ())
           + ((o['respawn'],) if o.get('respawn') else ()))
    state_path = st.path('lm_state.npz')
    resume = None
    if os.path.exists(state_path) and not a.fresh:
        with np.load(state_path, allow_pickle=False) as z:           # closed again: the file is replaced later
            saved = {k: np.array(z[k]) for k in z.files}
        if str(saved.pop('sig')) == sig:
            resume = saved
            log.info('resuming from %s (after iteration %d)', state_path, int(resume['iteration']))
        else:
            log.info('ignoring %s: saved for a different fit', state_path)

    def checkpoint(state):
        """Write the state atomically; a failure (a file held by OneDrive, say) is logged, never fatal."""
        tmp = state_path + '.tmp.npz'
        for k in range(5):
            try:
                np.savez(tmp, sig=np.array(sig), **state)
                os.replace(tmp, state_path)
                return
            except OSError as e:
                err = e
                time.sleep(2.0 * (k + 1))
        log.warning('could not save the search state after iteration %d: %s', int(state['iteration']), err)

    r = lm_multistart(res_cost.residuals, d['lb'], d['ub'], n_starts=starts, seed=int(o.get('seed', 0)),
                      h=float(o.get('h', 1e-3)), max_iter=iters, log=log.info, x0=x0, checkpoint=checkpoint,
                      resume=resume, prune=tuple(tuple(p) for p in o.get('prune', ((4, 16), (8, 8), (12, 4)))),
                      select=o.get('select'), names=d['names'], respawn=o.get('respawn'))
    x = r['x']
    f = float(cost(x[None, :])[0])
    out = dict(groups={**fixed, **dict(zip(d['names'], x))}, fitted=list(d['names']), fixed=fixed, lb=list(d['lb']),
               ub=list(d['ub']), cost_mV=f, sum_r2=r['f'], optimizer='lm_multistart',
               rmse_per_test=cost.rmse_per_test(x), n_eval=r['n_eval'], wall_s=time.time() - t0,
               history=r['history'], tests=[t['label'] for t in tests],
               starts=[dict(groups=dict(zip(d['names'], xs)), sum_r2=fs, iterations=n) for xs, fs, n in r['starts']])
    if o.get('select'):                # the in-search selection (lm.select): the best start's free set at the end
        out['select'] = dict(config=o['select'], free=r['free'], log=r['select_log'])
        log.info('in-search selection: %d of %d free at the end: %s', len(r['free']), len(d['names']), r['free'])
    ident.save(st.path('dynamics.json'), out)          # first without it: a failed analysis never loses the fit
    try:
        out['identifiability'] = identifiability.analyse(em, tests, d['names'], x, fixed)
        idf = out['identifiability']
        log.info('identifiability: collinearity index %.1f, max |corr| %.2f, insensitive %s, rel. std (lower bound) %s',
                 idf['collinearity_index'], idf['max_abs_correlation'], idf['insensitive'],
                 {k: '%.3g' % v for k, v in idf['rel_std_lower_bound'].items()})
        ident.save(st.path('dynamics.json'), out)
    except Exception as e:                              # noqa: BLE001
        log.warning('identifiability analysis failed: %s', e)
    for f_ in (state_path, state_path + '.tmp.npz'):
        if os.path.exists(f_):
            try:
                os.remove(f_)
            except OSError as e:
                log.warning('could not remove %s: %s', f_, e)
    srt = sorted(r['starts'], key=lambda z: z[1])
    log.info('fit_lm: cost %.2f mV (sum r^2 %.4g), %d evaluations in %.1f h | best starts sum r^2 %s | %s', f, r['f'],
             r['n_eval'], (time.time() - t0) / 3600, ['%.4g' % z[1] for z in srt[:6]],
             {k: '%.4g' % v for k, v in zip(d['names'], x)})


if __name__ == '__main__':
    main()
