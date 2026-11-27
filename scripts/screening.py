"""
screening.py
============
Helper of screening_local.py (TYPICAL, simulate, trajectories). Standalone: a global screening of the
parameters over a wide box (Morris elementary effects on the SIMULATED voltage, not on the cost); not used for the
results of the paper, whose screening is the local one (screening_local.py).

    python scripts/screening.py                                   # 26 parameters, r = 40 trajectories
    python scripts/screening.py --r 4 --tag smoke                 # quick check
    python scripts/screening.py --box literature --tag literature # the literature box (LITERATURE below)

Model: the multi-slice SPMe in groups with electrolyte_groups='material' (srcs/ms_model.py): kappa, D_e,
t+ and TDF of the study's electrolyte set, so the electrolyte is R_e and the thickness L_e (tau_e and R_ce
follow); Bruggeman fixed at 1.5. The box is TYPICAL below (typical values, not the optimiser's box), the
windows and OCPs come from --study (its static stage), every other parameter from its dynamics.json.

Design: r Morris trajectories on a p-level grid of the optimiser's unit cube (log scale for positive
bounds, sensitivity.Box): from a random grid point, each parameter in random order moves by
Delta = p / (2 (p - 1)) (down where up would leave the cube). r (d + 1) candidates, simulated on every test.

Effect of parameter i on test t, part s: d_i = RMS over the samples of s of the voltage change [mV] caused by
the move of i (one per trajectory). Reported per test and part: mu* = mean of d_i over the trajectories,
the median (robust to trajectories that end in a collapsed or diverged state) and the standard deviation
(interactions, nonlinearity). Parts: all, discharge (I > 0.02 A), charge (I < -0.02 A), rest, and the
low-voltage end (data V < 3.4 V under discharge).

Tests: C/20, HPPC (1 s, as measured) and the 1C charge of --cell25 (all at 25 C), and the first cycles
(VAL) of --val cells (0.2C, 1.5C, 2C at -5, 25, 45 C).
Writes runs/screening/<tag>/screening.json and screening.log.
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
from srcs import identification as ident, study as S        # noqa: E402
from srcs.ms_model import BatchMS                             # noqa: E402
from sensitivity import Box                                   # noqa: E402

log = logging.getLogger('screening')
I_REST = 0.02             # [A] below it a sample is a rest
V_LOW = 3.4               # [V] the low-voltage end of a discharge (data)
BRUGG = 1.5               # fixed (Li 2020, Oca 2021: 1.4-1.5)
# typical values (lb, ub) at 25 C for a ~5 Ah NMC/graphite pouch; log scale where lb > 0 (Box)
TYPICAL = {
    'theta_n': (10.0, 1.2e4),        # R^2 / Ds, R 1-11 um, Ds 1e-14-1e-13 m2/s (Li 2020, Table 1)          [s]
    'theta_p': (10.0, 1.2e4),
    'I0_n': (2.0, 200.0),            # A a L j0: A a L ~ 5-20 m2, j0 ~ 0.5-9 A/m2 (k of Li 2020; Oca 2021)   [A]
    'I0_p': (2.0, 200.0),
    'R_contact': (1e-4, 5e-3),       # tabs, collectors, film                                                [Ohm]
    'R_e': (5e-4, 1e-2),             # electrolyte ohmic resistance of a uniform split                        [Ohm]
    'L_e': (80e-6, 200e-6),          # total thickness, L_k 35-79 um, L_s 10-30 um (Li 2020)                  [m]
    'l_s': (0.05, 0.3),              # separator share of L_e (same)
    'r_np': (0.3, 0.7),              # L_n / (L_n + L_p) (same)
    'eps_n': (0.25, 0.5),            # electrolyte volume fractions (Li 2020)
    'eps_s': (0.4, 0.55),
    'eps_p': (0.25, 0.5),
    'Ea_Dn': (5e3, 8e4),             # activation energies [J/mol]: Oca 2021 9-17 kJ/mol, Ruan 2018 up to ~40
    'Ea_Dp': (5e3, 8e4),
    'Ea_r': (5e3, 8e4),
    'Ea_s': (0.0, 3e4),              # series resistance (Ruan 2018: ohmic ~4 kJ/mol)
    **{'dUdTk%d' % i: (-6e-4, 6e-4) for i in range(8)},   # full-cell entropic coefficient [V/K]
}
# literature box (--box literature): the dynamic groups bounded by the published values for graphite/NMC, measured
# and parameter sets, converted to this cell's groups (paper Table 5; the stack groups keep TYPICAL). Sources: theta
# = R^2/D_s per paper (Ai 2020 .. Chen 2020 average; NMC Hosseinzadeh 2018 .. Capron 2018); I0 = 3 Q j0 / (F c_max R)
# with Q_n, Q_p of step 1 (O'Regan 2022 .. Smith 2007; O'Regan 2022 .. Chen 2020); R_e from R Q = 0.007-0.125
# Ohm Ah, R_c from tab and contact resistances (Hosseinzadeh 2018, Das 2019); activation energies (O'Regan 2022,
# Ecker 2015, Chen 2020, Oca 2021); the cell's dU/dT over 10-90 % SOC (Hess 2026, Wind & Vie 2025).
LITERATURE = {**TYPICAL,
              'theta_n': (641.0, 2.0e4), 'theta_p': (250.0, 3.9e5), 'I0_n': (0.66, 1285.0), 'I0_p': (1.83, 10.95),
              'R_contact': (5e-5, 1.8e-3), 'R_e': (1.44e-3, 2.58e-2), 'Ea_Dn': (5e3, 6e4), 'Ea_Dp': (1e4, 8.06e4),
              'Ea_r': (1.18e4, 6.5e4), **{'dUdTk%d' % i: (-1.8e-4, 1.9e-4) for i in range(8)}}


def trajectories(r, d, p, rng):
    """(r, d + 1, d) Morris trajectories on the p-level grid of [0, 1]^d, and (r, d) the parameter moved
    at each of the d steps."""
    delta = p / (2.0 * (p - 1))
    levels = np.arange(p) / (p - 1)
    U = np.zeros((r, d + 1, d))
    order = np.zeros((r, d), int)
    for k in range(r):
        x = rng.choice(levels, d)
        U[k, 0] = x
        perm = rng.permutation(d)
        order[k] = perm
        for s, i in enumerate(perm):
            x = x.copy()
            x[i] = x[i] + delta if x[i] + delta <= 1.0 + 1e-12 else x[i] - delta
            U[k, s + 1] = x
    return U, order


def parts(t):
    I, V, m = t['I'], t['V'], t['mask']
    return {'all': m, 'discharge': m & (I > I_REST), 'charge': m & (I < -I_REST), 'rest': m & (np.abs(I) <= I_REST),
            'low V': m & (I > I_REST) & (V < V_LOW)}


def simulate(em, t, groups):
    """(P, K) voltage [V] of the candidates `groups` (dict of (P,) arrays) on test t, in GPU-sized chunks."""
    P = len(next(iter(groups.values())))
    c = ident.auto_chunk(len(t['I']), em.device)
    out = np.empty((P, len(t['I'])), np.float32)
    for a in range(0, P, c):
        g, q = ident.test_groups(t, {k: v[a:a + c] for k, v in groups.items()}, em.ocp)
        V = em.simulate(g, t['I'], t['dt'], T=t['T'], q_start=q)
        out[a:a + c] = V.cpu().numpy()
    return out


def effects(V, order, msk):
    """(r, d) RMS voltage change [mV] over the samples msk caused by each parameter's move, NaN where a
    simulation is not finite. V: (r, d + 1, K)."""
    r, d = order.shape
    dV = 1e3 * np.diff(V[:, :, msk].astype(np.float64), axis=1)          # (r, d, n)
    rms = np.sqrt(np.mean(dV ** 2, axis=2))                               # (r, d), step s
    E = np.full((r, d), np.nan)
    for k in range(r):
        E[k, order[k]] = rms[k]
    return np.where(np.isfinite(E), E, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--study', default='fitB')
    ap.add_argument('--r', type=int, default=40)
    ap.add_argument('--levels', type=int, default=6)
    ap.add_argument('--cell25', default='C01')
    ap.add_argument('--val', nargs='+', default=['C%02d' % c for c in range(1, 10)])
    ap.add_argument('--tag', default='a_priori')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--box', choices=('typical', 'literature'), default='typical')
    a = ap.parse_args()
    BOX = LITERATURE if a.box == 'literature' else TYPICAL

    out_dir = os.path.join(ROOT, 'runs', 'screening', a.tag)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(out_dir, 'screening.log'))])
    st = S.load(os.path.join(ROOT, 'configs', a.study + '.json'))
    m = st.cfg['model']
    em = BatchMS(st.cfg['surrogate'], st.pm, ocp=st.ocp(), n_slices=int(m.get('n_slices', 5)),
                 n_modes=int(m.get('n_modes', 8)), electrolyte_T=m['electrolyte_T'], entropic=m['entropic'],
                 electrolyte_set=m.get('electrolyte_set'), geometry='groups', electrolyte_groups='material')
    dyn = st.load('dynamics.json')
    names = list(BOX)
    lb, ub = [BOX[n][0] for n in names], [BOX[n][1] for n in names]
    box = Box(lb, ub)
    fixed = {**{k: v for k, v in dyn['groups'].items() if k not in names and k not in BatchMS.GROUPS}, 'brugg': BRUGG}
    specs = ([dict(cell=a.cell25, test=k) for k in ('C20', 'HPPC', 'CRATE1C')] +
             [dict(cell=c, test='VAL') for c in a.val])
    tests = ident.prepare_tests(st.ds, specs, st.windows(), em.ocp)
    rng = np.random.default_rng(a.seed)
    d = len(names)
    U, order = trajectories(a.r, d, a.levels, rng)
    X = box.phys(U.reshape(-1, d))
    groups = {**{k: np.full(len(X), v) for k, v in fixed.items()}, **{n: X[:, i] for i, n in enumerate(names)}}
    log.info('screening (%s): %d parameters, r = %d, p = %d -> %d candidates; window and OCPs of %s; tests %s',
             a.tag, d, a.r, a.levels, len(X), a.study, [t['label'] for t in tests])
    res = dict(study=a.study, box=a.box, names=names, lb=lb, ub=ub, r=a.r, levels=a.levels, electrolyte_groups='material',
               seed=a.seed, fixed=fixed, output='RMS voltage change [mV] per Morris step', tests={})
    for t in tests:
        t0 = time.time()
        V = simulate(em, t, groups).reshape(a.r, d + 1, -1)
        T_C = float(np.mean(t['T'])) - 273.15
        entry = dict(T_C=T_C, I_max=float(np.abs(t['I']).max()), samples=int(len(t['I'])),
                     not_finite=int((~np.isfinite(V).all(axis=2)).sum()), parts={})
        for pname, msk in parts(t).items():
            if msk.sum() < 5:
                continue
            E = effects(V, order, msk)
            entry['parts'][pname] = dict(n=int(msk.sum()), mu_star=np.nanmean(E, 0).tolist(),
                                         median=np.nanmedian(E, 0).tolist(), sigma=np.nanstd(E, 0).tolist())
        res['tests'][t['label']] = entry
        mu = np.array(entry['parts']['all']['median'])
        top = np.argsort(mu)[::-1][:8]
        log.info('%-12s %5.1f C  %.0f s  not finite %d | median effect [mV]: %s', t['label'], T_C, time.time() - t0,
                 entry['not_finite'], '  '.join('%s %.1f' % (names[i], mu[i]) for i in top))
        ident.save(os.path.join(out_dir, 'screening.json'), res)
    log.info('done: %s', os.path.join(out_dir, 'screening.json'))


if __name__ == '__main__':
    main()
