"""
heldout_capacity_fit.py
=======================
Case C without the C/20: the capacity of each held-out cell fitted on the first discharge of its first cycle
(VAL), then the whole cycle predicted. The held-out cell starts from the mean window of the calibration cells;
its electrode capacities Q_n and Q_p are both scaled by one factor s (the stoichiometric limits stay the mean),
as in case B, where s comes from the cell's C/20 capacity (configs/fitB.json).

s is chosen on a grid (--smin..--smax, --ds) by the RMSE of the first discharge alone (from t = 0 to the first
charging sample), batched over the grid in one simulation per cell, and refined by a parabola through the three
best points. The groups are those of the study given (its dynamics.json); the mean window is taken from
the windows.json of --mean-study and the C/20 factor of case B from the windows.json of --cap-study.

    python scripts/heldout_capacity_fit.py caseC --mean-study caseC --cap-study fitB --smin 0.96 --smax 1.05
        [--ds 0.0025] [--cells C01 C12]

Writes runs/studies/<study>/heldout_capacity_fit.csv: per held-out cell, s fitted, s from the C/20, RMSE of the
first discharge, of the rest of the cycle and of the whole cycle, with s fitted and with s = 1 (mean window).

s is observable only when the first discharge reaches the cutoff; on a partial first discharge (or a cycle that
starts with a charge) it runs to the grid edge. Table 7 of the paper (case C with the online capacity) applies s
where V_end_first < 3.2 V and keeps the mean capacity elsewhere.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from srcs import identification as ident, study as S       # noqa: E402


def cut(t, k):
    """Test t truncated to its first k samples."""
    u = dict(t)
    for key in ('I', 'V', 'T', 'mask', 't'):
        if isinstance(t.get(key), np.ndarray) and t[key].shape[:1] == t['I'].shape[:1]:
            u[key] = t[key][:k]
    if isinstance(t['dt'], np.ndarray) and t['dt'].shape == t['I'].shape:
        u['dt'] = t['dt'][:k]
    return u


def sim(em, t, g, s):
    """(B, N) voltage of test t for the capacity factors s (B,)."""
    s = np.atleast_1d(np.asarray(s, float))
    gg, q = ident.test_groups(t, {**g, 'win_Q_n': s, 'win_Q_p': s, 'win_x100': np.ones_like(s)}, em.ocp)
    V = em.simulate(gg, t['I'], t['dt'], T=t['T'], q_start=q)
    return ident.add_bias(V, t, gg, q).double().cpu().numpy()


def rmse(e, m):
    return 1e3 * float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('study')
    ap.add_argument('--smin', type=float, default=0.97)
    ap.add_argument('--smax', type=float, default=1.04)
    ap.add_argument('--ds', type=float, default=0.0025)
    ap.add_argument('--mean-study', default='caseC', help='study whose windows.json is the no-C/20 window')
    ap.add_argument('--cap-study', default='fitB', help='study whose windows.json has the C/20 capacity of each cell')
    ap.add_argument('--cells', nargs='+', help='held-out cells to fit (default: all)')
    a = ap.parse_args()
    st = S.load(os.path.join(ROOT, 'configs', a.study + '.json'))
    em = st.em()
    g = st.load('dynamics.json')['groups']
    mean = json.load(open(os.path.join(ROOT, 'runs', 'studies', a.mean_study, 'windows.json')))
    cap = json.load(open(os.path.join(ROOT, 'runs', 'studies', a.cap_study, 'windows.json')))
    fit = list(st.split['dynamics'])
    held = [c for c in st.ds.cells if c not in fit and c not in st.split.get('exclude', []) and st.ds.available(c, 'VAL')]
    if st.split.get('validation'):        # an explicit held-out list (LG)
        held = [c for c in st.split['validation'] if c in held]
    if a.cells:
        held = [c for c in held if c in a.cells]
    grid = np.arange(a.smin, a.smax + 0.5 * a.ds, a.ds)
    rows = []
    for t in ident.prepare_tests(st.ds, st.heldout_specs(held), mean, em.ocp):   # config heldout_tests
        c = t['cell']
        ch = np.flatnonzero(t['I'] < -0.02)
        k = int(ch[0]) if len(ch) else len(t['I'])            # first charging sample: end of the first discharge
        first = np.arange(len(t['I'])) < k
        t1 = cut(t, k)
        V1 = sim(em, t1, g, grid)
        r = np.array([rmse(V1[b] - t1['V'], t1['mask']) for b in range(len(grid))])
        i = int(np.argmin(r))
        s_fit = grid[i]
        if 0 < i < len(grid) - 1:                             # parabola through the three best points
            y0, y1, y2 = r[i - 1], r[i], r[i + 1]
            den = y0 - 2 * y1 + y2
            if den > 0:
                s_fit = grid[i] + 0.5 * a.ds * (y0 - y2) / den
        s_c20 = cap[c]['Q_n'] / mean[c]['Q_n']
        V = sim(em, t, g, [s_fit, 1.0, s_c20])
        e = V - t['V'][None, :]
        m = t['mask']
        row = dict(cell=c, T_C=float(np.mean(t['T']) - 273.15), I_dis_A=float(np.percentile(t['I'][t['I'] > 0.1], 50)),
                   q_first_Ah=float(np.sum((t['I'] * np.broadcast_to(t['dt'], t['I'].shape))[:k]) / 3600),
                   V_end_first=float(t['V'][max(k - 1, 0)]), s_fit=float(s_fit), s_c20=float(s_c20),
                   at_edge=bool(i in (0, len(grid) - 1)))
        for j, lab in enumerate(('fit', 'mean', 'c20')):
            row['first_' + lab] = rmse(e[j], m & first)
            row['rest_' + lab] = rmse(e[j], m & ~first)
            row['all_' + lab] = rmse(e[j], m)
        rows.append(row)
        print('%s %5.1f C  first discharge %.2f Ah to %.2f V | s fit %.4f  C/20 %.4f%s | whole cycle: fit %.1f  mean %.1f  '
              'C/20 %.1f mV | rest of cycle: fit %.1f  mean %.1f  C/20 %.1f mV' % (
                  c, row['T_C'], row['q_first_Ah'], row['V_end_first'], s_fit, s_c20, ' (edge)' if row['at_edge'] else '',
                  row['all_fit'], row['all_mean'], row['all_c20'], row['rest_fit'], row['rest_mean'], row['rest_c20']),
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(st.path('heldout_capacity_fit.csv'), index=False)
    df['T_group'] = np.where(df['T_C'] < 10, '-5', np.where(df['T_C'] > 37, '45', '25'))
    print(df.groupby('T_group')[['all_fit', 'all_mean', 'all_c20', 'rest_fit', 'rest_mean', 'rest_c20']].mean().round(1).to_string())
    print('worst cell, whole cycle: fit %.1f (%s), mean %.1f, C/20 %.1f' % (
        df['all_fit'].max(), df.loc[df['all_fit'].idxmax(), 'cell'], df['all_mean'].max(), df['all_c20'].max()))
    print('|s_fit - s_c20|: median %.4f, max %.4f' % ((df['s_fit'] - df['s_c20']).abs().median(), (df['s_fit'] - df['s_c20']).abs().max()))


if __name__ == '__main__':
    main()
