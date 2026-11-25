"""
fig_cases_heldout.py
====================
Held-out first-cycle (VAL) RMSE of every held-out pouch cell under the three cases A -> B -> C:
one group of three bars per cell (A blue, B yellow, C green), cells grouped by chamber temperature, the mean of
each case per temperature printed above its group. heldout() gives the per-cell values to the notebook, to Fig. 9
(fig_cases_errors.py) and to Fig. 10 (fig_lg_grid.py).

    A  measured anode OCP, each held-out cell's own C/20 capacity                  (fitA)
    B  generic graphite anode, each held-out cell's own C/20 capacity              (fitB)
    C  groups of B, every held-out cell at the mean capacity of the calibration cells (caseC)

RMSE over the scored samples of the test (the study's mask), no refit. The per-cell values are cached in
runs/studies/<study>/heldout_val_rmse.csv (delete it to recompute).

    python scripts/fig_cases_heldout.py [--cpu] [--cases A:<study> B:<study> C:<study>] [--out figures/fig_cases_heldout]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
import pandas as pd                 # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
from srcs import identification as ident, study as S       # noqa: E402

plt.style.use(os.path.join(ROOT, 'journal.mplstyle'))
mm = 1 / 25.4
DEFAULT = ('A:fitA', 'B:fitB', 'C:caseC')
COLORS = {'A': '#1F77B4', 'B': '#E1A100', 'C': '#2CA02C', 'C+': '#9467BD'}
LABELS = {'A': 'A: measured anode, C/20 capacity', 'B': 'B: generic anode, C/20 capacity',
          'C': 'C: generic anode, mean capacity', 'C+': 'C+: capacity from the first discharge'}


def tgroup(T):
    return -5 if T < 10 else (45 if T > 37 else 25)


def heldout(name):
    """Per held-out cell RMSE [mV] of the first cycle (VAL), cached."""
    st = S.load(os.path.join(ROOT, 'configs', name + '.json'))
    f = st.path('heldout_val_rmse.csv')
    if os.path.exists(f):
        return pd.read_csv(f)
    em = st.em()
    g = st.load('dynamics.json')['groups']
    W = st.windows()
    rows = []
    # each cell's test from config "heldout_tests" (default its whole VAL), grouped by its chamber temperature
    for t in ident.prepare_tests(st.ds, st.heldout_specs(st.split['validation']), W, em.ocp):
        gg, q = ident.test_groups(t, g, em.ocp)
        V = ident.add_bias(em.simulate(gg, t['I'], t['dt'], T=t['T'], q_start=q), t, gg, q)[0].cpu().numpy()
        m = t['mask'] & np.isfinite(V)
        T = float(np.mean(t['T']) - 273.15)
        Tn = st.nominal_T(t['cell'])
        rows.append(dict(cell=t['cell'], T_C=T, T_group=tgroup(T) if Tn is None else Tn,
                         rmse_mV=1e3 * float(np.sqrt(np.mean((V - t['V'])[m] ** 2)))))
        print(name, rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f, index=False)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cases', nargs='+', default=list(DEFAULT))
    ap.add_argument('--out', default=os.path.join('figures', 'fig_cases_heldout'))
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()
    cases = [c.split(':', 1) for c in a.cases]
    D = {k: heldout(s).set_index('cell') for k, s in cases}
    ref = D[cases[0][0]]
    order = ref.sort_values(['T_group', 'T_C']).index.tolist()
    groups = [(T, [c for c in order if ref.loc[c, 'T_group'] == T]) for T in (-5, 25, 45)]
    groups = [(T, cs) for T, cs in groups if cs]
    fs = 7
    fig, ax = plt.subplots(figsize=(180 * mm, 70 * mm))
    w = 0.27
    x0 = 0.0
    ticks, labels = [], []
    top = 1.08 * max(float(d.rmse_mV.max()) for d in D.values())
    for gi, (T, cs) in enumerate(groups):
        xs = x0 + np.arange(len(cs))
        for j, (k, _) in enumerate(cases):
            v = [D[k].loc[c, 'rmse_mV'] if c in D[k].index else np.nan for c in cs]
            ax.bar(xs + (j - 1) * w, v, w, color=COLORS.get(k, '#9467BD'), edgecolor='k', linewidth=0.3,
                   label=LABELS.get(k, k) if gi == 0 else None)
        means = ' / '.join('%.1f' % np.nanmean([D[k].loc[c, 'rmse_mV'] for c in cs if c in D[k].index]) for k, _ in cases)
        ax.text(xs.mean(), 1.02, '%s $^\\circ$C\nmean %s: %s mV' % (str(T).replace('-', '−'),'/'.join(k for k, _ in cases), means),
                transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=fs)
        if gi < len(groups) - 1:
            ax.axvline(xs[-1] + 0.75, color='0.6', lw=0.6, ls=':')
        ticks += list(xs)
        labels += [c.lstrip('C') for c in cs]
        x0 = xs[-1] + 1.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=fs)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=fs)
    ax.set_xlabel('held-out cell', fontsize=fs + 1)
    ax.set_ylabel('first-cycle RMSE [mV]', fontsize=fs + 1)
    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, x0 - 0.9)
    fig.legend(loc='upper center', ncol=3, frameon=False, fontsize=fs, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.78)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(ROOT, a.out + '.' + ext), dpi=300)
    for k, s in cases:
        d = D[k]
        print('%s %-34s -5/25/45: %s | mean %.1f | worst %.1f (%s)' % (
            k, s, ' / '.join('%.1f' % d[d.T_group == T].rmse_mV.mean() for T in (-5, 25, 45)),
            d.rmse_mV.mean(), d.rmse_mV.max(), d.rmse_mV.idxmax()))


if __name__ == '__main__':
    main()
