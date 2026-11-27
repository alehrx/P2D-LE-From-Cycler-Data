"""
fig_cases_errors.py
===================
Fig. 10 of the paper: held-out error of the three cases A -> B -> C in one figure:
    (a) first-cycle RMSE of every held-out cell, three bars per cell, cells grouped by chamber temperature
        (scripts/fig_cases_heldout.py, cached runs/studies/<study>/heldout_val_rmse.csv);
    (b)-(d) mean absolute error and 5-95 % range of every scored sample of the held-out first cycles, binned by
        C-rate, state of charge and chamber temperature (scripts/fig_error_summary.py, cached
        runs/studies/<study>/figures/error_bins.npz from scripts/error_bins.py).
Colours: A blue, B orange, C green.

    python scripts/fig_cases_errors.py [--cases A:<study> B:<study> C:<study>] [--out figures/fig_cases_errors]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import fig_cases_heldout as H       # noqa: E402
import fig_error_summary as E       # noqa: E402

mm = 1 / 25.4
DEFAULT = ('A:fitA', 'B:fitB', 'C:caseC')
MARK = {'A': 'o', 'B': 's', 'C': '^', 'C+': 'D'}


def bars(ax, cases, D, fs, temps=(-5, 25, 45)):
    ref = D[cases[0][0]]
    order = ref.sort_values(['T_group', 'T_C']).index.tolist()
    groups = [(T, [c for c in order if ref.loc[c, 'T_group'] == T]) for T in temps]
    w, x0, ticks, labels = 0.81 / len(cases), 0.0, [], []
    for gi, (T, cs) in enumerate([g for g in groups if g[1]]):
        xs = x0 + np.arange(len(cs))
        for j, (k, _) in enumerate(cases):
            v = [D[k].loc[c, 'rmse_mV'] if c in D[k].index else np.nan for c in cs]
            ax.bar(xs + (j - (len(cases) - 1) / 2) * w, v, w, color=H.COLORS[k], edgecolor='k', linewidth=0.3, label=H.LABELS[k] if gi == 0 else None)
        ax.text(xs.mean(), 0.97, '%s °C' % str(T).replace('-', '−'),     # no means (they are in Table 6)
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=fs - 1)
        if gi < 2:
            ax.axvline(xs[-1] + 0.75, color='0.6', lw=0.6, ls=':')
        ticks += list(xs)
        labels += [c.lstrip('C') for c in cs]
        x0 = xs[-1] + 1.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=fs)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, x0 - 0.9)
    ymax = max(float(d.rmse_mV.max()) for d in D.values())
    ax.set_ylim(0, 1.12 * ymax)
    ax.set_yticks(np.arange(0, 1.05 * ymax, 20))
    ax.set_xlabel('held-out cell', fontsize=fs + 1)
    ax.set_ylabel('RMSE [mV]', fontsize=fs + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cases', nargs='+', default=list(DEFAULT))
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'fig_cases_errors'))
    ap.add_argument('--temps', type=float, nargs=3, default=[-5, 25, 45], help='chamber temperatures [C] (LG: 10 25 40)')
    a = ap.parse_args()
    # panels (b)-(d): RMS of the error in each bin (marker), 5-95 % of |error| (whisker), so that every panel shows an RMSE
    # (the RMS can lie above the 95th percentile of a heavy-tailed bin: the whisker then reaches up to the marker)
    def stats(e):
        rms = float(np.sqrt(np.mean(np.square(e))))
        return rms, min(np.percentile(np.abs(e), 5), rms), max(np.percentile(np.abs(e), 95), rms)
    E.stats = stats

    def draw(ax, x, e, mk, col, lab=None, k=0):         # E.draw, with a marker past the axis drawn at its top
        m, lo, hi = stats(e)
        y, top = min(m, E.YMAX), min(hi, E.YMAX)
        ax.errorbar([x], [y], yerr=[[max(y - lo, 0)], [max(top - y, 0)]], fmt=mk, color=col, mfc=col, mec='k',
                    mew=0.4, ms=5, elinewidth=1.0, capsize=2.5 if hi <= E.YMAX else 0, label=lab)
        if hi > E.YMAX:
            ax.annotate('', xy=(x, E.YMAX), xytext=(x, E.YMAX - 8), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.0))
            dxt = 0.12 if ax.get_xlim()[1] < 20 else 1.5
            ax.text(x + dxt, E.YMAX - 4 - 6 * k, '%.0f' % (m if m > E.YMAX else hi), color=col,
                    fontsize=plt.rcParams['font.size'] - 2, va='top')
    E.draw = draw
    cases = [c.split(':', 1) for c in a.cases]
    D = {k: H.heldout(s).set_index('cell') for k, s in cases}
    data = {k: np.load(os.path.join(ROOT, 'runs', 'studies', s, 'figures', 'error_bins.npz')) for k, s in cases}
    fs = plt.rcParams['font.size'] - 1
    fig = plt.figure(figsize=(180 * mm, 120 * mm), layout='constrained')
    sf = fig.subfigures(2, 1, height_ratios=[1.0, 1.0])
    axb = sf[0].subplots()
    temps = [int(T) if float(T).is_integer() else T for T in a.temps]
    bars(axb, cases, D, fs, temps)
    axc, axs, axt = sf[1].subplots(1, 3, gridspec_kw=dict(width_ratios=[1.15, 1.25, 0.9]))
    axs.set_xlim(0, 100)
    CB = [i for i, lab in enumerate(E.C_LABELS) if lab != 'rest']
    axc.set_xlim(-0.5, len(CB) - 0.5)
    n = len(cases)
    for ik, (k, _) in enumerate(cases):
        mk, col, dx = MARK.get(k, 'o'), H.COLORS[k], 0.12 * (ik - (n - 1) / 2)
        d = data[k]
        e, soc, cr, tg = d['e'], d['soc'], d['crate'], d['tg']
        for j, i in enumerate(CB):      # the rests are left out of (b) (at rest after a discharge the
            m = (cr >= E.C_EDGES[i]) & (cr < E.C_EDGES[i + 1])     # error is the knee at its end, not a C-rate effect)
            if m.sum() >= E.NMIN:
                E.draw(axc, j + dx, e[m], mk, col, k=ik)
        for i in range(len(E.SOC_EDGES) - 1):
            m = (soc >= E.SOC_EDGES[i]) & (soc < E.SOC_EDGES[i + 1] + (1e-9 if i == len(E.SOC_EDGES) - 2 else 0))
            if m.sum() >= E.NMIN:
                E.draw(axs, 100 * 0.5 * (E.SOC_EDGES[i] + E.SOC_EDGES[i + 1]) + 2.2 * dx / 0.12, e[m], mk, col, k=ik)
        Ts = sorted(np.unique(tg), key=lambda g: float(g.split()[0].replace('−', '-')))
        for j, g in enumerate(Ts):
            E.draw(axt, j + dx, e[tg == g], mk, col, k=ik)
    axc.set(xlim=(-0.5, len(CB) - 0.5), xticks=range(len(CB)), xlabel='C-rate [1/h]',
            ylabel='RMSE [mV]')
    axc.set_xticklabels([E.C_LABELS[i] for i in CB], fontsize=fs, rotation=35, ha='right')
    axs.set(xlabel='State of charge [%]', xlim=(0, 100), xticks=range(0, 101, 20))
    axt.set(xlabel='Chamber temperature', xticks=[0, 1, 2], xlim=(-0.6, 2.6))
    axt.set_xticklabels(['%s °C' % str(T).replace('-', '−') for T in temps], fontsize=fs)
    for ax in (axc, axs, axt):
        ax.set_ylim(0, E.YMAX)
    for ax, lab in ((axb, 'a'), (axc, 'b'), (axs, 'c'), (axt, 'd')):
        ax.text(-0.01, 1.01, '(%s)' % lab, transform=ax.transAxes, fontweight='bold', va='bottom', ha='right')
    # legend above panel (a), as wide as its axes
    SHORT = {'A': 'case A', 'B': 'case B', 'C': 'case C', 'C+': 'case C+'}
    hs, ls = axb.get_legend_handles_labels()
    fig.legend(hs, [SHORT.get(l.split(':')[0], l) for l in ls], loc='outside upper center', ncol=n, frameon=False, fontsize=fs,
               handlelength=1.2, columnspacing=2.5)
    fig.savefig(a.out + '.pdf')
    fig.savefig(a.out + '.png', dpi=300)
    for k, _ in cases:
        d = data[k]
        soc = d['soc']; e = np.abs(d['e'])
        print('%s MAE %.1f mV | SOC>10%% %.1f | SOC<10%% %.1f | by T %s' % (
            k, e.mean(), e[soc >= 0.1].mean(), e[soc < 0.1].mean(),
            {g: round(float(e[d['tg'] == g].mean()), 1) for g in np.unique(d['tg'])}))
    print('written', a.out)


if __name__ == '__main__':
    main()
