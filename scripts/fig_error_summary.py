"""
fig_error_summary.py
====================
Panels (b)-(d) of Fig. 9 (drawn by fig_cases_errors.py): where the calibrated model errs on the held-out cells.
Mean absolute voltage error (marker) and the 5-95 % range of the absolute error (whisker) of every scored
sample of the held-out first cycles, binned (a) by C-rate, (b) by state of charge and (c) by chamber
temperature. Bins with fewer than 30 samples are omitted.

Reads the per-sample errors cached by scripts/error_bins.py (runs/studies/<study>/figures/error_bins.npz;
run that script first for a new study).

    python scripts/fig_error_summary.py [--pouch fitB] [--out figures/fig_error_summary]
    python scripts/fig_error_summary.py --cases A:fitA B:fitB C:caseC
        (one series per case, blue / yellow / green, with a legend)
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
plt.style.use(os.path.join(ROOT, 'journal.mplstyle'))
mm = 1 / 25.4
C_EDGES = np.array([0.0, 0.02, 0.25, 0.5, 1.0, 1.5, 2.2])
C_LABELS = ['rest', '<0.25', '0.25–0.5', '0.5–1', '1–1.5', '1.5–2']
SOC_EDGES = np.linspace(0, 1, 11)
CELLS = (('pouch', 'o', '#1F4E79', 0.0, None),)
CASE_STYLE = (('o', '#1F77B4'), ('s', '#E1A100'), ('^', '#2CA02C'))
CASE_NAMES = {'A': "A: cells' own anode, mean window", 'B': 'B: Ai2020 anode, own capacity',
              'C': 'C: Ai2020 anode, capacity from 1st discharge'}
NMIN = 30


def stats(a):
    a = np.abs(a)
    return a.mean(), np.percentile(a, 5), np.percentile(a, 95)


YMAX = 70.0


def draw(ax, x, a, mk, col, lab=None, k=0):
    m, lo, hi = stats(a)
    top = min(hi, YMAX)
    ax.errorbar([x], [m], yerr=[[m - lo], [top - m]], fmt=mk, color=col, mfc=col, mec='k', mew=0.4, ms=5,
                elinewidth=1.0, capsize=2.5 if hi <= YMAX else 0, label=lab)
    if hi > YMAX:                                   # whisker past the axis: arrow and its value
        ax.annotate('', xy=(x, YMAX), xytext=(x, YMAX - 8), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.0))
        dxt = 0.12 if ax.get_xlim()[1] < 20 else 1.5        # category axes vs the SOC axis
        ax.text(x + dxt, YMAX - 4 - 6 * k, '%.0f' % hi, color=col, fontsize=plt.rcParams['font.size'] - 2, va='top')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--pouch', default='fitB')
    ap.add_argument('--cases', nargs='+', help='KEY:study, one series each')
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'fig_error_summary'))
    a = ap.parse_args()
    global CELLS
    pairs = [c.split(':', 1) for c in a.cases] if a.cases else [('pouch', a.pouch)]
    if a.cases:
        n = len(pairs)
        CELLS = tuple((k, CASE_STYLE[i][0], CASE_STYLE[i][1], 0.12 * (i - (n - 1) / 2), CASE_NAMES.get(k, k))
                      for i, (k, _) in enumerate(pairs))
    data = {k: np.load(os.path.join(ROOT, 'runs', 'studies', s, 'figures', 'error_bins.npz')) for k, s in pairs}
    fig, (axc, axs, axt) = plt.subplots(1, 3, figsize=(180 * mm, 58 * mm), layout='constrained',
                                        gridspec_kw=dict(width_ratios=[1.15, 1.25, 0.9]))
    axs.set_xlim(0, 100)
    axc.set_xlim(-0.5, len(C_LABELS) - 0.5)
    for ik, (key, mk, col, dx, name) in enumerate(CELLS):
        d = data[key]
        e, soc, cr, tg = d['e'], d['soc'], d['crate'], d['tg']
        first = True
        for i in range(len(C_LABELS)):
            m = (cr >= C_EDGES[i]) & (cr < C_EDGES[i + 1])
            if m.sum() >= NMIN:
                draw(axc, i + dx, e[m], mk, col, name if first else None, k=ik)
                first = False
        for i in range(len(SOC_EDGES) - 1):
            m = (soc >= SOC_EDGES[i]) & (soc < SOC_EDGES[i + 1] + (1e-9 if i == len(SOC_EDGES) - 2 else 0))
            if m.sum() >= NMIN:
                draw(axs, 100 * 0.5 * (SOC_EDGES[i] + SOC_EDGES[i + 1]) + 2.2 * dx / 0.12, e[m], mk, col, k=ik)
        Ts = sorted(np.unique(tg), key=lambda g: float(g.split()[0].replace('−', '-')))
        for k, g in enumerate(Ts):                  # cold, 25 C, hot of each cell
            draw(axt, k + dx, e[tg == g], mk, col, k=ik)
    axc.set(xlim=(-0.5, len(C_LABELS) - 0.5), xticks=range(len(C_LABELS)), xlabel='C-rate [1/h]', ylabel='|model − data| [mV]')
    axc.set_xticklabels(C_LABELS, fontsize=plt.rcParams['font.size'] - 1, rotation=35, ha='right')
    axs.set(xlabel='State of charge [%]', xlim=(0, 100), xticks=range(0, 101, 20))
    axt.set(xlabel='Chamber temperature', xticks=[0, 1, 2], xlim=(-0.6, 2.6))
    axt.set_xticklabels([u'−5 °C', u'25 °C', u'45 °C'],
                        fontsize=plt.rcParams['font.size'] - 2)
    if a.cases:
        fig.legend(*axc.get_legend_handles_labels(), loc='outside upper center', ncol=len(CELLS), frameon=False,
                   fontsize=plt.rcParams['font.size'] - 1)
    for ax, lab in ((axc, 'a'), (axs, 'b'), (axt, 'c')):
        ax.set_ylim(0, YMAX)
        ax.text(-0.02, 1.02, '(%s)' % lab, transform=ax.transAxes, fontweight='bold', va='bottom', ha='right')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + '.pdf')
    fig.savefig(a.out + '.png', dpi=300)
    for key, *_ in CELLS:
        print(key, 'MAE %.1f mV' % np.abs(data[key]['e']).mean())
    print('written', a.out)


if __name__ == '__main__':
    main()
