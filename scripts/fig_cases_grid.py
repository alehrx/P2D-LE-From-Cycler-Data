"""
fig_cases_grid.py
=================
Fig. 9 of the paper: measured and predicted voltage of the first cycle (VAL) of fifteen held-out cells under the
three cases A, B, C. Rows: ageing protocol (Table 2, labelled on the left); columns: chamber temperature (-5, 25,
45 C, titled on top).

    P5  C/5 charge and discharge, 50 % DoD          cells 14, 13, 15
    P1  C/5 charge and discharge, 100 % DoD         cells 02, 01, 03
    P7  1.5C charge, drive-cycle discharge, 50 % DoD cells 20, 19, 21

Only the first cycle is drawn (up to the second discharge). Cell, protocol and RMSE of each case go to
stdout.

    python scripts/fig_cases_grid.py [--cells C14 C13 C15 C02 C01 C03 ...] [--cases A:fitA B:fitB C:caseC] [--out figures/fig_cases_grid]

Set CUDA_VISIBLE_DEVICES=-1 to run it on the CPU.
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import matplotlib.ticker            # noqa: E402,F401
import numpy as np                  # noqa: E402
from matplotlib.patches import Rectangle    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from fig_cases_voltage import CASES, N_MARK, mm, simulate, with_suffix     # noqa: E402  (also sets the style and cwd)

STYLE = {'A': ('#0072B2', 'o', 'case A'), 'B': ('#D55E00', 's', 'case B'), 'C': ('#009E73', '^', 'case C'),
         'C+': ('#CC79A7', 'D', 'case C+')}
# rows = protocols (Table 2), columns = chamber temperature (-5, 25, 45 C), 15 held-out cells
CELLS = ['C14', 'C13', 'C15', 'C02', 'C01', 'C03', 'C05', 'C04', 'C06', 'C11', 'C10', 'C12', 'C20', 'C19', 'C21']
ROWS = ['C/5 / C/5\n50 % DoD', 'C/5 / C/5\n100 % DoD', '1.5C / 1.5C\n100 % DoD', 'C/5 / 1.5C\n100 % DoD',
        '1.5C / drive cycle\n50 % DoD']
COLS = ['−5 °C', '25 °C', '45 °C']
NR, NC = len(ROWS), len(COLS)


def first_cycle(I, t):
    """Number of samples of the first cycle: up to the first discharge sample after the first charge, a charge being
    a negative current held for at least 10 min (the regenerative pulses of a drive cycle are not one); t in h."""
    neg = I < -0.1
    k = 0
    while k < len(I):
        if neg[k]:
            j = k
            while j < len(I) and neg[j]:
                j += 1
            if t[j - 1] - t[k] >= 1 / 6:
                nxt = np.flatnonzero((I > 0.1) & (np.arange(len(I)) >= j))
                return int(nxt[0]) if len(nxt) else len(I)
            k = j
        k += 1
    return len(I)


def zoom(x, res, c, n, win=None, box=(0.55, 0.05, 0.42, 0.42)):
    """Inset joined to the window it magnifies by the rectangle and two connectors (indicate_inset_zoom): by default
    the end of the first discharge of cell c (its lowest measured voltage), else the time window win = (t0, t1) [h],
    its voltage range from the data."""
    d0 = res[CASES[0][0]][c]
    t, V = d0['t'][:n], d0['Vd'][:n]
    if win is None:
        tmin = t[np.argmin(V)]
        x0, x1, y0, y1 = tmin - 0.6, tmin + 0.15, 2.9, 3.8          # a wider window
    else:
        x0, x1 = win
        ww = (t > x0) & (t < x1)
        y0, y1 = V[ww].min() - 0.03, V[ww].max() + 0.03
    w = (t > x0) & (t < x1)
    ins = x.inset_axes(list(box))
    ins.plot(t[w], V[w], color='k', lw=0.9)
    for ik, (k, _, col, mk, _) in enumerate(CASES):
        r = res[k][c]
        ins.plot(r['t'][:n][w], r['V'][:n][w], lw=0.8, ls='--', color=col, marker=mk, markevery=(3 * ik, 9), ms=1.8,
                 mfc='white', mec=col, mew=0.5)
    ins.set(xlim=(x0, x1), ylim=(y0, y1), xticks=[])
    ins.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
    ins.tick_params(labelsize=5, length=2, pad=1)
    for s_ in ins.spines.values():
        s_.set_linewidth(0.6)
        s_.set_edgecolor('0.3')
    x.indicate_inset_zoom(ins, edgecolor='0.3', linewidth=0.6, alpha=1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cells', nargs=15, default=CELLS)
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'fig_cases_grid'))
    ap.add_argument('--suffix', default='', help='appended to every case study name (before _cap)')
    ap.add_argument('--cases', nargs='+', help='KEY:study pairs replacing the default cases A, B, C (no suffix), e.g. '
                    'B:fitB C:caseC')
    a = ap.parse_args()
    global CASES
    if a.cases:
        CASES = tuple((k, st) + STYLE[k] for k, st in (c.split(':', 1) for c in a.cases))
        a.suffix = ''
    cache = a.out + '_sim.pkl'            # the simulations, kept so that a change of layout does not rerun them
    key = (tuple(a.cells), tuple((k, cfg, a.suffix) for k, cfg, *_ in CASES))
    res = None
    if os.path.exists(cache):
        ck, cr = pickle.load(open(cache, 'rb'))
        res = cr if ck == key else None
    if res is None:
        res = {k: simulate(with_suffix(cfg, a.suffix), a.cells) for k, cfg, *_ in CASES}
        pickle.dump((key, res), open(cache, 'wb'))
    fig, ax = plt.subplots(NR, NC, figsize=(180 * mm, 215 * mm), sharex='row', sharey=True, layout='constrained')
    for i, c in enumerate(a.cells):
        x = ax.flat[i]
        d0 = res[CASES[0][0]][c]
        n = first_cycle(d0['It'], d0['t'])                  # the first cycle only
        x.plot(d0['t'][:n], d0['Vd'][:n], color='k', lw=0.9, label='measured')
        lab = []
        for ik, (k, _, col, mk, name) in enumerate(CASES):
            r = res[k][c]
            step = max(n // N_MARK, 1)
            x.plot(r['t'][:n], r['V'][:n], lw=0.8, ls='--', label=name, color=col, marker=mk,
                   markevery=(ik * step // len(CASES), step), ms=2.3, mfc='white', mec=col, mew=0.7)
            e = (r['V'] - r['Vd'])[:n][r['m'][:n]]
            lab.append('%s %.0f (whole test %.0f)' % (k, 1e3 * np.sqrt(np.mean(e ** 2)), r['rmse']))
        if c == 'C20':          # the cold drive cycle: a stretch of the dynamic discharge
            zoom(x, res, c, n, win=(1.2, 1.4), box=(0.12, 0.05, 0.5, 0.4))
        L = 'abcdefghijklmno'[i]                         # row by row
        x.set_title('(%s)' % L, fontsize=8, loc='left', fontweight='bold')
        if i < NC:              # the temperature on top of each column, the protocol left of each row
            x.annotate(COLS[i], xy=(0.5, 1), xycoords='axes fraction', xytext=(0, 15), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        if i % NC == 0:
            x.annotate(ROWS[i // NC], xy=(0, 0.5), xycoords='axes fraction', xytext=(-42, 0), textcoords='offset points',
                       rotation=90, ha='right', va='center', fontsize=8, fontweight='bold', multialignment='center')
        x.set_ylim(2.9, 4.25)
        x.spines[['top', 'right']].set_visible(False)
        if i % NC == 0:
            x.set_ylabel('V [V]')
        if i >= NC * (NR - 1):
            x.set_xlabel('t [h]')
        print('(%s) cell %s, %.0f C, %.1f A | RMSE first cycle %s mV' % (L, c[1:], d0['T'], d0['I'], ', '.join(lab)))
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside upper center', ncol=4, fontsize=7, frameon=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + '.pdf')
    fig.savefig(a.out + '.png', dpi=300)
    print('written', a.out)


if __name__ == '__main__':
    main()
