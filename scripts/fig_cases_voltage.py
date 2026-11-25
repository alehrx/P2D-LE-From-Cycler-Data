"""
fig_cases_voltage.py
====================
Measured and predicted voltage of four held-out first cycles (VAL) under the three cases of the paper, each case
a study with its own windows and groups; also the case definitions, colours and markers that fig_cases_grid.py
and fig_lg_grid.py use:

    A  measured anode OCP; each held-out cell at its own C/20 capacity            configs/fitA.json
    B  generic graphite anode (Ai2020); each held-out cell at its own C/20 capacity  configs/fitB.json
    C  the groups of B, every held-out cell at the mean capacity of the calibration cells   configs/caseC.json

Top of each panel: voltage (data black, cases coloured); bottom: model - data. Each case also has its own marker
(o, s, ^ as in fig_error_summary.py), N_MARK per curve, staggered in time.

    python scripts/fig_cases_voltage.py [--cells C11 C04 C01 C12] [--out figures/fig_cases_voltage]
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from srcs import identification as ident, study as S       # noqa: E402

plt.style.use(os.path.join(ROOT, 'journal.mplstyle'))
mm = 1 / 25.4
CASES = (('A', 'fitA', '#1F77B4', 'o', 'case A'),
         ('B', 'fitB', '#E1A100', 's', 'case B'),
         ('C', 'caseC', '#2CA02C', '^', 'case C'))
N_MARK = 14             # markers per curve, staggered between cases so that they do not overlap


def simulate(cfg, cells):
    st = S.load(os.path.join(ROOT, 'configs', cfg + '.json'))
    em = st.em()
    g = st.load('dynamics.json')['groups']
    out = {}
    for t in ident.prepare_tests(st.ds, [dict(cell=c, test='VAL') for c in cells], st.windows(), em.ocp):
        gg, q = ident.test_groups(t, g, em.ocp)
        V = ident.add_bias(em.simulate(gg, t['I'], t['dt'], T=t['T'], q_start=q), t, gg, q)[0].cpu().numpy()
        m = t['mask']
        out[t['cell']] = dict(t=t['t'] / 3600, Vd=t['V'], V=V, m=m, T=float(np.mean(t['T']) - 273.15),
                              I=float(np.percentile(t['I'][t['I'] > 0.1], 50)),
                              It=np.asarray(t['I']),
                              rmse=1e3 * float(np.sqrt(np.mean((V - t['V'])[m] ** 2))))
    return out



def with_suffix(cfg, suffix):
    """The study of a case with a suffix appended (before a final '_cap')."""
    return cfg[:-4] + suffix + '_cap' if cfg.endswith('_cap') else cfg + suffix

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cells', nargs='+', default=['C11', 'C04', 'C01', 'C12'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'fig_cases_voltage'))
    ap.add_argument('--suffix', default='', help='appended to every case study name (and to the default --out)')
    a = ap.parse_args()
    if a.suffix and a.out == ap.get_default('out'):
        a.out += a.suffix
    res = {k: simulate(with_suffix(cfg, a.suffix), a.cells) for k, cfg, *_ in CASES}
    fig = plt.figure(figsize=(180 * mm, 130 * mm), layout='constrained')
    sf = np.atleast_2d(fig.subfigures(2, 2)).ravel()
    for i, c in enumerate(a.cells):
        ax = sf[i].subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1.7, 1]))
        d0 = res['A'][c]
        ax[0].plot(d0['t'], d0['Vd'], color='k', lw=0.9, label='data')
        lab = []
        for ik, (k, _, col, mk, name) in enumerate(CASES):
            r = res[k][c]
            n = len(r['t'])
            step = max(n // N_MARK, 1)
            every = (ik * step // len(CASES), step)
            sty = dict(color=col, marker=mk, markevery=every, ms=3.2, mfc='white', mec=col, mew=0.8)
            ax[0].plot(r['t'], r['V'], lw=0.8, ls='--', label=name, **sty)
            e = 1e3 * (r['V'] - r['Vd'])
            e[~r['m']] = np.nan
            ax[1].plot(r['t'], e, lw=0.7, **sty)
            lab.append('%s %.0f' % (k, r['rmse']))
        ax[0].set_ylim(2.9, 4.25)
        ax[0].set_ylabel('V [V]')
        ax[1].set_ylabel('error [mV]')
        ax[1].set_xlabel('t [h]')
        ax[1].axhline(0, color='0.6', lw=0.5)
        ax[1].set_ylim(-150, 150)
        ax[0].set_title('(%s)' % 'abcd'[i], fontsize=8, loc='left', fontweight='bold')   # cell, protocol: in the caption
        print('(%s) cell %s, %.0f C, %.1f A | RMSE %s mV' % ('abcd'[i], c[1:], d0['T'], d0['I'], ', '.join(lab)))
        for x in ax:
            x.spines[['top', 'right']].set_visible(False)
        if i == 0:
            h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside upper center', ncol=4, fontsize=7, frameon=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + '.pdf')
    fig.savefig(a.out + '.png', dpi=300)
    for c in a.cells:
        print(c, '  '.join('%s %.1f mV' % (k, res[k][c]['rmse']) for k, *_ in CASES))
    print('written', a.out)


if __name__ == '__main__':
    main()
