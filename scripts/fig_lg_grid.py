"""
fig_lg_grid.py
==============
Paper figure for the second cell type: measured and predicted
voltage of the six held-out LG M50T cells, as fig_cases_grid.py for the Mohtat cells. Rows: experiment (drive-cycle
ageing, WLTP part of the first cycles; 1C cycling, 1C part of the first cycle); columns: chamber temperature
(10, 25, 40 C). Each held-out test is the one of the config "heldout_tests" (srcs/study.Study.heldout_specs); the
measured voltage is drawn as recorded and the model with its OCV hysteresis term (V + Vh). Panel RMSE over the
scored samples (as heldout_summary.py) go to stdout. Below the voltages, the RMSE of each held-out cell, grouped by chamber temperature, as panel (a) of
fig_cases_errors.py (runs/studies/<study>/heldout_val_rmse.csv); --bar-cases sets its cases (default: those drawn).

    python scripts/fig_lg_grid.py [--cases A:lgA B:lgB C:lgC] [--bar-cases ...]
            [--out runs/fig_lg_grid]
"""
import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import matplotlib.ticker            # noqa: E402
import numpy as np                  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from fig_cases_voltage import N_MARK, mm          # noqa: E402  (also sets the style and cwd)
from fig_cases_grid import STYLE                  # noqa: E402
import fig_cases_errors as FE                     # noqa: E402
import fig_cases_heldout as H                     # noqa: E402
from srcs import identification as ident, study as S   # noqa: E402

CELLS = ['E4B', 'E4E', 'E4G', 'E5B', 'E5E', 'E5G']
ROWS = ['drive cycle (WLTP)', '1C cycling']
COLS = ['10 °C', '25 °C', '40 °C']
NR, NC = len(ROWS), len(COLS)
LEAD = 0.25     # [h] of the initial rest kept before the first discharge


def simulate(study, cells):
    st = S.load(os.path.join(ROOT, 'configs', study + '.json'))
    em = st.em()                                   # also sets ident.HYSTERESIS from the study
    g = st.load('dynamics.json')['groups']
    out = {}
    for t in ident.prepare_tests(st.ds, st.heldout_specs(cells), st.windows(), em.ocp):
        gg, q = ident.test_groups(t, g, em.ocp)
        V = ident.add_bias(em.simulate(gg, t['I'], t['dt'], T=t['T'], q_start=q), t, gg, q)[0].cpu().numpy()
        Vh = np.asarray(t.get('Vh', 0.0)) * np.ones_like(V)
        m = t['mask']
        out[t['cell']] = dict(t=t['t'] / 3600, Vd=np.asarray(t.get('V_meas', t['V'])), V=V + Vh, m=m,
                              T=float(np.mean(t['T']) - 273.15),
                              rmse=1e3 * float(np.sqrt(np.mean((V - t['V'])[m] ** 2))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--cases', nargs='+', default=['A:lgA', 'B:lgB', 'C:lgC'], help='KEY:study pairs (keys of STYLE)')
    ap.add_argument('--bar-cases', nargs='+', help='KEY:study pairs of the error bars (default: --cases)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'fig_lg_grid'))
    a = ap.parse_args()
    cases = [(k, s) + STYLE[k] for k, s in (c.split(':', 1) for c in a.cases)]
    cache = a.out + '_sim.pkl'
    key = (tuple(CELLS), tuple((k, s) for k, s, *_ in cases))
    res = None
    if os.path.exists(cache):
        ck, cr = pickle.load(open(cache, 'rb'))
        res = cr if ck == key else None
    if res is None:
        res = {k: simulate(s, CELLS) for k, s, *_ in cases}
        pickle.dump((key, res), open(cache, 'wb'))
    fig = plt.figure(figsize=(180 * mm, 165 * mm), layout='constrained')
    sf = fig.subfigures(2, 1, height_ratios=[2.0, 1.0])
    ax = sf[0].subplots(NR, NC, sharex='row', sharey=True)
    for i, c in enumerate(CELLS):
        x = ax.flat[i]
        d0 = res[cases[0][0]][c]
        # the rest at the top before the first discharge (about 4 h) is cut, LEAD h kept before it
        moved = np.flatnonzero(d0['Vd'] < np.maximum.accumulate(d0['Vd']) - 0.05)   # the first discharge step
        t0 = max(float(d0['t'][moved[0]]) - LEAD, float(d0['t'][0])) if len(moved) else float(d0['t'][0])
        w = d0['t'] >= t0
        x.plot(d0['t'][w] - t0, d0['Vd'][w], color='k', lw=0.9, label='measured')
        lab = []
        n = int(w.sum())
        for ik, (k, _, col, mk, name) in enumerate(cases):
            r = res[k][c]
            step = max(n // N_MARK, 1)
            x.plot(r['t'][w] - t0, r['V'][w], lw=0.8, ls='--', label=name, color=col, marker=mk,
                   markevery=(ik * step // len(cases), step), ms=2.3, mfc='white', mec=col, mew=0.7)
            lab.append('%s %.1f' % (k, r['rmse']))
        L = 'abcdef'[i]
        x.set_title('(%s)' % L, fontsize=8, loc='left', fontweight='bold')
        if i < NC:
            x.annotate(COLS[i], xy=(0.5, 1), xycoords='axes fraction', xytext=(0, 15), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        if i % NC == 0:
            x.annotate(ROWS[i // NC], xy=(0, 0.5), xycoords='axes fraction', xytext=(-42, 0), textcoords='offset points',
                       rotation=90, ha='right', va='center', fontsize=8, fontweight='bold')
            x.set_ylabel('V [V]')
        x.set_ylim(2.45, 4.25)
        x.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
        x.spines[['top', 'right']].set_visible(False)
        if i >= NC * (NR - 1):
            x.set_xlabel('t [h]')
        print('(%s) cell %s, %.1f C mean | RMSE %s mV' % (L, c, d0['T'], ', '.join(lab)))
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside upper center', ncol=4, fontsize=7, frameon=False)
    bc = [c.split(':', 1) for c in (a.bar_cases or a.cases)]
    D = {k: H.heldout(s).set_index('cell') for k, s in bc}
    axb = sf[1].subplots()
    FE.bars(axb, bc, D, plt.rcParams['font.size'] - 1, temps=(10, 25, 40))
    axb.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))    # LG errors < 40 mV: ticks every 10
    axb.set_title('(g)', fontsize=8, loc='left', fontweight='bold')
    if [k for k, _ in bc] != [k for k, *_ in cases]:     # bars with cases not drawn above: their own legend
        hb, _ = axb.get_legend_handles_labels()
        axb.legend(hb, ['case %s' % k for k, _ in bc], fontsize=6, frameon=False, ncol=len(bc), loc='upper left')
    for k, s in bc:
        print('bars %s (%s): %s | mean %.1f mV' % (k, s, ', '.join('%s %.1f' % (c, v) for c, v in D[k].rmse_mV.items()),
                                                   D[k].rmse_mV.mean()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + '.pdf')
    fig.savefig(a.out + '.png', dpi=300)
    print('written', a.out)


if __name__ == '__main__':
    main()
