"""
error_bins.py
=============
Where the calibrated models err on the held-out cells: the voltage error (model - data) of every scored
sample of the held-out first cycles (VAL), binned by state of charge (10 % bins) and by C-rate, per chamber
temperature, for both cells. Median and 5-95 % band per bin.

SOC = 1 - q / q_ocv, q the charge discharged from the top of the cell's window (the test's q_start plus the
counted charge), q_ocv the window's C/20 (Mohtat) or C/10 (LG) capacity; C-rate = |I| / 5 A (both cells
are 5 A.h), rests (< C/50) in their own bin.

    python scripts/error_bins.py configs/fitA.json configs/fitB.json configs/caseC.json [--cpu N] [--out runs]

The simulations are cached per study (runs/studies/<name>/figures/error_bins.npz); delete it to rerun.
Writes fig_error_bins.png/.pdf to --out (default runs/).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
if '--cpu' in sys.argv:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch                                                       # noqa: E402

from srcs import identification as ident, study as S                # noqa: E402

plt.style.use(os.path.join(ROOT, 'journal.mplstyle'))
mm = 1 / 25.4
Q_NOM = 5.0
SOC_EDGES = np.linspace(0, 1, 11)
C_EDGES = np.array([0.0, 0.02, 0.25, 0.5, 1.0, 1.5, 2.2])
C_LABELS = ['rest', '<0.25', '0.25-0.5', '0.5-1', '1-1.5', '1.5-2']
GROUPS = {'mohtat': (('-5 °C', '#2b6cb0'), ('25 °C', '#2f855a'), ('45 °C', '#c53030')),
          'lgm50': (('10 °C', '#2b6cb0'), ('25 °C', '#2f855a'), ('40 °C', '#c53030'))}


def tgroup(T_C, ds):
    if ds == 'lgm50':
        return '10 °C' if T_C < 17 else ('40 °C' if T_C > 32 else '25 °C')
    return '-5 °C' if T_C < 10 else ('45 °C' if T_C > 37 else '25 °C')


def simulate(cfg):
    st = S.load(cfg)
    cache = st.path(os.path.join('figures', 'error_bins.npz'))
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return st.ds.name, {k: z[k] for k in z.files}
    em = st.em()
    g = st.load('dynamics.json')['groups']
    fit = list(st.split['dynamics'])
    if 'validation' in st.split and st.split['validation']:
        held = list(st.split['validation'])
    else:
        held = [c for c in st.ds.cells if c not in fit and c not in st.split.get('exclude', [])]
    held = [c for c in held if st.ds.available(c, 'VAL')]
    val = next(t for t in st.cfg['dynamics']['tests'] if t['test'] == 'VAL')
    specs = [dict({k: v for k, v in val.items() if k == 'coarsen'}, cell=c, test='VAL') for c in held]
    lg_parts = bool(st.cfg.get('heldout_tests'))   # LG scores the WLTP / 1C part (config "heldout_tests"),
    if lg_parts:                                    # as heldout_val_rmse.csv; Mohtat (no such key) is unchanged
        specs = st.heldout_specs(held)
    E, SOC, CR, TG, CELL = [], [], [], [], []
    win = st.windows()
    for t in ident.prepare_tests(st.ds, specs, win, em.ocp):
        gg, q0 = ident.test_groups(t, g, em.ocp)
        V = em.simulate(gg, t['I'], t['dt'], T=t['T'], q_start=q0)
        V = (ident.add_bias(V, t, gg, q0) if lg_parts else V)[0].cpu().numpy()
        dt = np.broadcast_to(np.asarray(t['dt'], float), t['I'].shape)
        q = float(np.atleast_1d(q0)[0]) + np.r_[0.0, np.cumsum(t['I'][:-1] * dt[:-1])] / 3600
        m = t['mask']
        w = win[t['cell']]
        E.append(1e3 * (V - t['V'])[m])
        SOC.append((1 - q / float(w['q_ocv_Ah']))[m])
        CR.append((np.abs(t['I']) / Q_NOM)[m])
        T_C = float(np.mean(t['T']) - 273.15)
        TG += [tgroup(T_C, st.ds.name)] * int(m.sum())
        CELL += [t['cell']] * int(m.sum())
        print('%s %s %.1f C: rmse %.1f mV' % (st.name, t['cell'], T_C, np.sqrt(np.mean(E[-1] ** 2))), flush=True)
    out = dict(e=np.concatenate(E), soc=np.concatenate(SOC), crate=np.concatenate(CR), tg=np.array(TG), cell=np.array(CELL))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return st.ds.name, out


def band(x, e, edges):
    k = np.digitize(x, edges) - 1
    med, lo, hi, n = [], [], [], []
    for b in range(len(edges) - 1):
        v = e[k == b]
        n.append(len(v))
        if len(v) < 30:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
        else:
            med.append(np.median(v)); lo.append(np.percentile(v, 5)); hi.append(np.percentile(v, 95))
    return np.array(med), np.array(lo), np.array(hi), np.array(n)


def main():
    if '--cpu' in sys.argv:
        torch.set_num_threads(int(sys.argv[sys.argv.index('--cpu') + 1]))
    out = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(ROOT, 'runs')
    cfgs = [a for a in sys.argv[1:] if a.endswith('.json')]
    data = [simulate(c) for c in cfgs]
    fig, ax = plt.subplots(len(data), 2, figsize=(170 * mm, 62 * mm * len(data)), layout='constrained', squeeze=False,
                           gridspec_kw=dict(width_ratios=[1.25, 1]))
    title = {'mohtat': 'NMC532 pouch (Mohtat), held-out', 'lgm50': 'LG M50T, held-out'}
    for r, (ds, d) in enumerate(data):
        groups = GROUPS.get(ds, GROUPS['mohtat'])
        off = np.linspace(-0.25, 0.25, len(groups))
        for (tg, col), o in zip(groups, off):
            s = d['tg'] == tg
            if not s.any():
                continue
            med, lo, hi, n = band(d['soc'][s], d['e'][s], SOC_EDGES)
            xc = 100 * (0.5 * (SOC_EDGES[:-1] + SOC_EDGES[1:]) + 0.02 * o)
            ax[r, 0].errorbar(xc, med, yerr=[med - lo, hi - med], fmt='o-', color=col, ms=2.5, lw=0.8, elinewidth=0.6,
                              capsize=1.5, label='%s (%d cells)' % (tg, len(set(d['cell'][s]))))
            med, lo, hi, n = band(d['crate'][s], d['e'][s], C_EDGES)
            xc = np.arange(len(C_LABELS)) + o
            ax[r, 1].errorbar(xc, med, yerr=[med - lo, hi - med], fmt='o', color=col, ms=2.5, elinewidth=0.6, capsize=1.5)
        ax[r, 0].set_xlabel('state of charge [%]')
        ax[r, 1].set_xticks(range(len(C_LABELS)), C_LABELS, fontsize=6)
        ax[r, 1].set_xlabel('C-rate [1/h]')
        for a in ax[r]:
            a.axhline(0, color='0.6', lw=0.5)
            a.set_ylabel('model - data [mV]')
            a.spines[['top', 'right']].set_visible(False)
            a.set_ylim(-150, 150)
        ax[r, 0].legend(fontsize=6, frameon=False, ncol=3, loc='lower right')
        ax[r, 0].set_title('(%s) %s: by state of charge' % ('ac'[r] if r < 2 else '', title.get(ds, ds)), fontsize=8, loc='left')
        ax[r, 1].set_title('(%s) by C-rate' % ('bd'[r] if r < 2 else ''), fontsize=8, loc='left')
        for tg, _ in groups:
            s = d['tg'] == tg
            if s.any():
                print('%-7s %-6s: %7d samples, median %+.1f mV, 5-95 %% [%+.1f, %+.1f], rmse %.1f' % (
                    ds, tg, s.sum(), np.median(d['e'][s]), np.percentile(d['e'][s], 5), np.percentile(d['e'][s], 95),
                    np.sqrt(np.mean(d['e'][s] ** 2))))
    os.makedirs(out, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, 'fig_error_bins.' + ext))
    plt.close(fig)
    print('->', out)


if __name__ == '__main__':
    main()
