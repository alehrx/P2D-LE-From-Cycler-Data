"""
hysteresis_c10_gitt.py
======================
The OCV hysteresis of a silicon-graphite cell from its slow cycle and its rests (LG M50T): for each
window cell (step1 "window_cells", else split "dynamics") the slow cycle (step1 "ocv_test", C/10) gives a discharge leg
V_dis(q) and a charge leg V_ch(q) on the axis q of the charge discharged from the top, and the rest test (step1
"rest_test", GITT) the rested voltage V_rest(q) at the end of each rest after a discharge pulse, on the discharge
branch. With the same overpotential eta on the two legs of the slow cycle (V_dis = OCV_dis - eta, V_ch = OCV_ch + eta)
and OCV_dis = V_rest, the charge branch lies

    dV(q) = OCV_ch - OCV_dis = V_ch + V_dis - 2 V_rest

above the discharge branch. Each cell's points are mapped to the bulk anode stoichiometry x = x100 - q / Q_n of its
window (windows.json), averaged in --bins bins of x over the cells, clipped at 0 (a negative value is noise: the
rests are not fully relaxed) and written to runs/studies/<study>/hysteresis.json {"x", "dV"} for the config option
model.hysteresis (srcs/identification.hysteresis_voltage), with a figure.
    python scripts/hysteresis_c10_gitt.py configs/lgB.json [--bins 15]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
from srcs import study as S          # noqa: E402


def legs(ds, cell, test, i_min):
    """Discharge and charge legs of the slow cycle on the axis of the charge from the top (A h)."""
    df, _ = ds.load(cell, test)
    t, I, V = df['t_s'].to_numpy(), df['I_A'].to_numpy(), df['V_V'].to_numpy()
    q = np.cumsum(I * np.gradient(t)) / 3600.0
    d, c = I > i_min, I < -i_min
    qc = q[c] - q[c][-1]                    # the charge leg ends at the top: 0 there, growing backwards
    return (q[d], V[d]), (qc[::-1], V[c][::-1])


def rests(ds, cell, test, min_s):
    """Voltage at the end of each rest of at least min_s that follows a discharge, and its q from the top."""
    df, _ = ds.load(cell, test)
    t, I, V = df['t_s'].to_numpy(), df['I_A'].to_numpy(), df['V_V'].to_numpy()
    q = np.cumsum(I * np.gradient(t)) / 3600.0
    r = np.abs(I) < 0.01
    out, k = [], 0
    while k < len(I):
        if r[k]:
            j = k
            while j < len(I) and r[j]:
                j += 1
            if t[j - 1] - t[k] >= min_s and k > 0 and I[k - 1] > 0.01:
                out.append((q[j - 1], V[j - 1]))
            k = j
        else:
            k += 1
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config')
    ap.add_argument('--bins', type=int, default=15)
    a = ap.parse_args()
    st = S.load(os.path.join(ROOT, a.config) if not os.path.isabs(a.config) else a.config)
    s1 = st.cfg['step1']
    cells = list(s1.get('window_cells', st.split['dynamics']))
    W = json.load(open(st.path('windows.json')))
    i_min = 0.6 * float(st.ds.window_data.get('ocv_current_A', 0.5)) if hasattr(st.ds, 'window_data') else 0.3
    X, D = [], []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c in cells:
        (qd, Vd), (qc, Vc) = legs(st.ds, c, s1.get('ocv_test', 'C10'), i_min)
        R = rests(st.ds, c, s1.get('rest_test', 'GITT'), float(s1.get('rest_min_s', 1800)))
        R = R[(R[:, 0] > max(qd.min(), qc.min()) + 0.05) & (R[:, 0] < min(qd.max(), qc.max()) - 0.05)]
        dV = np.interp(R[:, 0], qc, Vc) + np.interp(R[:, 0], qd, Vd) - 2.0 * R[:, 1]
        x = W[c]['x100'] - R[:, 0] / W[c]['Q_n']
        X.append(x)
        D.append(dV)
        ax[0].plot(R[:, 0], 1e3 * dV, 'o-', ms=3, label=c)
        print('%s: %d rests, dV mean %.1f mV, range %.1f..%.1f mV' % (c, len(x), 1e3 * dV.mean(), 1e3 * dV.min(), 1e3 * dV.max()))
    X, D = np.concatenate(X), np.concatenate(D)
    edges = np.linspace(X.min(), X.max(), a.bins + 1)
    k = np.clip(np.digitize(X, edges) - 1, 0, a.bins - 1)
    xc = np.array([X[k == i].mean() for i in range(a.bins) if np.any(k == i)])
    dc = np.array([D[k == i].mean() for i in range(a.bins) if np.any(k == i)])
    dc = np.clip(dc, 0.0, None)
    o = np.argsort(xc)
    out = dict(x=[float(v) for v in xc[o]], dV=[float(v) for v in dc[o]], cells=cells,
               note='OCV_ch - OCV_dis = V_ch + V_dis - 2 V_rest (C/10 legs, GITT rests), binned in x, clipped at 0 '
                    '(scripts/hysteresis_c10_gitt.py)')
    json.dump(out, open(st.path('hysteresis.json'), 'w'), indent=1)
    ax[1].plot(X, 1e3 * D, '.', color='0.6', label='rests of all cells')
    ax[1].plot(xc[o], 1e3 * dc[o], 'k-o', ms=3, label='binned, clipped at 0')
    ax[0].set(xlabel='charge from the top [A h]', ylabel='OCV_ch - OCV_dis [mV]')
    ax[1].set(xlabel='bulk anode stoichiometry x', ylabel='OCV_ch - OCV_dis [mV]')
    for x_ in ax:
        x_.axhline(0, color='0.7', lw=0.5)
        x_.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(st.path('hysteresis.png'), dpi=130)
    print('wrote', st.path('hysteresis.json'), '|', ' '.join('%.3f:%.1f' % (x, 1e3 * d) for x, d in zip(xc[o], dc[o])))


if __name__ == '__main__':
    main()
