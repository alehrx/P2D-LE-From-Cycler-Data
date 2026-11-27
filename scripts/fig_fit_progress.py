"""
fig_fit_progress.py
===================
Fig. 8 of the paper: how the identification of case B brings the model onto the measurement:
    (a) sketch of the three phases in the space of the groups: pre-calibration from Sobol starts over the literature
        ranges, screening around the pre-calibrated point, final fit from starts in a box around it (toy cost);
    (b) the search: RMS of the scores [mV] (the square root of the sum of squares the Levenberg-Marquardt search
        minimises) of the best start and of the median of the starts, iteration by iteration, in the pre-calibration
        (lm.log of the centre study of --end-study; iteration 0 computed here) and in the final fit (lm.log of
        --end-study; iteration 0 computed here), with the three stages of (c)-(d);
    (c) the calibration test the final fit improves most (Mohtat: the 2C cycle at -5 C, cell 08): measured voltage
        and the model at the start of one path of the search, at an intermediate iteration and at the end, with
        their error, and an inset on the knee at the end of the first discharge;
    (d) the RMSE of each calibration test at the three stages.
The two fits have different scores (the pre-calibration fits only the tests at 25 C), so their costs are not on the
same footing. The path and its intermediate iteration come from the states kept by
    python scripts/fit_lm.py configs/fitB_trace.json --fresh --keep-states
(runs/studies/<study>/lm_state_<it>.npz): the intermediate iteration is the first whose best sum of squares falls
below the geometric mean of the first iteration's and the end's; the start is the Sobol start of the path best there.

    python scripts/fig_fit_progress.py [--study fitB_trace] [--end-study fitB] [--out figures/fig_fit_progress]
"""
import argparse
import glob
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import matplotlib.ticker as mt      # noqa: E402
import numpy as np                  # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402
from matplotlib.patches import Rectangle        # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'scripts')]
plt.style.use(os.path.join(ROOT, 'journal.mplstyle'))
from srcs import identification as ident, study as S          # noqa: E402
from srcs.optimizers import _Counted                         # noqa: E402
from fit_lm import start_box                                 # noqa: E402

mm = 1 / 25.4
MEAS = dict(color='0.72', lw=2.4, solid_capstyle='round', zorder=1)
STAGE = (('#E69F00', ':', 'o'), ('#56B4E9', '--', 's'), ('#0072B2', '-', 'D'))
PRE, SCR = '#009E73', '#CC79A7'                 # pre-calibration, screening
DENSE = {'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5}
GAP = 9                                         # width of the screening band in (b) [iterations]


def log_history(path):
    """(iteration, best sum r^2, median of the active, number of active starts) of the last run in an lm.log."""
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    last = max(i for i, ln in enumerate(lines) if 'fit_lm ' in ln and ' starts, ' in ln)
    pat = re.compile(r'lm iteration (\d+): (\d+) active.*best sum r\^2 ([\d.e+]+).*median of the active ([\d.e+]+)')
    out = [tuple(float(m.group(k)) for k in (1, 3, 4, 2)) for m in (pat.search(ln) for ln in lines[last:]) if m]
    return np.array(out)


KIND = {'HPPC': 'HPPC', 'C20': 'C/20', 'C10': 'C/10', 'GITT': 'GITT'}


def degC(T):
    return ('%g' % T).replace('-', '−') + ' °C'


def label(ax, letter, text=None):
    ax.text(0, 1.01, '(%s)' % letter, transform=ax.transAxes, fontweight='bold', ha='left', va='bottom')
    if text:
        ax.annotate(text, (0, 1.01), xycoords='axes fraction', xytext=(14, 0), textcoords='offset points',
                    ha='left', va='bottom')


def badge(ax, x, y, k, col, transform=None, size=6.5):
    ax.text(x, y, str(k), ha='center', va='center', color='white', fontweight='bold', fontsize=size, zorder=9,
            transform=transform or ax.transData, bbox=dict(boxstyle='round,pad=0.22,rounding_size=0.45', fc=col, ec='none'))


# ---------------------------------------------------------------------------------------------- (a) the sketch
def _valley(u1):
    return 0.16 + 0.66 * u1 + 0.3 * (u1 - 0.5) ** 2 - 0.06 * np.sin(5.0 * u1)


def _resid(U, u_opt):
    """Toy residuals of two groups: a curved valley (two compensating groups) with its minimum at u1 = u_opt."""
    U = np.atleast_2d(U)
    return np.stack([5.0 * (U[:, 1] - _valley(U[:, 0])), 0.9 * (U[:, 0] - u_opt)], 1)


def _descent(u, u_opt, n=400, dt=0.004):
    """Steepest-descent path on the toy cost (straight into the valley, then along it)."""
    p = [np.array(u, float)]
    for _ in range(n):
        u = p[-1]
        r = _resid(u, u_opt)[0]
        J = np.stack([(_resid(u + 1e-6 * e, u_opt)[0] - r) / 1e-6 for e in np.eye(2)], 1)
        g = J.T @ r
        p.append(np.clip(u - dt * g / max(1.0, np.linalg.norm(g) * dt / 0.01), 0.0, 1.0))
    return np.array(p)


def sketch(ax):
    from scipy.stats import qmc
    u_pre, u_fin, w, h = 0.5, 0.6, 0.2, 0.05
    P, E = np.array([u_pre, _valley(u_pre)]), np.array([u_fin, _valley(u_fin)])
    g = np.linspace(0, 1, 300)
    X, Y = np.meshgrid(g, g)
    Z = np.sum(_resid(np.stack([X.ravel(), Y.ravel()], 1), 0.55) ** 2, 1).reshape(X.shape)
    ax.contourf(X, Y, np.log10(Z + 2e-3), levels=12, cmap='Greys_r', alpha=0.25)
    ax.contour(X, Y, np.log10(Z + 2e-3), levels=12, colors='0.78', linewidths=0.25)
    # 1: pre-calibration, Sobol starts over the literature ranges (paths drawn for those outside the final box)
    S0 = qmc.Sobol(2, scramble=True, seed=4).random(16) * 0.92 + 0.04
    for s in S0:
        if np.all(np.abs(s - P) > w + 0.03) and not (s[0] > 0.7 and s[1] > 0.7):
            p = _descent(s, u_pre)
            ax.plot(p[:, 0], p[:, 1], color=PRE, lw=0.7, alpha=0.85, zorder=3)
        ax.plot(*s, 'o', color=PRE, ms=3.0, mec='white', mew=0.3, zorder=4)
    # 3: final fit, starts in a box of width 0.4 around the pre-calibrated point
    ax.add_patch(Rectangle(P - w, 2 * w, 2 * w, fc='white', alpha=0.35, ec='none', zorder=4))
    ax.add_patch(Rectangle(P - w, 2 * w, 2 * w, fc='none', ec=STAGE[2][0], lw=0.8, ls=(0, (3, 2)), zorder=5))
    S1 = P - w + 2 * w * (qmc.Sobol(2, scramble=True, seed=11).random(8) * 0.9 + 0.05)
    for s in S1:
        ax.plot(*s, 'o', color=STAGE[0][0], ms=3.2, mec='k', mew=0.3, zorder=6)
    # 2: screening in a small box around the pre-calibrated point
    ax.add_patch(Rectangle(P - h, 2 * h, 2 * h, fc=SCR, alpha=0.45, ec=SCR, lw=0.8, zorder=10))
    ax.plot(*P, '*', color=PRE, mec='k', mew=0.4, ms=9, zorder=11)
    ax.plot(*E, 'D', color=STAGE[2][0], mec='k', mew=0.4, ms=4.5, zorder=11)
    fs = 6.2
    ax.text(0.97, 0.03, 'literature ranges', ha='right', va='bottom', fontsize=fs, color='0.35', style='italic')
    badge(ax, 0.07, 0.94, '2.1', PRE)
    ax.text(0.145, 0.94, 'pre-calibration', ha='left', va='center', fontsize=fs)
    badge(ax, P[0] + h + 0.06, P[1] - h - 0.03, '2.2', SCR)
    badge(ax, P[0] - w + 0.04, P[1] + w + 0.05, '3', STAGE[2][0])
    ax.text(P[0] - w + 0.09, P[1] + w + 0.05, 'final fit', ha='left', va='center', fontsize=fs)
    ax.annotate('pre-calibrated\npoint', P, xytext=(0.2, 0.2), fontsize=fs, ha='center', va='top',
                arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3', shrinkB=4))
    ax.annotate('optimum', E, xytext=(0.84, 0.46), fontsize=fs, ha='center', va='top',
                arrowprops=dict(arrowstyle='-', lw=0.5, color='0.3', shrinkB=3))
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], xlabel='group $i$', ylabel='group $j$')
    label(ax, 'a', 'steps 2 and 3')


# ---------------------------------------------------------------------------------------------- (b) the search
def search(ax, pre_hist, pre_r0, hist, r0, marks, heads):
    """Best start and median of the starts: pre-calibration at x = iteration, final fit shifted by GAP."""
    n_pre = int(pre_hist[-1, 0])
    off = n_pre + GAP
    for hst, rr0, x0, col in ((pre_hist, pre_r0, 0, PRE), (hist, r0, off, '0.2')):
        it = np.r_[0, hst[:, 0]] + x0
        ax.plot(it, np.sqrt(np.r_[np.median(rr0 ** 2), hst[:, 2]]), color='0.6', lw=0.9, ls='--')
        ax.plot(it, np.sqrt(np.r_[np.min(rr0 ** 2), hst[:, 1]]), color=col, lw=1.2)
    ax.plot(n_pre, np.sqrt(pre_hist[-1, 1]), '*', color=PRE, mec='k', mew=0.4, ms=8, zorder=4)
    for (ix, rr), (col, _, mk) in zip(marks, STAGE):
        ax.plot(ix + off, rr, mk, color=col, ms=5, mec='k', mew=0.4, zorder=4)
    ax.axvspan(n_pre + 1.5, off - 1.5, color=SCR, alpha=0.18, lw=0)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, p: '%g' % v))
    ax.yaxis.set_minor_formatter(mt.FuncFormatter(lambda v, p: '%g' % v if round(v) in (20, 30, 50, 200, 500) else ''))
    lo = 0.8 * min(np.sqrt(pre_hist[:, 1].min()), np.sqrt(hist[:, 1].min()))
    hi = 1.6 * max(np.sqrt(np.median(pre_r0 ** 2)), np.sqrt(np.median(r0 ** 2)))
    ax.set_ylim(lo, hi)
    n_fin = int(hist[-1, 0])
    step = 10
    ticks = list(range(0, n_pre + 1, step)) + [off + k for k in range(0, n_fin + 1, step)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t if t <= n_pre else t - off) for t in ticks])
    ax.set(xlabel='iteration', ylabel='RMS of the scores [mV]', xlim=(-1, off + n_fin + 1))
    yh = lo * (hi / lo) ** 0.72                  # the phase headers, above the converged curves
    for x, y, k, col, txt in ((12.5, yh, '2.1', PRE, heads[0]), (off + 12.5, lo * (hi / lo) ** 0.45, '3', STAGE[2][0], heads[1])):
        badge(ax, x, y, k, col)
        ax.text(x + (2.0 if len(k) > 1 else 1.4), y, txt, ha='left', va='center', fontsize=6.2)
    badge(ax, (n_pre + off) / 2, hi / 1.3, '2.2', SCR)
    ax.text((n_pre + off) / 2, hi / 2.2, 'screening', rotation=90, ha='center', va='top', fontsize=6.2, color='0.25')
    ax.spines[['top', 'right']].set_visible(False)
    label(ax, 'b', 'the search')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--study', default='fitB_trace')
    ap.add_argument('--end-study', default='fitB', help='the study whose dynamics.json is the end of the fit')
    ap.add_argument('--mid', type=int, help='the intermediate iteration (default: see above)')
    ap.add_argument('--cycle', help='the test label of (c) (default: the one the fit improves most)')
    ap.add_argument('--val-name', help='name of the VAL cycle (default 2C for Mohtat, 1C otherwise)')
    ap.add_argument('--zoom', nargs=2, type=float, help='time window of the inset [h] (default: the end of the first '
                    'discharge); --zoom 0 0 for no inset')
    ap.add_argument('--inset', nargs=4, type=float, default=[0.45, 0.07, 0.42, 0.6], help='inset box (axes fraction)')
    ap.add_argument('--lead', type=float, default=0.25, help='time drawn before the first discharge [h] (a long initial rest is cut)')
    ap.add_argument('--all-cycles', dest='first_cycle', action='store_false', help='draw the whole test in (c), not '
                    'only its first cycle')
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'fig_fit_progress'))
    a = ap.parse_args()
    st = S.load(os.path.join(ROOT, 'configs', a.study + '.json'))
    em = st.em()
    d, o = st.cfg['dynamics'], st.cfg['lm']
    names = list(d['names'])
    fixed = {**st.thermal(d.get('thermal', 'reference')), **st.fixed_groups(d.get('fixed'))}
    tests = ident.prepare_tests(st.ds, st.expand('dynamics', d['tests'], 'dynamics'), st.windows(), em.ocp)
    rc = ident.ResidualCost(em, tests, names, fixed)
    rms = lambda X: np.sqrt(np.sum(np.square(rc.residuals(np.atleast_2d(X))), axis=1))   # [mV]
    C = _Counted(lambda X: np.zeros(len(X)), d['lb'], d['ub'], 0)
    mohtat = 'mohtat' in st.cfg['dataset'].lower()
    val_name = a.val_name or ('2C' if mohtat else '1C')

    end_st = S.load(os.path.join(ROOT, 'configs', a.end_study + '.json'))
    end = end_st.load('dynamics.json')
    x_end = np.array([end['groups'][n] for n in names])
    hist = log_history(end_st.path('lm.log'))
    F = {}
    for f in sorted(glob.glob(st.path('lm_state_*.npz'))):
        z = np.load(f)
        F[int(z['iteration'])] = (np.array(z['F']), np.array(z['U']))
    its = sorted(F)
    gm = np.sqrt(F[its[0]][0].min() * float(end['sum_r2']))
    mid = a.mid or next(i for i in its if F[i][0].min() < gm)
    b = int(np.argmin(F[mid][0]))
    x0 = np.asarray(start_box(o['center'], d, int(o['starts']), int(o.get('seed', 0)),
                              st.cfg.get('model', {}).get('theta_ref_C')), float)
    xs = [x0[b], C.phys(F[mid][1][b:b + 1])[0], x_end]
    cache = a.out + '_sim.pkl'          # the simulations, kept so that a change of layout does not rerun them
    key = (a.study, a.end_study, mid, tuple(t['label'] for t in tests), np.round(np.ravel(xs), 9).tobytes())
    ck = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else None
    if ck is None or ck[0] != key:
        ck = (key, rms(x0), {})                                      # iteration 0: every Sobol start
    r0, VM = ck[1], ck[2]
    rs = [float(r0[b]), float(np.sqrt(F[mid][0][b])), float(np.sqrt(end['sum_r2']))]
    labels = ['start', 'iteration %d' % mid, 'end (iteration %d)' % int(hist[-1, 0])]
    print('stages (RMS of the scores, mV):', list(zip(labels, np.round(rs, 1))))

    def sim(t, x):
        """Model voltage as measured (with the OCV hysteresis term where the model has one)."""
        g = {**fixed, **dict(zip(names, x))}
        gg, q = ident.test_groups(t, g, em.ocp)
        V = ident.add_bias(em.simulate(gg, t['I'], t['dt'], T=t['T'], q_start=q), t, gg, q)[0].double().cpu().numpy()
        return V + np.asarray(t.get('Vh', 0.0), float)

    def name(t):
        T = st.nominal_T(t['cell'])
        T = float(np.mean(t['T']) - 273.15) if T is None else T
        return KIND.get(t['test'], val_name), T

    # every calibration test at the three stages
    rows = []
    for t in tests:
        Vd = np.asarray(t.get('V_meas', t['V']), float)
        if t['label'] not in VM:
            VM[t['label']] = [sim(t, xp) for xp in xs]
        Vm = VM[t['label']]
        m = np.asarray(t['mask'], bool)
        e = [1e3 * np.sqrt(np.mean((v - Vd)[m] ** 2)) for v in Vm]
        kind, T = name(t)
        rows.append(dict(t=t, Vd=Vd, Vm=Vm, e=e, kind=kind, T=T))
        print('%-9s RMSE start / mid / end: %.1f / %.1f / %.1f mV' % (t['label'], *e))
    # the pre-calibration: its history and its Sobol starts (iteration 0), on its own tests; after the simulations
    # above, since loading another study changes the model settings shared by the module
    pre_st = S.load(os.path.join(ROOT, 'configs', o['center']['study'] + '.json'))
    pre_hist = log_history(pre_st.path('lm.log'))
    if '_pre_r0' not in VM:
        from scipy.stats import qmc
        pd_, po = pre_st.cfg['dynamics'], pre_st.cfg['lm']
        pem = pre_st.em()
        pfixed = {**pre_st.thermal(pd_.get('thermal', 'reference')), **pre_st.fixed_groups(pd_.get('fixed'))}
        ptests = ident.prepare_tests(pre_st.ds, pre_st.expand('dynamics', pd_['tests'], 'dynamics'), pre_st.windows(),
                                     pem.ocp)
        prc = ident.ResidualCost(pem, ptests, list(pd_['names']), pfixed)
        pC = _Counted(lambda X: np.zeros(len(X)), pd_['lb'], pd_['ub'], 0)
        U = qmc.Sobol(len(pd_['names']), scramble=True, seed=int(po.get('seed', 0))).random(int(po['starts'])) * (1 - 1e-3)
        VM['_pre_r0'] = np.sqrt(np.sum(np.square(prc.residuals(pC.phys(U))), axis=1))
    pre_r0 = VM['_pre_r0']
    pre_cells = sorted({c for t in pre_st.cfg['dynamics']['tests'] for c in t.get('cells', [])})

    pickle.dump(ck, open(cache, 'wb'))
    rows.sort(key=lambda r: (r['kind'] != val_name, r['T'] if r['kind'] == val_name else 0))
    sel = next(r for r in rows if r['t']['label'] == a.cycle) if a.cycle else max(rows, key=lambda r: r['e'][0] - r['e'][2])
    Ts = sorted({r['T'] for r in rows if r['kind'] == val_name})
    cell_txt = lambda c: c[1:] if mohtat else c                                             # noqa: E731
    heads = ('pre-calibration:\n%d groups,\n%s at 25 °C' % (len(pre_st.cfg['dynamics']['names']) - ('R_contact' in pre_st.cfg['dynamics']['names']),
                                                         'cell ' + ', '.join(cell_txt(c) for c in pre_cells)),
             'final fit:\n%d groups,\n%s to %s' % (len(names), degC(Ts[0]), degC(Ts[-1])))

    with plt.rc_context(DENSE):
        fig = plt.figure(figsize=(170 * mm, 122 * mm), layout='constrained')
        top, bot = fig.subfigures(2, 1, height_ratios=[1.0, 1.08], hspace=0.04)
        gt = top.add_gridspec(1, 2, width_ratios=[1.0, 1.7])
        sketch(top.add_subplot(gt[0, 0]))
        top.axes[0].set_box_aspect(1)
        marks = ((0, rs[0]), (mid, rs[1]), (int(hist[-1, 0]), rs[2]))
        search(top.add_subplot(gt[0, 1]), pre_hist, pre_r0, hist, ck[1], marks, heads)

        gb = bot.add_gridspec(2, 2, height_ratios=[3, 1.25], width_ratios=[1.7, 0.95], hspace=0.0)
        # (c) the test the final fit improves most
        t, V, Vms = sel['t'], sel['Vd'], sel['Vm']
        x = np.asarray(t['t'], float) / 3600
        av = bot.add_subplot(gb[0, 0])
        ae = bot.add_subplot(gb[1, 0], sharex=av)
        av.tick_params(labelbottom=False)
        av.plot(x, V, **MEAS, label='measured')
        for Vm, (col, ls, _), lab in zip(Vms, STAGE, labels):
            av.plot(x, Vm, color=col, ls=ls, lw=1.0, label=lab)
            ae.plot(x, np.clip(1e3 * (Vm - V), -120, 120), color=col, ls=ls, lw=0.9)
        if a.zoom is None:          # the knee at the end of the first discharge, down to its lowest voltage
            k_lo = int(np.argmin(np.where(x < 0.6, V, np.inf)))
            a.zoom = (x[k_lo] - 0.07, x[k_lo] + 0.004)
        x0z, x1z = a.zoom
        if x1z > x0z:
            w = (x > x0z) & (x < x1z)
            ins = av.inset_axes(list(a.inset))
            ins.plot(x[w], V[w], **MEAS)
            for Vm, (col, ls, _) in zip(Vms, STAGE):
                ins.plot(x[w], Vm[w], color=col, ls=ls, lw=1.0)
            lo = min(v[w].min() for v in [V] + Vms)
            hi = max(v[w].max() for v in [V] + Vms)
            ins.set(xlim=(x0z, x1z), ylim=(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)), xticks=[])
            yt = mt.MaxNLocator(3).tick_values(lo, hi)
            ins.set_yticks([v for v in yt if lo + 0.1 * (hi - lo) < v < hi])     # none on the lower edge
            ins.tick_params(labelsize=5.5, length=2, pad=1)
            ins.tick_params(which='minor', length=0)
            for sp in ins.spines.values():
                sp.set_linewidth(0.6)
                sp.set_edgecolor('0.35')
            av.indicate_inset_zoom(ins, edgecolor='0.35', linewidth=0.6, alpha=1.0)
        ae.axhline(0, color='0.5', lw=0.5, zorder=0)
        ae.set_ylim(-125, 125)
        k_on = np.flatnonzero(np.abs(np.asarray(t['I'], float)) > 0.1)
        I = np.asarray(t['I'], float)
        x_lo = x[k_on[0]] - a.lead if len(k_on) and x[k_on[0]] > a.lead + 0.25 else x[0]   # a long initial rest: cut it
        x_hi = x[-1]
        if a.first_cycle and len(k_on):              # stop before the second discharge
            s0 = np.sign(I[k_on[0]])
            k_ch = k_on[np.sign(I[k_on]) != s0]
            k_2 = k_on[(np.sign(I[k_on]) == s0) & (k_on > (k_ch[0] if len(k_ch) else len(I)))]
            if len(k_2):
                x_hi = x[k_2[0]] - 0.02
        av.set_xlim(x_lo, x_hi + 0.01 * (x_hi - x_lo))
        ae.yaxis.set_major_locator(mt.MultipleLocator(100))
        lo_v = min(np.nanmin(np.where(np.isfinite(v), v, np.inf)[x <= av.get_xlim()[1]]) for v in [V] + Vms)
        av.set_ylim(min(np.nanmin(V) - 0.12, lo_v - 0.04), max(np.nanmax(v) for v in [V] + Vms) + 0.05)
        ae.set_xlabel('t [h]')
        av.set_ylabel('V [V]')
        ae.set_ylabel('error [mV]')
        what = '%s cycle' % sel['kind'] if sel['kind'] == val_name else sel['kind']
        label(av, 'c', '%s at %s (cell %s)' % (what, degC(sel['T']), cell_txt(t['cell'])))
        for ax in (av, ae):
            ax.spines[['top', 'right']].set_visible(False)
        # (d) RMSE of every calibration test at the three stages
        ac = bot.add_subplot(gb[:, 1])
        for k, r in enumerate(rows):
            e = r['e']
            y = len(rows) - 1 - k
            ac.plot([e[0], e[2]], [y, y], color='0.8', lw=2.0, zorder=1, solid_capstyle='round')
            for ev, ms, (col, _, mk) in zip(e, (6.5, 5.2, 4.4), STAGE):      # nested where they overlap
                ac.plot(ev, y, mk, color=col, ms=ms, mec='k', mew=0.4, zorder=3)
        ac.set_yticks(range(len(rows)))
        ac.set_yticklabels(['%s, %s' % (r['kind'], degC(r['T'])) for r in rows][::-1])
        ac.tick_params(axis='y', which='both', length=0)
        ac.set(xlabel='RMSE [mV]', xlim=(0, 1.1 * max(max(r['e']) for r in rows)), ylim=(-0.6, len(rows) - 0.4))
        ac.grid(axis='x', color='0.92', lw=0.6)
        ac.set_axisbelow(True)
        label(ac, 'd', 'calibration tests')
        ac.spines[['top', 'right', 'left']].set_visible(False)
        st = [Line2D([], [], color=col, ls=ls, marker=mk, ms=4.5, mec='k', mew=0.4, lw=1.0, label=lab)
              for (col, ls, mk), lab in zip(STAGE, ('final-fit initial candidates', 'final fit: iteration %d' % mid,
                                                    'final fit: end'))]
        hd = [Line2D([], [], **{**MEAS, 'zorder': None}, label='measured'),
              Line2D([], [], color=PRE, ls='', marker='o', ms=3.5, mec='white', mew=0.3, label='initial candidates'),
              Line2D([], [], color=PRE, ls='', marker='*', ms=7, mec='k', mew=0.4, label='pre-calibrated point'),
              *st,
              Line2D([], [], color='0.2', lw=1.2, label='best candidate'),
              Line2D([], [], color='0.6', lw=0.9, ls='--', label='median of the candidates')]
        fig.legend(handles=hd, loc='outside lower center', ncol=4, frameon=False, columnspacing=1.5, handlelength=2.2)
        fig.savefig(a.out + '.png')
        fig.savefig(a.out + '.pdf')
    print('written', a.out)


if __name__ == '__main__':
    main()
