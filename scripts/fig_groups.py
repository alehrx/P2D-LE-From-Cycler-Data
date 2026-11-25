"""
fig_groups.py
=============
Fig. 4 of the paper (Section 2.3): the window groups, for the pouch cell.

One row of three panels:
  (a) the stoichiometries of the two electrodes against the discharged charge: the window x0-x100, y100-y0,
      and the electrode capacities Q_n, Q_p as the charge each electrode would need to cover 0..1
      (extensions outside the window, shaded);
  (b) the open-circuit potentials U_n(x), U_p(y), solid over the part the cell uses;
  (c) the open-circuit voltage U_p(y) - U_n(x) over the window.
The window and the OCPs are those of one calibration cell of the study (--study, --cell; per-cell windows).
panel_a (MPM-LE schematic) and panel_b (one particle) are kept but no longer drawn.

    python scripts/fig_groups.py [--study fitB] [--cell C07] [--out figures/fig_groups]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.patches import Circle, Rectangle      # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from srcs import study as S                          # noqa: E402

plt.style.use(os.path.join(ROOT, 'journal.mplstyle'))
AN, ANL = '#B03A2E', '#F6E3E1'
CA, CAL = '#1E8449', '#E3F1E7'
EL = '#E1EEF7'
mm = 1 / 25.4
TEMPERATURE_PANELS = False      # the temperature row (e, f); needs a third subfigure row if switched on


def label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, '(%s)' % s, transform=ax.transAxes, fontweight='bold', va='bottom', ha='right')


def panel_a(ax):
    ax.set_xlim(-0.6, 11.0)
    ax.set_ylim(-1.55, 2.55)
    ax.set_aspect('equal')
    ax.axis('off')
    yb, yt = 0.0, 1.0
    xa, xs, xc, xe = 0.3, 4.6, 5.9, 10.2           # anode | separator | cathode, collector width 0.25
    ax.add_patch(Rectangle((xa - 0.25, yb), 0.25, yt - yb, fc='#7F7F7F', ec='k', lw=0.6))
    ax.add_patch(Rectangle((xe, yb), 0.25, yt - yb, fc='#C8C8C8', ec='k', lw=0.6))
    ax.add_patch(Rectangle((xa, yb), xs - xa, yt - yb, fc=EL, ec='k', lw=0.6))
    ax.add_patch(Rectangle((xc, yb), xe - xc, yt - yb, fc=EL, ec='k', lw=0.6))
    ax.add_patch(Rectangle((xs, yb), xc - xs, yt - yb, fc='white', ec='k', lw=0.6, hatch='////'))
    N = 5
    for (x0, x1, col, colL, bus, sgn) in ((xa, xs, AN, ANL, r'$\varphi_{s,n}$', -1), (xc, xe, CA, CAL, r'$\varphi_{s,p}$', 1)):
        w = (x1 - x0) / N
        xs_ = x0 + w * (np.arange(N) + 0.5)
        for i in range(1, N):
            ax.plot([x0 + i * w] * 2, [yb, yt], ls='--', lw=0.6, color=col)
        d_sep = (xs_ - x0) / (x1 - x0) if sgn < 0 else (x1 - xs_) / (x1 - x0)   # 1 at the separator
        cmap = plt.cm.Reds if sgn < 0 else plt.cm.Greens
        for x, d in zip(xs_, d_sep):
            # radial stoichiometry in a discharge: anode depleted, cathode enriched at the surface, more near the separator
            g = 0.15 + 0.45 * d ** 2
            c0, c1 = (0.65, 0.65 - g) if sgn < 0 else (0.08, 0.08 + g)
            for k in range(20, 0, -1):
                rr = k / 20
                ax.add_patch(Circle((x, 0.62), 0.30 * rr, fc=cmap(c0 + (c1 - c0) * rr ** 2), ec='none'))
            ax.add_patch(Circle((x, 0.62), 0.30, fc='none', ec=col, lw=0.9))
            ax.plot([x, x], [0.92, 1.85], color=col, lw=0.8)
        ax.plot([xs_[0], xs_[-1]], [1.85, 1.85], color=col, lw=2.0)
        ax.text(xs_.mean(), 1.95, bus, color=col, ha='center', va='bottom')
        xt = xs_[0] - 0.85 if sgn < 0 else xs_[-1] + 0.85
        ax.plot([xs_[0] if sgn < 0 else xs_[-1], xt], [1.85, 1.85], color='k', lw=0.8)
        ax.plot(xt, 1.85, 'o', mfc='white', mec='k', ms=5)
        ax.text(xt, 2.05, r'$-$' if sgn < 0 else r'$+$', ha='center', va='bottom')
        ax.text(xs_[0] + 0.22, 1.40, r'$I_{0,%s}$' % ('n' if sgn < 0 else 'p'), color=col, ha='left', va='center')
        ax.text(xs_[1] - 0.05 if sgn < 0 else xs_[-2] + 0.05, 0.15, r'$\theta_%s$' % ('n' if sgn < 0 else 'p'),
                color=col, ha='center', va='center')
        ax.text((x0 + x1) / 2, 0.15, r'$\varepsilon_%s$' % ('n' if sgn < 0 else 'p'), ha='center', va='center')
    ax.text((xs + xc) / 2, 0.15, r'$\varepsilon_s$', ha='center', va='center', bbox=dict(fc='white', ec='none', pad=1))
    ax.text((xa + xe) / 2, -0.38, r'electrolyte: Bruggeman exponent $b$', ha='center', va='center', color='#0072B2')
    for (u, v, lab) in ((xa, xs, r'$L_n$'), (xs, xc, r'$L_s$'), (xc, xe, r'$L_p$')):
        ax.plot([u, v], [-0.85, -0.85], color='k', lw=0.6)
        for z in (u, v):
            ax.plot([z, z], [-0.95, -0.75], color='k', lw=0.6)
        ax.text((u + v) / 2, -1.05, lab, ha='center', va='top')


C_LIM = (0.70, 0.87)                           # stoichiometry range of panel (a), shared by the plot and the shading


def panel_b(ax_p, ax_c):
    """Upper-right quarter of the particle above the profile plot, on the same r/R axis: the colour of the quarter
    at radius r is the stoichiometry the t = 0.6 theta curve gives at that r."""
    # radial profile: sphere, constant delithiation flux, explicit FV in tau = t / theta
    n = 101
    rr = np.linspace(0, 1, n)
    dr = rr[1] - rr[0]
    c = np.full(n, 0.85)
    flux = 0.06                                # -dc/dr at r = 1 (dimensionless)
    dt = 0.2 * dr ** 2
    snaps, taus, t = [], [0.01, 0.12, 0.6], 0.0
    for tt in taus:
        while t < tt:
            lap = np.zeros(n)
            lap[1:-1] = (c[2:] - 2 * c[1:-1] + c[:-2]) / dr ** 2 + (c[2:] - c[:-2]) / (rr[1:-1] * dr)
            lap[0] = 6 * (c[1] - c[0]) / dr ** 2
            ghost = c[-2] - 2 * dr * flux
            lap[-1] = (ghost - 2 * c[-1] + c[-2]) / dr ** 2 + (c[-2] - ghost) / (2 * dr)
            c = c + dt * lap
            t += dt
        snaps.append(c.copy())
    fs = plt.rcParams['font.size'] - 1
    from matplotlib.colors import Normalize
    from matplotlib.patches import Wedge
    norm = Normalize(*C_LIM)
    cmap = plt.cm.Reds
    shade = snaps[-1]
    # quarter particle LEFT of the plot: centre at the bottom-right corner, radius pointing up along the plot's r axis
    for k in range(len(rr) - 1, 0, -1):         # one ring per node, coloured by c_s(r) at 0.6 theta
        ax_p.add_patch(Wedge((1, 0), rr[k], 90, 180, fc=cmap(norm(shade[k])), ec='none'))
    ax_p.add_patch(Wedge((1, 0), 1.0, 90, 180, fc='none', ec=AN, lw=1.0))
    for a in np.deg2rad([108, 135, 162]):       # the surface flux j
        ax_p.annotate('', xy=(1 + 1.2 * np.cos(a), 1.2 * np.sin(a)), xytext=(1 + np.cos(a), np.sin(a)),
                      arrowprops=dict(arrowstyle='-|>', color=AN, lw=0.9, mutation_scale=7), annotation_clip=False)
    ax_p.text(1 + 0.98 * np.cos(np.deg2rad(150)) - 0.05, 1.12 * np.sin(np.deg2rad(150)) + 0.12, r'$j$', color=AN,
              ha='right', va='bottom')
    ax_p.set_xlim(0, 1)
    ax_p.set_ylim(0, 1)
    ax_p.axis('off')
    # the profiles: r/R horizontal from the particle centre (left, next to the quarter's centre) to the surface
    for s_, col, tau, yl, va in zip(snaps, ('0.65', '0.4', cmap(0.85)), taus, (0.853, 0.840, 0.765),
                                    ('bottom', 'top', 'bottom')):
        ax_c.plot(rr, s_, color=col, lw=1.4 if tau < 0.5 else 2.0)
        ax_c.text(0.04, yl, r'$t=%g\,\theta$' % tau, color=col, ha='left', va=va, fontsize=fs)
    from matplotlib.cm import ScalarMappable
    cax = ax_c.inset_axes([1.06, 0.0, 0.06, 1.0])     # follows the square box of the plot
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_ticks([])
    cb.set_label(r'shading at $t=0.6\,\theta$', fontsize=fs)
    ax_c.axhline(0.85, color='0.5', lw=0.6, ls='--')
    # the surface gradient set by the flux: -D_s dc/dr = j/F
    s1 = snaps[1]
    g = (s1[-1] - s1[-6]) / (rr[-1] - rr[-6])
    rt = np.array([0.72, 1.0])
    ax_c.plot(rt, s1[-1] + g * (rt - 1.0), color='k', lw=0.8, ls=(0, (2, 1.5)))
    ax_c.annotate(r'$-D_s\,\partial c_s/\partial r=j/F$', xy=(0.86, s1[-1] + g * (0.86 - 1.0)), xytext=(0.5, 0.795),
                  fontsize=fs, ha='center', va='center', arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    ax_c.set(xlabel=r'$r/R$', ylabel=r'$c_s/c_\mathrm{max}$', xlim=(0, 1), ylim=C_LIM)
    for sp in ('top', 'right'):
        ax_c.spines[sp].set_visible(False)
    ax_c.tick_params(top=False, right=False, which='both')
    ax_c.set_box_aspect(1)                      # square: the quarter beside it has the same radius as the r axis


def panel_c(ax, x100, y100, Qn, Qp, Q):
    q_lo, q_hi = -y100 * Qp, x100 * Qn                     # where y = 0 and x = 0
    ax.axvspan(q_lo - 0.3, 0, color='0.93', lw=0)
    ax.axvspan(Q, (1 - y100) * Qp + 0.3, color='0.93', lw=0)
    ax.text((q_lo + 0) / 2, 0.60, 'outside the\nwindow', ha='center', va='center', color='0.45',
            fontsize=plt.rcParams['font.size'] - 1)
    qa = np.linspace(-(1 - x100) * Qn, q_hi, 50)
    qc = np.linspace(q_lo, (1 - y100) * Qp, 50)
    ax.plot(qa, x100 - qa / Qn, color=AN, lw=1.3, ls=(0, (3, 1.5)))
    ax.plot(qc, y100 + qc / Qp, color=CA, lw=1.3, ls=(0, (3, 1.5)))
    for qq, v, col, yarr in ((-(1 - x100) * Qn, 1.0, AN, -0.38), (q_hi, 0.0, AN, -0.38), (q_lo, 0.0, CA, -0.13),
                             ((1 - y100) * Qp, 1.0, CA, -0.13)):
        ax.plot(qq, v, 'o', mfc='white', mec=col, mew=1.2, ms=5, zorder=5)
        ax.plot([qq, qq], [yarr, v], color=col, lw=0.6, ls=':')
    q = np.array([0, Q])
    ax.plot(q, x100 - q / Qn, color=AN, lw=2.2)
    ax.plot(q, y100 + q / Qp, color=CA, lw=2.2)
    x0, y0 = x100 - Q / Qn, y100 + Q / Qp
    # labels inside the window, clear of the open circles at the stoichiometry ends (x100 ~ 1, x0 ~ 0 here)
    for (qq, v, col, lab, ha, dy) in ((0, x100, AN, r'$x_{100}$', 'right', -0.12), (Q, x0, AN, r'$x_0$', 'left', 0.15),
                                     (0, y100, CA, r'$y_{100}$', 'left', -0.10), (Q, y0, CA, r'$y_0$', 'left', -0.12)):
        ax.plot(qq, v, 'o', color=col, ms=4)
        ax.text(qq + (-0.12 if ha == 'right' else 0.12), v + dy, lab, color=col, ha=ha, va='center')
    ax.annotate('', xy=(q_lo, -0.13), xytext=((1 - y100) * Qp, -0.13),
                arrowprops=dict(arrowstyle='<->', color=CA, lw=0.9))
    fs = plt.rcParams['font.size'] - 1
    ax.text(q_lo + 0.9, -0.20, r'$Q_p$: $y$ from 0 to 1', color=CA, va='top', fontsize=fs)
    ax.annotate('', xy=(-(1 - x100) * Qn, -0.38), xytext=(q_hi, -0.38),
                arrowprops=dict(arrowstyle='<->', color=AN, lw=0.9))
    ax.text(q_hi - 0.2, -0.45, r'$Q_n$: $x$ from 1 to 0', color=AN, va='top', ha='right', fontsize=fs)
    ax.annotate('', xy=(0, 1.1), xytext=(Q, 1.1), arrowprops=dict(arrowstyle='<->', color='k', lw=0.9))
    ax.text(Q / 2, 1.07, r'$Q$', ha='center', va='top')
    ax.set(xlabel='Discharged charge [Ah]', ylabel='Stoichiometry', xlim=(q_lo - 0.3, (1 - y100) * Qp + 0.3), ylim=(-0.62, 1.22))
    ax.set_yticks([0, 0.5, 1])
    top = ax.secondary_xaxis('top', functions=(lambda z: 100 * (1 - z / Q), lambda s: Q * (1 - s / 100)))
    top.set_xticks([100, 50, 0])
    top.set_xlabel('SOC [%]')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--study', default='fitB')
    ap.add_argument('--cell', default='C07')
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'fig_groups'))
    a = ap.parse_args()
    st = S.load(os.path.join(ROOT, 'configs', a.study + '.json'))
    ocp = st.em().ocp
    W = json.load(open(st.path('windows.json')))[a.cell]
    x100, y100, Qn, Qp, x0 = W['x100'], W['y100'], W['Q_n'], W['Q_p'], W['x0']
    Q = Qn * (x100 - x0)
    y0 = y100 + Q / Qp

    # one row of three panels: (a) stoichiometries, (b) OCPs, (c) OCV. The MPM-LE schematic
    # (panel_a) and the particle (panel_b) are left out: Fig. 1 and the step schematic of Section 2.2 show them.
    fig, (axs, axd, axe) = plt.subplots(1, 3, figsize=(180 * mm, 62 * mm), layout='constrained')
    panel_c(axs, x100, y100, Qn, Qp, Q)
    label(axs, 'a', x=-0.12, y=1.10)
    s = np.linspace(0.005, 0.995, 400)
    U = lambda f, z: np.asarray(f(np.asarray(z, float)), float)          # noqa: E731
    for f, lo, hi, col, name in ((ocp.U_n, x0, x100, AN, r'$U_n(x)$'), (ocp.U_p, y100, y0, CA, r'$U_p(y)$')):
        axd.plot(s, U(f, s), color=col, lw=0.8, ls=(0, (3, 2)))
        m = (s >= lo) & (s <= hi)
        axd.plot(s[m], U(f, s[m]), color=col, lw=2.0, label=name)
    for z, f, col, lab, dx, dy in ((x0, ocp.U_n, AN, r'$x_0$', 0.04, 0.25), (x100, ocp.U_n, AN, r'$x_{100}$', 0.0, 0.25),
                                   (y100, ocp.U_p, CA, r'$y_{100}$', 0.0, 0.22), (y0, ocp.U_p, CA, r'$y_0$', 0.0, -0.42)):
        axd.plot(z, U(f, [z])[0], 'o', color=col, ms=4)
        axd.text(z + dx, U(f, [z])[0] + dy, lab, color=col, ha='center', va='center')
    axd.set(xlabel=r'Stoichiometry $x$, $y$', ylabel=r'Potential vs Li/Li$^+$ [V]', xlim=(0, 1), ylim=(0, 4.8))
    axd.legend(loc='center right', frameon=False)
    label(axd, 'b', x=-0.12, y=1.02)
    q = np.linspace(0, Q, 300)
    up, un = U(ocp.U_p, y100 + q / Qp), U(ocp.U_n, x100 - q / Qn)
    axe.plot(q, up, color=CA, lw=1.2, label=r'$U_p(y)$')
    axe.plot(q, un, color=AN, lw=1.2, label=r'$U_n(x)$')
    axe.plot(q, up - un, color='k', lw=2.0, label=r'OCV')
    axe.set(xlabel='Discharged charge [Ah]', ylabel='Potential [V]', xlim=(0, Q), ylim=(0, 4.5))
    axe.legend(loc='center right', frameon=False)
    label(axe, 'c', x=-0.12, y=1.02)
    # temperature panels (e: entropic OCV shift, f: Arrhenius) - left out
    if True:      # the temperature panels below need the old two-row layout (sf); kept for reference only
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        fig.savefig(a.out + '.pdf')
        fig.savefig(a.out + '.png', dpi=300)
        print('written %s.png/.pdf; x0 %.4f x100 %.4f y100 %.4f y0 %.4f Qn %.3f Qp %.3f Q %.3f' % (
            a.out, x0, x100, y100, y0, Qn, Qp, Q))
        return
    G = json.load(open(st.path('dynamics.json')))['groups']
    T_ref, Rg = 298.15, 8.314
    axT, axA = sf[2].subplots(1, 2)
    xq = x100 - q / Qn
    yq = y100 + q / Qp
    for T_C, col in ((-5.0, '#2166AC'), (45.0, '#B2182B')):
        dT = (T_C + 273.15) - T_ref
        dU = dT * (G['dUdT0'] + G['dUdT_n'] * (xq - 0.5) + G.get('dUdT_p', 0.0) * (yq - 0.5))
        axT.plot(q, 1e3 * dU, color=col, lw=1.6, label=r'%+.0f $^\circ$C' % T_C)
    axT.axhline(0, color='0.5', lw=0.6)
    axT.set(xlabel='Discharged charge [Ah]', ylabel=r'OCV$(T)-$OCV$(25\,^\circ$C$)$ [mV]', xlim=(0, Q))
    axT.legend(loc='lower left', frameon=False)
    label(axT, 'e', x=-0.12, y=1.02)
    inv = np.linspace(1000 / (273.15 + 50), 1000 / (273.15 - 10), 50)          # 1000/T [1/K]
    for key, name, col in (('Ea_Dn', r'$D_{s,n}$ ($1/\theta_n$)', AN), ('Ea_Dp', r'$D_{s,p}$ ($1/\theta_p$)', CA),
                           ('Ea_r', r'$I_{0,k}$', '#6A3D9A')):
        Ea = G[key]
        f = np.exp(-Ea / Rg * (inv / 1000 - 1 / T_ref))
        axA.semilogy(inv, f, color=col, lw=1.6, label=r'%s, $E_a$ = %.0f kJ/mol' % (name, Ea / 1000))
    for T_C in (-5, 25, 45):
        axA.axvline(1000 / (273.15 + T_C), color='0.6', lw=0.6, ls=':')
        axA.text(1000 / (273.15 + T_C), 12, r'%d $^\circ$C' % T_C, ha='center', va='bottom', color='0.4',
                 fontsize=plt.rcParams['font.size'] - 1)
    axA.set(xlabel=r'$1000/T$ [1/K]', ylabel=r'value / value at 25 $^\circ$C', ylim=(0.002, 30))
    axA.legend(loc='lower left', frameon=False, fontsize=plt.rcParams['font.size'] - 1)
    label(axA, 'f', x=-0.12, y=1.02)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out + '.pdf')
    fig.savefig(a.out + '.png', dpi=300)
    print('written %s.png/.pdf; x0 %.4f x100 %.4f y100 %.4f y0 %.4f Qn %.3f Qp %.3f Q %.3f' % (
        a.out, x0, x100, y100, y0, Qn, Qp, Q))


if __name__ == '__main__':
    main()
