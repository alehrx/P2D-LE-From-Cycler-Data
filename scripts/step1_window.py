"""
step1_window.py
===============
Step 1 of the two-step identification: each cell's OCV, its anode window, the cathode OCP by subtraction
and the per-cell windows (config key "step1").

    python scripts/step1_window.py configs/fitB.json

1. OCV(q) of a cell, q [A.h] below the top of charge after the CV (where the HPPC and the cycles start):
   the C/20 discharge (charge where the discharge was not logged), corrected by a smooth residual through
   the ends of the HPPC rests longer than rest_min_s, the last one past the knee included (cell_ocv).
2. Anode window (x100, Q_n): the graphite OCP U_n (literature, the only curve taken from it) is the one
   with staging features; the cathode's is smooth ("anode_window": "pair_fit" instead: least squares of the
   set's U_p - U_n on the OCV, for a cathode with features of its own, LCO). Grid search for the window whose remainder
   OCV(q) + U_n(x100 - q / Q_n) is the smoothest (least squared second difference).
3. U_p(y) by subtraction on a conventional cathode window (the cathode window is not identifiable with a
   free U_p): y = y100 + q / Q_p, with y100 and y = y_end at the mean C/20 capacity of the cells_U_p, the
   same Q_p for all of them; the remainders of those with the full C/20 discharge (the charge leg sits
   ~30 mV above the rests: hysteresis; the HPPC rests follow discharges) and an anode window inside the
   search grid are averaged on a y grid. Outside the observed range,
   Nernst-like tails that keep value and slope (or, "tails": "literature", the set's U_p shifted to join): U_hi + a ln((1 - y) / (1 - y_hi)) towards y -> 1,
   U_lo + b ln(y / y_lo) towards y -> 0.
4. Per-cell windows of every cell: (x100, Q_n, y100, Q_p) by least squares of U_p - U_n on its own OCV.
"ocv_mode": "pseudo": no rest test; the OCV is the mean of the low-rate discharge and charge branches.
"U_p_from": <study>: skip 3, take that study's U_p (ocp_correction.json) as it is (the cells of an ageing study).

The tests are the config's step1 "ocv_test" (default C20) and "rest_test" (default HPPC): LG M50T uses the
C/10 and the GITT. A cell without the rest test (LG M50T Expt 5: no GITT) takes the mean offset of the
ocv-test discharge from the anchors over the cells that have both, on the ocv test's q axis (the donors'
offsets moved by their shift).
"ocv_rest_anchors": true adds the ocv test's own long rests past the rest test's last anchor (the LG M50T
C/10: its 6 h rest after the discharge to 2.5 V ends at ~2.91 V, the GITT's last rest, 0.13 A.h above, at
~2.93 V; without it the OCV below that rest is the C/10 under load, down to 2.55 V); a borrowing cell's
offset then ramps from the donors' at their mean last rest-test anchor to its own at that rest.
"top_anchor_test": a test that every cell starts at the top with the same rest (LG M50T: "VAL", 4 h after
its CC-CV) sets a borrowing cell's OCV at q = 0: the donors' mean OCV(0) plus the cell's difference from
the donors' mean voltage at top_anchor_s into that rest (the Expt 5 C/10 is the processed curve, under load
from the first sample right after the CV: without it the top sits ~30 mV below the Expt 4 one). The offset
ramps from it to the borrowed one at the donors' first anchor below the top.
"U_n_bottom_anchor" (with a U_n table: "U_n" or "U_n_smooth"; LG M50T): the anode OCP past the observed range
(below the ref cells' mean x at the rest test's last anchor) is the table's, scaled about that point by the
factor that puts the ref cells' mean rested bottom (the ocv test's rest after its cut-off) on U_p - U_n. The
cathode cannot take it (its tail would have to rise towards y = 1); unscaled, the window's OCV at the C/10
cut-off sits ~100 mV below that rest, and the model relaxes to it. "bottom_weight": the bottom as that many
samples in the per-cell window fit (as "top_weight" for the top).
"bottom_vinf": the rests past the knee (the rest test's last one, the ocv test's own) as anchors at their
extrapolated V_inf (rest_ends) instead of their measured end: the LG M50T GITT's last 1 h rest is still ~50 mV
below its V_inf, which makes the OCV fall steeply into it and then flatten to the C/10's 6 h rest. (The rest's own end is no
anchor: it relaxes for hours, ~15 mV in the 4 h, and the cold cells' chamber cools during it.)

Writes runs/studies/<name>/windows.json, ocp_correction.json (the U_p table, srcs/ocp.py),
step1.json (per-cell details) and step1_<cell>.png / step1_U_p.png.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import numpy as np                  # noqa: E402
from scipy.interpolate import PchipInterpolator   # noqa: E402
from scipy.optimize import least_squares   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
from srcs import identification as ident, physics, study as S   # noqa: E402
from srcs.datasets.base import segments   # noqa: E402

plt.style.use(os.path.join(ROOT, 'journal.mplstyle'))
mm = 1 / 25.4
I_REST = 0.02
DQ = 0.005              # [A.h] OCV grid
DQ_ROUGH = 0.02         # [A.h] stencil of the roughness


def legs(df, dt):
    """q [A.h] below the top at each sample midpoint of a test that starts at the top of charge."""
    I = df['I_A'].to_numpy()
    return (np.r_[0.0, np.cumsum(I * dt)[:-1]] + 0.5 * I * dt) / 3600.0


def rest_ends(I, V, q, dt, rest_min_s):
    """(q, V_inf, V_end) of every rest longer than rest_min_s: V_inf extrapolates the relaxation with the
    long-time diffusion tail V(t) = V_inf - a / sqrt(t), fitted on the rest's second half (at the bottom of
    the HPPC the voltage still rises by ~0.3-0.5 mV/min after the hour of rest)."""
    rest = np.abs(I) <= I_REST
    edges = np.flatnonzero(np.diff(np.r_[0, rest.astype(int), 0]))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) * dt >= rest_min_s:
            k = np.arange((a + b) // 2, b)
            s = 1.0 / np.sqrt((k - a + 1) * dt)
            slope, V_inf = np.polyfit(s, V[k], 1)
            out.append((q[b - 1], V_inf, V[b - 1]))
    return np.array(out)


def cell_ocv(ds, cell, cfg):
    """OCV on a q grid (q = 0 at the HPPC's top, after the CV, where the cycles start too) and the pieces
    it was built from. The level comes from the measured ends of the HPPC rests, including the one after the
    discharge to the cutoff, past the knee (extrapolating the relaxation, rest_ends' V_inf, fits worse: OCV
    RMSE of the windows 5.6 against 1.3 mV, it lifts the mid-range rests too); the shape between them from ONE C/20 leg: the
    discharge, or the charge where the discharge was not logged in full (the charge sits ~30 mV above the
    rests: hysteresis). The C/20 axis is placed on the HPPC one by the shift that makes the leg's offset
    from the anchors below the top most constant (grid search); the C/20's own bottom (1 min after a
    discharge below 3 V) is not a rested point."""
    if cfg.get('ocv_mode') == 'pseudo':
        # no rest test (CALCE A123: only the C/20 test): the OCV is the mean of the discharge and charge branches of
        # the low-rate test on a common q grid (cancels the low-rate overpotential and most of the hysteresis)
        br = ds.ocv_branches(cell)
        qd, Vd = br['discharge']
        qc, Vc = br['charge']
        oc = np.argsort(qc)
        q_end = float(min(qd.max(), qc.max()))
        grid = np.arange(0.0, q_end, DQ)
        Vdis = np.interp(grid, qd, Vd)
        ocv = 0.5 * (Vdis + np.interp(grid, qc[oc], Vc[oc]))
        return dict(q=grid, ocv=ocv, Vleg=Vdis, offset=ocv - Vdis, leg='pseudo', q_chg=qc, V_chg=Vc, q_dis=qd, V_dis=Vd,
                    anchors=np.zeros((0, 2)), shift_Ah=0.0, bias_mV=np.nan, q_hi=q_end, relax_last_mV=np.nan,
                    knee_offset_mV=np.nan, q_last_rest=q_end, n_own_anchors=0, ql=qd, Vl=Vd)
    c20, c20cfg = ds.load(cell, cfg.get('ocv_test', 'C20'))
    I, V, dt = c20['I_A'].to_numpy(), c20['V_V'].to_numpy(), float(c20cfg['dt_s'])
    q = float(c20cfg.get('q_top_Ah', 0.0)) + (np.r_[0.0, np.cumsum(I * dt)[:-1]] + 0.5 * I * dt) / 3600.0
    chg, dis = I < -I_REST, I > I_REST
    # each leg is its longest contiguous run: an aged cell's C/20 (Mohtat RPTs) can start with a few discharge
    # samples at the cut-off, which would make the leg's q non-monotonic (np.interp needs it increasing)
    for m in (chg, dis):
        segs = segments(m)
        if len(segs) > 1:
            a0, b0 = max(segs, key=lambda s: s[1] - s[0])
            m[:] = False
            m[a0:b0] = True
    qc, Vc = q[chg], V[chg]
    qd, Vd = q[dis], V[dis]
    leg = 'discharge' if dis.any() and qd.max() - max(qd.min(), 0.0) > 0.9 * (qc.max() if chg.any() else 0.0) else 'charge'   # NASA RW: discharge only
    ql, Vl = (qd, Vd) if leg == 'discharge' else (qc[np.argsort(qc)], Vc[np.argsort(qc)])
    own = np.zeros((0, 2))
    if cfg.get('ocv_rest_anchors'):                              # the ocv test's own long rests (q, V_end)
        own = rest_ends(I, V, q, dt, cfg['rest_min_s'])
        own = own[:, [0, 2]] if len(own) else np.zeros((0, 2))
    if not ds.available(cell, cfg.get('rest_test', 'HPPC')):     # no anchors: main() borrows the mean offset
        q_end = ql.max()
        grid = np.arange(0.0, q_end, DQ)
        return dict(q=grid, ocv=None, Vleg=np.interp(grid, ql, Vl), leg=leg, q_chg=qc, V_chg=Vc, q_dis=qd, V_dis=Vd,
                    anchors=np.zeros((0, 2)), shift_Ah=0.0, bias_mV=np.nan, q_hi=float(q_end), relax_last_mV=np.nan,
                    knee_offset_mV=np.nan, borrowed=True, own_anchors=own, ql=ql, Vl=Vl)
    hp, hpcfg = ds.load(cell, cfg.get('rest_test', 'HPPC'))
    hdt = float(hpcfg['dt_s'])
    R = rest_ends(hp['I_A'].to_numpy(), hp['V_V'].to_numpy(), legs(hp, hdt), hdt, cfg['rest_min_s'])
    relax_last_mV = 1e3 * float(R[-1, 1] - R[-1, 2])
    R = R[:, [0, 2]]                                            # the measured end of each rest (see the docstring)
    mid = R[:, 0] > 0.3
    shifts = np.arange(-0.4, 0.4, 0.002)
    var = [np.var(R[mid, 1] - np.interp(R[mid, 0] + s, ql, Vl)) for s in shifts]
    sh = float(shifts[int(np.argmin(var))])
    bias = float(np.mean(R[mid, 1] - np.interp(R[mid, 0] + sh, ql, Vl)))
    # one anchor per level: the rests before the discharge pulse and after the charge pulse sit within a few
    # mA.h of each other (two points that close make the interpolated offset spike)
    A = R[np.argsort(R[:, 0])]
    grp = np.r_[0, np.cumsum(np.diff(A[:, 0]) > 0.02)]
    A = np.array([A[grp == g].mean(0) for g in np.unique(grp)])
    q_last_rest = float(A[-1, 0])
    if len(own):                                                 # past the rest test's last anchor only
        own = own[own[:, 0] - sh > A[-1, 0] + 0.02]
        A = np.vstack([A, np.c_[own[:, 0] - sh, own[:, 1]]])
    q_end = ql.max() - sh
    grid = np.arange(0.0, q_end, DQ)
    Vleg = np.interp(grid + sh, ql, Vl)
    # the leg's offset from the anchors, smooth (PCHIP: no kinks in dV/dq), held constant beyond them
    r = PchipInterpolator(A[:, 0], A[:, 1] - np.interp(A[:, 0] + sh, ql, Vl), extrapolate=False)(grid)
    r = np.where(grid < A[0, 0], A[0, 1] - np.interp(A[0, 0] + sh, ql, Vl), r)
    r = np.where(grid > A[-1, 0], A[-1, 1] - np.interp(A[-1, 0] + sh, ql, Vl), r)
    ocv = Vleg + r
    return dict(q=grid, ocv=ocv, Vleg=Vleg, offset=r, leg=leg, q_chg=qc - sh, V_chg=Vc, q_dis=qd - sh, V_dis=Vd, anchors=A,
                shift_Ah=sh, bias_mV=1e3 * bias, q_hi=float(q_end), relax_last_mV=relax_last_mV,
                knee_offset_mV=1e3 * float(A[-1, 1] - np.interp(A[-1, 0] + sh, ql, Vl)), q_last_rest=q_last_rest,
                n_own_anchors=int(len(own)))


def roughness(rem):
    k = int(round(DQ_ROUGH / DQ))
    d2 = rem[2 * k:] - 2 * rem[k:-k] + rem[:-2 * k]
    return float(np.mean(d2 ** 2))


def anode_window_pair(O, U_n, U_p, cfg):
    """(x100, Q_n) by least squares of the set's two curves, U_p(y100 + q/Q_p) - U_n(x100 - q/Q_n), on the
    cell's OCV ("anode_window": "pair_fit"). For a cathode with its own features (LCO: the ~3.93 V plateau and
    the order/disorder steps near 4.1 V) the smoothest remainder is no criterion: it hands those features to
    the anode. Only the anode window is kept; U_p is then derived from the data as for NMC."""
    from scipy.optimize import least_squares
    m = (O['q'] > 0.01) & (O['q'] < O['q_hi'] - 0.01)
    q, ocv = O['q'][m], O['ocv'][m]
    f = lambda p: U_p(p[2] + q / p[3]) - U_n(p[0] - q / p[1]) - ocv          # noqa: E731
    lo = [cfg['x100_range'][0], cfg['Q_n_range'][0], 0.05, 0.5 * O['q_hi']]
    hi = [cfg['x100_range'][1], cfg['Q_n_range'][1], 0.9, 10.0 * O['q_hi']]
    best = None
    for x0 in np.linspace(lo[0] + 0.02, hi[0] - 0.02, 4):
        for y0 in (0.3, 0.45, 0.6):
            p0 = [x0, max(lo[1] + 1e-3, min(hi[1] - 1e-3, O['q_hi'] / max(x0 - 0.02, 0.1))), y0, O['q_hi'] / max(0.95 - y0, 0.2)]
            p0 = list(np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6))
            try:
                r = least_squares(f, p0, bounds=(lo, hi))
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
    O['pair_fit'] = dict(x100=float(best.x[0]), Q_n=float(best.x[1]), y100=float(best.x[2]), Q_p=float(best.x[3]),
                         rmse_mV=float(1e3 * np.sqrt(np.mean(best.fun ** 2))))
    return float(best.x[0]), float(best.x[1])


def anode_window(O, U_n, cfg):
    """(x100, Q_n) of the smoothest remainder OCV + U_n on a grid, refined on a finer one. "x100_fixed": the
    anode's top by convention (a blend anode whose features fix x0 at the knee but not the scale: the search
    then runs along x100 - q_hi / Q_n = x0), Q_n searched alone."""
    m = (O['q'] > 0.02) & (O['q'] < O['q_hi'] - 0.02)
    q, ocv = O['q'][m], O['ocv'][m]
    best = None
    for level in (0, 1):
        if best is None and level == 1 and 'x100_fixed' in cfg:     # no Q_n of the range keeps x0 above x0_min
            x100 = float(cfg['x100_fixed'])
            return x100, O['q_hi'] / (x100 - cfg.get('x0_min', 0.002))
        if level == 0:
            xs, Qs = np.linspace(*cfg['x100_range'], 59), np.linspace(*cfg['Q_n_range'], 53)
        else:
            xs, Qs = np.linspace(best[1] - 0.01, best[1] + 0.01, 41), np.linspace(best[2] - 0.1, best[2] + 0.1, 41)
            Qs = Qs[(Qs >= cfg['Q_n_range'][0]) & (Qs <= cfg['Q_n_range'][1])] if 'x100_fixed' in cfg else Qs
        if 'x100_fixed' in cfg:
            xs = [float(cfg['x100_fixed'])]
        for x100 in xs:
            for Qn in Qs:
                if x100 - O['q_hi'] / Qn < cfg.get('x0_min', 0.002):      # "x0_min": a literature x0 below 0.002
                    continue
                r = roughness(ocv + U_n(x100 - q / Qn))
                if best is None or r < best[0]:
                    best = (r, x100, Qn)
    return best[1], best[2]


def tails(y, U, y_lo, y_hi, dy=0.01):
    """U on [1e-5, 1 - 1e-5] from its values on [y_lo, y_hi], with the module docstring's tails; their slopes
    from the last dy of the observed range."""
    yy = np.linspace(1e-5, 1 - 1e-5, 4001)
    Ui = np.interp(yy, y, U)
    k_lo, k_hi = np.searchsorted(y, y_lo + dy), np.searchsorted(y, y_hi - dy)
    s_lo = (U[k_lo] - U[0]) / (y[k_lo] - y[0])
    s_hi = (U[-1] - U[k_hi]) / (y[-1] - y[k_hi])
    U_lo, U_hi = np.interp(y_lo, y, U), np.interp(y_hi, y, U)
    lo, hi = yy < y_lo, yy > y_hi
    Ui[lo] = U_lo + s_lo * y_lo * np.log(yy[lo] / y_lo)
    Ui[hi] = U_hi - s_hi * (1 - y_hi) * np.log((1 - yy[hi]) / (1 - y_hi))
    return yy, Ui


def main():
    st = S.load(sys.argv[1])
    cfg = st.cfg['step1']
    os.makedirs(st.path(''), exist_ok=True)
    U_n, U_p_lit = physics.ocps(st.pm)
    if cfg.get('U_n_set'):                              # the anode OCP of ANOTHER PyBaMM set (no prior of this cell)
        U_n = physics.ocps(cfg['U_n_set'])[0]
    if cfg.get('U_n') or cfg.get('U_n_smooth') or cfg.get('U_n_set'):   # an analytic fit, or a set's table (smoothed: its slope is noisy)
        xn = np.linspace(1e-5, 1 - 1e-5, 8001)
        if cfg.get('U_n_set') and not cfg.get('U_n_smooth'):
            Un_tab = U_n(xn)
        elif cfg.get('U_n'):
            Un_tab = physics.ANALYTIC_OCP[cfg['U_n']](xn)
        else:
            from scipy.ndimage import gaussian_filter1d
            Un_tab = gaussian_filter1d(U_n(xn), float(cfg['U_n_smooth']) / (xn[1] - xn[0]), mode='nearest')
        U_n = lambda x: np.interp(np.clip(x, xn[0], xn[-1]), xn, Un_tab)
    cells = [c for c in st.ds.cells if st.ds.available(c, cfg.get('ocv_test', 'C20'))]
    O, win = {}, {}
    for c in cells:
        O[c] = cell_ocv(st.ds, c, cfg)
    donors = [c for c in cells if not O[c].get('borrowed') and O[c]['leg'] == 'discharge']
    for c in [c for c in cells if O[c].get('borrowed')]:
        if O[c]['leg'] != 'discharge':
            raise SystemExit('%s: no rest test and no full discharge leg - cannot borrow the offset' % c)
        # on the ocv test's own axis (the donors' q + their shift): the borrowing cell has no rest test to shift to
        offs = [np.interp(O[c]['q'], O[d]['q'] + O[d]['shift_Ah'], O[d]['offset']) for d in donors]
        off = np.mean(offs, 0)
        own = O[c]['own_anchors']
        qL = float(np.mean([O[d]['q_last_rest'] + O[d]['shift_Ah'] for d in donors]))
        O[c]['q_last_rest'] = qL
        if len(own):            # its own rests past the donors' last rest-test anchor: ramp the offset to them
            own = own[own[:, 0] > qL + 0.02]
            if len(own):
                Aq = np.r_[qL, own[:, 0]]
                Ar = np.r_[np.interp(qL, O[c]['q'], off), own[:, 1] - np.interp(own[:, 0], O[c]['ql'], O[c]['Vl'])]
                g = O[c]['q']
                ramp = PchipInterpolator(Aq, Ar, extrapolate=False)(g)
                off = np.where(g > qL, np.where(g > Aq[-1], Ar[-1], ramp), off)
                O[c]['anchors'] = np.c_[own[:, 0], own[:, 1]]
                O[c]['knee_offset_mV'] = 1e3 * float(Ar[-1])
        if cfg.get('top_anchor_test') and st.ds.available(c, cfg['top_anchor_test']):
            def v_at(cell):
                tdf, tcfg = st.ds.load(cell, cfg['top_anchor_test'])
                return float(np.interp(float(cfg.get('top_anchor_s', 3600.0)), tdf['t_s'].to_numpy(float), tdf['V_V'].to_numpy()))
            dd = [d for d in donors if st.ds.available(d, cfg['top_anchor_test'])]
            V0 = float(np.mean([O[d]['ocv'][0] for d in donors])) + v_at(c) - float(np.mean([v_at(d) for d in dd]))
            g = O[c]['q']
            q1 = float(np.mean([O[d]['anchors'][O[d]['anchors'][:, 0] > 0.05][0, 0] + O[d]['shift_Ah'] for d in donors]))
            q1 = max(q1, 0.05)
            top = V0 - float(O[c]['Vleg'][0])
            Aq = np.array([0.0, q1, q1 + 0.05])
            Ar = np.array([top, np.interp(q1, g, off), np.interp(q1 + 0.05, g, off)])
            off = np.where(g < q1, PchipInterpolator(Aq, Ar, extrapolate=False)(np.clip(g, 0, None)), off)
            O[c]['anchors'] = np.vstack([[[0.0, V0]], O[c]['anchors']]) if len(O[c]['anchors']) else np.array([[0.0, V0]])
            O[c]['top_offset_mV'] = 1e3 * top
            print('%s: OCV top %.4f V (donors %.4f V, its %s at %.0f s %+.1f mV from theirs; offset %+.1f mV, ramp to the '
                  'borrowed one at %.3f Ah)' % (c, V0, np.mean([O[d]['ocv'][0] for d in donors]), cfg['top_anchor_test'],
                                                float(cfg.get('top_anchor_s', 3600.0)), 1e3 * (V0 - np.mean([O[d]['ocv'][0] for d in donors])), 1e3 * top, q1))
        O[c]['offset'] = off
        O[c]['ocv'] = O[c]['Vleg'] + off
        O[c]['bias_mV'] = 1e3 * float(np.mean(offs))
        print('%s: no %s, offset borrowed from %s (mean %+.1f mV, spread %.1f mV)'
              % (c, cfg.get('rest_test', 'HPPC'), donors, O[c]['bias_mV'], 1e3 * float(np.mean(np.std(offs, 0)))))
    for c in cells:
        win[c] = (anode_window_pair(O[c], U_n, U_p_lit, cfg) if cfg.get('anode_window') == 'pair_fit'
                  else anode_window(O[c], U_n, cfg))
        if 'pair_fit' in O[c]:
            print('%s: pair fit of the two literature curves %s' % (c, {k: round(v, 4) for k, v in O[c]['pair_fit'].items()}))
        print('%s: %s leg, OCV to %.3f Ah, shift %+.3f Ah (offset %+.1f mV; at the last rest %+.1f mV, its '
              'relaxation %+.1f mV), x100 %.4f, Q_n %.3f, x0 %.4f'
              % (c, O[c]['leg'], O[c]['q_hi'], O[c]['shift_Ah'], O[c]['bias_mV'], O[c]['knee_offset_mV'],
                 O[c]['relax_last_mV'], win[c][0], win[c][1],
                 win[c][0] - O[c]['q_hi'] / win[c][1]), flush=True)
    if cfg.get('U_p_from'):                          # the U_p of an earlier study, frozen (an aged cell: only its windows)
        src = os.path.join(ROOT, 'runs', 'studies', cfg['U_p_from'])
        corr = json.load(open(os.path.join(src, 'ocp_correction.json'), encoding='utf-8'))
        if 'U_n_table' in corr:
            raise SystemExit('U_p_from: %s has a U_n table - not supported' % cfg['U_p_from'])
        yy, UU = np.asarray(corr['table']['y']), np.asarray(corr['table']['U'])
        U_p = lambda y: np.interp(np.clip(y, yy[0], yy[-1]), yy, UU)
        Q_p = float(corr['Q_p_convention'])
        # starts: the smoothest-remainder anode window, and every window of a grid in (x100, Q_n): with U_p fixed the
        # least squares has more than one valley on an aged cell, and the grid search alone lands on its edge
        starts = [win] + [{c: (x, qn) for c in cells} for x in (0.65, 0.75, 0.85, 0.95) for qn in (3.0, 3.5, 4.0, 5.5, 7.0)]
        wd, h = float(cfg.get('dvdq_weight', 0.0)), float(cfg.get('dvdq_h', 0.05))
        score = lambda w: (w['rmse_mV'] ** 2 + wd * (2 * h * w.get('dvdq_rmse_mV_per_Ah', 0.0)) ** 2)   # ~ the least-squares cost
        W, det = windows(cells, O, U_p, U_n, win, cfg, Q_p)
        for s0 in starts[1:]:
            W1, det1 = windows(cells, O, U_p, U_n, s0, cfg, Q_p)
            for c in cells:
                if score(W1[c]) < score(W[c]) - 1e-9:
                    W[c], det[c] = W1[c], det1[c]
        ident.save(st.path('ocp_correction.json'), corr)
        for c in cells:
            print('%s window: x100 %.4f Q_n %.3f y100 %.4f Q_p %.3f | x0 %.4f y0 %.4f | OCV RMSE %.2f mV (to %.3f Ah) | dV/dq RMSE %.1f mV/Ah'
                  % (c, W[c]['x100'], W[c]['Q_n'], W[c]['y100'], W[c]['Q_p'], W[c]['x0'], W[c]['y0'], W[c]['rmse_mV'],
                     W[c]['q_fit_Ah'], W[c].get('dvdq_rmse_mV_per_Ah', float('nan'))))
        ident.save(st.path('windows.json'), W)
        ident.save(st.path('windows_percell.json'), W)      # step 1b (anchor_window.py, mean_window.py) averages these
        ident.save(st.path('step1.json'), dict(config=cfg, Q_p_convention=Q_p, cells=det, U_p_from=cfg['U_p_from']))
        return
    # shared U_p on the conventional window
    lo_x, hi_x = cfg['x100_range']
    ref = [c for c in cfg['cells_U_p'] if O[c]['leg'] in ('discharge', 'pseudo')   # the charge leg carries ~30 mV of hysteresis
           and ('x100_fixed' in cfg or cfg.get('anode_window') == 'pair_fit'
                or lo_x + 0.015 < win[c][0] < hi_x - 0.015)]   # and not a grid-edge window
    print('U_p from %s (cells with the full %s discharge)' % (ref, cfg.get('ocv_test', 'C20')))
    Q_ref = float(np.mean([O[c]['q_hi'] for c in ref]))
    Q_p = Q_ref / (cfg['y_end'] - cfg['y100'])
    yg = np.arange(0.0, 1.0, 0.0025)
    anode = dict(win)
    # one pass: U_p from the smoothest-remainder anode windows, then the windows on U_p. Iterating (U_p from
    # the fitted windows) lowers the OCV RMSE but lets the two co-adapt: the graphite's stage-2 peak drifts
    # off the cell's and leaves a bump in the cathode curve, which has no staging
    for it in range(1):
        stack = []
        for c in ref:
            # "U_p_to_last_rest": observed only down to the rest test's last anchor (below it the OCV is a ramp
            # between two post-cutoff rests, not resolved enough to subtract the anode's steep end from)
            q_obs = O[c]['q_last_rest'] if cfg.get('U_p_to_last_rest') else O[c]['q_hi']
            q = O[c]['q'][(O[c]['q'] > 0) & (O[c]['q'] < q_obs)]
            rem = np.interp(q, O[c]['q'], O[c]['ocv']) + U_n(anode[c][0] - q / anode[c][1])
            y = cfg['y100'] + q / Q_p
            stack.append(np.where((yg >= y[0]) & (yg <= y[-1]), np.interp(yg, y, rem), np.nan))
        stack = np.array(stack)
        n_ok = np.sum(np.isfinite(stack), 0)
        keep = n_ok >= min(len(ref), max(2, len(ref) // 2))   # (one reference cell: its own curve)
        y_obs, U_obs = yg[keep], np.nanmean(stack[:, keep], 0)
        spread = 1e3 * np.nanstd(stack[:, keep], 0)
        if cfg.get('U_p_smooth'):                       # Gaussian sigma in y (the knee's remainder is noisy)
            from scipy.ndimage import gaussian_filter1d
            U_obs = gaussian_filter1d(U_obs, float(cfg['U_p_smooth']) / (yg[1] - yg[0]), mode='nearest')
        yy, UU = tails(y_obs, U_obs, y_obs[0], y_obs[-1], float(cfg.get('tail_dy', 0.01)))
        if cfg.get('tails') == 'literature':         # the set's U_p beyond the observed range, joined continuously
            lo, hi = yy < y_obs[0], yy > y_obs[-1]   # (the subtraction leaves a flat end where the anode takes the knee)
            UU[lo] = U_p_lit(yy[lo]) - U_p_lit(y_obs[0]) + U_obs[0]
            UU[hi] = U_p_lit(yy[hi]) - U_p_lit(y_obs[-1]) + U_obs[-1]
        if cfg.get('U_n_bottom_anchor'):             # the anode's unobserved end scaled to the rested bottom
            if not (cfg.get('U_n') or cfg.get('U_n_smooth')):
                raise SystemExit('U_n_bottom_anchor needs "U_n" or "U_n_smooth" (a U_n table)')
            U_p0 = lambda y: np.interp(np.clip(y, yy[0], yy[-1]), yy, UU)
            xa = np.array([anode[c][0] - O[c]['q_last_rest'] / anode[c][1] for c in ref])
            xb = np.array([anode[c][0] - O[c]['q_hi'] / anode[c][1] for c in ref])
            Unb = np.array([U_p0(cfg['y100'] + O[c]['q_hi'] / Q_p) - O[c]['ocv'][-1] for c in ref])
            x_a, x_b, U_nb = float(np.mean(xa)), float(np.mean(xb)), float(np.mean(Unb))
            U_na = float(U_n(x_a))
            U_nb0 = float(U_n(x_b))
            k_n = (U_nb - U_na) / (U_nb0 - U_na)
            if not 0.0 < k_n:
                raise SystemExit('U_n_bottom_anchor: scale %.3f <= 0 (the bottom needs a falling anode end)' % k_n)
            lo = xn < x_a
            Un_tab = Un_tab.copy()
            Un_tab[lo] = U_na + k_n * (Un_tab[lo] - U_na)
            print('U_n bottom anchor: below x %.4f (the last rest) the anode tail scaled by %.3f: U_n(%.4f) %.4f V '
                  '(cells %s) instead of %.4f V' % (x_a, k_n, x_b, U_nb, ', '.join('%.4f' % u for u in Unb), U_nb0))
            U_n_scaled = dict(x_from=x_a, scale=k_n)
            U_n = lambda x, t=Un_tab: np.interp(np.clip(x, xn[0], xn[-1]), xn, t)
        U_p = lambda y: np.interp(np.clip(y, yy[0], yy[-1]), yy, UU)
        W, det = windows(cells, O, U_p, U_n, win, cfg, Q_p)
        print('pass %d: cell-to-cell spread of U_p mean %.1f mV; OCV RMSE of the windows mean %.2f mV, max %.2f mV'
              % (it, np.mean(spread), np.mean([w['rmse_mV'] for w in W.values()]),
                 np.max([w['rmse_mV'] for w in W.values()])), flush=True)
        anode = {c: (W[c]['x100'], W[c]['Q_n']) for c in cells}
    corr = dict(electrode='p', table=dict(y=yy.tolist(), U=UU.tolist()), y100_convention=cfg['y100'],
                y_end_convention=cfg['y_end'], Q_p_convention=Q_p, y_observed=[float(y_obs[0]), float(y_obs[-1])],
                cells=ref, spread_mV_mean=float(np.mean(spread)), spread_mV_max=float(np.max(spread)))
    if cfg.get('U_n') or cfg.get('U_n_smooth') or cfg.get('U_n_set'):
        src_pm = cfg.get('U_n_set') or st.pm
        corr['U_n_table'] = dict(source=cfg.get('U_n') or ('%s table, Gaussian sigma %g in x' % (src_pm, cfg['U_n_smooth'])
                                                           if cfg.get('U_n_smooth') else '%s table' % src_pm),
                                 x=xn[::4].tolist(), U=Un_tab[::4].tolist())
        if cfg.get('U_n_bottom_anchor'):
            corr['U_n_table']['bottom_anchor'] = U_n_scaled
    ident.save(st.path('ocp_correction.json'), corr)
    for c in cells:
        print('%s window: x100 %.4f Q_n %.3f y100 %.4f Q_p %.3f | x0 %.4f y0 %.4f | OCV RMSE %.2f mV (to %.3f Ah) | '
              'bottom OCV %.3f V, model %.3f V'
              % (c, W[c]['x100'], W[c]['Q_n'], W[c]['y100'], W[c]['Q_p'], W[c]['x0'], W[c]['y0'], W[c]['rmse_mV'],
                 W[c]['q_fit_Ah'], W[c]['ocv_bottom'], W[c]['ocv_bottom_model']))
    ident.save(st.path('windows.json'), W)
    ident.save(st.path('windows_percell.json'), W)          # step 1b (anchor_window.py, mean_window.py) averages these
    ident.save(st.path('step1.json'), dict(config=cfg, Q_p_convention=Q_p, cells=det))
    figures(st, ref, O, W, yg, stack, yy, UU, y_obs, spread, U_p, U_n, Q_p, cfg.get('ocv_test', 'C20'))


def window_residuals(Oc, U_p, U_n, cfg):
    """(ft, f, fd) of one cell's window p = (x100, Q_n, y100, Q_p): ft the least-squares residual, f the OCV
    residual [V], fd the dV/dq residual [V/A.h] (None without "dvdq_weight").
    "dvdq_weight" w, "dvdq_h" h [A.h, default 0.05]: adds the differential voltage, central differences over
    +-h of the OCV and of the window's U_p - U_n (the same stencil smooths both), as a block of weight w relative to
    the OCV block (w = 1: equal sums of squares per sample count), its residual scaled by 2h to a voltage. The
    graphite's staging peaks sit at fixed x, so their positions on the q axis pin x100 and Q_n, which the OCV itself
    bounds only from below (a 1-2 mV change on a plateau)."""
    q_fit = Oc['q_last_rest'] if cfg.get('U_p_to_last_rest') else Oc['q_hi']   # (see main)
    m = (Oc['q'] > 0) & (Oc['q'] < q_fit)
    q, ocv = Oc['q'][m], Oc['ocv'][m]
    Vm = lambda p, x: U_p(p[2] + x / p[3]) - U_n(p[0] - x / p[1])
    f = lambda p: Vm(p, q) - ocv
    # "top_weight": the rested top (q = 0) as that many samples - the window's OCV(0) is where every test starts
    wt = np.sqrt(float(cfg.get('top_weight', 0.0)))
    wb = np.sqrt(float(cfg.get('bottom_weight', 0.0)))   # "bottom_weight": the same for the rested bottom (q_hi)
    qb, Vb = Oc['q_hi'], float(Oc['ocv'][-1])
    wd = float(cfg.get('dvdq_weight', 0.0))
    if wd > 0:
        h = float(cfg.get('dvdq_h', 0.05))
        qd = q[(q > h) & (q < q_fit - h)]
        dd = (np.interp(qd + h, q, ocv) - np.interp(qd - h, q, ocv)) / (2 * h)
        fd = lambda p: (Vm(p, qd + h) - Vm(p, qd - h)) / (2 * h) - dd
        sd = np.sqrt(wd * len(q) / len(qd)) * 2 * h
    else:
        fd, sd = None, 0.0
    ft = lambda p: np.r_[f(p), wt * (Vm(p, 0.0) - Oc['ocv'][0]), wb * (Vm(p, qb) - Vb),
                         sd * fd(p) if fd is not None else np.zeros(0)]
    return ft, f, fd, q_fit


def windows(cells, O, U_p, U_n, win, cfg, Q_p):
    """Per-cell windows (x100, Q_n, y100, Q_p) of U_p - U_n on each cell's OCV (least squares)."""
    W, det = {}, {}
    for c in cells:
        ft, f, fd, q_fit = window_residuals(O[c], U_p, U_n, cfg)
        qmin = min(3.0, 0.6 * O[c]['q_hi'])           # 3 A.h for the 5 A.h cells; scaled for a 2 A.h 18650
        x_max = float(cfg.get('x100_max', 1.0))       # "x100_max": keeps the anode's top off the graphite's steep end
        p = least_squares(ft, [min(win[c][0], x_max - 1e-6), win[c][1], cfg['y100'], Q_p],
                          bounds=([0.5, qmin, 0.05, qmin], [x_max, 12.0, 0.9, 20.0])).x
        rmse = 1e3 * float(np.sqrt(np.mean(f(p) ** 2)))
        W[c] = dict(x100=float(p[0]), Q_n=float(p[1]), y100=float(p[2]), Q_p=float(p[3]), rmse_mV=rmse,
                    q_ocv_Ah=O[c]['q_hi'], C20_leg=O[c]['leg'], x0=float(p[0] - O[c]['q_hi'] / p[1]), y0=float(p[2] + O[c]['q_hi'] / p[3]),
                    q_fit_Ah=float(q_fit), ocv_bottom_model=float(U_p(p[2] + O[c]['q_hi'] / p[3]) - U_n(p[0] - O[c]['q_hi'] / p[1])),
                    ocv_bottom=float(O[c]['ocv'][-1]))
        if fd is not None:
            W[c]['dvdq_rmse_mV_per_Ah'] = 1e3 * float(np.sqrt(np.mean(fd(p) ** 2)))
        det[c] = dict(**W[c], x100_smooth=float(win[c][0]), Q_n_smooth=float(win[c][1]), hppc_shift_Ah=O[c]['shift_Ah'],
                      hppc_bias_mV=O[c]['bias_mV'], anchors=O[c]['anchors'].tolist(), offset_borrowed=bool(O[c].get('borrowed', False)))
    return W, det


def figures(st, ref, O, W, yg, stack, yy, UU, y_obs, spread, U_p, U_n, Q_p, ocv_test='C20'):
    """The shared U_p with the cells' remainders, and each of the first three reference cells' construction."""
    fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 65 * mm), layout='constrained')
    for k, c in enumerate(ref):
        ax[0].plot(yg, stack[k], lw=0.6, alpha=0.6)
    ax[0].plot(yy, UU, 'k', lw=1.0, label='shared $U_p$ (with tails)')
    ax[0].set_xlim(max(0.0, y_obs[0] - 0.1), 1.0)
    ax[0].set_ylim(np.nanmin(UU[(yy > y_obs[0] - 0.1) & (yy < 0.995)]) - 0.05, np.nanmax(UU[yy > y_obs[0] - 0.1]) + 0.05)
    ax[0].set_xlabel('$y$ (conventional)')
    ax[0].set_ylabel('$U_p$ [V]')
    ax[0].legend(fontsize=6)
    ax[1].plot(y_obs, spread, 'k', lw=0.8)
    ax[1].set_xlabel('$y$')
    ax[1].set_ylabel('cell-to-cell spread [mV]')
    for a in ax:
        a.spines[['top', 'right']].set_visible(False)
    for ext in ('png', 'pdf'):
        fig.savefig(st.path('step1_U_p.%s' % ext))
    plt.close(fig)
    for c in ref[:3]:
        o, w = O[c], W[c]
        fig, ax = plt.subplots(1, 2, figsize=(170 * mm, 65 * mm), layout='constrained')
        ax[0].plot(o['q_chg'], o['V_chg'], lw=0.6, label='%s charge' % ocv_test.replace('C', 'C/', 1))
        ax[0].plot(o['q_dis'], o['V_dis'], lw=0.6, label='%s discharge' % ocv_test.replace('C', 'C/', 1))
        ax[0].plot(o['q'], o['ocv'], 'k', lw=0.8, label='OCV')
        ax[0].plot(o['anchors'][:, 0], o['anchors'][:, 1], 'o', ms=2.5, label='rested anchors')
        qq = o['q']
        ax[0].plot(qq, U_p(w['y100'] + qq / w['Q_p']) - U_n(w['x100'] - qq / w['Q_n']), 'r--', lw=0.6, label='window')
        ax[0].set_xlabel('$q$ below the top [Ah]')
        ax[0].set_ylabel('$V$ [V]')
        ax[0].legend(fontsize=6)
        dv = -np.gradient(o['ocv'], o['q'])
        ax[1].plot(o['q'], dv, 'k', lw=0.6, label='cell OCV')
        ax[1].plot(qq, -np.gradient(U_p(w['y100'] + qq / w['Q_p']), qq), lw=0.6, label='cathode')
        ax[1].plot(qq, np.gradient(U_n(w['x100'] - qq / w['Q_n']), qq), lw=0.6, label='anode')
        ax[1].set_ylim(0, 1.0)
        ax[1].set_xlabel('$q$ below the top [Ah]')
        ax[1].set_ylabel('$-dV/dq$ [V/Ah]')
        ax[1].legend(fontsize=6)
        for a in ax:
            a.spines[['top', 'right']].set_visible(False)
        for ext in ('png', 'pdf'):
            fig.savefig(st.path('step1_%s.%s' % (c, ext)))
        plt.close(fig)
    print('Q_p (convention) %.3f Ah, U_p observed on y in [%.3f, %.3f], cell-to-cell spread mean %.1f mV, max %.1f mV'
          % (Q_p, y_obs[0], y_obs[-1], np.mean(spread), np.max(spread)))


if __name__ == '__main__':
    main()
