"""ecm.py
======
Thevenin 2-RC equivalent circuit model, the baseline of the study, tuned on the SAME budget as the EM:

    V = OCV(q) - I R0 - v1 - v2,     dv_k/dt = (R_k I - v_k) / tau_k,     q = charge below the top [A.h]

    ocv_curve(q, v)          OCV as a function of q from the cell's own OCV data (the points the EM's
                             window is fitted to: rest voltages and / or the low-rate pseudo-OCV);
                             linear extrapolation beyond the measured range
    fit_q / simulate_q       the five parameters (R0, R1, tau1, R2, tau2) by bounded least squares on
                             the cell's budget pulse test (HPPC, GITT)

The older SOC-based functions (extract_ecm_ocv, simulate_ecm, fit_ecm_parameters) are kept for the
existing figures.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.interpolate import interp1d

ECM_P0 = [0.005, 0.005, 10.0, 0.010, 100.0]
ECM_BOUNDS = ([1e-5, 1e-5, 0.5, 1e-5, 5.0], [0.1, 0.1, 100.0, 0.1, 2000.0])


def ocv_curve(q, v):
    """Monotone-in-q OCV table from points (q [A.h] below the top, V [V]), linear extrapolation."""
    o = np.argsort(q)
    q, v = np.asarray(q, float)[o], np.asarray(v, float)[o]
    q, i = np.unique(q, return_index=True)
    v = v[i]
    return interp1d(q, v, kind='linear', fill_value='extrapolate', assume_sorted=True)


def _branch(I, dt, R, tau):
    """RC voltage v_k = a v_(k-1) + R (1 - a) I_(k-1), a = exp(-dt_(k-1) / tau); dt one step or one per sample
    (piecewise constant: one lfilter per stretch of equal steps, the state carried across)."""
    if np.ndim(dt) == 0:
        a = np.exp(-dt / tau)
        return lfilter([0.0, R * (1.0 - a)], [1.0, -a], I)
    u = np.r_[0.0, I[:-1]]                                       # the current that drives sample k
    d = np.r_[dt[0], dt[:-1]]                                    # ... over the step of sample k - 1
    v, prev = np.empty(len(I)), 0.0
    cuts = np.r_[0, np.flatnonzero(np.diff(d) != 0) + 1, len(I)]
    for s, e in zip(cuts[:-1], cuts[1:]):
        a = np.exp(-d[s] / tau)
        v[s:e], _ = lfilter([R * (1.0 - a)], [1.0, -a], u[s:e], zi=[a * prev])
        prev = v[e - 1]
    return v


def simulate_q(params, I, dt, ocv_q, q0=0.0):
    """dt [s]: one step, or one per sample (data.coarsen, align=True)."""
    R0, R1, tau1, R2, tau2 = params
    v1, v2 = _branch(I, dt, R1, tau1), _branch(I, dt, R2, tau2)
    q = q0 + np.concatenate(([0.0], np.cumsum(I[:-1] * np.broadcast_to(dt, I.shape)[:-1]))) / 3600.0
    return ocv_q(q) - I * R0 - v1 - v2


def fit_q(I, V, dt, ocv_q, q0=0.0, mask=None, downsample_s=5.0):
    """(R0, R1, tau1, R2, tau2) on one test, every `downsample_s` of it."""
    step = max(1, int(round(downsample_s / float(np.median(dt)))))
    m = np.ones(len(I), bool) if mask is None else np.asarray(mask, bool)
    keep = np.zeros(len(I), bool)
    keep[::step] = True
    keep &= m
    res = least_squares(lambda p: (simulate_q(p, I, dt, ocv_q, q0) - V)[keep], ECM_P0, bounds=ECM_BOUNDS,
                        method='trf', x_scale='jac')
    return res.x

def extract_ecm_ocv(q_pts, v_pts):
    """Construct a monotonic OCV(SOC) function from quasi-equilibrium OCV points."""
    Q_tot = float(q_pts.max() + 0.2)
    soc_pts = 1.0 - q_pts / Q_tot
    idx = np.argsort(soc_pts)
    soc_sorted, v_sorted = soc_pts[idx], v_pts[idx]
    if soc_sorted[0] > 0.0:
        soc_sorted = np.insert(soc_sorted, 0, 0.0)
        v_sorted = np.insert(v_sorted, 0, min(2.5, v_sorted[0] - 0.5))
    if soc_sorted[-1] < 1.0:
        soc_sorted = np.append(soc_sorted, 1.0)
        v_sorted = np.append(v_sorted, max(4.2, v_sorted[-1] + 0.02))
    ocv_fn = interp1d(soc_sorted, v_sorted, kind='linear', fill_value='extrapolate')
    return ocv_fn, Q_tot

def simulate_ecm(params, I_A, dt_s, ocv_fn, Q_tot, soc0=1.0):
    """Simulate Thevenin 2-RC terminal voltage."""
    R0, R1, tau1, R2, tau2 = params
    a1 = np.exp(-dt_s / tau1)
    a2 = np.exp(-dt_s / tau2)
    b1 = R1 * (1.0 - a1)
    b2 = R2 * (1.0 - a2)
    v1 = lfilter([0.0, b1], [1.0, -a1], I_A)
    v2 = lfilter([0.0, b2], [1.0, -a2], I_A)
    q = np.concatenate(([0.0], np.cumsum(I_A[:-1]) * dt_s)) / 3600.0
    soc = soc0 - q / Q_tot
    return ocv_fn(soc) - I_A * R0 - v1 - v2

def fit_ecm_parameters(I_A, V_V, dt_s, ocv_fn, Q_tot, downsample_s=5.0):
    """Calibrate (R0, R1, tau1, R2, tau2) from characterization test."""
    down = max(1, int(round(downsample_s / dt_s)))
    Ih_sub, Vh_sub, dt_sub = I_A[::down], V_V[::down], dt_s * down
    
    def res_fn(p):
        return simulate_ecm(p, Ih_sub, dt_sub, ocv_fn, Q_tot) - Vh_sub
    
    p0 = [0.005, 0.005, 10.0, 0.010, 100.0]
    bounds = ([1e-5, 1e-5, 0.5, 1e-5, 5.0], [0.1, 0.1, 100.0, 0.1, 1000.0])
    res = least_squares(res_fn, p0, bounds=bounds, method='trf', ftol=1e-5, xtol=1e-5)
    return res.x
