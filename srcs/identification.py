"""
identification.py
=================
Tests, initial state and cost, shared by every stage and every dataset.

A test is a dict:
    I [A] (> 0 discharge, mean over each interval), V [V] (interval midpoint), T [K], dt [s]
    mask        samples that are scored (EIS sweeps excluded)
    cell, test, label, cfg (the prepared folder's config.json), coarsen
    window      the cell's capacity window (Q_n, Q_p, x100, y100) from the static stage
    q_start     charge below the top of the window at t = 0 [A.h]
    ref         (relative tests only) the reference sample of every sample, see below

Relative tests. A test spec with "relative": true scores every pulse, and the rest after it, relative to
the last rest sample before the pulse: the residual is (V_sim[k] - V_sim[ref[k]]) - (V[k] - V[ref[k]]).
An offset between the model's OCV and the cell's rest voltage that is constant across a pulse (an OCP
mismatch, a hysteresis branch) cancels; what is scored is the pulse's overpotential and its relaxation,
plus the OCV step across the pulse. The samples before the first pulse are not scored.

Window correction factors. A parameter win_<group> (e.g. win_y100, win_Q_p), fitted or fixed like any
other, scales that group of EVERY cell's own window: a stage can correct a bias the window stage shares
across cells (the LG M50T C/10 pseudo-OCV) on the tests it already fits. The windows on
disk stay as the window stage found them; test_groups() applies the factors for every simulation, and a
test that starts at rest gets its initial charge recomputed from the corrected window, per candidate.

The cost of a parameter set is the voltage RMSE [mV] over the masked samples (relative ones for a
relative test), averaged over the tests
of the stage, so a long test does not outweigh a short one. A population is evaluated as one batch by
srcs.em_model.BatchEM, in chunks sized to the free GPU memory.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from srcs import data

WINDOW = ('Q_n', 'Q_p', 'x100', 'y100')
FACTOR = 'win_'                     # win_<group>: a correction factor on every cell's window group


# ------------------------------------------------------------------ data
def clean_T(T_C, dt, smooth_s=60.0, dT_q=0.2, spike_C=3.0, median_s=300.0):
    """Measured temperature [K] as the model sees it.
    1. thermocouple dropouts: a sample further than spike_C from the rolling median over median_s is
       replaced by that median (e.g. LG M50 cell E4H reads -1 C for a moment in a 25 C chamber);
    2. smoothed over smooth_s and quantised to dT_q, so the model sees long stretches of constant T
       (the cell's thermal time constant is minutes; each change of T opens a new time window)."""
    T = pd.Series(np.asarray(T_C, dtype=float))
    w = max(1, int(round(median_s / dt))) | 1
    med = T.rolling(w, center=True, min_periods=1).median()
    T = T.where((T - med).abs() <= spike_C, med).to_numpy() + 273.15
    w = max(1, int(round(smooth_s / dt)))
    if w > 1 and len(T) > w:
        T = np.convolve(np.pad(T, (w // 2, w - 1 - w // 2), mode='edge'), np.ones(w) / w, mode='valid')
    return np.round(T / dT_q) * dT_q


def load_test(ds, cell: str, test: str, coarsen: int = 1, T_C: float | None = None, fine: int = 0,
              from_discharge: bool = False, from_q: float = 0.0, duration_s: float = 0.0, align: bool = True,
              v_mean: bool = False, adaptive: dict | None = None, from_s: float = 0.0) -> dict:
    """One prepared test. coarsen: odd factor (current averaged, V and T at the coarse midpoint), aligned
    to the changes of current (data.coarsen), keeping the first `fine` samples after each change at the
    test's step: dt is then one step per sample, t the sample start times; T_C: a fixed temperature
    instead of the measured one (the isothermal plant); from_discharge: start at the rest before the
    first discharge (e.g. the C/20's rested top, leaving out its charge from an unrested bottom);
    from_q [A.h]: then start where that much has been discharged (the end of a slow discharge, its knee;
    the test's initial charge must then be given, prepare_tests); duration_s: keep only that long from the
    start (after the cuts above; 0: all), e.g. the C/10's knee and its rest without the charge after it.
    align=False: the fixed grid of `coarsen` samples (data.coarsen), for a drive cycle whose current changes
    almost every sample, where the aligned grid keeps nearly every sample at the test's step (`fine` unused); v_mean: then V is
    the mean over each interval (data.coarsen). from_s [s]: drop the samples before that time first (e.g. a cycle that starts
    at the end of a CV, which is the top of the window: the test then starts under current, at q = 0)."""
    df, cfg = ds.load(cell, test)
    if from_s > 0:
        df = df.iloc[int(np.searchsorted(df['t_s'].to_numpy() - float(df['t_s'].iloc[0]), from_s)):].reset_index(drop=True)
    if from_discharge:
        I = df['I_A'].to_numpy()
        k = int(np.flatnonzero(I > 0.02)[0])
        while k > 0 and abs(I[k - 1]) <= 0.02:
            k -= 1
        df = df.iloc[k:].reset_index(drop=True)
    if from_q > 0:
        qd = np.cumsum(np.clip(df['I_A'].to_numpy(), 0.0, None)) * float(cfg['dt_s']) / 3600.0
        df = df.iloc[int(np.searchsorted(qd, from_q)):].reset_index(drop=True)
    if duration_s > 0:
        df = df.iloc[:int(np.searchsorted(df['t_s'].to_numpy() - float(df['t_s'].iloc[0]), duration_s))].reset_index(drop=True)
    if adaptive:          # {"fast": odd, "slow": odd, "fine_s": s, "dv_mV": mV} (data.coarsen_adaptive)
        d, dt = data.coarsen_adaptive(df, float(cfg['dt_s']), int(adaptive.get('fast', 1)), int(adaptive.get('slow', 3)),
                                      float(adaptive.get('fine_s', 60.0)), 1e-3 * float(adaptive.get('dv_mV', 3.0)))
    else:
        d, dt = data.coarsen(df, float(cfg['dt_s']), int(coarsen), align=bool(align), fine=int(fine),
                             v_mean=bool(v_mean))
    if T_C is not None:
        T = np.full(len(d), float(T_C) + 273.15)
    elif 'T_C' in d.columns:
        T = clean_T(d['T_C'].to_numpy(), float(np.median(dt)))
    else:
        T = np.full(len(d), float(cfg.get('temperature_C', 25.0)) + 273.15)
    mask = ~d['eis'].to_numpy().astype(bool) if 'eis' in d.columns else np.ones(len(d), bool)
    return dict(I=d['I_A'].to_numpy(), V=d['V_V'].to_numpy(), T=T, dt=dt, mask=mask, cell=cell, test=test,
                t=d['t_s'].to_numpy() - float(d['t_s'].iloc[0]),
                label='%s %s' % (cell, test), cfg=cfg, coarsen=int(coarsen))


def initial_charges(test: dict, w: dict, ocp, i_rest: float = 0.02, n: int = 2000):
    """Charge [A.h] below the top of the window at t = 0, (B,), for windows whose groups are floats or
    (B,) arrays. Every prepared test starts at the top of charge (datasets/base.py): a test that starts
    under current starts AT the top (the discharge that follows the CV); one that starts at rest is placed
    by inverting the window's OCV at its first voltage (grid of n points, refined linearly), which puts a
    rested top a few mV below V_max at q ~ 0."""
    W = {k: np.atleast_1d(np.asarray(w[k], dtype=float)) for k in WINDOW}
    B = max(len(v) for v in W.values())
    if abs(test['I'][0]) > i_rest:
        return np.zeros(B)
    W = {k: np.broadcast_to(v, (B,))[:, None] for k, v in W.items()}
    q = np.linspace(0.0, 1.0, n)[None, :] * 0.99 * np.minimum(W['Q_n'] * W['x100'], W['Q_p'] * (1 - W['y100']))
    v = ocp.U_p(W['y100'] + q / W['Q_p']) - ocp.U_n(W['x100'] - q / W['Q_n'])
    V0, r = float(test['V'][0]), np.arange(B)
    i = np.argmin(np.abs(v - V0), axis=1)
    j = np.clip(np.where(v[r, i] > V0, i + 1, i - 1), 0, n - 1)
    den = v[r, j] - v[r, i]
    frac = np.where(np.abs(den) > 1e-12, (V0 - v[r, i]) / np.where(np.abs(den) > 1e-12, den, 1.0), 0.0)
    qs = np.clip(q[r, i] + np.clip(frac, 0.0, 1.0) * (q[r, j] - q[r, i]), 0.0, None)
    return np.where(V0 < v[:, 0], qs, 0.0)


def initial_charge(test: dict, w: dict, ocp, i_rest: float = 0.02) -> float:
    """initial_charges for one window."""
    return float(initial_charges(test, w, ocp, i_rest)[0])


def test_groups(t: dict, g: dict, ocp):
    """(groups, q_start) to simulate test t with the groups g: t's window under g, with g's win_<group>
    factors applied to it; q_start is t's own unless a factor moves the window of a test that starts at
    rest, then one per candidate."""
    if THETA_REF_K is not None:       # theta_n, theta_p given at THETA_REF_K: back to 25 C through their Arrhenius laws
        g = dict(g)
        for th, ea in (('theta_n', 'Ea_Dn'), ('theta_p', 'Ea_Dp')):
            if th in g and ea in g:
                g[th] = np.asarray(g[th], float) * np.exp(-np.asarray(g[ea], float) / 8.314462618
                                                          * (1.0 / THETA_REF_K - 1.0 / 298.15))
    f = {k[len(FACTOR):]: v for k, v in g.items() if k.startswith(FACTOR)}
    if not f:
        return {**t['window'], **g}, t['q_start']
    w = {k: t['window'][k] * np.asarray(f[k], dtype=float) if k in f else t['window'][k] for k in WINDOW}
    q = t['q_start'] if abs(t['I'][0]) > 0.02 else initial_charges(t, w, ocp)
    return {**g, **w}, q


# ------------------------------------------------------------------ output bias dV(y)
BIAS = 'dVy@'        # groups "dVy@<y>" [V]: a piecewise-linear voltage bias on the cathode's mean stoichiometry


def output_bias(t: dict, g: dict, q, B: int):
    """(B, N) voltage bias [V] of test t for the groups g (test_groups' merged groups): piecewise linear in the
    cathode's mean stoichiometry y = y100 + q / Q_p (coulomb counting from q_start, sample midpoints), knots at
    the y of the groups named BIAS + "<y>", flat outside them; None if there is no such group. A static
    counterpart of an augmented bias state: it takes the SOC-dependent voltage errors (OCV, entropy, shape)
    off the dynamic groups."""
    knots = sorted((float(k[len(BIAS):]), k) for k in g if k.startswith(BIAS))
    if not knots:
        return None
    ky = np.array([k[0] for k in knots])
    vals = np.stack([np.broadcast_to(np.asarray(g[k], dtype=float), (B,)) for _, k in knots], 1)      # (B, K)
    dt = np.broadcast_to(np.asarray(t['dt'], dtype=float), t['I'].shape)
    qn = (np.cumsum(t['I'] * dt) - 0.5 * t['I'] * dt) / 3600.0
    y100 = np.broadcast_to(np.asarray(g['y100'], dtype=float), (B,))
    Qp = np.broadcast_to(np.asarray(g['Q_p'], dtype=float), (B,))
    q0 = np.broadcast_to(np.asarray(q, dtype=float), (B,))
    y = y100[:, None] + (q0[:, None] + qn[None, :]) / Qp[:, None]
    return np.stack([np.interp(y[b], ky, vals[b]) for b in range(B)])


THETA_REF_K = None      # [K], study config model.theta_ref_C: the fitted theta_n, theta_p are the
                        # diffusion times at this temperature instead of 25 C (test_groups converts them back)
CORE_HEAT = None        # [k K/A^2, tau s], study config model.core_heat: prepare_tests adds to the
                        # measured surface temperature a core rise k I^2 through a first-order lag of time constant tau
CLAMP_CUTOFF = None     # [V], study config model.clamp_cutoff (srcs/study.Study.em): see clamp_cutoff below
HYSTERESIS = None       # study config model.hysteresis (srcs/study.Study.em): {"x": knots, "dV": [V], "q_h_Ah",
                        # "h0"} - the OCV hysteresis of a silicon-graphite anode, see hysteresis_voltage below


def hysteresis_voltage(t, h0=None):
    """OCV hysteresis of a test [V], per sample. The model's OCV is the DISCHARGE branch (step 1: its shape
    from the slow discharge, its level from the rests after discharge pulses); after a charge the cell sits on the
    charge branch, dV(x) above it (HYSTERESIS: piecewise linear on the bulk anode stoichiometry
    x = x100 - (q_start + q) / Q_n, measured on the slow cycles and rests, scripts/hysteresis_c10_gitt.py). The branch
    state H in [0, 1] (1: charge branch) relaxes towards 1 while charging and towards 0 while discharging, by
    exp(-|charge moved| / q_h_Ah), and stays at rest (a first-order hysteresis of the charge throughput). H(t) and x(t)
    depend only on the current and the cell's window, not on the fitted groups, so the term is computed once per test."""
    hy = HYSTERESIS
    I = np.asarray(t['I'], float)
    dt = np.broadcast_to(np.asarray(t['dt'], float), I.shape)
    dq = I * dt / 3600.0
    q = np.cumsum(dq) - 0.5 * dq                                  # at the sample midpoint
    w = t['window']
    x = w['x100'] - (t['q_start'] + q) / w['Q_n']
    dV = np.interp(x, np.asarray(hy['x'], float), np.asarray(hy['dV'], float))
    H = np.empty(len(I))
    h = float(hy.get('h0', 1.0) if h0 is None else h0)
    k = np.exp(-np.abs(dq) / float(hy['q_h_Ah']))
    for i in range(len(I)):
        if I[i] < -0.02:
            h = 1.0 - (1.0 - h) * k[i]
        elif I[i] > 0.02:
            h = h * k[i]
        H[i] = h
    return H * dV


def clamp_cutoff(V, t: dict):
    """The model voltage held at the discharge cutoff once it goes below it, as the cycler stops the discharge
    there. The current is imposed, so a model that reaches the cutoff before the cell keeps
    discharging past its window and collapses by hundreds of mV in a few samples: a step in the cost that the
    Levenberg-Marquardt search cannot cross. Held, its error grows smoothly with how early it gets there.
    The voltage held is the lower of CLAMP_CUTOFF and the test's lowest measured voltage, so a test that never
    reaches the cutoff (a partial discharge) keeps its full error above it."""
    if CLAMP_CUTOFF is None:
        return V
    v = min(float(CLAMP_CUTOFF), float(np.min(t['V'])))
    if isinstance(V, torch.Tensor):
        return torch.clamp(V, min=v)
    return np.maximum(V, v)


def add_bias(V, t: dict, g: dict, q):
    """V (B, N) torch tensor or numpy array plus output_bias (unchanged without bias groups), held at the test's
    cutoff when CLAMP_CUTOFF (clamp_cutoff)."""
    V = clamp_cutoff(V, t)
    b = output_bias(t, g, q, V.shape[0])
    if b is None:
        return V
    if isinstance(V, torch.Tensor):
        return V + torch.as_tensor(b, dtype=V.dtype, device=V.device)
    return V + b


def pulse_refs(I, i_rest: float = 0.02) -> np.ndarray:
    """For every sample, the index of the last rest sample (|I| <= i_rest) before the pulse it belongs to
    (a pulse and the rest after it share one); -1 before the first pulse."""
    rest = np.abs(np.asarray(I, float)) <= i_rest
    k = np.arange(len(rest))
    start = np.zeros(len(rest), bool)
    start[1:] = rest[:-1] & ~rest[1:]
    return np.maximum.accumulate(np.where(start, k - 1, -1))


def relative(V, t):
    """V (..., K) relative to the test's reference samples for a relative test, else V itself."""
    return V - V[..., t['ref']] if 'ref' in t else V


def knee_slopes(t, knee_V=3.45, s_min=0.3, dq=0.01, settle_s=60.0):
    """(mask, inv_slope [A.h/V]) of the knee samples of test t: discharge samples below knee_V (data), at
    least settle_s into their stretch of constant sign, with the data's slope |dV/dq| over +-dq [A.h] at
    least s_min [V/A.h]. A voltage error e divided by that slope is the charge by which the model reaches
    the same voltage early or late: at the knee's steep end that is the capacity error, which a voltage
    error there (100+ mV for a few tens of mA.h) measures badly."""
    I, V = t['I'], t['V']
    dt = np.broadcast_to(np.asarray(t['dt'], float), I.shape)
    tt = np.cumsum(dt)
    inv = np.zeros(len(I))
    m = np.zeros(len(I), bool)
    dis = (I > 0.02).astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, dis, 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a < 3:
            continue
        q = np.cumsum(I[a:b] * dt[a:b]) / 3600.0
        Vs = V[a:b]
        hi, lo = np.minimum(q + dq, q[-1]), np.maximum(q - dq, q[0])
        s = (np.interp(hi, q, Vs) - np.interp(lo, q, Vs)) / np.maximum(hi - lo, 1e-9)
        ok = (Vs < knee_V) & (tt[a:b] - tt[a] >= settle_s) & (-s >= s_min)
        m[a:b] = ok
        inv[a:b] = np.where(ok, 1.0 / np.maximum(-s, s_min), 0.0)
    return m, inv


def prepare_tests(ds, specs, windows, ocp) -> list:
    """Load the tests of a stage. spec: {"cell": "E4D", "test": "GITT", "coarsen": 5, "T_C": 25.0,
    "V_min": 3.0, "align": true}; the cell's window and the initial charge are attached. V_min [V]: samples below it are
    left out of the residual (not of the simulation) - at the bottom of charge the OCV is so steep that
    a coulomb-counting error of a few mA over a long test moves the voltage by ~100 mV.
    "relative": true scores the test relative to the rest before each pulse (module docstring); a sample
    is scored only if its reference sample passes the mask as well."""
    tests = []
    for s in specs:
        t = load_test(ds, s['cell'], s['test'], s.get('coarsen', 1), s.get('T_C'), s.get('fine', 0),
                      s.get('from_discharge', False), s.get('from_q', 0.0), s.get('duration_s', 0.0),
                      s.get('align', True), s.get('v_mean', False), s.get('adaptive'), s.get('from_s', 0.0))
        if CORE_HEAT is not None:          # the core above the measured surface: k I^2 through a first-order lag
            k_h, tau_h = (float(v) for v in CORE_HEAT)
            dt_h = np.broadcast_to(np.asarray(t['dt'], float), t['I'].shape)
            rise, r = np.zeros(len(t['I'])), 0.0
            for i, (ii, dd) in enumerate(zip(t['I'], dt_h)):
                r += min(dd / tau_h, 1.0) * (k_h * ii * ii - r)
                rise[i] = r
            t['T'] = np.asarray(t['T'], float) + rise
        if 'V_min' in s:
            t['mask'] = t['mask'] & (t['V'] >= float(s['V_min']))
        if 'V_max' in s:                 # likewise above it (an unrelaxed top after a CV, the steep end of an LFP)
            t['mask'] = t['mask'] & (t['V'] <= float(s['V_max']))
        if 'skip_s' in s:                # the first seconds left out of the residual (a test that starts unrelaxed)
            t['mask'] = t['mask'] & (t['t'] >= float(s['skip_s']))
        if 'exclude_I' in s:             # samples whose current lies in [lo, hi] A left out of the score
            lo, hi = (float(v) for v in s['exclude_I'])   # (LG: the slow 0.15 A step of the first cycles)
            t['mask'] = t['mask'] & ~((t['I'] >= lo) & (t['I'] <= hi))
        if s.get('rests_only'):          # a GITT-like score: only the samples at rest (the relaxations)
            t['mask'] = t['mask'] & (np.abs(t['I']) <= 0.02)
            t['label'] += ' (rests)'
        if s.get('q_windows'):           # [[q_lo, q_hi], ...] A.h of net charge discharged since the test start
            qn = np.cumsum(t['I'] * np.broadcast_to(np.asarray(t['dt'], float), t['I'].shape)) / 3600.0
            inw = np.zeros(len(qn), bool)
            for lo, hi in s['q_windows']:
                inw |= (qn >= float(lo)) & (qn < float(hi))
            t['mask'] = t['mask'] & inw
            t['label'] += ' (q-windows)'
        if s.get('relative'):
            ref = pulse_refs(t['I'])
            t['ref'] = np.where(ref >= 0, ref, np.arange(len(ref)))
            t['mask'] = t['mask'] & (ref >= 0) & t['mask'][t['ref']]
            t['label'] += ' (rel.)'
        # scores: (name, mask, weight) summed into the cost; "transients_s": a second score on the samples
        # within that time of a change of current (pulses and the start of the rest after them)
        t['scores'] = [('all', t['mask'], float(s.get('weight', 1.0)))]
        if 'transients_s' in s:
            I = t['I']
            start = np.r_[True, np.abs(np.diff(I)) > 0.05 * max(np.abs(I).max(), 1e-9)]
            t0 = np.maximum.accumulate(np.where(start, t['t'], -np.inf))
            near = (t['t'] - t0) < float(s['transients_s'])
            t['scores'].append(('transients', t['mask'] & near, float(s.get('weight_transients', 1.0))))
        # per-sample weights in the scores: "knee_weight" k on the discharge samples below "knee_V" (data), where
        # the error peaks (the end of a discharge); 1 elsewhere
        t['sample_w'] = np.ones(len(t['I']))
        if 'knee_weight' in s:
            knee = (t['I'] > 0.02) & (t['V'] < float(s.get('knee_V', 3.45)))
            t['sample_w'] = np.where(knee, float(s['knee_weight']), 1.0)
        # adaptive grid: each sample weighs its interval's duration, so the score stays the time-mean squared
        # error of the uniform grid (otherwise the coarse plateaus count less and the transients more)
        if s.get('adaptive') and s['adaptive'].get('time_weight', True):
            dt = np.broadcast_to(np.asarray(t['dt'], float), t['I'].shape)
            t['sample_w'] = t['sample_w'] * dt / dt.min()
        # "knee_q_weight": a score on the knee in charge (knee_slopes): the voltage error over the data's slope,
        # in mA.h, counted as mV at that weight; uniform sample weights
        t['score_scale'] = {}
        if 'knee_q_weight' in s:
            km, inv = knee_slopes(t, float(s.get('knee_V', 3.45)))
            if km.any():
                t['scores'].append(('knee_q', t['mask'] & km, float(s['knee_q_weight'])))
                t['score_scale']['knee_q'] = inv
        t['window'] = {k: float(windows[s['cell']][k]) for k in WINDOW}
        if 'from_q' in s:            # below the window's top: the cut, less the test's axis shift from it (step 1)
            t['q_start'] = float(s['from_q']) - float(s.get('axis_shift', {}).get(s['cell'], 0.0))
        else:
            # "q_start" of the spec, or the test's own (a charge that ends at the top: its charge, config q_start_Ah)
            q0 = s.get('q_start', t['cfg'].get('q_start_Ah'))
            t['q_start'] = float(q0) if q0 is not None else initial_charge(t, t['window'], ocp)
        if HYSTERESIS is not None:       # the model has no hysteresis, so the target is the measured
            t['Vh'] = hysteresis_voltage(t, s.get('h0'))     # voltage less its charge-branch excess (V_meas = V + Vh)
            t['V_meas'] = t['V']
            t['V'] = t['V'] - t['Vh']
        tests.append(t)
    return tests


# ------------------------------------------------------------------ cost
PHI_LIMIT = 3.0          # |Phi| the particle network is trained for (mionet-clean hpo_rollout PHI_CLIP)
PHI_PENALTY_MV = 100.0   # cost added per unit of |Phi| beyond it: the surrogate is not trusted there, so a
                         # candidate must not earn its cost from it (srcs/ms_model.py superstep)


def auto_chunk(K: int, device: str, frac: float = 0.4, bytes_per_sample: int = 160) -> int:
    """Candidates per GPU call for a test of K samples: `frac` of the free memory over an empirical
    footprint per candidate and sample (both particles' reports and states, kinetics, electrolyte)."""
    if device != 'cuda' or not torch.cuda.is_available():
        return 512
    free, _ = torch.cuda.mem_get_info()
    return int(np.clip(frac * free / (K * bytes_per_sample), 32, 20000))


class Cost:
    """cost(X) -> mean voltage RMSE [mV] over the tests' scores (prepare_tests: one per test, plus one on
    the transients where asked; weighted mean; within a score the samples carry the test's sample_w) for a
    (P, d) array of the parameters `names`; every other group is taken from `fixed`, and each test's own
    window is merged in. With a multi-slice model in superstep mode, PHI_PENALTY_MV per unit of the largest
    |Phi| beyond PHI_LIMIT is added."""

    def __init__(self, em, tests, names, fixed, chunk=None):
        self.em, self.tests, self.names, self.fixed = em, tests, list(names), dict(fixed)
        self.chunk = {t['label']: chunk or auto_chunk(len(t['I']), em.device) for t in tests}
        self.n_calls = 0
        for t in tests:
            t.setdefault('scores', [('all', t['mask'], 1.0)])
        self.w_total = sum(w for t in tests for _, _, w in t['scores'])

    def groups(self, X, t):
        """(groups, q_start) of the candidates X on test t (test_groups)."""
        return test_groups(t, {**self.fixed, **{n: X[:, i] for i, n in enumerate(self.names)}}, self.em.ocp)

    def __call__(self, X):
        X = np.atleast_2d(X)
        out = np.zeros(len(X))
        phi = np.zeros(len(X))
        self.n_calls += 1
        for t in self.tests:
            Vt = torch.as_tensor(relative(t['V'], t), dtype=torch.float32, device=self.em.device)[None, :]
            ms = self._score_weights(t)
            r = torch.as_tensor(t['ref'], device=self.em.device) if 'ref' in t else None
            c = self.chunk[t['label']]
            for a in range(0, len(X), c):
                g, q = self.groups(X[a:a + c], t)
                V = add_bias(self.em.simulate(g, t['I'], t['dt'], T=t['T'], q_start=q), t, g, q)
                e = ((V - V[:, r]) if r is not None else V) - Vt
                for m, sw, w, sc in ms:
                    out[a:a + c] += w / self.w_total * 1e3 * torch.sqrt(((e[:, m] * sc) ** 2 * sw).sum(1)).cpu().numpy()
                if getattr(self.em, 'last_phi_max', None) is not None and getattr(self.em, 'superstep', False):
                    phi[a:a + c] = np.maximum(phi[a:a + c], self.em.last_phi_max.cpu().numpy())
        self.last_phi_max = phi
        out = out + PHI_PENALTY_MV * np.clip(phi - PHI_LIMIT, 0.0, None)
        return np.where(np.isfinite(out), out, np.inf)

    def _score_weights(self, t):
        """[(sample indices, normalised sample weights, score weight, error scale)] of test t's scores on the
        device. A score with an error scale (t['score_scale'], e.g. knee_q: mA.h per mV) has uniform weights."""
        sw = t.get('sample_w', np.ones(len(t['I'])))
        scale = t.get('score_scale', {})
        out = []
        for name, m, w in t['scores']:
            k = np.flatnonzero(m)
            wk = np.ones(len(k)) if name in scale else sw[k]
            sc = scale[name][k] if name in scale else np.ones(len(k))
            f32 = lambda z: torch.as_tensor(z, dtype=torch.float32, device=self.em.device)
            out.append((torch.as_tensor(k, device=self.em.device), f32(wk / wk.sum()), w, f32(sc)))
        return out

    def simulate(self, x, t):
        """Voltage of ONE parameter set on test t (numpy)."""
        g, q = test_groups(t, {**self.fixed, **{n: float(x[i]) for i, n in enumerate(self.names)}}, self.em.ocp)
        return add_bias(self.em.simulate(g, t['I'], t['dt'], T=t['T'], q_start=q), t, g, q)[0].cpu().numpy()

    def rmse_per_test(self, x):
        """RMSE [mV] of one parameter set on each test's scores (relative for a relative test)."""
        out = {}
        for t in self.tests:
            e = relative(self.simulate(x, t) - t['V'], t)
            scale = t.get('score_scale', {})
            for name, m, _ in t['scores']:
                sc = scale[name][m] if name in scale else 1.0
                out[t['label'] if name == 'all' else '%s (%s)' % (t['label'], name)] = float(1e3 * np.sqrt(np.mean((e[m] * sc) ** 2)))
        return out


class ResidualCost(Cost):
    """residuals(X) -> (P, M) weighted voltage residuals [mV] of every score's samples, concatenated, whose
    sum of squares is the weighted mean of the scores' (sample-weighted) mean squared errors: the least-
    squares form of Cost (srcs/optimizers.lm_multistart). The superstep |Phi| penalty is not applied."""

    def residuals(self, X):
        X = np.atleast_2d(X)
        cols = []
        self.n_calls += 1
        for t in self.tests:
            Vt = torch.as_tensor(relative(t['V'], t), dtype=torch.float32, device=self.em.device)[None, :]
            ms = self._score_weights(t)
            r = torch.as_tensor(t['ref'], device=self.em.device) if 'ref' in t else None
            c = self.chunk[t['label']]
            blocks = [[] for _ in ms]
            for a in range(0, len(X), c):
                g, q = self.groups(X[a:a + c], t)
                V = add_bias(self.em.simulate(g, t['I'], t['dt'], T=t['T'], q_start=q), t, g, q)
                e = ((V - V[:, r]) if r is not None else V) - Vt
                for b, (m, sw, w, sc) in zip(blocks, ms):
                    b.append((1e3 * e[:, m] * sc * torch.sqrt(sw * w / self.w_total)).cpu().numpy())
            cols += [np.vstack(b) for b in blocks]
        R = np.hstack(cols)
        return np.where(np.isfinite(R), R, 1e3)


def save(path, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=conv)
