"""loader.py
=========
Unified dataset loading for real cells and simulated plant data.
"""

import os
import json
import numpy as np
import pandas as pd

from srcs.paths import ROOT as REPO_ROOT, DATA as DATA_ROOT

def load_real_cell(cell_id, test_name):
    """Load cycles.csv and config.json for a real cell and characterization/validation test.
    
    Parameters
    ----------
    cell_id : int
        Cell number (1 to 21).
    test_name : str
        'HPPC', 'C20', 'VAL', or 'CRATE1C'.
        
    Returns
    -------
    df : pd.DataFrame
    config : dict
    """
    folder = os.path.join(DATA_ROOT, f"DB_REAL_C{cell_id:02d}_{test_name}_BOL")
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}. Did you run prepare.py?")
    
    df = pd.read_csv(os.path.join(folder, 'cycles.csv'))
    cfg = json.load(open(os.path.join(folder, 'config.json'), 'r'))
    return df, cfg

def load_plant_test(test_name, temp_tag="25C"):
    """Load cycles.csv and config.json for simulated plant data.
    
    Parameters
    ----------
    test_name : str
        e.g. 'ECM_HPPC', 'HPPC_MULTI', 'VAL_CC', 'VAL_DST'.
    temp_tag : str
        e.g. '25C', '10C', '40C'.
    """
    folder = os.path.join(DATA_ROOT, f"DB_IDENT_{test_name}_{temp_tag}")
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Plant dataset folder not found: {folder}")
    df = pd.read_csv(os.path.join(folder, 'cycles.csv'))
    cfg = json.load(open(os.path.join(folder, 'config.json'), 'r'))
    return df, cfg

def extract_hppc_rest_ends(df, dt_s, min_rest_s=1200.0):
    """Extract (throughput_Ah, Voltage_V) at the end of clean HPPC rests."""
    I = df['I_A'].to_numpy()
    V = df['V_V'].to_numpy()
    eis = df['eis'].to_numpy().astype(bool) if 'eis' in df.columns else np.zeros(len(I), bool)
    
    q = np.concatenate(([0.0], np.cumsum(I[:-1]) * dt_s)) / 3600.0
    rest = np.abs(I) < 0.02
    edges = np.flatnonzero(np.diff(rest.astype(int)) != 0) + 1
    
    pts = []
    for r in np.split(np.arange(len(I)), edges):
        if rest[r[0]] and len(r) * dt_s >= min_rest_s:
            clean = r[~eis[r]]
            if len(clean) and clean[-1] >= r[-1] - 60 / dt_s:
                pts.append((q[clean[-1]], V[clean[-1]]))
    pts = np.array(pts)
    return pts[:, 0], pts[:, 1]


def current_segments(I, jump=0.05):
    """Start indices of the stretches of constant current: a new one opens where the current moves by more
    than jump * max|I| from its value at the stretch's start (em_model.windows uses the same rule)."""
    I = np.asarray(I, dtype=float)
    tol = jump * max(float(np.max(np.abs(I))), 1e-9)
    starts, a = [0], I[0]
    for k in range(1, len(I)):
        if abs(I[k] - a) > tol:
            starts.append(k)
            a = I[k]
    return np.array(starts)


def coarsen_adaptive(df, dt_s, fast=1, slow=3, fine_s=60.0, dv=0.003, jump=0.05):
    """Adaptive resampling with two steps only (few runs of constant dt for the solver). Within each stretch
    of constant current (current_segments): the first `fine_s` seconds after the change at `fast` samples per
    interval (the transient), then intervals of `slow` samples wherever the measured voltage moves by less
    than `dv` [V] across the interval (plateaus, the end of rests), otherwise `fast` (the knee, steep parts).
    `fast`, `slow` odd; the current is the mean over each interval, V and T the midpoint sample, as in coarsen.
    Returns (DataFrame, dt array)."""
    fast, slow = int(fast), int(slow)
    if fast % 2 == 0 or slow % 2 == 0:
        raise ValueError('fast and slow must be odd, got %d and %d' % (fast, slow))
    I, V = df['I_A'].to_numpy(), df['V_V'].to_numpy()
    seg = current_segments(I, jump)
    A, S = [], []
    n_fine = int(round(float(fine_s) / dt_s))
    for a, b in zip(seg, np.r_[seg[1:], len(I)]):
        k = a
        while k < b:
            if k - a < n_fine or b - k < slow or np.ptp(V[k:k + slow]) >= dv:
                L = min(fast, b - k)
                if L % 2 == 0:
                    L = 1
            else:
                L = slow
            A.append(k)
            S.append(L)
            k += L
    A, S = np.asarray(A, int), np.asarray(S, int)
    mid = A + S // 2
    out = pd.DataFrame({'t_s': df['t_s'].to_numpy()[A], 'I_A': np.add.reduceat(I, A) / S, 'V_V': V[mid]})
    if 'T_C' in df.columns:
        out['T_C'] = df['T_C'].to_numpy()[mid]
    if 'eis' in df.columns:
        out['eis'] = np.maximum.reduceat(df['eis'].to_numpy().astype(int), A)
    return out, S * float(dt_s)


def coarsen(df, dt_s, factor, align=True, jump=0.05, fine=0, v_mean=False):
    """Resample by `factor`, keeping the sampling convention: the current is the mean over each coarse
    interval and V, T are the values at its midpoint. `factor` must be odd, so the midpoint of a coarse
    interval is the midpoint of its central fine sample (no interpolation).

    align=True (default): no coarse interval straddles a change of current. Each stretch of constant current
    (current_segments) is cut into intervals of `factor` samples from its start, and its last L mod factor
    samples are kept at dt_s. The step is then one value per sample (an array, constant within a stretch),
    which em_model.simulate and ecm.simulate_q accept; the fixed grid instead averages a pulse edge into
    one interval, an error of 0.4-1.2 mV RMSE with 17-42 mV at the edges of the Mohtat HPPC and LG M50T
    GITT pulses. Returns (DataFrame, dt) with dt a float when uniform.
    fine (align=True): the first `fine` samples of every stretch stay at dt_s (the transient after each
    change of current, e.g. a whole 10 s pulse and the first minute of the rest after it).
    align=False: the fixed grid of dt_s * factor. v_mean (align=False): V is the mean over each interval
    instead of its midpoint value, consistent with the mean current to first order. On a drive cycle the
    midpoint current differs from the mean, and the midpoint voltage then carries a full I*R error (LG M50T
    WLTP, 9 s grid: 10 mV RMSE at the midpoint, 0.9 mV with the mean; runs/val_grid_check.md)."""
    factor = int(factor)
    if factor <= 1:
        return df.reset_index(drop=True), dt_s
    if factor % 2 == 0:
        raise ValueError('factor must be odd, got %d' % factor)
    if align:
        I = df['I_A'].to_numpy()
        seg = current_segments(I, jump)
        A, S = [], []
        for a, b in zip(seg, np.r_[seg[1:], len(I)]):
            nf = min(int(fine), b - a)
            A.append(a + np.arange(nf))
            S.append(np.ones(nf, int))
            a = a + nf
            nb = (b - a) // factor
            A += [a + factor * np.arange(nb), a + nb * factor + np.arange(b - a - nb * factor)]
            S += [np.full(nb, factor), np.ones(b - a - nb * factor, int)]
        A, S = np.concatenate(A).astype(int), np.concatenate(S).astype(int)
        mid = A + S // 2
        out = pd.DataFrame({'t_s': df['t_s'].to_numpy()[A], 'I_A': np.add.reduceat(I, A) / S,
                            'V_V': df['V_V'].to_numpy()[mid]})
        if 'T_C' in df.columns:
            out['T_C'] = df['T_C'].to_numpy()[mid]
        if 'eis' in df.columns:                                   # excluded if any of its fine samples is
            out['eis'] = np.maximum.reduceat(df['eis'].to_numpy().astype(int), A)
        dt = S * float(dt_s)
        return out, (float(dt[0]) if np.all(dt == dt[0]) else dt)
    n = (len(df) // factor) * factor
    I = df['I_A'].to_numpy()[:n].reshape(-1, factor).mean(axis=1)
    mid = np.arange(factor // 2, n, factor)
    V = df['V_V'].to_numpy()[:n].reshape(-1, factor).mean(axis=1) if v_mean else df['V_V'].to_numpy()[mid]
    out = pd.DataFrame({'t_s': df['t_s'].to_numpy()[:n:factor], 'I_A': I, 'V_V': V})
    for col in ('T_C', 'eis'):
        if col in df.columns:
            out[col] = df[col].to_numpy()[mid]
    return out, dt_s * factor
