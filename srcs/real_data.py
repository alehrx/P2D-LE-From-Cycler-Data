"""
prepare_real.py
===============
Phase 2. Convert the raw UofM (Mohtat) characterisation and cycling files of a cell into the same
dataset format as the simulated plant (simulate_plant.py), so the identification scripts run
unchanged on real data.

Per cell, beginning of life (cycle 0 of the characterisation files, the first cycles of ageing):
    HPPC   Resistance.csv: from the rest after the first CC-CV (top of charge) to the end.
           Per ~10 % SOC: C/2 step, 30 min rest, 1C discharge pulse 9 s, 10 min rest, 1C charge
           pulse 9 s, ~33 min rest with EIS. 1 s grid.
    C20    OCV_wExpansion.csv: C/20 charge from the bottom to 4.2 V (no CV), 1 h rest, C/20
           discharge to 3.0 V. 10 s grid. `q_top_Ah` in config = throughput of the charge leg,
           i.e. where "top" sits on this test's axis.
    VAL    cycling_wExpansion.csv: from the start of the first discharge after a CV (top of
           charge, not rested) over N_VAL_DISCHARGES discharges. 10 s grid. Held out.

Conventions (as the plant data): I > 0 discharge [A]; I_A is the MEAN current over
[t, t + dt), V_V and T_C are taken at t + dt/2. Raw BioLogic current is + on charge and in mA.
EIS rows (Frequency > 0, Resistance.csv) are small sinusoids during rests: their current is set to
zero and their samples are flagged `eis = 1`, to be excluded from voltage residuals.

Outputs -> <repo>/data/DB_REAL_C<cell>_<TEST>_BOL/ : cycles.csv, config.json, metadata.json
(existing folders are skipped, never overwritten)
"""
import os
import sys
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from srcs.paths import ROOT as REPO_ROOT, DATA as DATA_ROOT   # noqa: E402
from srcs import physics as gm   # noqa: E402

# ============================ CONFIG ============================
RAW_ROOT = os.environ.get('MOHTAT_RAW_ROOT',
                          r'C:\datasets\Datasets Batteries\Mohtat NMC\data')
CELLS = tuple(range(1, 22))   # all 21 cells: room (1,4,7,10,13,16,19), cold (2,5,8,11,14,17,20), hot (3,6,9,12,15,18,21)
DT = {'HPPC': 1.0, 'C20': 10.0, 'VAL': 10.0, 'CRATE1C': 10.0}
N_VAL_DISCHARGES = 2
REST_TOP_MIN_S = 600.0       # a rest this long at >= 4.19 V after a CV marks the HPPC top
USE_Q = False                # charge from the cycler's Q [Ah] column instead of the trapezoid of the logged
                             # current: the 10 s rows put a step start inside an interval whose left sample
                             # is 0 A, so the trapezoid loses ~half an interval per step (3.4 mAh per C/2 step
                             # of the HPPC, 32-36 mAh by its bottom).
FOLDER = 'DB_REAL_{cell}_{test}_BOL'
# ===============================================================

COLS = {'OCV': ['t', 'I_mA', 'V', 'exp', 'T', 'Q', 'Cap', 'cyc', 'tot'],
        'Crate': ['t', 'I_mA', 'V', 'exp', 'T', 'Q', 'Cap', 'cyc', 'tot'],
        'cycling': ['t', 'I_mA', 'V', 'exp', 'T', 'Q', 'Cap', 'cyc', 'tot'],
        'Resistance': ['t', 'I_mA', 'V', 'f', 'mag', 'ph', 'Re', 'mIm', 'T', 'Q', 'cyc', 'tot']}
FILES = {'OCV': 'OCV_wExpansion.csv', 'Resistance': 'Resistance.csv', 'cycling': 'cycling_wExpansion.csv',
         'Crate': 'Crate_wExpansion.csv'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def read_raw(cell, test, max_cycle=None, nrows=None):
    p = os.path.join(RAW_ROOT, f'{cell:02d}', FILES[test])
    d = pd.read_csv(p, skiprows=1, header=None, names=COLS[test], nrows=nrows)
    if max_cycle is not None:
        d = d[d['cyc'] <= max_cycle]
    d = d.drop_duplicates('t').sort_values('t').reset_index(drop=True)
    d['I'] = -d['I_mA'] / 1000.0            # BioLogic + = charge  ->  + = discharge, A
    return d


def to_grid(d, dt, eis=None, use_Q=False):
    """Charge-preserving resampling onto a uniform grid (see module docstring). use_Q: take the
    charge from the cycler's own running coulomb count (column Q [Ah], + on charge) instead of
    integrating the logged current samples - needed for the drive cycle, whose current changes
    every second while it is logged every 10 s. The resampling itself is datasets.base.to_grid,
    shared with the LG M50 preparation."""
    from srcs.datasets.base import to_grid as grid
    q = None
    if use_Q:
        dq = -np.diff(d['Q'].to_numpy()) * 3600.0                          # A.s discharged per raw interval
        I = d['I'].to_numpy()
        idle = (np.abs(I[1:]) < 1e-3) & (np.abs(I[:-1]) < 1e-3)           # logged 0 A at both ends: a rest
        if eis is not None:                                                # the counter integrates the EIS sine
            idle |= np.asarray(eis[1:], bool) | np.asarray(eis[:-1], bool)
        q = np.concatenate(([0.0], np.cumsum(np.where(idle, 0.0, dq))))
    return grid(d['t'].to_numpy(), d['I'].to_numpy(), d['V'].to_numpy(), d['T'].to_numpy(), dt, eis=eis, q=q)


def hppc(cell, cyc=None):
    """cyc: the characterisation (RPT) cycle, default the first (beginning of life)."""
    d = read_raw(cell, 'Resistance')
    d = d[d['cyc'] == (d['cyc'].min() if cyc is None else cyc)].reset_index(drop=True)
    eis = (d['f'] > 0).to_numpy()
    # top of charge: first long rest at >= 4.19 V
    rest = (np.abs(d['I']) < 0.02).to_numpy() & ~eis
    t = d['t'].to_numpy()
    start = None
    i = 0
    while i < len(d):
        if rest[i] and d['V'].iloc[i] >= 4.19:
            j = i
            while j + 1 < len(d) and rest[j + 1]:
                j += 1
            if t[j] - t[i] >= REST_TOP_MIN_S:
                start = i
                break
            i = j + 1
        else:
            i += 1
    if start is None:
        raise RuntimeError(f'cell {cell}: no top-of-charge rest found in the HPPC file')
    d = d.iloc[start:].reset_index(drop=True)
    return to_grid(d, DT['HPPC'], eis=eis[start:], use_Q=USE_Q), dict(start_time_raw_s=float(t[start]), use_Q=USE_Q)


def c20(cell, cyc=None):
    d = read_raw(cell, 'OCV')
    d = d[d['cyc'] == (d['cyc'].min() if cyc is None else cyc)].reset_index(drop=True)
    g = to_grid(d, DT['C20'], use_Q=USE_Q)
    q = np.concatenate(([0.0], np.cumsum(g['I_A'].to_numpy()[:-1]) * DT['C20'])) / 3600.0
    i_top = int(np.argmin(q))                 # most charged point (q is discharged A.h, + = discharge)
    return g, dict(q_top_Ah=float(-q[i_top]), i_top=i_top,
                   note='starts at the bottom; charge to 4.2 V (no CV), rest, discharge to 3.0 V')


def val(cell, cyc=None):
    """cyc: start from the ageing cycle cyc (the first cycles after the RPT of that number), default 0."""
    c0 = 0 if cyc is None else int(cyc)
    d = read_raw(cell, 'cycling', max_cycle=c0 + N_VAL_DISCHARGES + 2)
    d = d[d['cyc'] >= c0].reset_index(drop=True)
    I = d['I'].to_numpy()
    V = d['V'].to_numpy()
    # first discharge that follows a CV hold (charging at ~4.2 V with a small current)
    cv = (I < 0) & (V >= 4.195)
    dis = I > 0.5
    k0 = next(k for k in range(1, len(d)) if dis[k] and cv[:k].any())
    # end: after N_VAL_DISCHARGES complete discharges
    starts = np.flatnonzero(dis[1:] & ~dis[:-1]) + 1
    starts = starts[starts >= k0]
    k1 = starts[N_VAL_DISCHARGES] if len(starts) > N_VAL_DISCHARGES else len(d)
    seg = d.iloc[k0:k1].reset_index(drop=True)
    return to_grid(seg, DT['VAL'], use_Q=USE_Q), dict(start_cycle=int(d['cyc'].iloc[k0]), use_Q=USE_Q,
                                         note='starts at top of charge after a CV, NOT rested')


def crate_1c(cell):
    """C-rate test, cycle 0: the 1 h rest at the bottom, the continuous 1C charge to 4.2 V (no CV)
    and the 1 h rest after it. Starts RESTED at the bottom (initial state by OCV inversion)."""
    d = read_raw(cell, 'Crate')
    d = d[d['cyc'] == d['cyc'].min()].reset_index(drop=True)
    I = d['I'].to_numpy()
    chg = np.abs(I + 5.0) < 0.5                  # 1C = 5 A charge (I > 0 is discharge)
    k = np.flatnonzero(chg)
    i0 = k[0]
    i1 = i0
    while i1 + 1 < len(I) and chg[i1 + 1]:       # the FIRST contiguous 1C block only: the hot cells'
        i1 += 1                                  # C-rate test continues with a C/2 discharge and a 2C charge
    rest = np.abs(I) < 0.02
    a = i0
    while a > 0 and rest[a - 1]:
        a -= 1
    b = i1
    while b + 1 < len(d) and rest[b + 1]:
        b += 1
    seg = d.iloc[a:b + 1].reset_index(drop=True)
    return to_grid(seg, DT['CRATE1C'], use_Q=USE_Q), dict(use_Q=USE_Q, note='rest at the bottom, 1C charge to 4.2 V (no CV), rest; '
                                                  'starts rested at the bottom',
                                             v_rest_start=float(seg['V'].iloc[0]))


def val_drive(cell):
    """Cell 19-21 ageing: 1.5C CC-CV charge, DRIVE-CYCLE discharge over 0-50 % DOD. From the first
    discharge that starts right after a CV (top of charge, not rested) over N_VAL_DISCHARGES
    discharge + charge cycles. Current from the coulomb count (see to_grid)."""
    d = read_raw(cell, 'cycling', max_cycle=N_VAL_DISCHARGES + 3)
    I, cyc = d['I'].to_numpy(), d['cyc'].to_numpy()
    # cycle 0 is the initial full charge; each later cycle = one drive discharge + one CC-CV charge.
    # (A 'discharge right after 4.19 V' rule fires inside the drive cycle: regen near the top.)
    c0 = 1
    k0 = int(np.flatnonzero((cyc == c0) & (I > 0.05))[0])
    k1 = int(np.flatnonzero(cyc <= c0 + N_VAL_DISCHARGES - 1)[-1]) + 1
    seg = d.iloc[k0:k1].reset_index(drop=True)
    return to_grid(seg, DT['VAL'], use_Q=True), dict(start_cycle=int(d['cyc'].iloc[k0]),
                                                     note='drive-cycle discharge (0-50 % DOD) + 1.5C CC-CV '
                                                          'charge; starts at top after a CV, NOT rested; '
                                                          'current from the coulomb count, 10 s logging')


def save(cell, test, df, extra):
    out = os.path.join(DATA_ROOT, FOLDER.format(cell=f'C{cell:02d}', test=test))
    if os.path.exists(out):
        log.info('%s exists - skipped', out)
        return
    os.makedirs(out)
    df.to_csv(os.path.join(out, 'cycles.csv'), index=False, float_format='%.7g')
    json.dump(dict(PM='Mohtat2020', source='UofM/Mohtat raw', cell=cell, protocol=test, dt_s=DT[test],
                   temperature_C=float(df['T_C'].mean()), T_min_C=float(df['T_C'].min()),
                   T_max_C=float(df['T_C'].max()), sampling='I = mean over [t, t+dt); V, T at t + dt/2',
                   sign='I > 0 discharge', raw_root=RAW_ROOT,
                   literature_groups=gm.truth_groups('Mohtat2020'), **extra),
              open(os.path.join(out, 'config.json'), 'w'), indent=2)
    json.dump(dict(date=datetime.now(timezone.utc).isoformat(),
                   description=f'real cell {cell:02d}, {test} at beginning of life, resampled for EM group fitting',
                   dataset=os.path.join(RAW_ROOT, f'{cell:02d}'), n_samples=len(df),
                   duration_h=float(df['t_s'].iloc[-1] / 3600)),
              open(os.path.join(out, 'metadata.json'), 'w'), indent=2)
    log.info('cell %02d %-4s -> %s (%d samples, %.1f h, T %.1f-%.1f C)', cell, test, out, len(df),
             df['t_s'].iloc[-1] / 3600, df['T_C'].min(), df['T_C'].max())


def main():
    os.makedirs(DATA_ROOT, exist_ok=True)
    for cell in CELLS:
        val_fn = val_drive if cell in (19, 20, 21) else val
        for test, fn in (('HPPC', hppc), ('C20', c20), ('VAL', val_fn), ('CRATE1C', crate_1c)):
            df, extra = fn(cell)
            save(cell, test, df, extra)


if __name__ == '__main__':
    main()
