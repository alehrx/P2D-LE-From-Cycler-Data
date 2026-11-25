"""
datasets/base.py
================
One interface for every dataset the identification runs on. A dataset is described by a JSON file in
configs/datasets/ and implemented by a subclass that knows how to PREPARE its raw files into the common
format, and where the prepared files are:

    data/<folder>/cycles.csv     t_s, I_A, V_V, T_C, eis
    data/<folder>/config.json    dt_s, temperatures, protocol, raw file and its sha256
    data/<folder>/metadata.json  date, description, dataset

Conventions of the common format:
    I_A > 0 on discharge [A], the MEAN current over [t, t + dt)
    V_V, T_C read at the interval midpoint t + dt/2
    eis = 1 on samples that are excluded from voltage residuals (EIS sweeps during rests)
    every prepared test STARTS AT THE TOP OF CHARGE: rested (the long rest after the CV) or under
    current (the discharge that follows the CV); identification.initial_charge relies on it

Vocabulary shared by every dataset
    cell     a string id ('C01' Mohtat, 'E4D' LG M50 Expt 4 cell D)
    test     a protocol name ('HPPC', 'C20', 'C10', 'GITT', ...)
    budget   the tests an equivalent-circuit model is tuned on. A VALIDATION cell contributes these and
             nothing else to the identification (its own capacity window); study.py enforces it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from srcs.paths import DATA

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ common helpers
def to_grid(t, I, V, T, dt, eis=None, q=None):
    """Charge-preserving resampling onto a uniform grid of step dt [s].

    t [s], I [A] (> 0 discharge), V [V], T [C] are the raw samples (any, even irregular, spacing).
    The charge is integrated from the raw samples (trapezoid) unless `q` [A.s discharged] is given,
    e.g. the cycler's own coulomb count; the grid current is the mean over each interval, so the
    charge of every interval is exact. V and T are interpolated at the interval midpoints. eis: raw
    samples to exclude; their current is set to zero and the grid intervals they touch are flagged."""
    t = np.asarray(t, float) - float(t[0])
    I = np.asarray(I, float).copy()
    if eis is not None:
        I[eis] = 0.0
    if q is None:
        q = np.concatenate(([0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))))
    edges = np.arange(0.0, t[-1] - dt + 1e-9, dt)
    edges = np.append(edges, edges[-1] + dt)
    Ig = np.diff(np.interp(edges, t, q)) / dt
    mid = edges[:-1] + dt / 2
    out = pd.DataFrame({'t_s': edges[:-1], 'I_A': Ig, 'V_V': np.interp(mid, t, np.asarray(V, float)),
                        'T_C': np.interp(mid, t, np.asarray(T, float))})
    flag = np.zeros(len(out), int)
    if eis is not None and np.any(eis):
        te = t[eis]
        k = np.clip(np.searchsorted(edges, te, side='right') - 1, 0, len(out) - 1)
        flag[k] = 1
        for a, b in zip(te[:-1], te[1:]):                    # an EIS sweep is continuous: flag the whole span
            if b - a < 60.0:
                flag[int(a // dt):int(b // dt) + 1] = 1
    out['eis'] = flag
    return out


def throughput(I, dt):
    """Charge discharged since the first sample [A.h] at the START of every sample."""
    return np.concatenate(([0.0], np.cumsum(np.asarray(I, float)[:-1]) * dt)) / 3600.0


def segments(mask):
    """[(a, b)] index ranges where `mask` is True."""
    m = np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(m.astype(int)) != 0) + 1
    return [(a, b) for a, b in zip(np.r_[0, edges], np.r_[edges, len(m)]) if m[a]]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ the dataset
class Dataset:
    """A dataset described by configs/datasets/<name>.json. Subclasses implement prepare()."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.name = spec['name']
        self.pm = spec['pm']
        self.cells = list(spec['cells'])
        self.tests = dict(spec['tests'])
        self.budget = list(spec['budget'])
        self.window_data = dict(spec.get('window_data', {}))
        self.V_min, self.V_max = float(spec['V_min']), float(spec['V_max'])
        self.nominal_Ah = float(spec.get('nominal_Ah', np.nan))
        unknown = [t for t in self.budget if t not in self.tests]
        if unknown:
            raise ValueError('%s: budget tests %s are not in "tests"' % (self.name, unknown))

    # ---- files
    def folder(self, cell: str, test: str) -> str:
        return os.path.join(DATA, self.spec['folder'].format(cell=cell, test=test))

    def available(self, cell: str, test: str) -> bool:
        return os.path.exists(os.path.join(self.folder(cell, test), 'cycles.csv'))

    def load(self, cell: str, test: str):
        """(DataFrame in the common format, the test's config.json)."""
        if cell not in self.cells:
            raise KeyError('%s: unknown cell %r' % (self.name, cell))
        if test not in self.tests:
            raise KeyError('%s: unknown test %r' % (self.name, test))
        f = self.folder(cell, test)
        if not self.available(cell, test):
            raise FileNotFoundError('%s missing - run: python scripts/prepare_data.py %s' % (f, self.name))
        df = pd.read_csv(os.path.join(f, 'cycles.csv'))
        cfg = json.load(open(os.path.join(f, 'config.json'), encoding='utf-8'))
        return df, cfg

    def data_hash(self, cell: str, test: str) -> str:
        return sha256(os.path.join(self.folder(cell, test), 'cycles.csv'))

    def cell_T_C(self, cell: str) -> float:
        """The cell's ageing / test temperature from the dataset description [C]."""
        return float(self.spec['cells'][cell].get('T_C', 25.0))

    # ---- what the capacity window is fitted to
    def rest_points(self, cell: str):
        """(q [A.h] below the top, V [V]) at the end of every clean long rest of the window rest test."""
        test = self.window_data['rest_test']
        df, cfg = self.load(cell, test)
        dt = float(cfg['dt_s'])
        I, V = df['I_A'].to_numpy(), df['V_V'].to_numpy()
        eis = df['eis'].to_numpy().astype(bool) if 'eis' in df else np.zeros(len(I), bool)
        q = throughput(I, dt)
        min_rest = float(self.window_data.get('min_rest_s', 1200.0))
        pts = []
        for a, b in segments(np.abs(I) < float(self.window_data.get('i_rest', 0.02))):
            if (b - a) * dt < min_rest:
                continue
            clean = np.arange(a, b)[~eis[a:b]]
            if len(clean) and clean[-1] >= b - 1 - 60.0 / dt:
                pts.append((q[clean[-1]], V[clean[-1]]))
        pts = np.array(pts)
        return pts[:, 0], pts[:, 1]

    def ocv_branches(self, cell: str):
        """{'discharge': (q, V), 'charge': (q, V)} of the low-rate OCV test, q [A.h] below the most
        charged point of the test. A branch the test does not contain is omitted."""
        test = self.window_data['ocv_test']
        df, cfg = self.load(cell, test)
        dt = float(cfg['dt_s'])
        I, V = df['I_A'].to_numpy(), df['V_V'].to_numpy()
        q = throughput(I, dt)
        q = q - q.min()
        i0 = float(self.window_data['ocv_current_A'])
        band = lambda x: (x > 0.7 * i0) & (x < 1.3 * i0)     # the OCV-test current only, not a pre-conditioning step
        out = {}
        for name, m in (('discharge', band(I)), ('charge', band(-I))):
            seg = max(segments(m), key=lambda s: s[1] - s[0], default=None)
            if seg is not None and seg[1] - seg[0] > 10:
                a, b = seg
                out[name] = (q[a:b], V[a:b])
        return out

    # ---- preparation
    def prepare(self, cells=None, force=False):
        raise NotImplementedError

    def _write(self, cell, test, df, extra, description):
        out = self.folder(cell, test)
        if os.path.exists(out) and os.path.exists(os.path.join(out, 'cycles.csv')):
            log.info('%s exists - skipped', os.path.basename(out))
            return False
        os.makedirs(out, exist_ok=True)
        df.to_csv(os.path.join(out, 'cycles.csv'), index=False, float_format='%.7g')
        json.dump(dict(dataset=self.name, cell=cell, protocol=test, PM=self.pm,
                       dt_s=float(df['t_s'].iloc[1] - df['t_s'].iloc[0]),
                       temperature_C=float(df['T_C'].mean()), T_min_C=float(df['T_C'].min()),
                       T_max_C=float(df['T_C'].max()), sampling='I = mean over [t, t+dt); V, T at t + dt/2',
                       sign='I > 0 discharge', starts='top of charge', **extra),
                  open(os.path.join(out, 'config.json'), 'w'), indent=2)
        json.dump(dict(date=datetime.now(timezone.utc).isoformat(), description=description,
                       dataset=extra.get('raw_file', self.name), n_samples=len(df),
                       duration_h=float(df['t_s'].iloc[-1] / 3600)),
                  open(os.path.join(out, 'metadata.json'), 'w'), indent=2)
        log.info('%s %-8s -> %s (%d samples, %.1f h, T %.1f-%.1f C)', cell, test, os.path.basename(out), len(df),
                 df['t_s'].iloc[-1] / 3600, df['T_C'].min(), df['T_C'].max())
        return True
