"""
datasets/lgm50.py
=================
LG M50T 21700 cells (NMC811 / graphite-SiOx, 5 Ah) of Kirkaldy et al., J. Power Sources 2024
(zenodo 10.5281/zenodo.10637534): the beginning-of-life performance checks and the first ageing cycles,
read from the raw BioLogic files (.mpt exports and .mpr binaries, I + on charge) and, where the raw files
are missing, from the dataset's processed curves.

Cells are '<experiment><letter>', e.g. 'E4D' = Expt 4, cell D (drive-cycle ageing), 'E5D' = Expt 5, cell D
(1C standard cycling). Cells A-C age at 10 C, D-E at 25 C, F-H at 40 C. Every BoL check ran at ~25 C.

Tests (configs/datasets/lgm50.json; "source" picks the reader):
    C10      bol        C/10 (0.5 A) discharge to 2.5 V, 6 h rest, C/10 charge to 4.2 V          E4
             processed  the same from the processed RPT0 curves (discharge, then rest + charge)  E5
    GITT     bol        ~24 x 0.2 A.h discharge pulses at C/2, each followed by a 1 h rest       E4
    C2, GITT_C2, HYB_C2, HYB_1C   bol (see the json)                                             E4
    VAL      cycling    the first ageing cycles at the chamber temperature, 1 s, from the 4 h rest
                        at the top (after the first CC-CV charge) to the end of the CV that follows
                        the n_full-th discharge to the bottom of the window:
                        E4: two 1-h WLTP discharges + 0.3C CC-CV recharges, a 0.15 A discharge step,
                            a full WLTP discharge (~5 h), 0.3C CC-CV
                        E5: a 1C discharge, 0.3C CC-CV, a 0.15 A discharge to ~3.9 V (~10 h), a 1C
                            discharge to 2.5 V, 0.3C CC-CV

Every test is cut to start at the top of charge: the first rest of at least `top_rest_s` at or above
`top_V` - so it starts rested at the top (q_start = 0); the processed C/10 starts with its discharge,
right after the CV.
"""
from __future__ import annotations

import glob
import logging
import os
import re

import numpy as np
import pandas as pd

from srcs.datasets.base import Dataset, segments, sha256, to_grid

log = logging.getLogger(__name__)
V_OK = (2.0, 4.5)        # [V] raw samples outside are logging glitches (seen in the .mpr files: -0.2 V, 10 V)


def read_mpt(path):
    """Raw BioLogic export -> (t [s], I [A] > 0 discharge, V [V], T [C])."""
    d = pd.read_csv(path, sep='\t', encoding='latin-1', low_memory=False)
    tcol = next(c for c in d.columns if c.startswith('Temperature'))
    d = d[['time/s', 'I/mA', 'Ecell/V', tcol]].apply(pd.to_numeric, errors='coerce').dropna()
    d = d.drop_duplicates('time/s').sort_values('time/s')
    return (d['time/s'].to_numpy(), -d['I/mA'].to_numpy() / 1000.0, d['Ecell/V'].to_numpy(),
            d[tcol].to_numpy())


def read_mpr(path):
    """Raw BioLogic binary -> (t [s], I [A] > 0 discharge, V [V], T [C], q [A.s discharged, the cycler's
    coulomb count]); samples with a glitched voltage and repeated time stamps are dropped (the charge comes
    from the cycler's count, so it is not affected)."""
    from galvani import BioLogic
    d = pd.DataFrame(BioLogic.MPRfile(path).data)
    tcol = next(c for c in d.columns if c.startswith('Temperature'))
    t = d['time/s'].to_numpy(float)
    q = -d['(Q-Qo)/mA.h'].to_numpy(float) * 3.6
    keep = (d['Ewe/V'].to_numpy() > V_OK[0]) & (d['Ewe/V'].to_numpy() < V_OK[1])
    keep &= np.r_[True, np.diff(t) > 0]
    return (t[keep], -d['I/mA'].to_numpy(float)[keep] / 1000.0, d['Ewe/V'].to_numpy(float)[keep],
            d[tcol].to_numpy(float)[keep], q[keep])


class LGM50(Dataset):

    def tests_of(self, cell: str):
        """The tests that exist for a cell (its experiment's)."""
        return [k for k, v in self.tests.items() if cell[:2] in v.get('experiments', ['E4'])]

    def raw_file(self, cell: str, test: str) -> str:
        info = self.src(cell, test)
        expt, letter = cell[:2], cell[2:]
        base = os.path.join(self.spec['raw_root'], self.spec['experiments'][expt])
        if info['source'] == 'bol':
            root = os.path.join(base, 'Raw Data', 'Performance Checks', info['check'], info['speed'].capitalize())
            pat = '*cell %s - %s (%s)_%02d_*.mpt' % (letter, info['check'], info['speed'], int(info['part']))
        elif info['source'] == 'cycling':
            root = os.path.join(base, 'Raw Data', 'Degradation Cycling', 'Set %d' % int(info.get('set', 1)))
            pat = '*cell %s_*.mpr' % letter                       # the first part of the set (not "(part 2)")
        else:
            raise ValueError('lgm50 %s %s: unknown source %r' % (cell, test, info['source']))
        hits = [h for h in glob.glob(os.path.join(root, pat)) if 'part' not in os.path.basename(h)]
        if len(hits) != 1:
            raise FileNotFoundError('%s %s: %d files match %s in %s' % (cell, test, len(hits), pat, root))
        return hits[0]

    def src(self, cell, test):
        """The test's description for this cell's experiment ("by_experiment" overrides)."""
        info = dict(self.tests[test])
        info.update(info.pop('by_experiment', {}).get(cell[:2], {}))
        info.setdefault('source', 'bol')
        return info

    def _top_start(self, t, I, V):
        """Index of the first sample of the first long rest at the top of charge."""
        top_V, top_rest = float(self.spec['top_V']), float(self.spec['top_rest_s'])
        for a, b in segments(np.abs(I) < 0.01):
            if t[b - 1] - t[a] >= top_rest and V[a] >= top_V:
                return a
        raise RuntimeError('no rest of %.0f s at >= %.2f V' % (top_rest, top_V))

    @staticmethod
    def _cycles_end(t, I, V, a, n_full, V_bottom=2.6, i_eps=0.02, charge_A=1.0, run_s=600.0):
        """Index after the charge (CC + CV) that follows the n_full-th discharge to the bottom (V < V_bottom)
        from sample a: the charge starts at the first run of more than run_s at < -charge_A after the bottom
        (a WLTP's regenerative pulses are shorter) and ends at the first discharge sample or the first rest
        longer than run_s after it."""
        pos = a
        for n in range(1, n_full + 1):
            low = np.flatnonzero((V[pos:] < V_bottom) & (I[pos:] > i_eps))
            if not len(low):
                break
            k = pos + int(low[0])
            k0 = None
            for cs, ce in segments(I[k:] < -charge_A):            # the recharge after that bottom
                if t[k + ce - 1] - t[k + cs] > run_s:
                    k0 = k + cs
                    break
            if k0 is None:
                break
            if n < n_full:
                pos = k0
                continue
            j = k0 + int(np.argmax(I[k0:] > i_eps)) if np.any(I[k0:] > i_eps) else len(I)
            for rs, r_e in segments(np.abs(I[k0:j]) < i_eps):
                if t[k0 + r_e - 1] - t[k0 + rs] > run_s:
                    return k0 + rs
            return j
        raise RuntimeError('no charge after the %d-th discharge to %.2f V' % (n_full, V_bottom))

    def prepare(self, cells=None, force=False):
        for cell in cells or self.cells:
            for test in self.tests_of(cell):
                info = self.src(cell, test)
                if self.available(cell, test) and not force:
                    log.info('%s %s exists - skipped', cell, test)
                    continue
                dt = float(info['dt_s'])
                if info['source'] == 'bol':
                    path = self.raw_file(cell, test)
                    t, I, V, T = read_mpt(path)
                    a = self._top_start(t, I, V)
                    df = to_grid(t[a:], I[a:], V[a:], T[a:], dt)
                    extra = dict(raw_file=path, raw_sha256=sha256(path), raw_start_s=float(t[a]))
                elif info['source'] == 'cycling':
                    path = self.raw_file(cell, test)
                    t, I, V, T, q = read_mpr(path)
                    a = self._top_start(t, I, V)
                    b = self._cycles_end(t, I, V, a, int(info.get('n_full', 1)))
                    df = to_grid(t[a:b], I[a:b], V[a:b], T[a:b], dt, q=q[a:b] - q[a])
                    extra = dict(raw_file=path, raw_sha256=sha256(path), raw_start_s=float(t[a]), raw_end_s=float(t[b - 1]))
                elif info['source'] == 'processed':
                    legs, paths = [], []
                    for leg in info['legs']:
                        base = os.path.join(self.spec['raw_root'], self.spec['experiments'][cell[:2]],
                                            'Processed Timeseries Data', info['folder'], 'Cell %s' % cell[2:])
                        hits = glob.glob(os.path.join(base, '*cell %s - %s - * %s data.csv' % (cell[2:], info['rpt'], leg)))
                        if len(hits) != 1:
                            raise FileNotFoundError('%s %s %s: %d files in %s' % (cell, test, leg, len(hits), base))
                        c = pd.read_csv(hits[0])
                        legs.append(c)
                        paths.append(hits[0])
                    c = pd.concat(legs).sort_values('Time (s)').drop_duplicates('Time (s)')
                    t = c['Time (s)'].to_numpy(float)
                    I = -c['Current (mA)'].to_numpy(float) / 1000.0
                    df = to_grid(t, I, c['Voltage (V)'].to_numpy(float), c['Temperature (degC)'].to_numpy(float), dt)
                    path = ';'.join(paths)
                    extra = dict(raw_file=path, raw_sha256=';'.join(sha256(p) for p in paths), raw_start_s=float(t[0]))
                else:
                    raise ValueError(info['source'])
                extra.update(protocol_note=info['note'], cell_ageing_T_C=self.cell_T_C(cell))
                self._write(cell, test, df, extra, 'LG M50T %s, %s (Kirkaldy 2024), resampled to %.3g s' % (cell, test, dt))
