"""
anchor_window.py
================
One window per cell type with the bottom anchored: every cell keeps the mean x100, y100 (the top of charge) of
the calibration cells, and its Q_n, Q_p are scaled so that at its OWN OCV capacity q_ocv (step 1, from its own
budget tests) it ends at the mean bottom stoichiometries x0, y0 of the calibration cells:

    Q_n,c = q_ocv,c / (x100 - x0),   Q_p,c = q_ocv,c / (y0 - y100)

With scripts/mean_window.py's plain mean (one Q_n, Q_p for all) a cell with more capacity than the mean runs
its anode past x0 (with the Ai2020 graphite, Q_n 5.1 A.h: x < 0 for C01, C07, C12) and the model collapses at
the end of the discharge. Reads a study's windows_percell.json (step 1) and writes windows.json of another
study (the other step-1 files copied).

    python scripts/anchor_window.py fitB fitB      # step 1b of case B (Mohtat), in place
"""
import argparse
import json
import os
import shutil

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--cells', nargs='+', default=['C07', 'C08', 'C09'])
    a = ap.parse_args()
    S, D = (os.path.join(ROOT, 'runs', 'studies', s) for s in (a.src, a.dst))
    os.makedirs(D, exist_ok=True)
    P = json.load(open(os.path.join(S, 'windows_percell.json')))
    m = {k: float(np.mean([P[c][k] for c in a.cells])) for k in ('x100', 'y100', 'x0', 'y0')}
    out = {}
    for c, p in P.items():
        q = float(p['q_ocv_Ah'])
        out[c] = {**p, 'x100': m['x100'], 'y100': m['y100'], 'Q_n': q / (m['x100'] - m['x0']), 'Q_p': q / (m['y0'] - m['y100']),
                  'x0': m['x0'], 'y0': m['y0'],
                  'window_source': 'top (x100, y100) and bottom (x0, y0) of the mean of %s; Q_n, Q_p from the cell\'s own q_ocv'
                                   % ','.join(a.cells)}
        print('%s q_ocv %.3f -> Q_n %.3f Q_p %.3f' % (c, q, out[c]['Q_n'], out[c]['Q_p']))
    json.dump(out, open(os.path.join(D, 'windows.json'), 'w'), indent=2)
    for f in ('windows_percell.json', 'ocp_correction.json', 'step1.json'):
        if os.path.abspath(S) != os.path.abspath(D):         # in place: the files are already there
            shutil.copy(os.path.join(S, f), os.path.join(D, f))
    print('mean top/bottom', {k: round(v, 4) for k, v in m.items()}, '->', os.path.join(D, 'windows.json'))


if __name__ == '__main__':
    main()
