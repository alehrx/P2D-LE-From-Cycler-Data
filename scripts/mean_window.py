"""
mean_window.py
==============
One electrode window per cell type: replaces the window groups (x100, Q_n, y100, Q_p) of EVERY cell in a
study's windows.json by their mean over the calibration cells (split "dynamics", or step1 "window_cells"), after step 1. The per-cell
windows are kept as windows_percell.json. The initial charge of each test still comes from the cell's own
rested voltage at its start (identification.prepare_tests), and the step-2 fit may correct the shared window
by the factors win_Q_n and win_x100.

    python scripts/mean_window.py configs/lgB.json      # step 1b of the LG M50T cases
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
KEYS = ('x100', 'Q_n', 'y100', 'Q_p')


def main():
    path = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(ROOT, sys.argv[1])
    c = json.load(open(path, encoding='utf-8'))
    sd = os.path.join(ROOT, 'runs', 'studies', c['name'])
    wp = os.path.join(sd, 'windows.json')
    keep = os.path.join(sd, 'windows_percell.json')
    if not os.path.exists(keep):
        shutil.copy(wp, keep)
    w = json.load(open(keep))
    # the cells the mean is taken over: step1 "window_cells" or the calibration cells; the mean goes to every cell of the dataset, also to the cells
    # step 1 could not window (no slow discharge)
    cal = list(c.get('step1', {}).get('window_cells', c['split']['dynamics']))
    mean = {k: float(np.mean([w[x][k] for x in cal])) for k in KEYS}
    spec = json.load(open(os.path.join(ROOT, 'configs', 'datasets', c['dataset'] + '.json'), encoding='utf-8'))
    out = {cell: {**w.get(cell, {}), **mean, 'window_source': 'mean of %s' % ','.join(cal)}
           for cell in sorted(set(w) | set(spec['cells']), key=lambda s: (len(s), s))}
    # step1 "capacity_test" (LG M50T: "C10"): every cell
    # keeps the mean stoichiometric window, with Q_n and Q_p scaled by its own discharge capacity in that test (the
    # first discharge above "capacity_I_A") over the mean of the window cells - its own slow cycle, as for a cell
    # that departs from its type
    ct = c.get('step1', {}).get('capacity_test')
    if ct:
        sys.path.insert(0, ROOT)
        from srcs import datasets
        from srcs.datasets.base import segments
        ds = datasets.get(c['dataset'])
        i_min = float(c['step1'].get('capacity_I_A', 0.5))

        def cap(cell):
            if not ds.available(cell, ct):
                return None
            df, tc = ds.load(cell, ct)
            I = df['I_A'].to_numpy()
            a, b = segments(I > i_min)[0]
            return float(I[a:b].sum() * float(tc['dt_s']) / 3600.0)
        Qc = {cell: cap(cell) for cell in out}
        Q_ref = float(np.mean([Qc[x] for x in cal]))
        for cell, v in out.items():
            if Qc[cell] is not None:
                f = Qc[cell] / Q_ref
                v.update(Q_n=mean['Q_n'] * f, Q_p=mean['Q_p'] * f, capacity_scale=f, capacity_Ah=Qc[cell],
                         window_source=v['window_source'] + ', capacity scaled by its %s discharge' % ct)
        print('capacity scale (%s, mean of the window cells %.3f Ah): %s' % (ct, Q_ref, {k: round(v.get('capacity_scale', 1.0), 3) for k, v in out.items()}))
    json.dump(out, open(wp, 'w'), indent=1)
    print('%s: mean window of %s on all %d cells: %s' % (c['name'], cal, len(out), {k: round(v, 4) for k, v in mean.items()}))
    for cell, v in w.items():
        print('  %-4s own %s' % (cell, ' '.join('%s %.4f' % (k, v[k]) for k in KEYS)))


if __name__ == '__main__':
    main()
