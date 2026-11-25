"""
make_case_c.py
==============
Case C of a final fit: a study folder (default <study>_C) with the groups of <study> and every cell's Q_n, Q_p
scaled to the mean C/20 capacity of the calibration cells (the stoichiometric limits unchanged), as
caseC was built from fitB. Also writes its config (the config of <study>, renamed).

    python scripts/make_case_c.py fitB [caseC]
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src + '_C'     # optional name of case C (e.g. caseC)
    cfg = json.load(open(os.path.join(ROOT, 'configs', src + '.json')))
    cal = cfg['split']['dynamics']
    sd, dd = os.path.join(ROOT, 'runs', 'studies', src), os.path.join(ROOT, 'runs', 'studies', dst)
    os.makedirs(dd, exist_ok=True)
    for f in ('dynamics.json', 'step1.json', 'windows_percell.json', 'ocp_correction.json', 'hysteresis.json'):
        if os.path.exists(os.path.join(sd, f)) or f != 'hysteresis.json':      # hysteresis.json: LG
            shutil.copy(os.path.join(sd, f), os.path.join(dd, f))
    W = json.load(open(os.path.join(sd, 'windows.json')))
    qm = float(np.mean([W[c]['q_ocv_Ah'] for c in cal]))
    for c, w in W.items():
        s = qm / w['q_ocv_Ah']
        w['Q_n'], w['Q_p'] = w['Q_n'] * s, w['Q_p'] * s
    json.dump(W, open(os.path.join(dd, 'windows.json'), 'w'), indent=1)
    cfg['name'] = dst
    cfg['description'] = 'Case C of %s: its groups, every cell the mean C/20 capacity of %s (%.3f Ah).' % (src, cal, qm)
    json.dump(cfg, open(os.path.join(ROOT, 'configs', dst + '.json'), 'w'), indent=1)
    print('wrote', dst, '| mean capacity %.4f Ah of %s' % (qm, cal))


if __name__ == '__main__':
    main()
