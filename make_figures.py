"""
make_figures.py
===============
The result figures of the paper for the Mohtat cells (Fig. 8: measured and predicted voltage of fifteen held-out
cells; Fig. 9: held-out error per cell and by C-rate, state of charge and temperature) for the chosen parameter
sets. Fig. 10 (LG M50T): python scripts/fig_lg_grid.py --cases A:lgA B:lgB C:lgC --out figures/fig10_lg_grid

    python make_figures.py                 # cases A, B and C of the paper
    python make_figures.py --sets B        # one set only
    python make_figures.py --sets B C --out figures/my_run

Parameter sets (configs/<study>.json, identified groups in runs/studies/<study>/dynamics.json):
    A   fitA    measured anode OCP, capacity of each cell from its C/20
    B   fitB    generic graphite OCP (Ai2020), capacity of each cell from its C/20
    C   caseC  groups of B, every held-out cell at the mean capacity of the calibration cells
A study of your own (same format) can be given as KEY:study, e.g. --sets B:fitB X:my_study.
Runs on a CUDA GPU if one is available, otherwise on the CPU (much slower).
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SETS = {'A': 'fitA', 'B': 'fitB', 'C': 'caseC'}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--sets', nargs='+', default=list(SETS))
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures', 'fig'))
    a = ap.parse_args()
    cases = [s if ':' in s else '%s:%s' % (s, SETS[s]) for s in a.sets]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    # the error samples of each set (runs/studies/<study>/figures/error_bins.npz), which Fig. 9 bins
    studies = [c.split(':')[1] for c in cases]
    cmd = [sys.executable, os.path.join(ROOT, 'scripts', 'error_bins.py'),
           *[os.path.join(ROOT, 'configs', s + '.json') for s in studies], '--out', os.path.join(ROOT, 'runs')]
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    for script, tag in (('fig_cases_grid.py', '_voltage_grid'), ('fig_cases_errors.py', '_heldout_errors')):
        cmd = [sys.executable, os.path.join(ROOT, 'scripts', script), '--cases', *cases, '--out', a.out + tag]
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    print('written %s_voltage_grid.pdf/.png and %s_heldout_errors.pdf/.png' % (a.out, a.out))


if __name__ == '__main__':
    main()
