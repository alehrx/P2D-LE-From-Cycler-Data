"""
heldout_summary.py
==================
Held-out first-cycle RMSE of one or more studies: per study the mean over its held-out cells, the mean
per chamber temperature and the worst cell, with the fit's cost. Uses fig_cases_heldout.heldout (cached in
runs/studies/<study>/heldout_val_rmse.csv).

    python scripts/heldout_summary.py fitB caseC [--exclude C01 C02]
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
import fig_cases_heldout as H     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('studies', nargs='+')
    ap.add_argument('--exclude', nargs='*', default=[])
    a = ap.parse_args()
    for s in a.studies:
        h = H.heldout(s)
        h = h[~h.cell.isin(a.exclude)]
        d = json.load(open(os.path.join('runs', 'studies', s, 'dynamics.json')))
        Ts = sorted(set(h.T_group))                      # the chamber temperatures (-5/25/45 Mohtat, 10/25/40 LG)
        per_T = ' / '.join('%.1f' % h[h.T_group == T].rmse_mV.mean() for T in Ts)
        w = h.loc[h.rmse_mV.idxmax()]
        print('%-22s cost %5.2f mV | held-out %d cells: mean %.1f mV, %s C %s, worst %s %.1f mV'
              % (s, d['cost_mV'], len(h), h.rmse_mV.mean(), '/'.join('%g' % T for T in Ts), per_T, w.cell, w.rmse_mV),
              flush=True)
        print('    ' + ', '.join('%s %.1f' % (c, r) for c, r in zip(h.cell, h.rmse_mV)), flush=True)


if __name__ == '__main__':
    main()
