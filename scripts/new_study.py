"""
new_study.py
============
A new study (configs/<name>.json) from one of the paper's cases, with the cells of your choice.

    python scripts/new_study.py fitB my_study --cal C04 C05 C06 --up C04 C05 C06 [--exclude C19]

    --cal      calibration cells: their first cycles (VAL) enter the final fit of step 3
    --up       cells whose C/20 and HPPC rests give the cathode OCP and the stoichiometric limits (step 1);
               the first one that logged a full C/20 discharge gives U_p (default: --cal)
    --ref      the cell whose HPPC and C/20 enter step 3 (default: the first --cal cell)
    --exclude  cells left out of everything; every other cell is held out (validation)

Then, for the new study:

    python scripts/step1_window.py configs/my_study.json      # step 1: windows and cathode OCP
    python scripts/fit_lm.py configs/my_study.json --fresh     # step 3: final fit (GPU recommended)
    python make_figures.py --sets B:fitB X:my_study    # step 3: held-out figures, side by side with B

The model, groups, bounds and search settings are those of the base case.
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('base')
    ap.add_argument('name')
    ap.add_argument('--cal', nargs='+', required=True)
    ap.add_argument('--up', nargs='+')
    ap.add_argument('--ref')
    ap.add_argument('--exclude', nargs='*', default=[])
    a = ap.parse_args()
    c = json.load(open(os.path.join(ROOT, 'configs', a.base + '.json'), encoding='utf-8'))
    ref = a.ref or a.cal[0]
    c['name'] = a.name
    c['description'] = 'From %s (scripts/new_study.py): calibration cells %s, step-1 cells %s, HPPC and C/20 of %s.' % (
        a.base, a.cal, a.up or a.cal, ref)
    c['split'] = {'dynamics': list(a.cal)}
    if a.exclude:
        c['split']['exclude'] = list(a.exclude)
    if 'step1' in c:
        c['step1']['cells_U_p'] = list(a.up or a.cal)
    for t in c['dynamics']['tests']:          # the cell-specific tests (HPPC, C/20) move to the reference cell
        if 'cells' in t:
            t['cells'] = [ref]
        if 'axis_shift' in t and ref not in t['axis_shift']:
            # the C/20 charge axis relative to the HPPC top of the new reference cell: after step 1, set it to
            # runs/studies/<name>/step1.json cells.<ref>.hppc_shift_Ah (0 until then)
            t['axis_shift'] = {ref: 0.0}
    c.get('lm', {}).pop('warm_from', None)
    out = os.path.join(ROOT, 'configs', a.name + '.json')
    if os.path.exists(out):
        raise SystemExit('%s exists' % out)
    json.dump(c, open(out, 'w', encoding='utf-8'), indent=2)
    print('wrote', out)


if __name__ == '__main__':
    main()
