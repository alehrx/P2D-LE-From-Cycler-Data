"""
select_groups_rule.py
=====================
The group selection rule of the paper applied automatically to a screening, to write the config of
the final fit:
  S_j > 15 mV       fitted (of two of them with |rho| > 0.9, only the more sensitive);
  5 <= S_j <= 15 mV fitted only if its correlation with every other candidate stays below 0.5 in magnitude (rounded
                    to two digits, as in Table 4); of a correlated pair, only the more sensitive is fitted, and none
                    when that one cannot be (b, fixed at the Bruggeman value) or the partner is already fitted;
  S_j < 5 mV        fixed at the pre-calibrated value.
The temperature terms are always fitted. The new config copies the base config (its tests, model, split, step 1,
start box) with the selected names, the bounds of the pre-calibration for the 25 C groups and of the base for the
temperature terms, and "freeze_from" the pre-calibration; the step-1 files of the base study are copied.
    python scripts/select_groups_rule.py <screening json> <pre-calibration config> <base config> <new name>
"""
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
T_TERMS = ('Ea_Dn', 'Ea_Dp', 'Ea_r', 'dUdT0', 'dEa_De')
NEVER = ('brugg',)                     # fixed by physics (1.5), never fitted
LO, HI, RHO = 5.0, 15.0, 0.5


def main():
    scr_p, pre_p, base_p, name = sys.argv[1:5]
    scr = json.load(open(scr_p))
    pre = json.load(open(pre_p, encoding='utf-8'))
    base = json.load(open(base_p, encoding='utf-8'))
    names = [n for n in scr['names'] if n != 'R_contact']
    ix = [scr['names'].index(n) for n in names]
    C = np.round(np.array(scr['corr_median'], float)[np.ix_(ix, ix)], 2)
    S = {n: float(scr['S_local_mV'][n][1] if isinstance(scr['S_local_mV'][n], list) else scr['S_local_mV'][n]) for n in names}
    fit, why = set(), {}
    for n in names:                                      # above HI
        if S[n] > HI and n not in NEVER:
            fit.add(n)
            why[n] = 'S_j > %g mV' % HI
    for i, n in enumerate(names):                        # strongly correlated pairs above HI: keep the more sensitive
        for j, m in enumerate(names):
            if n in fit and m in fit and n != m and abs(C[i, j]) > 0.9 and S[n] < S[m]:
                fit.discard(n)
                why[n] = 'pair with %s (rho %+.2f), less sensitive' % (m, C[i, j])
    for i, n in enumerate(sorted(names, key=lambda k: -S[k])):     # the band, the most sensitive first
        i = names.index(n)
        if n in fit or n in why:
            continue
        if S[n] < LO:
            why[n] = 'S_j < %g mV' % LO
            continue
        if S[n] > HI:                                    # only NEVER groups
            why[n] = 'fixed by physics'
            continue
        r = np.abs(C[i].copy())
        r[i] = 0.0
        k = int(np.argmax(r))
        m = names[k]
        if r[k] < RHO and n not in NEVER:
            fit.add(n)
            why[n] = 'S_j %g-%g mV, max |rho| %.2f < %.1f' % (LO, HI, r[k], RHO)
        elif m in fit:
            why[n] = 'reproduced by %s (rho %+.2f), fitted' % (m, C[i, k])
        elif S[n] >= S[m] and n not in NEVER:
            fit.add(n)
            why[n] = 'pair with %s (rho %+.2f), the more sensitive' % (m, C[i, k])
        else:
            why[n] = 'pair with %s (rho %+.2f), %s' % (m, C[i, k], 'fixed by physics' if n in NEVER else
                                                          ('partner fixed by physics' if m in NEVER else 'less sensitive'))
    for n in sorted(names, key=lambda k: -S[k]):
        print('%-8s S %5.1f mV  %-7s %s' % (n, S[n], 'FITTED' if n in fit else 'fixed', why[n]))
    sel = [n for n in names if n in fit] + list(T_TERMS)
    pd_ = pre['dynamics']
    PB = dict(zip(pd_['names'], zip(pd_['lb'], pd_['ub'])))
    BB = dict(zip(base['dynamics']['names'], zip(base['dynamics']['lb'], base['dynamics']['ub'])))
    c = json.loads(json.dumps(base))
    c['name'] = name
    c['description'] = ('%s with the groups chosen by scripts/select_groups_rule.py on %s (S_j < %g mV fixed, > %g mV fitted, '
                        'in between fitted if |rho| < %.1f): %s.' % (base['name'], os.path.basename(os.path.dirname(scr_p)),
                                                                     LO, HI, RHO, ', '.join(sel)))
    c['dynamics']['names'] = sel
    c['dynamics']['lb'] = [PB[n][0] if n in PB else BB[n][0] for n in sel]
    c['dynamics']['ub'] = [PB[n][1] if n in PB else BB[n][1] for n in sel]
    c['dynamics']['freeze_from'] = pre['name']
    out = os.path.join(ROOT, 'configs', name + '.json')
    json.dump(c, open(out, 'w', encoding='utf-8'), indent=2)
    sd, dd = os.path.join(ROOT, 'runs', 'studies', base['name']), os.path.join(ROOT, 'runs', 'studies', name)
    os.makedirs(dd, exist_ok=True)
    for f in ('windows.json', 'windows_percell.json', 'step1.json', 'ocp_correction.json', 'hysteresis.json'):
        if os.path.exists(os.path.join(sd, f)):
            shutil.copy(os.path.join(sd, f), os.path.join(dd, f))
    same = sorted(sel) == sorted(base['dynamics']['names'])
    print('wrote', out, '| selected:', sel, '| same as %s: %s' % (base['name'], same))


if __name__ == '__main__':
    main()
