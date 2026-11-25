"""
check_mode_smoothness.py
========================
Is the simulated voltage a smooth function of the groups? Fine sweep of one group around the case B optimum,
the others fixed, on the 2C first cycle of cell 07 at 25 C and of cell 08 at -5 C; reports the RMS of the first and
second differences of V along the sweep and the largest second difference relative to the typical one. A step
between neighbouring candidates makes the forward-difference Jacobian of the fit follow the step, not the model.
With the original solver settings (4/8 Newton iterations, restart from the uniform split after a change of
current) such steps reached 1-4 mV at 25 C and 140 mV at the end of the cold 2C discharge; with
"newton": [12, 24], "jump_start": "shift" they stay below 0.4 mV. Set NEWTON, NEWTON_JUMP and JUMP_START in the
environment to compare.

    python scripts/check_mode_smoothness.py [--study fitB] [--n 201] [--rel 0.05]
Runs on the CPU (CUDA hidden).
"""
import argparse
import os
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np    # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import srcs.ms_model as MM                                   # noqa: E402
MM.NEWTON, MM.NEWTON_JUMP = int(os.environ.get("NEWTON", MM.NEWTON)), int(os.environ.get("NEWTON_JUMP", MM.NEWTON_JUMP))
MM.JUMP_START = os.environ.get("JUMP_START", MM.JUMP_START)
from srcs import identification as ident, study as S       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--study', default='fitB')
    ap.add_argument('--n', type=int, default=201)
    ap.add_argument('--rel', type=float, default=0.05, help='half-width of the sweep, relative')
    ap.add_argument('--groups', nargs='+', default=['eps_n', 'r_np', 'De_f', 'R_e'])
    a = ap.parse_args()
    st = S.load(os.path.join(ROOT, 'configs', a.study + '.json'))
    em, W = st.em(), st.windows()
    g0 = st.load('dynamics.json')['groups']
    tests = ident.prepare_tests(st.ds, [dict(cell='C07', test='VAL'), dict(cell='C08', test='VAL')], W, em.ocp)
    B = a.n
    for name in a.groups:
        x = g0[name] * (1 + np.linspace(-a.rel, a.rel, B))
        for t in tests:
            gb = {**g0, name: x, 'win_Q_n': np.ones(B), 'win_Q_p': np.ones(B), 'win_x100': np.ones(B)}
            gg, q = ident.test_groups(t, gb, em.ocp)
            V = ident.add_bias(em.simulate(gg, t['I'], t['dt'], T=t['T'], q_start=q), t, gg, q).double().cpu().numpy()
            m = np.asarray(t['mask'])
            V = V[:, m]
            d1 = np.diff(V, axis=0)                      # (B-1, K): change between neighbours
            d2 = np.diff(V, 2, axis=0)
            r1 = np.sqrt(np.mean(d1 ** 2, axis=1))       # per step, RMS over samples
            r2 = np.sqrt(np.mean(d2 ** 2, axis=1))
            print('%-6s %s %-4s  step %.2e: dV RMS %.3f uV (min %.3f, max %.3f) | d2V RMS median %.4f uV, max %.4f uV '
                  '(x%.0f), at %s = %.4g' % (
                      name, t['cell'], 'VAL', x[1] - x[0], 1e6 * np.median(r1), 1e6 * r1.min(), 1e6 * r1.max(),
                      1e6 * np.median(r2), 1e6 * r2.max(), r2.max() / max(np.median(r2), 1e-12), name,
                      x[1 + int(np.argmax(r2))]), flush=True)


if __name__ == '__main__':
    main()
