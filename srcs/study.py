"""
study.py
========
A study is one JSON config (configs/<name>.json) and one folder (runs/studies/<name>/). This module reads
the config, resolves the split of the cells, enforces the held-out rules, runs the stages and records
what every result was computed from.

Config keys
    name, description        the study folder is runs/studies/<name>/; a name is never reused with a
                             different config (the copy in the folder is compared at every start)
    dataset                  configs/datasets/<dataset>.json (cells, tests, budget, OCP parameter set)
    surrogate                the mionet-clean run (runs/v2_nondim/<surrogate>)
    pm                       optional: the PyBaMM set for OCPs and temperature laws (default: the dataset's)
    T_ref_C                  the temperature the dynamic groups are given at
    model                    optional: {"dT_window_K": 1.0} the temperature drift that opens a new
                             particle window (em_model.windows; error measured in STATUS.md);
                             "electrolyte": "rc" (default: one branch, lumped groups R0, R_c, tau_e) or
                             "spme" (modal SPMe electrolyte on the parameter set's geometry, physical
                             parameters Ds_n, Ds_p, j0_n, j0_p, R_contact, alpha_De, beta_ce);
                             "ohmic" (the electrolyte as a series resistance R0, with the particle and
                             kinetic parameters of "spme": Ds_n, Ds_p, j0_n, j0_p, R0);
                             "n_modes": electrolyte modes kept with "spme" (default em_model.N_MODES);
                             "electrolyte_T": with "spme", "Ea_e" (default: the electrolyte and R_contact
                             share the fitted Ea_e) or "set" (the electrolyte on the set's own kappa(T),
                             D_e(T); R_contact on its own Ea_s);
                             "electrolyte_set": with "spme", a PyBaMM set whose bulk electrolyte replaces
                             pm's on pm's geometry (e.g. "ORegan2022": Landesfeind 2019, LiPF6 in EC:EMC
                             3:7); the optional parameter "tort" scales its effective transport;
                             "entropic": "set" (default: s_ocv x the set's dU/dT) or "cell" (a cell
                             coefficient of its own, dUdT0 + dUdT_n (x_s - 0.5) + dUdT_p (y_s - 0.5)
                             [V/K], em_model docstring);
                             "multislice": the multi-slice model (srcs/ms_model.py) with the parameters
                             and options of "spme"; "n_slices" per electrode (default ms_model.N_SLICES),
                             "n_modes" default ms_model.N_MODES_MS, "slice_ratio" (widest / narrowest
                             slice, geometric, narrowest at the separator; default 1, uniform;
                             geometry "groups" only)
    split                   {"dynamics": [...], "temperature": [...], "validation": [...], "exclude": [...]}
                             validation defaults to every remaining cell
    window, dynamics, temperature, benchmark, evaluation    the stages (see the stage functions)
    source                   instead of identifying: import a parameters/*.json set (stage "import")

Held-out rules, checked before anything is computed:
    every cell has exactly one role (dynamics, temperature, validation, exclude)
    window        every non-excluded cell, from its OWN budget tests only
    dynamics      only the dynamics cells;  temperature   only the temperature cells
    evaluation    validation cells, on held-out tests only (a test outside the dataset budget); fitting
                  cells only with "include_fitting_cells": true, and then flagged in_sample

Reproducibility: every stage writes its result and an entry in provenance.json - inputs (the sha256 of
every data file and of the upstream results it used), the git commit and whether the tree was dirty,
package versions, device, seeds, wall time. A stage whose result exists is skipped, but only if its
recorded upstream inputs are unchanged; otherwise it refuses to run on stale results.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from srcs import datasets, ecm, identifiability, identification as ident, physics, window as win
from srcs.datasets.base import sha256
from srcs.em_model import DT_WINDOW, N_MODES, BatchEM
from srcs.ocp import OCPSet
from srcs.optimizers import optimize_seeds
from srcs.paths import ROOT, STUDIES

log = logging.getLogger('study')
STAGES = ('import', 'window', 'dynamics', 'temperature', 'benchmark', 'evaluation')


# ------------------------------------------------------------------ provenance
def _git(*args):
    try:
        return subprocess.run(['git', '-C', ROOT, *args], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


def environment(device=None) -> dict:
    import scipy
    import torch
    env = dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
               scipy=scipy.__version__, torch=torch.__version__, cuda=torch.version.cuda,
               device=device, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               git_commit=_git('rev-parse', 'HEAD'),
               git_dirty_files=len([l for l in (_git('status', '--porcelain', '--', '.') or '').splitlines() if l]))
    try:
        import pybamm
        env['pybamm'] = pybamm.__version__
    except Exception:
        pass
    return env


# ------------------------------------------------------------------ the study
class Study:
    def __init__(self, cfg: dict, config_path: str | None = None):
        self.cfg = cfg
        self.name = cfg['name']
        self.dir = os.path.join(STUDIES, self.name)
        self.ds = datasets.get(cfg['dataset']) if 'dataset' in cfg else None
        self.pm = cfg.get('pm', self.ds.pm if self.ds else 'Mohtat2020')
        self.T_ref = float(cfg.get('T_ref_C', 25.0)) + 273.15
        self.config_path = config_path
        self._em = None
        if self.ds is not None:
            self.split = self.resolve_split()

    # ---------------------------------------------------------------- folder, config copy, log
    def open(self):
        os.makedirs(self.dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True,
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler(os.path.join(self.dir, 'study.log'), encoding='utf-8')])
        copy = os.path.join(self.dir, 'config.json')
        if os.path.exists(copy):
            if json.load(open(copy, encoding='utf-8')) != self.cfg:
                raise SystemExit('runs/studies/%s/config.json differs from %s - give the study a new "name"'
                                 % (self.name, self.config_path))
        else:
            json.dump(self.cfg, open(copy, 'w', encoding='utf-8'), indent=2)
            json.dump(dict(date=datetime.now(timezone.utc).isoformat(), description=self.cfg.get('description', ''),
                           dataset=self.cfg.get('dataset', self.cfg.get('source')), config=self.config_path),
                      open(os.path.join(self.dir, 'metadata.json'), 'w'), indent=2)
        if self.ds is not None:
            ident.save(self.path('split.json'), self.split)
        return self

    def path(self, f):
        return os.path.join(self.dir, f)

    def load(self, f):
        return json.load(open(self.path(f), encoding='utf-8'))

    # ---------------------------------------------------------------- split and held-out rules
    def resolve_split(self) -> dict:
        sp, cells = self.cfg.get('split', {}), self.ds.cells
        roles = {r: list(sp.get(r, [])) for r in ('dynamics', 'temperature', 'exclude')}
        listed = [c for r in roles.values() for c in r]
        unknown = sorted(set(listed + list(sp.get('validation', []))) - set(cells))
        if unknown:
            raise SystemExit('split: cells %s are not in dataset %s' % (unknown, self.ds.name))
        dup = sorted({c for c in listed if listed.count(c) > 1})
        if dup:
            raise SystemExit('split: cells %s have more than one role' % dup)
        rest = [c for c in cells if c not in listed]
        val = list(sp.get('validation', rest))
        if set(val) & set(listed):
            raise SystemExit('split: validation cells %s also fit or are excluded' % sorted(set(val) & set(listed)))
        unassigned = [c for c in rest if c not in val]
        if unassigned:
            raise SystemExit('split: cells %s have no role (add them to "validation" or "exclude")' % unassigned)
        return dict(roles, validation=val, budget=self.ds.budget,
                    window=[c for c in cells if c not in roles['exclude']])

    def expand(self, stage: str, items, role: str):
        """Test specs of a stage. item: "GITT" or {"test": "GITT", "coarsen": 5, "cells": [...], "T_C": 25};
        without "cells"/"cell" it applies to every cell of `role`. Cells outside the role are refused."""
        allowed, specs = self.split[role], []
        for it in items:
            it = {'test': it} if isinstance(it, str) else dict(it)
            cells = it.pop('cells', None) or ([it.pop('cell')] if 'cell' in it else allowed)
            bad = [c for c in cells if c not in allowed]
            if bad:
                raise SystemExit('%s: cells %s are not %s cells %s - held-out rule' % (stage, bad, role, allowed))
            if it['test'] not in self.ds.tests:
                raise SystemExit('%s: unknown test %r for dataset %s' % (stage, it['test'], self.ds.name))
            for c in cells:
                if self.ds.available(c, it['test']):
                    specs.append(dict(it, cell=c))
                else:
                    log.warning('%s: %s %s is not prepared - skipped', stage, c, it['test'])
        if not specs:
            raise SystemExit('%s: no test to fit (are the data prepared?)' % stage)
        return specs

    def heldout_specs(self, cells) -> list:
        """The test of each held-out cell: the first spec of config "heldout_tests" whose "cells" list it, else its
        whole first cycle, {"test": "VAL"} (LG: only the 1C part of an E5 cycle, the WLTP part of an E4 one)."""
        out = []
        for c in cells:
            sp = next((s for s in self.cfg.get('heldout_tests', []) if c in s.get('cells', [])), {'test': 'VAL'})
            out.append({**{k: v for k, v in sp.items() if k != 'cells'}, 'cell': c})
        return out

    def nominal_T(self, cell):
        """The chamber temperature of a cell in the dataset config [C], or None."""
        try:
            return float(self.ds.cell_T_C(cell))
        except Exception:
            return None

    # ---------------------------------------------------------------- model pieces
    def ocp(self) -> OCPSet:
        p = self.path('ocp_correction.json')
        return OCPSet(self.pm, json.load(open(p)) if os.path.exists(p) else None)

    def em(self) -> BatchEM:
        if self._em is None:
            m = self.cfg.get('model', {})
            ident.CLAMP_CUTOFF = m.get('clamp_cutoff')     # [V] or None: identification.clamp_cutoff
            ident.CORE_HEAT = m.get('core_heat')           # [k, tau] or None: identification.prepare_tests
            ident.THETA_REF_K = None if m.get('theta_ref_C') is None else float(m['theta_ref_C']) + 273.15
            ident.HYSTERESIS = None        # model.hysteresis {"q_h_Ah", "h0"} with <study>/hysteresis.json ({"x", "dV"},
            if m.get('hysteresis'):        # scripts/hysteresis_c10_gitt.py): identification.hysteresis_voltage
                hy = json.load(open(self.path(m['hysteresis'].get('file', 'hysteresis.json'))))
                ident.HYSTERESIS = {**hy, **{k: v for k, v in m['hysteresis'].items() if k != 'file'}}
            if m.get('electrolyte') == 'multislice':
                from srcs import ms_model
                from srcs.ms_model import N_MODES_MS, N_SLICES, BatchMS
                ms_model.J0_STOICH = bool(m.get('j0_stoich', True))     # False: j0 independent of x_s (a test)
                # Newton of the current split: [iterations, after a change of current], and where it restarts after
                # a change ('uniform' | 'shift', ms_model.JUMP_START). The defaults (4/8, 'uniform') leave voltage
                # islands between neighbouring candidates; the paper's fits use "newton": [12, 24],
                # "jump_start": "shift" (scripts/check_mode_smoothness.py).
                ms_model.NEWTON, ms_model.NEWTON_JUMP = (int(v) for v in m.get('newton', (4, 8)))
                ms_model.JUMP_START = str(m.get('jump_start', 'uniform'))
                kw = dict(ocp=self.ocp(), dT_window=float(m.get('dT_window_K', DT_WINDOW)),
                          n_slices=int(m.get('n_slices', N_SLICES)), n_modes=int(m.get('n_modes', N_MODES_MS)),
                          electrolyte_T=m.get('electrolyte_T', 'Ea_e'), entropic=m.get('entropic', 'set'),
                          electrolyte_set=m.get('electrolyte_set'), geometry=m.get('geometry', 'set'),
                          electrolyte_groups=m.get('electrolyte_groups', 'free'), graphs=bool(m.get('graphs', True)),
                          slice_ratio=float(m.get('slice_ratio', 1.0)))
                if m.get('particles', 'mionet') == 'fv':          # the exact finite-volume particles (srcs/ms_fv.py)
                    from srcs.ms_fv import BatchMSFV
                    s1 = self.cfg.get('step1', {})                  # the D_p(y) law on the conventional cathode window
                    self._em = BatchMSFV(None, self.pm, n_fine=int(m.get('n_fine', 51)),
                                         d_law=bool(m.get('d_law', False)),
                                         d_window=(float(s1.get('y100', 0.40)), float(s1.get('y_end', 0.95))), **kw)
                else:
                    self._em = BatchMS(self.cfg['surrogate'], self.pm, **kw)
                self._em.T_per_window = bool(m.get('T_per_window', False))   # ms_model.simulate
                self._em.tdf_T = bool(m.get('tdf_T', False))     # chi(c_e0, T) in the concentration overpotential
                return self._em
            self._em = BatchEM(self.cfg['surrogate'], self.pm, ocp=self.ocp(),
                               dT_window=float(m.get('dT_window_K', DT_WINDOW)),
                               electrolyte=m.get('electrolyte', 'rc'), n_modes=int(m.get('n_modes', N_MODES)),
                               electrolyte_T=m.get('electrolyte_T', 'Ea_e'), entropic=m.get('entropic', 'set'),
                               electrolyte_set=m.get('electrolyte_set'))
        return self._em

    def windows(self):
        return self.load('windows.json')

    def literature(self):
        """The parameter set's values of every group and physical parameter at T_ref (both models). For a
        synthetic plant (srcs/datasets/msplant.py) the plant's own values override them: they are the
        truth the benchmark's parameter errors are measured against."""
        out = {**physics.truth_groups(self.pm, self.T_ref), **physics.literature_physical(self.pm, self.T_ref)}
        if hasattr(self.ds, 'truth'):
            tr = self.ds.truth()
            out.update({k: v for k, v in tr.items() if isinstance(v, float)})
            if 'R_contact' in tr and 'R0_electrolyte' in out:      # the series resistance of an 'ohmic' model
                out['R0'] = float(tr['R_contact']) + out['R0_electrolyte']
            if os.path.exists(self.path('windows.json')):     # the window factors that would be exact
                W, cells = self.windows(), self.split.get('dynamics', [])
                for k in ident.WINDOW:
                    r = [out[k] / W[c][k] for c in cells if c in W]
                    if r:
                        out[ident.FACTOR + k] = float(np.mean(r))
        if getattr(self.em(), 'geometry', 'set') == 'groups':      # the multi-slice model in groups: the reference as groups
            out.update(self.em().set_groups({'alpha_De': 1.0, 'beta_ce': 1.0, **out}))
        return out

    def thermal(self, mode):
        """Temperature groups used while fitting the dynamics: none, or the parameter set's own laws."""
        if mode == 'none':
            return {'Ea_Dn': 0.0, 'Ea_Dp': 0.0, 'Ea_r': 0.0, 'Ea_e': 0.0, 'Ea_s': 0.0, 's_ocv': 1.0}
        if mode == 'reference':
            return physics.reference_activation_energies(self.pm, self.T_ref)
        raise SystemExit('dynamics: unknown thermal mode %r (none | reference)' % mode)

    def fixed_groups(self, spec: dict) -> dict:
        """{"tau_e": 30.0, "R_c": "literature"} -> numbers."""
        lit = None
        out = {}
        for k, v in (spec or {}).items():
            if v == 'literature':
                lit = lit or self.literature()
                v = lit[k]
            out[k] = float(v)
        return out

    # ---------------------------------------------------------------- provenance
    def record(self, stage, t0, inputs, **extra):
        p = self.path('provenance.json')
        prov = json.load(open(p)) if os.path.exists(p) else {}
        prov[stage] = dict(finished=datetime.now(timezone.utc).isoformat(), wall_s=time.time() - t0,
                           inputs=inputs, environment=environment(self._em.device if self._em else None), **extra)
        ident.save(p, prov)

    def upstream_ok(self, stage):
        """True if `stage` has a result whose recorded upstream inputs are unchanged."""
        p = self.path('provenance.json')
        prov = json.load(open(p)) if os.path.exists(p) else {}
        if stage not in prov:
            return False
        for f, h in prov[stage]['inputs'].items():
            if f.startswith('study:'):
                g = self.path(f[6:])
                if not os.path.exists(g) or sha256(g) != h:
                    raise SystemExit('%s: its input %s changed since it was computed - delete %s/%s.* and re-run'
                                     % (stage, f[6:], self.name, stage))
        return True

    def data_inputs(self, specs):
        return {'data:%s/%s' % (s['cell'], s['test']): self.ds.data_hash(s['cell'], s['test'])
                for s in {(x['cell'], x['test']): x for x in specs}.values()}

    def study_inputs(self, *files):
        return {'study:%s' % f: sha256(self.path(f)) for f in files if os.path.exists(self.path(f))}

    # ---------------------------------------------------------------- run
    def run(self, stages=None):
        self.open()
        stages = stages or [s for s in STAGES if s in self.cfg or (s == 'import' and 'source' in self.cfg)]
        log.info('study %s | dataset %s | stages %s', self.name, self.cfg.get('dataset'), stages)
        if self.ds is not None:
            log.info('split: %s', {k: v for k, v in self.split.items() if k != 'window'})
        for s in stages:
            getattr(self, 'stage_' + s)()
        log.info('done: runs/studies/%s', self.name)

    # ================================================================ stages
    def stage_import(self):
        """A parameters/*.json set (schema 1.0) as windows / dynamics / temperature results."""
        if os.path.exists(self.path('dynamics.json')):
            return
        t0 = time.time()
        src = os.path.join(ROOT, self.cfg['source'])
        ps = json.load(open(src, encoding='utf-8'))
        G = ps['groups']
        windows = {}
        for c, w in G['static_window']['per_cell'].items():
            key = 'C%02d' % int(c) if str(c).isdigit() else str(c)
            windows[key] = {k: w[k] for k in win.WINDOW}
            windows[key]['rmse_mV'] = w.get('ocv_fit_rmse_mV')
        ident.save(self.path('windows.json'), windows)
        dyn = dict(G['dynamic']['values'])
        ref = physics.reference_activation_energies(self.pm, self.T_ref)
        family = {'Ea_Dn': 'theta', 'Ea_Dp': 'theta', 'Ea_r': 'kinetics', 'Ea_e': 'ohmic'}
        ex = G['temperature']['exponents']
        thermal = {k: ex[f] * ref[k] for k, f in family.items()}
        thermal['s_ocv'] = ex['ocv']
        ident.save(self.path('dynamics.json'), dict(groups={**ref, **dyn}, fitted=list(dyn), source=self.cfg['source'],
                                                     provenance=G['dynamic']['provenance']))
        ident.save(self.path('temperature.json'), dict(groups={**dyn, **thermal}, fitted=list(thermal),
                                                        source=self.cfg['source'], exponents=ex, Ea_ref=ref,
                                                        provenance=G['temperature']['provenance']))
        self.record('import', t0, {'source:%s' % self.cfg['source']: sha256(src)})
        log.info('imported %s: %d windows', self.cfg['source'], len(windows))

    def stage_window(self):
        """Capacity window of every non-excluded cell from its own budget tests (srcs/window.py).
        -> windows.json, window_profiles.csv, ocp_correction.json (optional)"""
        if os.path.exists(self.path('windows.json')) and self.upstream_ok('window'):
            log.info('window: done before')
            return
        t0 = time.time()
        wcfg = self.cfg['window']
        used = {k for k in list(wcfg['fit_on']) + list(wcfg.get('check_on', []))}
        tests = {'rest': self.ds.window_data.get('rest_test'), 'pseudo_ocv': self.ds.window_data.get('ocv_test'),
                 'discharge': self.ds.window_data.get('ocv_test'), 'charge': self.ds.window_data.get('ocv_test')}
        for k in used:
            if tests[k] not in self.ds.budget:
                raise SystemExit('window: %s comes from %s, which is not a budget test' % (k, tests[k]))
        cells = self.split['window']
        ocp = OCPSet(self.pm)
        windows, profiles = win.fit_cells(self.ds, cells, wcfg, ocp, seed=int(wcfg.get('seed', 0)))
        if wcfg.get('ocp_correction'):
            fit_cells = wcfg['ocp_correction'].get('cells', 'dynamics')
            fit_cells = self.split[fit_cells] if isinstance(fit_cells, str) else list(fit_cells)
            ocp, _, report = win.fit_correction(self.ds, fit_cells, wcfg, ocp, windows)
            ident.save(self.path('ocp_correction.json'), {k: report[k] for k in ('electrode', 'knots', 'dU')})
            ident.save(self.path('ocp_correction_report.json'), report)
            windows, profiles = win.fit_cells(self.ds, cells, wcfg, ocp, seed=int(wcfg.get('seed', 0)))
        ident.save(self.path('windows.json'), windows)
        pd.DataFrame([dict(cell=c, Q_n=q, rmse_mV=r) for c, pr in profiles.items() for q, r in pr]).to_csv(
            self.path('window_profiles.csv'), index=False, float_format='%.6g')
        specs = [dict(cell=c, test=tests[k]) for c in cells for k in used]
        self.record('window', t0, self.data_inputs(specs), Q_n_mode=str(wcfg.get('Q_n', 'free')),
                    adopted_outside_valley=[c for c, w in windows.items() if w.get('adopted_in_valley') is False])

    def _fit(self, stage, specs, names, lb, ub, fixed, st):
        """Run the optimiser for every seed of the stage (merged GPU batches) and analyse the optimum."""
        em = self.em()
        tests = ident.prepare_tests(self.ds, specs, self.windows(), em.ocp)
        cost = ident.Cost(em, tests, names, fixed)
        seeds = list(st.get('seeds', [0]))
        log.info('%s: %d tests %s | %d parameters | %s, budget %d x %d seeds', stage, len(tests),
                 [t['label'] for t in tests], len(names), st['optimizer'], st['budget'], len(seeds))
        runs = optimize_seeds(st['optimizer'], cost, lb, ub, st['budget'], seeds, **st.get('options', {}))
        best = min(runs, key=lambda r: r['f'])
        X = np.array([r['x'] for r in runs])
        spread = {n: (float(np.std(np.log(X[:, i]))) if np.all(X[:, i] > 0)                  # log; signed: rel.
                      else float(np.std(X[:, i]) / max(np.mean(np.abs(X[:, i])), 1e-30))) if len(runs) > 1 else None
                  for i, n in enumerate(names)}
        out = dict(groups={**fixed, **dict(zip(names, best['x']))}, fitted=list(names), fixed=fixed,
                   lb=list(lb), ub=list(ub), cost_mV=best['f'], best_seed=best['seed'],
                   rmse_per_test=cost.rmse_per_test(best['x']),
                   seeds=[dict(seed=r['seed'], f=r['f'], groups=dict(zip(names, r['x'])), n_eval=r['n_eval'])
                          for r in runs],
                   seed_spread_log=spread, n_shared_calls=best['n_shared_calls'], wall_s=best['wall_s'],
                   history=best['history'], tests=[t['label'] for t in tests])
        if st.get('identifiability', True):
            out['identifiability'] = identifiability.analyse(em, tests, names, best['x'], fixed)
            idf = out['identifiability']
            log.info('%s identifiability: collinearity index %.1f, max |corr| %.2f, rel. std (lower bound) %s',
                     stage, idf['collinearity_index'], idf['max_abs_correlation'],
                     {k: '%.3g' % v for k, v in idf['rel_std_lower_bound'].items()})
        log.info('%s: %.2f mV (seeds %s) | %s', stage, best['f'], ['%.2f' % r['f'] for r in runs],
                 {k: '%.4g' % v for k, v in dict(zip(names, best['x'])).items()})
        return out

    def stage_dynamics(self):
        """theta_n, theta_p, I*_n, I*_p, R0, R_c, tau_e at T_ref on the dynamics cells' tests.
        config: tests, names, lb, ub, fixed ({group: value | "literature"}), thermal (reference | none),
        optimizer, budget, options, seeds, identifiability."""
        if os.path.exists(self.path('dynamics.json')) and self.upstream_ok('dynamics'):
            log.info('dynamics: done before')
            return
        t0 = time.time()
        st = self.cfg['dynamics']
        specs = self.expand('dynamics', st['tests'], 'dynamics')
        fixed = {**self.thermal(st.get('thermal', 'reference')), **self.fixed_groups(st.get('fixed'))}
        missing = [k for k in self.em().dynamic if k not in st['names'] and k not in fixed]
        if missing:
            raise SystemExit('dynamics: groups %s are neither fitted nor fixed' % missing)
        out = self._fit('dynamics', specs, st['names'], st['lb'], st['ub'], fixed, st)
        ident.save(self.path('dynamics.json'), out)
        self.record('dynamics', t0, {**self.data_inputs(specs), **self.study_inputs('windows.json', 'ocp_correction.json')},
                    seeds=st.get('seeds', [0]))

    def stage_temperature(self):
        """mode "fit": activation energies on the temperature cells' tests, dynamic groups fixed;
        "reference": the parameter set's own laws (no data used); "none": isothermal."""
        if os.path.exists(self.path('temperature.json')) and self.upstream_ok('temperature'):
            log.info('temperature: done before')
            return
        t0 = time.time()
        st = self.cfg['temperature']
        dyn = self.load('dynamics.json')['groups']
        mode = st.get('mode', 'fit')
        if mode == 'fit':
            specs = self.expand('temperature', st['tests'], 'temperature')
            fixed = {k: v for k, v in dyn.items() if k not in st['names']}
            out = self._fit('temperature', specs, st['names'], st['lb'], st['ub'], fixed, st)
            inputs = self.data_inputs(specs)
        elif mode in ('reference', 'none'):  # activation energies fitted with the dynamics (or fixed there, "reference") are kept
            dj = self.load('dynamics.json')
            keep = set(dj.get('fitted', [])) | (set(dj.get('fixed', {})) if mode == 'reference' else set())
            th = {k: v for k, v in self.thermal(mode).items() if k not in keep}
            kept = [k for k in dj.get('fitted', []) if k.startswith(('Ea_', 'dUdT')) or k == 's_ocv']
            note = 'not identified in this stage: %s' % ('the parameter set %s temperature laws' % self.pm
                                                         if mode == 'reference' else 'isothermal')
            if kept:
                note = '%s fitted in the dynamics stage (dynamics.json); the rest (%s): %s' % (
                    ', '.join(kept), ', '.join(sorted(th)) or 'none', note)
            out = dict(groups={**dyn, **th}, fitted=[], mode=mode, note=note)
            inputs = {}
            log.info('temperature: %s %s', mode, {k: '%.4g' % v for k, v in th.items()})
        else:
            raise SystemExit('temperature: unknown mode %r (fit | reference | none)' % mode)
        ident.save(self.path('temperature.json'), out)
        self.record('temperature', t0, {**inputs, **self.study_inputs('dynamics.json')}, mode=mode)

    def stage_benchmark(self):
        """Every optimiser of the config at the same budget over the seeds, on the dynamics tests; scored
        on the evaluation tests and against the reference groups (the plant)."""
        if os.path.exists(self.path('benchmark.csv')):
            log.info('benchmark: done before')
            return
        t0 = time.time()
        st, bm = self.cfg['dynamics'], self.cfg['benchmark']
        em = self.em()
        specs = self.expand('dynamics', st['tests'], 'dynamics')
        tests = ident.prepare_tests(self.ds, specs, self.windows(), em.ocp)
        val = ident.prepare_tests(self.ds, self.eval_specs(), self.windows(), em.ocp)
        fixed = {**self.thermal(st.get('thermal', 'none')), **self.fixed_groups(st.get('fixed'))}
        truth = self.literature()
        cost, vcost = ident.Cost(em, tests, st['names'], fixed), ident.Cost(em, val, st['names'], fixed)
        rows = []
        for method, options in bm['methods'].items():
            for r in optimize_seeds(method, cost, st['lb'], st['ub'], bm['budget'], bm['seeds'], **options):
                row = dict(method=method, seed=r['seed'], train_mV=r['f'], wall_s=r['wall_s'], n_eval=r['n_eval'],
                           n_batches=len(r['history']), n_shared_calls=r['n_shared_calls'],
                           **{'val_%s' % k: v for k, v in vcost.rmse_per_test(r['x']).items()})
                for n, v in zip(st['names'], r['x']):
                    row[n] = v
                    row['err_%s' % n] = 100 * (v / truth[n] - 1) if n in truth else np.nan
                rows.append(row)
                log.info('%-10s seed %d: train %.2f mV, %.0f s, %d batches', method, r['seed'], r['f'], r['wall_s'],
                         len(r['history']))
            pd.DataFrame(rows).to_csv(self.path('benchmark.csv'), index=False)
        self.record('benchmark', t0, {**self.data_inputs(specs), **self.study_inputs('windows.json')})

    # ---------------------------------------------------------------- evaluation
    def eval_specs(self):
        ev = self.cfg['evaluation']
        cells = list(self.split['validation'])
        if ev.get('include_fitting_cells'):
            cells += self.split['dynamics'] + self.split['temperature']
        tests = ev.get('tests') or [t for t in self.ds.tests if t not in self.ds.budget]
        specs = []
        for it in tests:
            it = {'test': it} if isinstance(it, str) else dict(it)
            if it['test'] in self.ds.budget:
                raise SystemExit('evaluation: %s is a budget test - the held-out set must not contain it' % it['test'])
            for c in cells:
                if self.ds.available(c, it['test']):
                    specs.append(dict(it, cell=c))
                else:
                    log.warning('evaluation: %s %s is not prepared - skipped', c, it['test'])
        return specs

    def stage_evaluation(self):
        """EM (identified groups, the cell's own window, measured temperature - nothing refitted) against
        the ECM tuned on the cell's own budget tests, on every held-out test of the evaluation cells.
        -> evaluation.csv, evaluation_traces.npz"""
        t0 = time.time()
        ev = self.cfg['evaluation']
        em = self.em()
        groups = self.load('temperature.json' if os.path.exists(self.path('temperature.json')) else 'dynamics.json')['groups']
        windows = self.windows()
        specs = self.eval_specs()
        ecm_cfg = ev.get('ecm')
        ocv_kinds = [k for k in self.cfg['window']['fit_on'] if k in ('rest', 'pseudo_ocv')] if 'window' in self.cfg else ['rest']
        ecm_fits, rows, traces = {}, [], {}
        truth = self.literature() if ev.get('reference_groups') else None
        v_min = self.fit_V_min()
        for s in specs:
            t = ident.prepare_tests(self.ds, [s], windows, em.ocp)[0]
            g, q = ident.test_groups(t, groups, em.ocp)                       # with any window factors
            V_em = em.simulate(g, t['I'], t['dt'], T=t['T'], q_start=q)[0].cpu().numpy()
            role = 'validation' if s['cell'] in self.split['validation'] else 'in_sample'
            row = dict(cell=s['cell'], test=s['test'], role=role, T_mean_C=float(np.mean(t['T']) - 273.15),
                       n=int(t['mask'].sum()), **err('em', V_em, t, v_min))
            if v_min is not None:
                row.update(V_min=v_min, n_above_vmin=int((t['mask'] & (t['V'] >= v_min)).sum()))
            cols = [V_em]
            if ecm_cfg:
                if s['cell'] not in ecm_fits:
                    ecm_fits[s['cell']] = self.fit_ecm(s['cell'], ocv_kinds, ecm_cfg)
                p, ocv_q = ecm_fits[s['cell']]
                V_ecm = ecm.simulate_q(p, t['I'], t['dt'], ocv_q, t['q_start'])
                row.update(err('ecm', V_ecm, t, v_min), **{'ecm_%s' % k: v for k, v in zip(('R0', 'R1', 'tau1', 'R2', 'tau2'), p)})
                cols.append(V_ecm)
            if truth is not None:
                g, q = ident.test_groups(t, {**groups, **{k: truth[k] for k in em.dynamic}}, em.ocp)
                V_ref = em.simulate(g, t['I'], t['dt'], T=t['T'], q_start=q)[0].cpu().numpy()
                row.update(err('reference', V_ref, t, v_min))
                cols.append(V_ref)
            rows.append(row)
            traces['%s__%s' % (s['cell'], s['test'])] = np.vstack([t['t'], t['I'], t['V'], *cols,
                                                                 t['T'], t['mask']])
            log.info('%-4s %-8s %-10s (%5.1f C): EM %6.2f mV%s%s', s['cell'], s['test'], role, row['T_mean_C'], row['em_mV'],
                     ' | ECM %6.2f mV' % row['ecm_mV'] if 'ecm_mV' in row else '',
                     ' | above %.1f V: EM %.2f%s' % (v_min, row['em_above_vmin_mV'], ', ECM %.2f mV' % row['ecm_above_vmin_mV']
                                                    if 'ecm_above_vmin_mV' in row else ' mV') if v_min is not None else '')
        df = pd.DataFrame(rows)
        df.to_csv(self.path('evaluation.csv'), index=False, float_format='%.6g')
        np.savez_compressed(self.path('evaluation_traces.npz'), **traces)
        for role, g in df.groupby('role'):
            log.info('%s (%d tests): EM %.2f mV%s%s', role, len(g), g.em_mV.mean(),
                     ' | ECM %.2f mV' % g.ecm_mV.mean() if 'ecm_mV' in g else '',
                     ' | above %.1f V: EM %.2f%s' % (v_min, g.em_above_vmin_mV.mean(), ', ECM %.2f mV' % g.ecm_above_vmin_mV.mean()
                                                    if 'ecm_above_vmin_mV' in g else ' mV') if v_min is not None else '')
        self.record('evaluation', t0, {**self.data_inputs(specs), **self.study_inputs('windows.json', 'dynamics.json',
                                                                                      'temperature.json', 'ocp_correction.json')})

    def fit_V_min(self):
        """The lowest V_min of the dynamics tests, or None. The evaluation also reports every RMSE above it
        (<tag>_above_vmin_mV): below the fit's range the OCV is so steep that a fraction of a mAh of initial
        charge moves the deep-discharge tail by mV, and that tail is not what was fitted."""
        vs = [float(s['V_min']) for s in self.cfg.get('dynamics', {}).get('tests', []) if isinstance(s, dict) and 'V_min' in s]
        return min(vs) if vs else None

    def fit_ecm(self, cell, ocv_kinds, ecm_cfg):
        """ECM of one cell on its OWN budget: OCV from the window's OCV data, R/RC on the budget pulse test."""
        if ecm_cfg['fit_on'] not in self.ds.budget:
            raise SystemExit('evaluation: the ECM must be tuned on a budget test, not %s' % ecm_cfg['fit_on'])
        bl = win.blocks(self.ds, cell, ocv_kinds, self.cfg.get('window', {}).get('ocv_V_range'))
        ocv_q = ecm.ocv_curve(np.concatenate([b.q for b in bl]), np.concatenate([b.v for b in bl]))
        t = ident.load_test(self.ds, cell, ecm_cfg['fit_on'], ecm_cfg.get('coarsen', 1))
        p = ecm.fit_q(t['I'], t['V'], t['dt'], ocv_q, 0.0, t['mask'], ecm_cfg.get('downsample_s', 5.0))
        return p, ocv_q


def err(tag, V, t, v_min=None):
    """RMSE, 95th percentile and max |error| [mV] over the scored samples; with v_min, also the RMSE over the
    scored samples whose measured voltage is at least v_min."""
    e = (V - t['V'])[t['mask']]
    out = {'%s_mV' % tag: float(1e3 * np.sqrt(np.mean(e ** 2))),
           '%s_p95_mV' % tag: float(1e3 * np.percentile(np.abs(e), 95)),
           '%s_max_mV' % tag: float(1e3 * np.max(np.abs(e)))}
    if v_min is not None:
        m = t['mask'] & (t['V'] >= v_min)
        out['%s_above_vmin_mV' % tag] = float(1e3 * np.sqrt(np.mean((V - t['V'])[m] ** 2))) if m.any() else np.nan
    return out


def load(config_path: str) -> Study:
    cfg = json.load(open(config_path, encoding='utf-8'))
    return Study(cfg, os.path.relpath(config_path, ROOT))
