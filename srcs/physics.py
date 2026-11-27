"""
physics.py
==========
What the identification takes from the PyBaMM parameter set, and nothing else:

    truth_groups     the lumped groups implied by a set at temperature T (the plant's reference values)
    geometry         what is taken as known: area, thicknesses, radii, specific areas, c_max, c_e0
    literature_physical   the set's values of the 7 physical parameters of the SPMe model (em_model)
    ocps, entropic   open-circuit potentials and their temperature coefficients
    arrhenius_ratio  how each group scales with T under the set's own laws (reference activation energies)

The particle itself is solved by the MIONet surrogate (srcs/em_model.py).
"""
from __future__ import annotations

import os
import logging
from functools import lru_cache

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

log = logging.getLogger(__name__)

R_GAS = 8.314462618
F_FARADAY = 96485.33212
T_REF = 298.15

# Order used for every parameter vector in this folder.
GROUPS = ('Q_n', 'Q_p', 'x100', 'y100', 'theta_n', 'theta_p', 'Istar_n', 'Istar_p', 'R0',
          'R_c', 'tau_e')


# ------------------------------------------------------------------ reference groups
def _ev(v, *args):
    """Evaluate a PyBaMM parameter that may be a number, a callable or a symbol."""
    if callable(v):
        v = v(*args)
    return float(v.evaluate()) if hasattr(v, 'evaluate') else float(v)


def truth_groups(pm: str = 'Mohtat2020', T: float = T_REF) -> dict:
    """The groups implied by a PyBaMM parameter set at temperature T [K].

    Q, x100/y100, theta and I* are exact translations of the set. R0, R_c and tau_e have no
    unique P2D counterpart: the values returned are the SPMe leading-order ones (electrolyte
    + solid ohmic; first salt-diffusion mode), i.e. a REFERENCE, not a truth.
    """
    import pybamm
    p = pybamm.ParameterValues(pm)
    area = float(p['Electrode width [m]']) * float(p['Electrode height [m]'])
    n_el = float(p['Number of electrodes connected in parallel to make a cell'])
    A = area * n_el
    g = {'pm': pm, 'T_K': T}
    for el, tag in (('Negative', 'n'), ('Positive', 'p')):
        Rp = float(p[f'{el} particle radius [m]'])
        eps = float(p[f'{el} electrode active material volume fraction'])
        L = float(p[f'{el} electrode thickness [m]'])
        cmax = float(p[f'Maximum concentration in {el.lower()} electrode [mol.m-3]'])
        Ds = _ev(p[f'{el} particle diffusivity [m2.s-1]'], 0.5, T)
        g[f'Q_{tag}'] = F_FARADAY * cmax * eps * L * A / 3600.0
        g[f'theta_{tag}'] = Rp ** 2 / Ds
        j0 = _ev(p[f'{el} electrode exchange-current density [A.m-2]'],
                 float(p['Initial concentration in electrolyte [mol.m-3]']), 0.5 * cmax, cmax, T)
        g[f'Istar_{tag}'] = A * (3.0 * eps / Rp) * L * j0

    # top/bottom of charge from PyBaMM's own electrode-SOH solver
    param = pybamm.LithiumIonParameters()
    Q_Li = p.evaluate(param.Q_Li_particles_init)
    esoh = pybamm.lithium_ion.ElectrodeSOHSolver(p, param)
    s = esoh.solve({'V_min': float(p['Lower voltage cut-off [V]']),
                    'V_max': float(p['Upper voltage cut-off [V]']),
                    'Q_n': g['Q_n'], 'Q_p': g['Q_p'], 'Q_Li': Q_Li})
    for k in ('x_100', 'y_100', 'x_0', 'y_0', 'Q'):
        g[k.replace('_', '')] = float(np.asarray(s[k]).ravel()[0])

    # SPMe leading-order ohmic resistance (electrolyte + solid), Bruggeman as in the set
    ce0 = float(p['Initial concentration in electrolyte [mol.m-3]'])
    kappa = _ev(p['Electrolyte conductivity [S.m-1]'], ce0, T)
    D_e = _ev(p['Electrolyte diffusivity [m2.s-1]'], ce0, T)
    t_plus = float(p['Cation transference number'])
    Ls = {k: float(p[f'{n} thickness [m]']) for k, n in
          (('n', 'Negative electrode'), ('s', 'Separator'), ('p', 'Positive electrode'))}
    por = {k: float(p[f'{n} porosity']) for k, n in
           (('n', 'Negative electrode'), ('s', 'Separator'), ('p', 'Positive electrode'))}
    brug = {k: float(p[f'{n} Bruggeman coefficient (electrolyte)']) for k, n in
            (('n', 'Negative electrode'), ('s', 'Separator'), ('p', 'Positive electrode'))}
    r_e = (Ls['n'] / (3 * kappa * por['n'] ** brug['n']) + Ls['s'] / (kappa * por['s'] ** brug['s'])
           + Ls['p'] / (3 * kappa * por['p'] ** brug['p'])) / A
    r_s = 0.0
    for el, tag in (('Negative', 'n'), ('Positive', 'p')):
        sig = float(p[f'{el} electrode conductivity [S.m-1]'])
        eps_s = float(p[f'{el} electrode active material volume fraction'])
        b_s = float(p[f'{el} electrode Bruggeman coefficient (electrode)'])
        r_s += Ls[tag] / (3 * sig * eps_s ** b_s) / A
    g['R0'] = r_e + r_s + float(p['Contact resistance [Ohm]'])
    g['R0_electrolyte'] = r_e
    g['R0_solid'] = r_s
    # first salt-diffusion mode: tau ~ L_tot^2 / (pi^2 D_eff); R_c from the SPMe steady state
    D_eff = {k: D_e * por[k] ** brug[k] for k in por}
    L_tot = Ls['n'] + Ls['s'] + Ls['p']
    g['tau_e'] = L_tot ** 2 / (np.pi ** 2 * np.mean([D_eff[k] for k in D_eff]))
    R_ce = (Ls['n'] / (3 * D_eff['n']) + Ls['s'] / D_eff['s']
            + Ls['p'] / (3 * D_eff['p'])) / A
    # eta_c ~ (2RT/F)(1 - t+) * dc/c_e0, dc = (1 - t+) I R_ce / F  ->  R_c = d eta_c / d I
    g['R_c'] = 2 * R_GAS * T / F_FARADAY * (1 - t_plus) ** 2 * R_ce / (F_FARADAY * ce0)
    return g


# ------------------------------------------------------------------ known geometry
@lru_cache(maxsize=4)
def geometry(pm: str = 'Mohtat2020') -> dict:
    """What the SPMe model takes as known from the parameter set: total electrode area A [m2]
    (width x height x electrodes in parallel), thicknesses L_n, L_s, L_p [m], particle radii R_n, R_p
    [m], specific areas a_n, a_p = 3 eps_s / R [1/m], c_max per electrode and c_e0 [mol/m3]. The
    electrolyte's porosities and Bruggeman exponents are read by srcs_EM.Electrolyte itself."""
    import pybamm
    p = pybamm.ParameterValues(pm)
    A = (float(p['Electrode width [m]']) * float(p['Electrode height [m]'])
         * float(p['Number of electrodes connected in parallel to make a cell']))
    out = {'A': A, 'L_s': float(p['Separator thickness [m]']),
           'c_e0': float(p['Initial concentration in electrolyte [mol.m-3]'])}
    for el, k in (('Negative', 'n'), ('Positive', 'p')):
        R = float(p[f'{el} particle radius [m]'])
        out['R_%s' % k] = R
        out['L_%s' % k] = float(p[f'{el} electrode thickness [m]'])
        out['a_%s' % k] = 3.0 * float(p[f'{el} electrode active material volume fraction']) / R
        out['c_max_%s' % k] = float(p[f'Maximum concentration in {el.lower()} electrode [mol.m-3]'])
    return out


def groups_from_physical(phys: dict, geo: dict) -> dict:
    """theta_k = R_k^2 / D_s,k and I*_k = A a_k L_k j0_k (exchange current of the whole electrode at
    c_e0, half stoichiometry) from the physical parameters and the known geometry. Arrays pass through."""
    out = {}
    for k in ('n', 'p'):
        out['theta_%s' % k] = geo['R_%s' % k] ** 2 / np.asarray(phys['Ds_%s' % k], dtype=float)
        out['Istar_%s' % k] = geo['A'] * geo['a_%s' % k] * geo['L_%s' % k] * np.asarray(phys['j0_%s' % k], dtype=float)
    return out


def literature_physical(pm: str = 'Mohtat2020', T: float = T_REF) -> dict:
    """The set's values of the 7 shared parameters of the SPMe model at T [K]:
        Ds_n, Ds_p   solid diffusivity at half stoichiometry [m2/s]
        j0_n, j0_p   exchange-current density at c_e0, half stoichiometry [A/m2]
        R_contact    series resistance not in the electrolyte: contact + solid ohmic [Ohm]
        alpha_De     multiplier on the set's electrolyte diffusivity (1 by definition)
        beta_ce      multiplier on (1 - t+)^2 TDF, the gain of the concentration overpotential (1)"""
    g = truth_groups(pm, T)
    geo = geometry(pm)
    out = {}
    for k in ('n', 'p'):
        out['Ds_%s' % k] = geo['R_%s' % k] ** 2 / g['theta_%s' % k]
        out['j0_%s' % k] = g['Istar_%s' % k] / (geo['A'] * geo['a_%s' % k] * geo['L_%s' % k])
    out['R_contact'] = g['R0'] - g['R0_electrolyte']
    out['alpha_De'] = 1.0
    out['beta_ce'] = 1.0
    return out


# Analytic OCPs from the literature, for step 1 (scripts/step1_window.py "U_n"): a tabulated curve's interpolant
# can carry measurement noise in its slope, which the smoothest-remainder window search then follows.
ANALYTIC_OCP = {
    # Chen et al., J. Electrochem. Soc. 167 (2020) 080534, eq. 8: LG M50 graphite-SiOx (within 2.2 mV rms of
    # PyBaMM's tabulated graphite_LGM50_ocp_Chen2020 on x in [0.1, 0.9])
    'Chen2020_graphite_SiOx': lambda x: (1.9793 * np.exp(-39.3631 * x) + 0.2482 - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
                                         - 0.04478 * np.tanh(14.9159 * (x - 0.2769)) - 0.0205 * np.tanh(30.4444 * (x - 0.6103))),
}


def _msmr_inverse(U0, X, w, T=298.15):
    """U(x) of a multi-site multi-reaction (MSMR) electrode, x(U) = sum_j X_j / (1 + exp(F (U - U0_j) / (w_j R T))),
    inverted on a fine potential grid (x decreases monotonically with U)."""
    f = 96485.33212 / (8.314462618 * T)
    U0, X, w = (np.asarray(v, float)[:, None] for v in (U0, X, w))
    Ug = np.linspace(-0.2, 2.5, 200001)
    xg = np.sum(X / (1.0 + np.exp(np.clip(f * (Ug[None, :] - U0) / w, -700, 700))), axis=0)
    xs, Us = xg[::-1], Ug[::-1]                       # ascending x
    return lambda x: np.interp(x, xs, Us)


# Verbrugge, Baker, Koch, Xiao, Gu, J. Electrochem. Soc. 164 (2017) E3243, graphite (six reactions); a generic
# graphite half cell, not the LG M50 (PyBaMM's MSMR_example_set takes these values from Verbrugge 2017)
ANALYTIC_OCP['MSMR_Verbrugge2017_graphite'] = _msmr_inverse(
    U0=[0.08843, 0.12799, 0.14331, 0.16984, 0.21446, 0.36325],
    X=[0.43336, 0.23963, 0.15018, 0.05462, 0.06744, 0.05476],
    w=[0.08611, 0.08009, 0.72469, 2.53277, 0.09470, 5.97354])


def _blend_ocp(graphite_pm: str, si_pm: str, w: float, n: int = 4001):
    """A graphite-silicon blend anode at equilibrium: both phases at the same potential U, their
    lithium adding up, x = (1 - w) x_g(U) + w x_s(U), w the silicon share of the anode capacity. Graphite: the
    anode OCP of the PyBaMM set graphite_pm; silicon: the secondary (delithiation) OCP of the set si_pm
    (Chen2020_composite). Returns U(x) as a clipped interpolant."""
    import pybamm
    xs = np.linspace(1e-4, 1.0 - 1e-4, n)
    Ug = ocps(graphite_pm)[0](xs)
    p = pybamm.ParameterValues(si_pm)
    key = next(k for k in ('Secondary: Negative electrode delithiation OCP [V]', 'Secondary: Negative electrode OCP [V]') if k in p.keys())
    Us = np.array([_ev(p[key], float(s)) for s in xs])
    mono = lambda U: np.minimum.accumulate(U)                       # U(x) non-increasing
    Ug, Us = mono(Ug), mono(Us)
    Ugrid = np.linspace(min(Ug.min(), Us.min()), max(Ug.max(), Us.max()), n)   # a phase beyond its range: x at 0 or 1
    inv = lambda U, tab: np.interp(-Ugrid, -tab, xs)                 # x(U) of one phase
    x = (1.0 - w) * inv(Ugrid, Ug) + w * inv(Ugrid, Us)
    o = np.argsort(x)
    xg, Ux = x[o], Ugrid[o]
    return lambda q: np.interp(np.clip(q, xg[0], xg[-1]), xg, Ux)


class _OCPRegistry(dict):
    """ANALYTIC_OCP, plus blends on demand: 'blend:<graphite set>:<silicon set>:<w>'."""
    def __missing__(self, key):
        if isinstance(key, str) and key.startswith('blend:'):
            _, g, s, w = key.split(':')
            self[key] = _blend_ocp(g, s, float(w))
            return self[key]
        raise KeyError(key)


ANALYTIC_OCP = _OCPRegistry(ANALYTIC_OCP)


@lru_cache(maxsize=4)
def ocps(pm: str = 'Mohtat2020', n: int = 8001):
    """Tabulated (U_n, U_p) as fast clipped interpolants."""
    import pybamm
    p = pybamm.ParameterValues(pm)
    xs = np.linspace(1e-5, 1.0 - 1e-5, n)
    tabs = []
    for el in ('Negative', 'Positive'):
        f = p[f'{el} electrode OCP [V]']
        tabs.append(np.array([_ev(f, float(s)) for s in xs]))
    U_n_tab, U_p_tab = tabs

    def U_n(x):
        return np.interp(np.clip(x, xs[0], xs[-1]), xs, U_n_tab)

    def U_p(y):
        return np.interp(np.clip(y, xs[0], xs[-1]), xs, U_p_tab)
    return U_n, U_p


def literature_electrolyte(e, source: str):
    """Give the srcs_EM.Electrolyte `e` (built on a set's geometry) the bulk electrolyte of another PyBaMM
    set `source`: D_e(c, T), kappa(c, T), TDF(c, T) and t+ (at c_e0, T_REF). The geometry, porosities,
    Bruggeman exponents and c_e0 stay those of `e`; the modes are rebuilt (the source term carries
    1 - t+), the property tables are cleared."""
    import pybamm
    p = pybamm.ParameterValues(source)
    de_fn, ka_fn = p['Electrolyte diffusivity [m2.s-1]'], p['Electrolyte conductivity [S.m-1]']
    td_fn, tp_fn = p.get('Thermodynamic factor', None), p['Cation transference number']

    def at(fn, c, T):
        return _ev(fn, float(c), float(T)) if callable(fn) else float(fn)

    e._de_func = lambda c, T: at(de_fn, c, T)
    e._kappa_func = lambda c, T: at(ka_fn, c, T)
    e._tdf_func = lambda c, T: 1.0 if td_fn is None else at(td_fn, c, T)
    e.de_ref, e.kappa_ref = e._de_func(e.c_e0, T_REF), e._kappa_func(e.c_e0, T_REF)
    e.tdf_ref, e.t_plus = e._tdf_func(e.c_e0, T_REF), at(tp_fn, e.c_e0, T_REF)
    e.param_set = '%s geometry, %s electrolyte' % (e.param_set, source)
    e._tables = {}
    e._build_modes()
    return e


@lru_cache(maxsize=4)
def entropic(pm: str = 'Mohtat2020', n: int = 2001):
    """Tabulated entropic coefficients (dU_n/dT, dU_p/dT) [V/K]; zero where the set has none.
    The OCP at temperature T is U(s) + (T - T_REF) dU/dT(s), as in PyBaMM."""
    import pybamm
    p = pybamm.ParameterValues(pm)
    xs = np.linspace(1e-5, 1.0 - 1e-5, n)
    out = []
    for el in ('Negative', 'Positive'):
        f = p.get(f'{el} electrode OCP entropic change [V.K-1]')
        if f is None:
            tab = np.zeros(n)
        else:
            try:
                tab = np.array([_ev(f, float(s), T_REF) for s in xs])
            except TypeError:
                tab = np.array([_ev(f, float(s)) for s in xs])
        out.append(tab)
    dn, dp = out
    return (lambda x: np.interp(np.clip(x, xs[0], xs[-1]), xs, dn),
            lambda y: np.interp(np.clip(y, xs[0], xs[-1]), xs, dp))


def perturbed_ocps(pm: str = 'Mohtat2020', spec: dict | None = None):
    """(U_n, U_p) with a controlled error, to emulate imperfect OCP knowledge.

    spec keys (all optional):
        n_source / p_source   take that electrode's curve from another PyBaMM parameter set
        n_shift / p_shift     add a constant [V]
        n_stretch / p_stretch evaluate the curve at stretch * s (a stoichiometry-scale error,
                              e.g. a half-cell capacity reference off by that factor)
    """
    spec = spec or {}
    U_n0, U_p0 = ocps(pm)
    if 'n_source' in spec:
        U_n0 = ocps(spec['n_source'])[0]
    if 'p_source' in spec:
        U_p0 = ocps(spec['p_source'])[1]
    sn, sp = spec.get('n_stretch', 1.0), spec.get('p_stretch', 1.0)
    dn, dp = spec.get('n_shift', 0.0), spec.get('p_shift', 0.0)

    def U_n(x):
        return U_n0(np.asarray(x) * sn) + dn

    def U_p(y):
        return U_p0(np.asarray(y) * sp) + dp
    return U_n, U_p


@lru_cache(maxsize=4)
def _arrhenius_table(pm: str = 'Mohtat2020'):
    """ln(group(T)) of the dynamic groups on a T grid, from the parameter set's own temperature laws
    (diffusivity, exchange current, electrolyte conductivity / diffusivity)."""
    Tg = np.linspace(255.0, 335.0, 17)
    keys = ('theta_n', 'theta_p', 'Istar_n', 'Istar_p', 'R0', 'R_c', 'tau_e')
    tab = {k: [] for k in keys}
    for T in Tg:
        g = truth_groups(pm, float(T))
        for k in keys:
            tab[k].append(np.log(g[k]))
    return Tg, {k: np.array(v) for k, v in tab.items()}


def arrhenius_ratio(key: str, T, T_ref: float, pm: str = 'Mohtat2020'):
    """group(T) / group(T_ref) under the parameter set's temperature laws (T may be an array) [K]."""
    Tg, tab = _arrhenius_table(pm)
    return np.exp(np.interp(T, Tg, tab[key]) - np.interp(T_ref, Tg, tab[key]))


def reference_activation_energies(pm: str = 'Mohtat2020', T_ref: float = T_REF, dT: float = 5.0) -> dict:
    """Activation energies [J/mol] implied by the set's own temperature laws at T_ref, in the form the
    model uses: theta ~ exp(+Ea_D/R (1/T - 1/T_ref)), I* ~ exp(-Ea_r/R (...)), R0 ~ exp(+Ea_e/R (...)).
    Ea_s (the series resistance outside the electrolyte) has no law in the sets: it takes R0's."""
    a, b = T_ref - dT, T_ref + dT
    x = 1.0 / a - 1.0 / b

    def slope(key):                       # d ln(group) / d(1/T)
        return np.log(arrhenius_ratio(key, a, T_ref, pm) / arrhenius_ratio(key, b, T_ref, pm)) / x
    return {'Ea_Dn': R_GAS * slope('theta_n'), 'Ea_Dp': R_GAS * slope('theta_p'),
            'Ea_r': -R_GAS * 0.5 * (slope('Istar_n') + slope('Istar_p')), 'Ea_e': R_GAS * slope('R0'),
            'Ea_s': R_GAS * slope('R0'), 's_ocv': 1.0}
