"""
em_model.py
===========
The electrochemical model of the identification, written in lumped groups, with both particles
advanced by the parameter-free physics-informed MIONet (mionet-clean, package srcs_EM). Every
quantity carries a leading batch dimension B, so a whole swarm of candidate parameter sets is
simulated in one pass on the GPU.

    V = U_p(c_p,s) - U_n(c_n,s) - eta_n - eta_p - I R0 - R_c u_e + s_ocv (T - T_ref) (dU_p/dT - dU_n/dT)

    particles   dimensionless sphere, tau = t / theta_k(T), surface flux
                Phi_n = +I theta_n / (3 Q_n 3600),  Phi_p = -I theta_p / (3 Q_p 3600)
    kinetics    eta_k = (2RT/F) asinh(I / (2 I*_k(T) sqrt(c(1-c))/0.5))      symmetric Butler-Volmer
    electrolyte du_e/dt = (I - u_e) / tau_e(T)                             one salt-diffusion mode
    temperature theta_k(T) = theta_k exp(Ea_D,k/R (1/T - 1/T_ref)), I*_k(T) = I*_k exp(-Ea_r/R (...)),
                R0, R_c, tau_e scale with exp(Ea_e/R (...)); s_ocv scales the entropic term

Groups (per candidate): Q_n, Q_p, x100, y100 (capacity window), theta_n, theta_p, Istar_n, Istar_p,
R0, R_c, tau_e (dynamics at T_ref), Ea_Dn, Ea_Dp, Ea_r, Ea_e [J/mol], s_ocv (temperature).

Three electrolyte models (BatchEM(electrolyte=...)):

    'rc'    the single branch above - the model of the imported parameter set
    'spme'  the leading-order SPMe electrolyte on the KNOWN geometry of the parameter set :
            the modal state space of srcs_EM.Electrolyte (eigenmodes of the three-domain slab, built
            once), batched over the candidates. The shared parameters are physical:

                Ds_n, Ds_p   solid diffusivity [m2/s]          -> theta_k = R_k^2 / Ds_k
                j0_n, j0_p   exchange-current density at c_e0, half stoichiometry [A/m2]
                                                               -> I*_k = A a_k L_k j0_k
                R_contact    series resistance outside the electrolyte [Ohm]
                alpha_De     multiplier on the set's D_e: every eigenvalue scales with it
                beta_ce      multiplier on (1 - t+)^2 TDF: (1 - t+) -> sqrt(beta_ce) (1 - t+) in both
                             the salt source and the concentration overpotential, TDF fixed

            V = U_p - U_n - eta_n - eta_p - I (R_contact + R_e) + eta_c + entropic term
                eta_k  Butler-Volmer with j0 ~ sqrt(c_e,k / c_e0) (c_e,k the electrode's mean salt)
                R_e    leading-order electrolyte resistance from kappa(c_e0) on the geometry (fixed, D3)
                eta_c  (2RT/F)(1 - t+) TDF ln(c_e,p / c_e,n)
            Temperature: D_e and 1/kappa, R_contact follow Ea_e; Ds_k follow Ea_D,k; j0 follows Ea_r.
            With electrolyte_T='set' the electrolyte follows instead the set's own D_e(c_e0, T) and
            kappa(c_e0, T) (no fitted parameter), and R_contact its own activation energy Ea_s.
    'ohmic' the electrolyte as a series resistance only: no salt dynamics, no concentration
            overpotential, j0 at c_e0. The particles and kinetics are those of 'spme' (Ds_n, Ds_p,
            j0_n, j0_p on the geometry of pm), and
                R0           the whole series resistance: electrolyte, solid and contacts [Ohm]
            V = U_p - U_n - eta_n - eta_p - I R0 + entropic term; R0 follows Ea_e.

Entropic term (BatchEM(entropic=...)): 'set' (default) s_ocv (T - T_ref) (dU_p/dT - dU_n/dT), the set's
own coefficients scaled by s_ocv; 'cell' a cell coefficient of its own, linear in the electrodes'
surface stoichiometries (where the OCPs are read),
    (T - T_ref) (dUdT0 + dUdT_n (x_s - 0.5) + dUdT_p (y_s - 0.5))
dUdT0, dUdT_n, dUdT_p [V/K] are temperature parameters (signed; 0 unless given), s_ocv is not used.
'knots' (multi-slice model only): a free-form OCV change with temperature on the BULK anode stoichiometry
x_b = x100 - q / Q_n (coulomb count, the same for every slice), piecewise linear on the knots T_KNOTS,
    (T - T_ref) interp(x_b, T_KNOTS, (dUdTk0, ..., dUdTk7))
so that U(x_b, T) = U(x_b, T_ref) + that: a fitted OCV curve at every temperature, not an overpotential.

Time stepping. The current is cut into windows over which it is constant (a 5 % change of the
peak current, or a temperature change, opens a new window). Each candidate crosses a window in its
own equal sub-steps of at most TAU_TARGET in tau - its diffusion time sets how many - and the
voltage at every sample inside the window is read from the network's trunk at that sample's tau,
without feeding back into the state, exactly as in deployment. Once a candidate has spent TAU_RELAX
in a window, the transients of the linear particle problem are below exp(-lam_1^2 TAU_RELAX) =
exp(-20): it stops being stepped, its profile shape is frozen (flat in a rest) and only its level
falls, by exactly -3 Phi per unit tau (BatchEM.particle). The lithium-balance projection of the
surrogate is applied after every sub-step.

Sampling convention of the data: I_k is the mean current over [t_k, t_k + dt_k) and V_k is read at the
interval midpoint t_k + dt_k/2. dt is one step for every sample or one per sample (data.coarsen with
align=True); a window never spans two different steps.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from srcs import physics
from srcs.paths import MIONET

R_GAS, F_FARADAY, T_REF = physics.R_GAS, physics.F_FARADAY, physics.T_REF
TAU_TARGET = 0.05        # largest sub-step in tau (the window rule of the surrogate paper)
TAU_RELAX = 1.0          # a rest longer than this (in tau) is fully relaxed
JUMP = 0.05              # a current change above this fraction of the peak current opens a window
DT_WINDOW = 1.0          # [K] a temperature drift above this from the window start opens a window
DYNAMIC = ('theta_n', 'theta_p', 'Istar_n', 'Istar_p', 'R0', 'R_c', 'tau_e')          # electrolyte 'rc'
DYNAMIC_SPME = ('Ds_n', 'Ds_p', 'j0_n', 'j0_p', 'R_contact', 'alpha_De', 'beta_ce')  # electrolyte 'spme'
DYNAMIC_OHMIC = ('Ds_n', 'Ds_p', 'j0_n', 'j0_p', 'R0')                                 # electrolyte 'ohmic'
T_KNOTS = tuple(np.linspace(0.02, 0.84, 8))   # entropic='knots': anode bulk stoichiometries of dUdTk0..7
THERMAL = ('Ea_Dn', 'Ea_Dp', 'Ea_r', 'Ea_e', 'Ea_s', 's_ocv',
           'dUdT0', 'dUdT_n', 'dUdT_p') + tuple('dUdTk%d' % i for i in range(len(T_KNOTS)))
# Ea_s: R_contact with electrolyte_T='set'; dUdT0/_n/_p: entropic='cell'; dUdTk*: entropic='knots'
OPTIONAL = {'tort': 1.0}  # 'spme': tortuosity factor on the effective transport (D_e,eff / tort, R_e * tort)
WINDOW = ('Q_n', 'Q_p', 'x100', 'y100')
N_MODES = 4               # electrolyte modes kept (the steady split is captured to < 1 %, Electrolyte.mode_capture)


# mionet-clean renamed its runs (same networks, see its README):
# studies recorded with the old names still load
RENAMED = {**{'%s_s%d%s' % (o, s, x): 'mionet_300k_s%d%s' % (s, x) for o in ('wk25b', 'wk25c', 'trained_model')
              for s in (42, 43, 44, 45) for x in ('', '_lbfgs')},
           **{'%s_s%d%s' % (o, s, x): 'mionet_77k_s%d%s' % (s, x) for o in ('wk25r', 'wk25rcv')
              for s in (42, 43, 44, 45) for x in ('', '_lbfgs')}}


def load_surrogate(run: str, device: str):
    """The trained network of a mionet-clean run (runs/v2_nondim/<run>)."""
    if MIONET not in sys.path:
        sys.path.insert(0, MIONET)
    from srcs_EM import Surrogate
    run = RENAMED.get(run, run)
    return Surrogate.from_run(os.path.join(MIONET, 'runs', 'v2_nondim', run), device=device)


class Table:
    """A tabulated function of a stoichiometry in [0, 1], evaluated on the GPU by linear interpolation."""

    def __init__(self, f, device, n=8001):
        self.x = np.linspace(0.0, 1.0, n)
        self.y = torch.tensor(f(self.x), dtype=torch.float32, device=device)
        self.n = n

    def __call__(self, s):
        u = torch.clamp(s, 0.0, 1.0) * (self.n - 1)
        i = torch.clamp(u.floor().long(), 0, self.n - 2)
        w = u - i
        return self.y[i] * (1 - w) + self.y[i + 1] * w


def windows(I: np.ndarray, T: np.ndarray, jump: float = JUMP, dT: float = DT_WINDOW, dt=None):
    """[(start, stop)] sample ranges of constant current, within which the temperature stays within dT
    of its value at the window start. Only the particles see a window's temperature (its mean, through
    theta(T)); kinetics, R0 and the electrolyte branch use every sample's own temperature. dT = 0 opens
    a window at every change of T. dt (one per sample): a change of the step opens a window too, so the
    samples of a window are evenly spaced."""
    tol = jump * max(float(np.max(np.abs(I))), 1e-9)
    dt = np.zeros(len(I)) if dt is None else np.broadcast_to(np.asarray(dt, dtype=float), (len(I),))
    out, a = [], 0
    for k in range(1, len(I) + 1):
        if k == len(I) or abs(I[k] - I[a]) > tol or abs(T[k] - T[a]) > dT + 1e-9 or (dT == 0 and T[k] != T[a]) \
                or dt[k] != dt[a]:
            out.append((a, k))
            a = k
    return out


class BatchEM:
    """ocp: a srcs.ocp.OCPSet (the parameter set's curves, possibly corrected); default the plain curves of pm.
    dT_window [K]: the temperature drift that opens a new window (see windows()).
    electrolyte: 'rc' (one branch, lumped groups), 'spme' (modal SPMe electrolyte on the geometry of pm,
    physical parameters) or 'ohmic' (a series resistance, physical particle parameters); n_modes:
    electrolyte modes kept in 'spme'. electrolyte_T ('spme' only): 'Ea_e' (the electrolyte and R_contact
    share one fitted activation energy) or 'set' (the electrolyte follows the set's kappa(T), D_e(T);
    R_contact follows Ea_s). entropic: 'set' (s_ocv x the set's coefficients) or 'cell' (dUdT0, dUdT_n,
    dUdT_p; module docstring). electrolyte_set ('spme' only): a PyBaMM set whose bulk electrolyte
    (kappa, D_e, TDF, t+) replaces pm's on pm's geometry (physics.literature_electrolyte); the optional
    parameter 'tort' (default 1) then scales the effective transport: D_e,eff / tort and R_e * tort.
    self.dynamic: the shared dynamic parameters of the chosen model."""

    def __init__(self, run: str, pm: str = 'Mohtat2020', device: str | None = None, ocp=None,
                 dT_window: float = DT_WINDOW, electrolyte: str = 'rc', n_modes: int = N_MODES,
                 electrolyte_T: str = 'Ea_e', entropic: str = 'set', electrolyte_set: str | None = None):
        from srcs.ocp import OCPSet
        if electrolyte not in ('rc', 'spme', 'ohmic'):
            raise ValueError("electrolyte must be 'rc', 'spme' or 'ohmic', not %r" % electrolyte)
        if electrolyte_T not in ('Ea_e', 'set'):
            raise ValueError("electrolyte_T must be 'Ea_e' or 'set', not %r" % electrolyte_T)
        if entropic not in ('set', 'cell', 'knots'):
            raise ValueError("entropic must be 'set', 'cell' or 'knots', not %r" % entropic)
        self.entropic, self.electrolyte_set = entropic, electrolyte_set
        self.electrolyte, self.electrolyte_T = electrolyte, electrolyte_T
        self.dynamic = {'rc': DYNAMIC, 'spme': DYNAMIC_SPME, 'ohmic': DYNAMIC_OHMIC}[electrolyte]
        self.dT_window = float(dT_window)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if run is None:      # exact finite-volume particles (srcs/ms_fv.py): no network; ms_fv sets its own grid
            self.sur = self.net = None
            self.n, self.tau_max = 21, 0.2
            self.xi = torch.linspace(0.0, 1.0, self.n, dtype=torch.float32, device=self.device)
            self.vm = torch.full((self.n,), 1.0 / self.n, dtype=torch.float32, device=self.device)
        else:                # the MIONet surrogate particles (not used in the paper; needs mionet-clean)
            self.sur = load_surrogate(run, self.device)
            self.net = self.sur.net
            self.n = self.sur.n
            self.tau_max = self.sur.tau_max
            self.xi = torch.tensor(self.sur.sensors, dtype=torch.float32, device=self.device)
            self.vm = torch.tensor(self.sur.vm_weights, dtype=torch.float32, device=self.device)
        self.ocp = ocp if ocp is not None else OCPSet(pm)
        self.U_n, self.U_p = Table(self.ocp.U_n, self.device), Table(self.ocp.U_p, self.device)
        self.dU_n, self.dU_p = Table(self.ocp.dU_n, self.device), Table(self.ocp.dU_p, self.device)
        if electrolyte == 'spme':
            self._build_spme(pm, int(n_modes))
        elif electrolyte == 'ohmic':
            self.geo = physics.geometry(pm)

    def _build_spme(self, pm: str, n_modes: int):
        """The electrolyte's modal state space on the known geometry, from srcs_EM.Electrolyte: eigenvalues
        lam_hat (the eigenvalue is D_e lam_hat, < 0), the salt source per ampere g (proportional to
        1 - t+), and the rows that average the modes over each electrode (a_n, a_p). Built once; a
        candidate's D_e only rescales the eigenvalues, its (1 - t+) the source."""
        from srcs.electrolyte_modes import Electrolyte     # the modal electrolyte (from mionet-clean srcs_EM)
        e = Electrolyte.from_pybamm(pm, n_modes=n_modes, tdf_mode='ref')
        if self.electrolyte_set:
            physics.literature_electrolyte(e, self.electrolyte_set)
        f = lambda v: torch.tensor(np.asarray(v, dtype=float), dtype=torch.float32, device=self.device)
        self.ely = e
        self.geo = physics.geometry(pm)
        self.e_lam_hat, self.e_g = f(e._lam_hat), f(e._g)
        self.e_an, self.e_ap = f(e._w[0] @ e._phi), f(e._w[2] @ e._phi)
        self.e_c0, self.e_tplus, self.e_tdf = float(e.c_e0), float(e.t_plus), float(e.tdf())
        self.e_De, self.e_Re = float(e.De()), float(e.r_ohmic())          # at c_e0, T_REF

    def electrolyte_factors(self, T):
        """The set's own temperature law at c_e0: (D_e(T) / D_e(T_REF), R_e(T) / R_e(T_REF)) for T [K]
        (numpy), from srcs_EM.Electrolyte's tabulated kappa and D_e."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if not hasattr(self, '_e_table'):
            grid = np.arange(228.15, 348.15 + 0.25, 0.5)
            self._e_table = (grid, np.log([self.ely.De(t) / self.e_De for t in grid]),
                             np.log([self.ely.r_ohmic(t) / self.e_Re for t in grid]))
        grid, ld, lr = self._e_table
        return np.exp(np.interp(T, grid, ld)), np.exp(np.interp(T, grid, lr))

    def tdf_factor(self, T):
        """The set's thermodynamic factor at c_e0 and T over its value at T_REF, for T [K] (numpy). Used by
        the multi-slice model when tdf_T is on (the concentration overpotential then follows the cell
        temperature through chi, as kappa and D_e do); off, chi stays at T_REF."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if not hasattr(self, '_tdf_table'):
            grid = np.arange(228.15, 348.15 + 0.25, 0.5)
            self._tdf_table = (grid, np.log([self.ely.tdf(t) / self.e_tdf for t in grid]))
        grid, lt = self._tdf_table
        return np.exp(np.interp(T, grid, lt))

    # ------------------------------------------------------------------ network
    def _branch(self, y, phi):
        """Branch product for centre-pinned profiles y (B, n) and fluxes phi (B,) -> (B, h)."""
        return self.net._product(y, phi[:, None])

    def _trunk(self, tau, xi):
        """Trunk features at the dimensionless times tau and radii xi (same shape) -> (..., h)."""
        return self.net.trunk(self.net._trunk_in((tau / self.tau_max)[..., None], xi[..., None], tau.dim()))

    def _out(self, prod, tr, phi, xi2_half):
        """Network output from branch products (..., h), trunk features (..., h), flux phi and xi^2/2."""
        out = (prod * tr).sum(-1)
        if self.net.output_bias:
            out = out + self.net.b0
        if self.net.hard_bc:
            out = out - phi * xi2_half
        return out

    # ------------------------------------------------------------------ particles
    def particle(self, c0, theta, sign, Q, I, dt, wins, Tw=None, report_chunk: int = 1 << 18):
        """Surface stoichiometry at the sample midpoints, (B, K), for a batch of particles.

        Every row is an independent particle: c0 (B,) initial uniform stoichiometry; theta (B, n_windows)
        the diffusion time in each window; sign +1 anode, -1 cathode, a float or one per row; Q (B,)
        capacity [A.h]. simulate() stacks the negative and positive particles of every candidate into
        ONE batch, so both electrodes advance in the same sequential loop.

        Long windows. The particle problem is linear diffusion with a constant surface flux over a
        window, whose solution is <x0> - 3 Phi tau + (a zero-mean quasi-steady shape) + transients that
        decay as exp(-lam_n^2 tau), lam_1 = 4.493 (mionet-clean srcs_EM/exact.py). After tau = TAU_RELAX
        the transients are below exp(-20); from there the shape is frozen and only its level falls, by
        exactly -3 Phi per unit tau. So each candidate is stepped by the network up to TAU_RELAX at most
        and translated analytically after it - in a rest (Phi = 0) the frozen shape is flat, the profile
        is set to its mean. This bounds the sub-steps of any window to TAU_RELAX / TAU_TARGET = 20,
        whatever the diffusion time; without it a fast candidate over a long constant-current window
        (e.g. theta = 100 s over a 9 h C/10 discharge) needs thousands of sequential sub-steps.

        Cost. The sequential loop over sub-steps is what a call costs on the GPU, so it does as little
        per sub-step as possible and never waits for the device:
          * within a window a row's sub-step length is constant, so the trunk at the sensor radii is
            evaluated once per window;
          * the rows are sorted once per window by their number of sub-steps (descending), so the rows
            still stepping at sub-step s are the first n_act[s] - a slice with a host-side length, no
            per-step nonzero / searchsorted and no host-device synchronisation inside the loop;
          * a sub-step only evaluates the branch and stores its products; the voltage samples are read
            after the loop, all at once (in chunks of report_chunk), each from the products of the
            sub-step that contains it (s_j = floor(tau_j / tau_s)) and the trunk at its own tau."""
        B, K, dev = c0.shape[0], len(I), self.device
        dt = np.broadcast_to(np.asarray(dt, dtype=float), (K,))               # one step per sample
        sign = torch.as_tensor(sign, dtype=torch.float32, device=dev).expand(B)
        y = c0[:, None].repeat(1, self.n)                                     # state on the sensors
        surf = torch.empty(B, K, device=dev)
        xi = self.xi
        xi2h = 0.5 * xi ** 2
        for w, (a, b) in enumerate(wins):
            n = b - a
            Iw = float(np.mean(I[a:b]))
            th = theta[:, w]
            phi = sign * Iw * th / (3.0 * Q * 3600.0)
            dtau = n * float(dt[a]) / th                                      # (B,)
            rest = abs(Iw) < 1e-9
            dtau_eff = torch.clamp(dtau, max=TAU_RELAX)
            nsub = torch.clamp(torch.ceil(dtau_eff / TAU_TARGET), min=1)      # (B,)
            tau_s = dtau_eff / nsub
            # rows by decreasing number of sub-steps: at sub-step s the rows still stepping are a prefix
            nsub_o, order = torch.sort(nsub, descending=True, stable=True)
            ns = nsub_o.long().cpu().numpy()                                  # one host sync per window
            n_steps = int(ns[0])
            n_act = (ns[None, :] > np.arange(n_steps)[:, None]).sum(1).tolist()
            inv = torch.empty_like(order)
            inv[order] = torch.arange(B, device=dev)
            yo, phio, tau_so = y[order], phi[order], tau_s[order]
            tr_state = self._trunk(tau_so[:, None].expand(B, self.n), xi.expand(B, self.n))  # (B, n, h)
            bc_state = phio[:, None] * xi2h[None, :]
            prods, cpins, shifts = None, None, None
            for s in range(n_steps):
                A = n_act[s]
                ya = yo[:A]
                c_pin = ya[:, :1]
                prod = self._branch(ya - c_pin, phio[:A])                    # (A, h)
                z = self._out(prod[:, None, :], tr_state[:A], 1.0, bc_state[:A]) + c_pin
                shift = -3.0 * phio[:A] * tau_so[:A] - ((z - ya) @ self.vm)  # lithium balance
                z = z + shift[:, None]
                if prods is None:                                             # (n_steps, B, h), rows sorted
                    prods = torch.zeros(n_steps, B, prod.shape[1], device=dev)
                    cpins = torch.zeros(n_steps, B, device=dev)
                    shifts = torch.zeros(n_steps, B, device=dev)
                prods[s, :A] = prod
                cpins[s, :A] = c_pin[:, 0]
                shifts[s, :A] = shift
                yo[:A] = z                                                    # yo is a fresh gather, safe in place
            y = yo[inv]
            # every voltage sample of the window, from the sub-step that contains it
            f = (torch.arange(n, device=dev, dtype=torch.float32) + 0.5) / n
            tau_j = f[None, :] * dtau[:, None]                                # (B, n)
            covered = tau_j <= dtau_eff[:, None] + 1e-9                       # a relaxed window stops at TAU_RELAX
            s_j = torch.minimum(torch.floor(tau_j / tau_s[:, None]), nsub[:, None] - 1).long()
            pb, pj = covered.nonzero(as_tuple=True)
            for c in range(0, pb.numel(), report_chunk):
                rb, rj = pb[c:c + report_chunk], pj[c:c + report_chunk]
                rs, ro = s_j[rb, rj], inv[rb]
                tl = torch.clamp(tau_j[rb, rj] - rs * tau_s[rb], min=0.0)
                trs = self._trunk(tl, torch.ones_like(tl))
                zs = self._out(prods[rs, ro], trs, phi[rb], 0.5) + cpins[rs, ro]
                surf[rb, a + rj] = zs + shifts[rs, ro] * (tl / tau_s[rb])
            beyond = dtau > TAU_RELAX                                         # the asymptote, see the docstring
            if beyond.any():
                ub, uj = (~covered).nonzero(as_tuple=True)
                if rest:
                    mean = y @ self.vm
                    y = torch.where(beyond[:, None], mean[:, None].expand(B, self.n), y)
                    surf[ub, a + uj] = mean[ub]
                else:
                    surf[ub, a + uj] = y[ub, -1] - 3.0 * phi[ub] * (tau_j[ub, uj] - dtau_eff[ub])
                    drop = torch.where(beyond, -3.0 * phi * (dtau - dtau_eff), torch.zeros_like(phi))
                    y = y + drop[:, None]
        return surf

    # ------------------------------------------------------------------ electrolyte (spme)
    def spme_electrolyte(self, alpha, sqrt_beta, de_T, I, dt, wins):
        """Mean salt concentration of each electrode at the sample midpoints, (B, K) each [mol/m3].

        Modal state q (B, M), driven by the current, exact at constant current over a window:
            dq/dt = lam q + g_eff I,   lam = alpha D_e(T_w) lam_hat,   g_eff = sqrt(beta) g
            q(t) = q_inf + (q_0 - q_inf) exp(lam t),   q_inf = -g_eff I / lam
        de_T (B, n_windows): D_e(T_w) / D_e(T_REF) at each window's mean temperature, as the particles
        (exp(-Ea_e (1/T - 1/T_ref)/R), or the set's own law).
        c_e,n = c_e0 + a_n . q, c_e,p = c_e0 + a_p . q (the slab averages of the modes). Accumulated mode
        by mode, so that nothing larger than (B, window length) is formed."""
        B, K, dev = alpha.shape[0], len(I), self.device
        dt = np.broadcast_to(np.asarray(dt, dtype=float), (K,))
        M = self.e_lam_hat.numel()
        q = torch.zeros(B, M, device=dev)
        cn = torch.full((B, K), self.e_c0, device=dev)
        cp = torch.full((B, K), self.e_c0, device=dev)
        g_eff = sqrt_beta[:, None] * self.e_g[None, :]                                    # (B, M)
        for w, (a, b) in enumerate(wins):
            Iw = float(np.mean(I[a:b]))
            lam = (alpha * self.e_De * de_T[:, w])[:, None] * self.e_lam_hat[None, :]   # < 0
            q_inf = -g_eff * Iw / lam
            tj = (torch.arange(b - a, device=dev, dtype=torch.float32) + 0.5) * float(dt[a])  # to each midpoint
            for m in range(M):
                qm = q_inf[:, m:m + 1] + (q[:, m:m + 1] - q_inf[:, m:m + 1]) * torch.exp(lam[:, m:m + 1] * tj[None, :])
                cn[:, a:b] += self.e_an[m] * qm
                cp[:, a:b] += self.e_ap[m] * qm
            q = q_inf + (q - q_inf) * torch.exp(lam * ((b - a) * float(dt[a])))
        return cn, cp

    # ------------------------------------------------------------------ cell
    @torch.no_grad()
    def simulate(self, g: dict, I, dt: float, T=None, T_ref: float = T_REF, q_start: float = 0.0,
                 electrolyte: bool = True, parts: bool = False):
        """Terminal voltage (B, K) at the sample midpoints for current I [A] (> 0 discharge).

        g: the window groups, the model's dynamic parameters (self.dynamic) and the temperature
        parameters, each a float or a (B,) array/tensor. T: the cell temperature [K], a float or one
        value per sample (default T_ref, isothermal). q_start: charge [A.h] below the top of the window
        (a float or one per candidate)
        at t = 0, from rest. electrolyte=False drops the electrolyte's dynamics (RC: no branch; SPMe: salt
        frozen at c_e0, its ohmic drop kept; ohmic: no effect). parts=True returns a dict of the voltage's
        terms instead."""
        dev = self.device
        spme = self.electrolyte == 'spme'
        I = np.asarray(I, dtype=float)
        K = len(I)
        Tn = np.full(K, T_ref) if T is None else np.broadcast_to(np.asarray(T, dtype=float), (K,)).copy()
        dt = np.broadcast_to(np.asarray(dt, dtype=float), (K,)).copy()       # one step per sample
        g = {k: v for k, v in g.items() if k in WINDOW + self.dynamic + THERMAL + tuple(OPTIONAL)}
        missing = [k for k in WINDOW + self.dynamic if k not in g]
        if missing:
            raise KeyError('simulate (%s electrolyte): missing %s' % (self.electrolyte, missing))
        B = max(int(np.size(v)) for v in g.values())
        G = {k: torch.as_tensor(np.broadcast_to(np.asarray(v, dtype=np.float64), (B,)).copy(),
                                dtype=torch.float32, device=dev) for k, v in g.items()}
        for k in THERMAL:
            G.setdefault(k, torch.zeros(B, device=dev) if k != 's_ocv' else torch.ones(B, device=dev))
        for k, v in OPTIONAL.items():
            G.setdefault(k, torch.full((B,), v, device=dev))
        if self.electrolyte != 'rc':                                          # physical -> particle groups
            geo = self.geo
            for k in ('n', 'p'):
                G['theta_%s' % k] = geo['R_%s' % k] ** 2 / G['Ds_%s' % k]
                G['Istar_%s' % k] = geo['A'] * geo['a_%s' % k] * geo['L_%s' % k] * G['j0_%s' % k]
        wins = windows(I, Tn, dT=self.dT_window, dt=dt)
        Tw = torch.tensor([Tn[a:b].mean() for a, b in wins], dtype=torch.float32, device=dev)   # window mean
        arr = (1.0 / Tw - 1.0 / T_ref) / R_GAS                                # (n_windows,)
        th_n = G['theta_n'][:, None] * torch.exp(G['Ea_Dn'][:, None] * arr[None, :])
        th_p = G['theta_p'][:, None] * torch.exp(G['Ea_Dp'][:, None] * arr[None, :])
        qs = torch.as_tensor(np.broadcast_to(np.asarray(q_start, dtype=np.float64), (B,)).copy(),
                             dtype=torch.float32, device=dev)                 # a float or one per candidate
        x0 = G['x100'] - qs / G['Q_n']
        y0 = G['y100'] + qs / G['Q_p']
        # both electrodes of every candidate in one batch of 2B independent particles: one sequential loop
        sign = torch.cat([torch.ones(B, device=dev), -torch.ones(B, device=dev)])
        surf = self.particle(torch.cat([x0, y0]), torch.cat([th_n, th_p]), sign, torch.cat([G['Q_n'], G['Q_p']]),
                             I, dt, wins, Tw)
        xs, ys = surf[:B], surf[B:]
        Tt = torch.tensor(Tn, dtype=torch.float32, device=dev)[None, :]
        ark = (1.0 / Tt - 1.0 / T_ref) / R_GAS                                # (1, K)
        It = torch.tensor(I, dtype=torch.float32, device=dev)[None, :]
        xc, yc = torch.clamp(xs, 1e-6, 1 - 1e-6), torch.clamp(ys, 1e-6, 1 - 1e-6)
        kin = torch.exp(-G['Ea_r'][:, None] * ark)
        i_n = G['Istar_n'][:, None] * kin * torch.sqrt(xc * (1 - xc)) / 0.5
        i_p = G['Istar_p'][:, None] * kin * torch.sqrt(yc * (1 - yc)) / 0.5
        kf = 2.0 * R_GAS * Tt / F_FARADAY
        ohm = torch.exp(G['Ea_e'][:, None] * ark)
        if self.entropic == 'knots':
            raise NotImplementedError("entropic='knots' is implemented in the multi-slice model only")
        if self.entropic == 'cell':                                         # linear in x_s, y_s
            entropic = (Tt - physics.T_REF) * (G['dUdT0'][:, None] + G['dUdT_n'][:, None] * (xs - 0.5)
                                               + G['dUdT_p'][:, None] * (ys - 0.5))
        else:
            entropic = G['s_ocv'][:, None] * (Tt - physics.T_REF) * (self.dU_p(ys) - self.dU_n(xs))
        out = {'U': self.U_p(ys) - self.U_n(xs), 'entropic': entropic}
        if self.electrolyte == 'rc':
            out['eta_kin'] = -kf * torch.asinh(It / (2 * i_n)) - kf * torch.asinh(It / (2 * i_p))
            out['ohmic'] = -It * G['R0'][:, None] * ohm
            if electrolyte:                                                   # closed form in each window
                u = torch.zeros(B, device=dev)
                um = torch.empty(B, K, device=dev)
                e_w = torch.exp(G['Ea_e'][:, None] * arr[None, :])            # (B, n_windows)
                for w, (a, b) in enumerate(wins):
                    Iw = float(np.mean(I[a:b]))
                    tau = G['tau_e'] * e_w[:, w]
                    j = torch.arange(b - a, device=dev, dtype=torch.float32)[None, :]
                    decay = torch.exp(-(j + 0.5) * float(dt[a]) / tau[:, None])  # to each midpoint
                    um[:, a:b] = Iw + (u[:, None] - Iw) * decay
                    u = Iw + (u - Iw) * torch.exp(-(b - a) * float(dt[a]) / tau)
                out['electrolyte'] = -G['R_c'][:, None] * ohm * um
        elif self.electrolyte == 'ohmic':
            out['eta_kin'] = -kf * torch.asinh(It / (2 * i_n)) - kf * torch.asinh(It / (2 * i_p))
            out['ohmic'] = -It * G['R0'][:, None] * ohm
        else:
            sb = torch.sqrt(G['beta_ce'])
            if electrolyte:
                if self.electrolyte_T == 'set':
                    de_T = torch.tensor(self.electrolyte_factors(Tw.cpu().numpy())[0], dtype=torch.float32,
                                        device=dev)[None, :].expand(B, -1)
                else:
                    de_T = torch.exp(-G['Ea_e'][:, None] * arr[None, :])
                cn, cp = self.spme_electrolyte(G['alpha_De'] / G['tort'], sb, de_T, I, dt, wins)
                cn, cp = torch.clamp(cn, min=1.0), torch.clamp(cp, min=1.0)
                out['electrolyte'] = (kf * (1.0 - self.e_tplus) * sb[:, None] * self.e_tdf
                                      * torch.log(cp / cn))                   # concentration overpotential
                out['c_e_n'], out['c_e_p'] = cn, cp
                i_n = i_n * torch.sqrt(cn / self.e_c0)                        # j0 ~ sqrt(c_e)
                i_p = i_p * torch.sqrt(cp / self.e_c0)
            out['eta_kin'] = -kf * torch.asinh(It / (2 * i_n)) - kf * torch.asinh(It / (2 * i_p))
            if self.electrolyte_T == 'set':                                   # each on its own law
                r_T = torch.tensor(self.electrolyte_factors(Tn)[1], dtype=torch.float32, device=dev)[None, :]
                out['ohmic'] = -It * (G['R_contact'][:, None] * torch.exp(G['Ea_s'][:, None] * ark)
                                      + self.e_Re * G['tort'][:, None] * r_T)
            else:
                out['ohmic'] = -It * (G['R_contact'][:, None] + self.e_Re * G['tort'][:, None]) * ohm
        V = out['U'] + out['entropic'] + out['eta_kin'] + out['ohmic']
        if 'electrolyte' in out:
            V = V + out['electrolyte']
        if parts:
            out['V'] = V
            out['x_surf'], out['y_surf'] = xs, ys
            return out
        return V
