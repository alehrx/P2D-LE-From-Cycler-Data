"""
ms_model.py
===========
The multi-slice model of the identification, batched over the candidates on the GPU: the
multi-slice model of mionet-clean (srcs_EM/multislice.py, the model of the plant) with the same
physical parameters as em_model's 'spme', so that the identification pipeline runs unchanged.

Per electrode, N slices through the thickness, each a particle advanced by the parameter-free MIONet,
share one solid potential and couple through the modal electrolyte:

    j_i = 2 j0_i sinh(F eta_i / 2RT),   eta_i = phi_s - phi_e,i - U(x_s,i),   mean_i j_i = +-I / (a L A)
    phi_e,i = R_T (P j)_i + (2RT/F)(1 - t+) sqrt(beta_ce) TDF ln(c_e,i / c_e0)
    dq/dt = alpha_De D_e(T) lam_hat q + sqrt(beta_ce) G j                (modal salt, exact per step)
    V = phi_s,p - phi_s,n - I R_contact + entropic term

    j0_i = j0_k(T) sqrt(x_s(1 - x_s)) / 0.5 sqrt(c_e,i / c_e0)          (as em_model)
    P    the ohmic electrolyte response at c_e0 (phi_e per unit slice current, srcs_EM MultiSlice._transport),
         scaled by R_T (the set's kappa(T) law, or Ea_e) and by 'tort'
    G    the salt source of each slice on the modes (MultiSlice._build_maps)

It differs from the plant (srcs/msplant.py) by the MIONet particles (exact there), the electrolyte
linearised at c_e0 (D_e, kappa, TDF constant; local D_e(c), kappa(c), TDF(c) there), the number of
slices and modes, and the time stepping described below. N = 1 is the SPMe of em_model.

Time stepping. Closed loop: the slices' currents depend on the state, so the model steps sample by
sample (not window by window as em_model). At each sample k (current I_k over [t_k, t_k + dt_k)):
  1. predictor: the particles to the end of the step t_k + dt_k at a guess of the split (the previous
     one, shifted to the new mean current);
  2. the current split AT THE END OF THE STEP, implicit in the particles and in the salt (backward
     Euler): a slice's surface responds to its own current over the step, x_s(j) = x_s,guess - (Phi(j)
     - Phi(guess)) h(dtau), with h the exact surface response of a sphere to a unit flux (linear
     diffusion, any initial state); the salt at the end of the step is linear in the currents (the
     modes' exact response), c_e(j) = c0 + C (q E + (1 - E) q_inf(j)). Newton on (j, phi_s) of both
     electrodes at once (they share the salt), batched over the candidates (NEWTON iterations,
     NEWTON_JUMP after a current change). Explicit in the particles, the split oscillates where the OCV
     is steep (the end of a discharge) as soon as dt exceeds the redistribution time; balanced at the
     midpoint (split='mid', implicit midpoint) it still rings, sample to sample, wherever the
     redistribution is much faster than dt (fast kinetics: the 45 C cells), since the midpoint rule
     damps a stiff mode by a factor -> -1. Explicit in the salt (c_e at the guessed split) it diverges
     at long steps where the kinetics are fast and the concentration overpotential strong (beta_ce 2.3
     with the 45 C j0 at dt = 10 s: jumps of 100s of mV);
  3. the particles advance by dt_k at the split's currents, in nsub equal sub-steps of at most TAU_SUB
     (odd, so that the midpoint falls in the middle of one), the electrolyte exactly;
  4. the voltage at the midpoint (the data convention): surfaces from 3., salt at the midpoint, the
     split's currents; phi_s of each electrode the mean over its slices of U + phi_e + eta.
The trunk of the network is evaluated once per run of constant dt and temperature.

Cost. A sample is a few hundred small kernels whatever the number of candidates, so a call is bound by
the kernel launches, not by the batch. On CUDA the step of a sample is therefore captured as a CUDA
graph (one per number of sub-steps and per current change or not) on a workspace of fixed buffers per
batch size: the parameters, the state, the per-run quantities and the per-sample inputs, read at a
sample counter that the graph itself advances. graphs=False steps the same code eagerly.
"""
from __future__ import annotations

import numpy as np
import torch

from srcs import physics
from srcs.em_model import BatchEM, OPTIONAL, THERMAL, T_KNOTS, WINDOW, F_FARADAY, R_GAS, T_REF, Table, windows

# optional groups of the multi-slice model, with the value that leaves the model unchanged when a config does not
# name them. De_f (electrolyte_groups='material'): a factor on the literature D_e(c_e0, T_REF), which sets tau_e
# (L_e^2 / D_e) and the salt amplitude R_ce (1 / D_e); its temperature law stays the set's
MS_OPTIONAL = {'De_f': 1.0,
               # dEa_rp: the cathode's reaction activation energy minus the anode's (Ea_r is the anode's then) [J/mol]
               'dEa_rp': 0.0,
               # dEa_e: an activation energy [J/mol] on top of the electrolyte set's resistance law,
               # R_e(T) = R_e,set(T) exp(dEa_e (1/T - 1/T_ref)/R) (electrolyte_T='set' only); > 0 = more ohmic drop when cold
               'dEa_e': 0.0,
               # dEa_De: the same on the salt diffusivity, D_e(T) = D_e,set(T) exp(-dEa_De (1/T - 1/T_ref)/R)
               # (electrolyte_T='set' only); > 0 = slower salt diffusion, hence more concentration polarisation, when cold
               'dEa_De': 0.0}

N_SLICES = 5             # slices per electrode
N_MODES_MS = 8           # electrolyte modes (the slices resolve the profile, so keep more than em_model's 4)
TAU_SUB = 0.2            # largest particle sub-step in tau (the network's training horizon)
DTAU_MIN = 0.002         # smallest step the network is rolled out with (its roll-out training range, 0.002-0.03)
N_TRUNK = 1200           # log-spaced times of the trunk table (superstep), 1e-7 .. TAU_SUB
NEWTON = 4               # Newton iterations of the split at a sample of unchanged current
NEWTON_JUMP = 8          # ... after a change of the current (from the uniform split)
# where the split restarts after a change of the current. 'uniform' (the original): from the uniform
# split. 'shift': from the previous split moved to the new mean current, as at an unchanged current. In a CV phase
# the current falls by about the jump tolerance at every sample, so 'uniform' restarted some samples and not others
# and neighbouring candidates ended on different branches (voltage islands of 1-4 mV; scripts/check_mode_smoothness.py).
JUMP_START = 'uniform'
J0_SLOPES = True         # the slopes of j0 in the Jacobian (False: the earlier, inexact Newton)
NEWTON_DAMP = True       # step limit of the split (x_s inside (0, 1), f eta by <= 2 per iteration)
X_EDGE = 1e-3            # the split starts with every surface stoichiometry at least this far from 0 and 1
J0_STOICH = True         # j0 ~ sqrt(x_s (1 - x_s)) / 0.5 (False: j0 independent of the surface stoichiometry; a test)
D_LAW = {'Dp_drop': 0.0, 'Dp_shape': 1.0,    # cathode D_s(y) (srcs/ms_fv.py, finite-volume particles only):
         'Dn_slope': 0.0, 'Dp_wsurf': 0.0}   # D_p = D_p(y100) 10^(-Dp_drop u^Dp_shape), u = (y - y100) / (y_end - y100);
                         # anode D_n = D_n(x = 0.5) 10^(Dn_slope (x - 0.5)), y and x the mean
                         # stoichiometries at the start of the step (the cathode's y moved towards its surface
                         # value by Dp_wsurf in [0, 1]); the defaults: constant


def surface_response(tau):
    """h(tau): the fall of the surface stoichiometry of a sphere under a unit dimensionless flux
    (dc/dr = -1 at r = 1, the mean falls by 3 per unit tau), from rest:
        h = 3 tau + 1/5 - 2 sum_n exp(-lam_n^2 tau) / lam_n^2,   tan lam_n = lam_n
    (2 sqrt(tau / pi) at short times). Linear diffusion: the response to a flux change from any state."""
    from scipy.optimize import brentq
    lam = np.array([brentq(lambda x: np.tan(x) - x, (n + 0.5) * np.pi - 0.5, (n + 0.5) * np.pi - 1e-9)
                    for n in range(1, 400)] + [(n + 0.5) * np.pi - 1.0 / ((n + 0.5) * np.pi) for n in range(400, 20000)])
    tau = np.asarray(tau, dtype=float)
    out = np.empty_like(tau)
    short = tau < 1e-6
    out[short] = 2.0 * np.sqrt(tau[short] / np.pi)
    t = tau[~short]
    s = (np.exp(-np.outer(t, lam ** 2)) / lam ** 2).sum(1)
    out[~short] = 3.0 * t + 0.2 - 2.0 * s
    return out


class BatchMS(BatchEM):
    """BatchEM(electrolyte='spme') with N slices per electrode (module docstring). Same parameters
    (DYNAMIC_SPME, THERMAL, 'tort'), same electrolyte_T, entropic and electrolyte_set options.

    geometry='groups': no geometry from pm. The parameters are the groups the voltage sees (GROUPS):
    theta_k = R_k^2 / Ds_k [s], I0_k = A a_k L_k j0_k [A], R_contact, and the electrolyte as three
    groups at 25 C, R_e (ohmic resistance of a uniform split) [Ohm], tau_e (slowest time constant) [s]
    and R_ce (steady concentration polarisation of a uniform split, linearised) [Ohm], plus its shape
    through the thickness: l_s (separator share of the thickness), r_np (L_n / (L_n + L_p)), eps_n,
    eps_s, eps_p (porosities) and brugg (Bruggeman exponent). The maps (modes, slice sources, ohmic
    response) are then built per candidate (_group_maps); pm's radii, areas and thicknesses only carry
    the groups to the internal Ds, j0 and slice current densities, and cancel. Kept from the electrolyte
    set: c_e0, the gain (1 - t+) TDF of the concentration overpotential (R_ce sets the salt amplitude),
    and the temperature laws (electrolyte_T).

    electrolyte_groups='material' (geometry='groups'): the electrolyte's material properties at c_e0 and
    25 C (kappa, D_e, t+, TDF of electrolyte_set) are taken as known, and tau_e and R_ce follow from R_e and
    the total thickness L_e [m] (GROUPS_MATERIAL): on the shape's dimensionless grid x / L_e,
    R_e = L_e / (kappa A) r_hat (so A = L_e r_hat / (kappa R_e)), tau_e = L_e^2 / (D_e |lam_hat_1|) and
    R_ce = gain (1 - t+) L_e / (F A D_e) c_hat = gain (1 - t+) kappa R_e c_hat / (F D_e r_hat), with
    r_hat, lam_hat, c_hat the shape's ohmic response, slowest eigenvalue and steady salt difference.
    R_ce / R_e is then a property of the material and the shape alone."""

    GROUPS = ('theta_n', 'theta_p', 'I0_n', 'I0_p', 'R_contact', 'R_e', 'tau_e', 'R_ce',
              'l_s', 'r_np', 'eps_n', 'eps_s', 'eps_p', 'brugg')
    GROUPS_MATERIAL = ('theta_n', 'theta_p', 'I0_n', 'I0_p', 'R_contact', 'R_e', 'L_e',
                       'l_s', 'r_np', 'eps_n', 'eps_s', 'eps_p', 'brugg')
    FV_GROUPS = (10, 12)         # geometry='groups': finite volumes per slice of an electrode, and in the separator

    def __init__(self, run: str, pm: str = 'Mohtat2020', device: str | None = None, ocp=None,
                 n_slices: int = N_SLICES, n_modes: int = N_MODES_MS, electrolyte_T: str = 'Ea_e',
                 entropic: str = 'set', electrolyte_set: str | None = None, graphs: bool = True,
                 split: str = 'end', geometry: str = 'set', electrolyte_groups: str = 'free', superstep: bool = True,
                 slice_ratio: float = 1.0, **kw):
        super().__init__(run, pm, device=device, ocp=ocp, electrolyte='spme', n_modes=n_modes,
                         electrolyte_T=electrolyte_T, entropic=entropic, electrolyte_set=electrolyte_set, **kw)
        self.electrolyte = 'multislice'
        if split not in ('end', 'mid'):
            raise ValueError('multislice: split %r (end | mid)' % split)
        if geometry not in ('set', 'groups'):
            raise ValueError('multislice: geometry %r (set | groups)' % geometry)
        if electrolyte_groups not in ('free', 'material'):
            raise ValueError('multislice: electrolyte_groups %r (free | material)' % electrolyte_groups)
        if electrolyte_groups == 'material' and geometry != 'groups':
            raise ValueError("multislice: electrolyte_groups='material' needs geometry='groups'")
        self.split, self.geometry, self.e_groups = split, geometry, electrolyte_groups
        if geometry == 'groups':
            self.dynamic = self.GROUPS_MATERIAL if electrolyte_groups == 'material' else self.GROUPS
        self.n_slices = int(n_slices)
        # slice widths: slice_ratio = widest / narrowest slice of an electrode, geometric, narrowest at the
        # separator (where the reaction concentrates); 1 = uniform (the original code paths, unchanged).
        # Non-uniform: every average over an electrode's slices is weighted by the slice widths (_avg).
        self.slice_ratio = float(slice_ratio)
        self._uniform = self.slice_ratio == 1.0 or self.n_slices == 1
        if not self._uniform and geometry != 'groups':
            raise ValueError("multislice: slice_ratio != 1 needs geometry='groups'")
        Ns = self.n_slices
        wn = self.slice_ratio ** (-np.arange(Ns) / max(Ns - 1, 1))       # anode: collector -> separator
        wn = wn / wn.sum()
        self._sw64 = {'n': wn, 'p': wn[::-1].copy()}                      # cathode: separator -> collector
        self._sw = {k: torch.tensor(v, dtype=torch.float32, device=self.device) for k, v in self._sw64.items()}
        self.graphs = bool(graphs) and self.device == 'cuda'
        if self.graphs and 2 * Ns + 2 > 16:
            # the Newton system has 2 N + 2 rows: past 16 torch's default batched solve goes to MAGMA, which a
            # CUDA graph cannot capture (seen with 10 slices and a batch of 64+); cuSOLVER's can (all sizes, batches
            # to 16k tested). Process-wide, so set only when needed: 5-slice runs keep the cuBLAS path unchanged.
            torch.backends.cuda.preferred_linalg_library('cusolver')
        self._t_knots = torch.tensor(T_KNOTS, dtype=torch.float32, device=self.device)   # entropic='knots'
        self._t_h = float(T_KNOTS[1] - T_KNOTS[0])
        self._ws = {}                                                       # workspaces by batch size
        self._build_slices()
        self.xi2h = 0.5 * self.xi ** 2
        slope = lambda U: (lambda s: np.gradient(U(s), s))
        self.dUdx_n, self.dUdx_p = Table(slope(self.ocp.U_n), self.device), Table(slope(self.ocp.U_p), self.device)
        lt = np.linspace(-9.0, 2.0, 700)                        # h(tau) on a log grid of tau
        self._h_lt, self._h = lt, np.log(surface_response(10.0 ** lt))
        self.superstep = bool(superstep)
        self.T_per_window = False          # simulate: the particles' temperature per run or per window
        if self.superstep and split != 'end':
            raise ValueError("multislice: superstep needs split='end'")
        self._ss_tab = None                                     # trunk and h tables (_ss_tables), built at first use

    def _avg(self, v, k: str, keepdim: bool = False):
        """Average of v (B, N) over the slices of electrode k ('n' or 'p'), weighted by the slice widths."""
        if self._uniform:
            return v.mean(1, keepdim=keepdim)
        a = v @ self._sw[k]
        return a[:, None] if keepdim else a

    def h(self, tau):
        """surface_response(tau) for a tensor tau (interpolated in log-log)."""
        lt = torch.log10(torch.clamp(tau, min=1e-9)).cpu().numpy()
        return torch.tensor(np.exp(np.interp(lt, self._h_lt, self._h)), dtype=torch.float32, device=self.device)

    def _build_slices(self):
        """Slice maps on the electrolyte's grid (as srcs_EM MultiSlice._build_maps): per slice the
        source on the modes G[:, i], the row that averages the modes over it C[i], and the ohmic
        response P (slice by slice, anode slices first) at c_e0, T_REF."""
        e, geo, N = self.ely, self.geo, self.n_slices
        dx, _, d_eff, dom = e._grid()
        phi = e._phi
        kap = float(e.kappa(T_REF)) * d_eff

        def cumint(f):
            return np.cumsum(f * dx) - 0.5 * f * dx

        G, C, Pcol, cells_all, w_all = [], [], [], [], []
        for key, d in (('n', 0), ('p', 2)):
            idx = np.flatnonzero(dom == d)
            s = np.cumsum(dx[idx]) - 0.5 * dx[idx]
            edges = np.linspace(0.0, dx[idx].sum(), N + 1)
            grp = np.clip(np.searchsorted(edges, s) - 1, 0, N - 1)
            a = geo['a_%s' % key]
            for k in range(N):
                c = idx[grp == k]
                if c.size == 0:
                    raise ValueError('empty slice %d in electrode %s: fewer slices or more cells' % (k, key))
                w = dx[c] / dx[c].sum()
                b = np.zeros(len(dx))
                b[c] = dx[c] * (1.0 - e.t_plus) * a / F_FARADAY
                G.append(phi.T @ b)
                C.append(w @ phi[c])
                u = np.zeros(len(dx))
                u[c] = a
                Pcol.append(cumint(-cumint(u) / kap))
                cells_all.append(c)
                w_all.append(w)
        Pfull = np.column_stack(Pcol)                                          # (n_fv, 2N)
        P = np.array([w @ Pfull[c] for w, c in zip(w_all, cells_all)])         # (2N, 2N): phi_e at slice i per j_m
        f = lambda v: torch.tensor(np.asarray(v, dtype=float), dtype=torch.float32, device=self.device)
        self.ms_G = f(np.column_stack(G))                                      # (M, 2N)
        self.ms_C = f(np.array(C))                                             # (2N, M)
        self.ms_P = f(P)
        Pj = P.copy()
        Pj[:N, N:] = 0.0                                                       # the split's: no cathode share at the anode
        self.ms_Pj = f(Pj)
        self.ms_aLA = {k: geo['A'] * geo['a_%s' % k] * geo['L_%s' % k] for k in ('n', 'p')}
        # leading-order check: P of a uniform split against the SPMe's R_e
        jn, jp = 1.0 / self.ms_aLA['n'], -1.0 / self.ms_aLA['p']
        jv = np.r_[np.full(N, jn), np.full(N, jp)]
        pe = P @ jv
        self.ms_Re_check = float(pe[:N].mean() - pe[N:].mean())               # [Ohm] vs self.e_Re

    def _set_maps(self, ws):
        """geometry='set': pm's maps, the same for every candidate, with alpha_De, beta_ce and tort."""
        B = ws['sb'].numel()
        ex = lambda t: t[None].expand(B, *t.shape)
        ws['P'].copy_(ex(self.ms_P))
        ws['Pj'].copy_(ex(self.ms_Pj))
        ws['G'].copy_(ws['sb'][:, None, None] * ex(self.ms_G))
        ws['C'].copy_(ex(self.ms_C))
        ws['lam25'].copy_((ws['alpha_De'] / ws['tort'] * self.e_De)[:, None] * self.e_lam_hat[None, :])
        ws['sg'].copy_(ws['sb'])

    def _group_maps(self, ws):
        """geometry='groups': the maps of every candidate from its shape (l_s, r_np, eps_k, brugg), scaled
        to its R_e, tau_e and R_ce (class docstring). Built as _build_slices on a dimensionless grid
        x / L in [0, 1] with a fixed number of finite volumes per slice (FV_GROUPS), so that the slices
        are the same cells for every candidate; the eigenproblem is solved batched, in float64."""
        dev, N, M = self.device, self.n_slices, self.e_lam_hat.numel()
        f64 = lambda k: ws[k].double()
        B = ws['sb'].numel()
        npc, ns = self.FV_GROUPS
        # dimensionless thicknesses and per-cell properties (B, n_fv): anode slices, separator, cathode slices
        l_s = f64('l_s')
        l_n, l_p = (1.0 - l_s) * f64('r_np'), (1.0 - l_s) * (1.0 - f64('r_np'))
        nn = N * npc
        dom = torch.cat([torch.zeros(nn), torch.ones(ns), torch.full((nn,), 2.0)]).to(dev)
        if self._uniform:
            dx = torch.cat([(l_n / nn)[:, None].expand(B, nn), (l_s / ns)[:, None].expand(B, ns),
                            (l_p / nn)[:, None].expand(B, nn)], 1)
        else:                                                   # npc equal cells inside each slice of its width
            cw = {k: torch.tensor(np.repeat(v, npc) / npc, dtype=torch.float64, device=dev)
                  for k, v in self._sw64.items()}
            dx = torch.cat([l_n[:, None] * cw['n'][None, :], (l_s / ns)[:, None].expand(B, ns),
                            l_p[:, None] * cw['p'][None, :]], 1)
        eps = torch.cat([f64('eps_n')[:, None].expand(B, nn), f64('eps_s')[:, None].expand(B, ns),
                         f64('eps_p')[:, None].expand(B, nn)], 1)
        deff = eps ** f64('brugg')[:, None]
        n = dx.shape[1]
        # salt operator, K c with no flux at the collectors (harmonic face transport, as srcs_EM), mass eps dx
        coef = 2.0 / (dx[:, :-1] / deff[:, :-1] + dx[:, 1:] / deff[:, 1:])                    # (B, n-1)
        diag = torch.zeros(B, n, dtype=torch.float64, device=dev)
        diag[:, :-1] -= coef
        diag[:, 1:] -= coef
        K = torch.diag_embed(diag) + torch.diag_embed(coef, 1) + torch.diag_embed(coef, -1)
        mh = (eps * dx) ** -0.5
        lam, V = torch.linalg.eigh(mh[:, :, None] * K * mh[:, None, :])                     # ascending, <= 0
        lam, V = lam[:, -(M + 1):-1].flip(1), V[:, :, -(M + 1):-1].flip(2)                    # slowest M, no zero mode
        phi = mh[:, :, None] * V                                                              # eps dx-orthonormal
        # slices: indicator (B, n, 2N) of unit current per slice (u integrates to 1 over the slice)
        slab = torch.zeros(n, 2 * N, dtype=torch.float64, device=dev)
        for i in range(N):
            slab[i * npc:(i + 1) * npc, i] = 1.0
            slab[nn + ns + i * npc:nn + ns + (i + 1) * npc, N + i] = 1.0
        width = (dx[:, :, None] * slab[None]).sum(1)                                          # (B, 2N)
        u = slab[None] / width[:, None, :]
        W = (slab[None] * dx[:, :, None] / width[:, None, :]).transpose(1, 2)                 # (B, 2N, n) averages
        G = phi.transpose(1, 2) @ (dx[:, :, None] * u)                                       # (B, M, 2N) source
        C = W @ phi                                                                           # (B, 2N, M)
        cumint = lambda v: torch.cumsum(v * dx[:, :, None], 1) - 0.5 * v * dx[:, :, None]
        P = W @ cumint(-cumint(u) / deff[:, :, None])                                         # (B, 2N, 2N) per slice current
        # scales: uniform split of 1 A
        if self._uniform:
            jv = torch.cat([torch.full((N,), 1.0 / N), torch.full((N,), -1.0 / N)]).double().to(dev)
            wavg = lambda v, k: v.mean(1)
        else:                                                   # a uniform j: each slice's share of 1 A is its width
            w64 = {k: torch.tensor(v, dtype=torch.float64, device=dev) for k, v in self._sw64.items()}
            jv = torch.cat([w64['n'], -w64['p']])
            wavg = lambda v, k: v @ w64[k]
        pe = P @ jv
        r_hat = wavg(pe[:, :N], 'n') - wavg(pe[:, N:], 'p')
        P = P * (f64('R_e') / r_hat)[:, None, None]
        gain = 2.0 * R_GAS * T_REF / F_FARADAY * (1.0 - self.e_tplus) * self.e_tdf / self.e_c0
        if self.e_groups == 'material':                                                       # class docstring
            dch = C @ (-(G @ jv) / lam)[:, :, None]
            c_hat = -(wavg(dch[:, N:, 0], 'p') - wavg(dch[:, :N, 0], 'n'))
            De = self.e_De * f64('De_f')                                    # MS_OPTIONAL: 1 unless fitted
            tau_e = f64('L_e') ** 2 / (De * lam[:, 0].abs())
            R_ce = (gain * (1.0 - self.e_tplus) * float(self.ely.kappa(T_REF)) * f64('R_e') * c_hat
                    / (F_FARADAY * De * r_hat))
        else:
            tau_e, R_ce = f64('tau_e'), f64('R_ce')
        lam = lam / (tau_e[:, None] * lam[:, :1].abs())                                      # [1/s] at 25 C
        dc = C @ (-(G @ jv) / lam)[:, :, None]                                                # steady salt per unit source
        Rce_unit = -gain * (wavg(dc[:, N:, 0], 'p') - wavg(dc[:, :N, 0], 'n'))
        G = G * (R_ce / Rce_unit)[:, None, None]
        # per unit slice current density j of the model: I_slice = A a L j / N (non-uniform: A a L j w_i)
        aLA = torch.tensor([self.ms_aLA['n']] * N + [self.ms_aLA['p']] * N, dtype=torch.float64, device=dev)
        aLA = aLA / N if self._uniform else aLA * torch.cat([w64['n'], w64['p']])
        P, G = P * aLA, G * aLA
        Pj = P.clone()
        Pj[:, :N, N:] = 0.0
        for k, v in (('P', P), ('Pj', Pj), ('G', G), ('C', C), ('lam25', lam)):
            ws[k].copy_(v.float())
        ws['sg'].fill_(1.0)

    def set_groups(self, g: dict) -> dict:
        """The GROUPS of physical parameters on pm's geometry (Ds_k, j0_k, R_contact, alpha_De, beta_ce,
        tort; scalars), at 25 C: what geometry='groups' must be given to reproduce geometry='set'."""
        e, geo, N = self.ely, self.geo, self.n_slices
        tort = float(g.get('tort', 1.0))
        sb = float(np.sqrt(g['beta_ce']))
        lam = (g['alpha_De'] / tort * self.e_De) * self.e_lam_hat.double().cpu().numpy()
        jv = np.r_[np.full(N, 1.0 / self.ms_aLA['n']), np.full(N, -1.0 / self.ms_aLA['p'])]
        dc = self.ms_C.double().cpu().numpy() @ (-(sb * self.ms_G.double().cpu().numpy() @ jv) / lam)
        gain = 2.0 * R_GAS * T_REF / F_FARADAY * (1.0 - self.e_tplus) * self.e_tdf * sb / self.e_c0
        Lt = e.L_n + e.L_s + e.L_p
        out = {'theta_%s' % k: geo['R_%s' % k] ** 2 / g['Ds_%s' % k] for k in ('n', 'p')}
        out.update({'I0_%s' % k: self.ms_aLA[k] * g['j0_%s' % k] for k in ('n', 'p')})
        out.update(R_contact=g['R_contact'], R_e=tort * self.ms_Re_check, tau_e=float(-1.0 / lam[0]),
                   R_ce=float(-gain * (dc[N:].mean() - dc[:N].mean())), L_e=Lt, l_s=e.L_s / Lt,
                   r_np=e.L_n / (e.L_n + e.L_p), eps_n=e.eps_n, eps_s=e.eps_s, eps_p=e.eps_p, brugg=e.b_n)
        return {k: float(v) for k, v in out.items()}

    # ------------------------------------------------------------------ split
    def _newton(self, j, ph, xg, kap, j0k, cb, Ms, Pj, r, dfac, tgt, f, iters):
        """Batched Newton on the split of both electrodes (j (B, 2N), anode slices first; phi_s (B, 2))
        where it balances (self.split):
            j - 2 j0(x_s, c_e) sinh(f eta) = 0,  mean j_n = tgt_n,  mean j_p = tgt_p,
            eta = phi_s - r (P j) - dfac ln(c_e / c_e0) - U(x_s),  x_s = xg - kap (j - jg),  c_e = cb + Ms j
        (module docstring, step 2), jg the initial j, P = Pj (B, 2N, 2N). j0 = j0k sqrt(x_s (1 - x_s)) / 0.5
        sqrt(c_e / c_e0). The Jacobian is exact (the slopes of j0 in x_s and c_e included, zero where a clamp
        is active): without the j0 slopes the iteration converges only linearly and wanders at the end of a
        discharge, where x_s (1 - x_s) is steep."""
        B, N2 = j.shape
        N = N2 // 2
        c0 = self.e_c0
        U = lambda s: torch.cat([self.U_n(s[:, :N]), self.U_p(s[:, N:])], 1)
        dU = lambda s: torch.cat([self.dUdx_n(s[:, :N]), self.dUdx_p(s[:, N:])], 1)
        eye = torch.eye(N2, device=j.device)
        J = torch.zeros(B, N2 + 2, N2 + 2, device=j.device)
        if self._uniform:
            J[:, N2, :N] = 1.0 / N
            J[:, N2 + 1, N:N2] = 1.0 / N
        else:                                                   # the totals are width-weighted means
            J[:, N2, :N] = self._sw['n']
            J[:, N2 + 1, N:N2] = self._sw['p']
        Pr = r[:, None, None] * Pj
        jg = j
        # a start inside (0, 1): a slice the guessed split takes past a limit (x_s is linear in j) starts at
        # X_EDGE from it, its excess current spread over its electrode's other slices; the damping below then
        # keeps every x_s inside (it only stops a crossing, so a start outside stayed outside, at the clamped
        # U and j0, e.g. the separator-side anode slice at x_s > 1 in a fast charge)
        xt = torch.clamp(xg, X_EDGE, 1.0 - X_EDGE)
        c = (xg - xt) / kap.clamp(min=1e-12)
        free = (c == 0).float()
        if self._uniform:
            cs = torch.cat([c[:, :N].sum(1, keepdim=True).expand(B, N), c[:, N:].sum(1, keepdim=True).expand(B, N)], 1)
            nf = torch.cat([free[:, :N].sum(1, keepdim=True).expand(B, N), free[:, N:].sum(1, keepdim=True).expand(B, N)], 1)
        else:                                                   # width-weighted: the electrode's total is kept
            cs = torch.cat([self._avg(c[:, :N], 'n', True).expand(B, N), self._avg(c[:, N:], 'p', True).expand(B, N)], 1)
            nf = torch.cat([self._avg(free[:, :N], 'n', True).expand(B, N),
                            self._avg(free[:, N:], 'p', True).expand(B, N)], 1)
        j = jg + c - free * cs / nf.clamp(min=1.0)
        for _ in range(iters):
            xr = xg - kap * (j - jg)
            xs = torch.clamp(xr, 1e-6, 1.0 - 1e-6)
            cr = cb + (Ms @ j[:, :, None])[..., 0]
            c = torch.clamp(cr, min=1.0)
            j0 = j0k * (torch.sqrt(xs * (1.0 - xs)) / 0.5 if J0_STOICH else 1.0) * torch.sqrt(c / c0)
            # (not ph.repeat_interleave: with an int it copies a host tensor, which a CUDA graph replays from freed memory)
            eta = ph[:, :, None].expand(B, 2, N).reshape(B, N2) - (Pr @ j[:, :, None])[..., 0] - dfac * torch.log(c / c0) - U(xs)
            fe = f[:, None] * eta
            z = torch.clamp(fe, -30.0, 30.0)
            sz = torch.sinh(z)
            res = torch.cat([j - 2.0 * j0 * sz, (self._avg(j[:, :N], 'n') - tgt[0])[:, None],
                             (self._avg(j[:, N:], 'p') - tgt[1])[:, None]], dim=1)
            # slopes at the clamped values, not zeroed: a depleted slice still learns to shed current
            D = 2.0 * j0 * f[:, None] * torch.cosh(z)
            ic = (cr == c).float()
            dj0x = j0 * (1.0 - 2.0 * xs) / (2.0 * xs * (1.0 - xs)) * (J0_SLOPES and J0_STOICH)  # d j0 / d x_s
            dj0c = j0 / (2.0 * c) * ic * J0_SLOPES                          # d j0 / d c_e
            Ax = Pr + (dfac / c * ic)[:, :, None] * Ms                       # - d eta / d j but the OCV
            J[:, :N2, :N2] = (eye + D[:, :, None] * Ax + torch.diag_embed(-D * dU(xs) * kap + 2.0 * sz * dj0x * kap)
                              - (2.0 * sz * dj0c)[:, :, None] * Ms)
            J[:, :N, N2] = -D[:, :N]
            J[:, N:N2, N2 + 1] = -D[:, N:]
            delta = torch.linalg.solve_ex(J, -res)[0]                        # no sync: graph-safe
            dj, dph = delta[:, :N2], delta[:, N2:]
            # damping, per candidate: x_s keeps at least 10 % of its distance to 0 and 1, f eta moves by <= 2
            dx = -kap * dj
            ax = torch.where((dx < 0) & (xr > 0), 0.9 * xr / (-dx).clamp(min=1e-30),
                             torch.where((dx > 0) & (xr < 1), 0.9 * (1.0 - xr) / dx.clamp(min=1e-30), torch.ones_like(dx)))
            deta = dph[:, :, None].expand(B, 2, N).reshape(B, N2) - (Ax @ dj[:, :, None])[..., 0] - dU(xs) * dx
            az = 2.0 / (f[:, None] * deta).abs().clamp(min=1e-30)
            a = torch.clamp(torch.minimum(ax, az).amin(1, keepdim=True), max=1.0) if NEWTON_DAMP else 1.0
            j, ph = j + a * dj, ph + a * dph
        return j, ph

    def _kap(self, ws):
        """(kap_n, kap_p): the surface stoichiometry's fall per unit j over the step, at the split (the
        run's values; srcs/ms_fv.py recomputes it from the state when D_s depends on it)."""
        return ws['kap_n'], ws['kap_p']

    def _on_window(self, ws, tau_s):
        """Per-run set-up of the particle propagators (outside any CUDA graph); the network needs none."""

    def _nsub(self, dtau, dtk):
        """Sub-steps of a run (odd): every row's sub-step <= TAU_SUB."""
        nsub = int(np.ceil(float(dtau.max()) / TAU_SUB))
        return nsub + 1 - nsub % 2

    def _advance(self, y, phir, tr_state, tr_half, bc_xi, tau_s, n_steps, m_mid):
        """n_steps sub-steps of the particles (rows of y) at fluxes phir; the surface at the middle of
        sub-step m_mid (read from the trunk at tau_s / 2). Returns (y after n_steps, surface at the midpoint)."""
        y_mid = None
        bc = phir[:, None] * bc_xi[None, :]
        for s in range(n_steps):
            c_pin = y[:, :1]
            prod = self._branch(y - c_pin, phir)
            z = self._out(prod[:, None, :], tr_state, 1.0, bc) + c_pin
            shift = -3.0 * phir * tau_s - ((z - y) @ self.vm)
            if s == m_mid:
                y_mid = self._out(prod, tr_half, phir, 0.5) + c_pin[:, 0] + 0.5 * shift
            y = z + shift[:, None]
        return y, y_mid

    # ------------------------------------------------------------------ superstep
    # The network is rolled out with steps of DTAU_MIN or more: many shorter steps drift (at -5 C, dt / theta_n
    # ~ 4e-4 per 10 s sample put the anode surface hundreds of mV off the exact particle). With superstep, every
    # particle keeps a base state y_b, the time te elapsed since it and the integral of its flux; a sample
    # evaluates the particle ONCE from y_b over te at the mean flux since y_b (the trunk read at any time from
    # a table), and the particle commits (y_b := that state) only when te >= DTAU_MIN, before a change of the
    # current and at the end of a run. A particle whose own step is long enough commits at every sample, in its
    # own number of sub-steps of <= TAU_SUB (masked: a fast candidate no longer cuts the others' steps).
    # The flux of a sample is spread over te: its surface response is h(te) dt / te instead of h(dt), so the
    # split's slope kap is that too (consistent with what the particle then does).
    def _ss_tables(self):
        """Trunk features on a log grid of times (N_TRUNK, n, h) at the sensors and (N_TRUNK, h) at the
        surface, and log h(tau) on its grid, as device tensors."""
        if self._ss_tab is None:
            dev = self.device
            lt = torch.linspace(np.log(1e-7), np.log(TAU_SUB * 1.0001), N_TRUNK, device=dev)
            t = torch.exp(lt)
            with torch.no_grad():
                TT = self._trunk(t[:, None].expand(-1, self.n), self.xi[None, :].expand(N_TRUNK, -1))
                TS = self._trunk(t, torch.ones_like(t))
            self._ss_tab = dict(l0=float(lt[0]), dl=float(lt[1] - lt[0]), TT=TT, TS=TS,
                                h_l0=float(self._h_lt[0]), h_dl=float(self._h_lt[1] - self._h_lt[0]),
                                h=torch.tensor(self._h, dtype=torch.float32, device=dev))
        return self._ss_tab

    @staticmethod
    def _lerp(tab, u):
        """Rows of tab interpolated at the fractional indices u (graph-safe: index_select, no sync)."""
        u = torch.clamp(torch.nan_to_num(u, nan=0.0), 0.0, tab.shape[0] - 1.0)   # a diverged candidate gives NaN
        i0 = torch.clamp(u.floor().long(), 0, tab.shape[0] - 2)     # (not via u: n - 1 - 1e-6 is n - 1 in float32)
        w = (u - i0).view(-1, *([1] * (tab.dim() - 1)))
        return tab.index_select(0, i0) * (1.0 - w) + tab.index_select(0, i0 + 1) * w

    def _h_t(self, tau):
        """surface_response(tau) for a tensor tau, on device (log-log interpolation of its table)."""
        T = self._ss_tables()
        u = (torch.log10(torch.clamp(tau, min=1e-9)) - T['h_l0']) / T['h_dl']
        return torch.exp(self._lerp(T['h'], u))

    def _advance_ss(self, yb, phir, te, t_mid, bc_xi, n_slots):
        """The particles from their base states yb over te (per row) at the constant fluxes phir, in
        n_r = odd(ceil(te / TAU_SUB)) sub-steps per row (n_slots >= max n_r; slots past a row's n_r leave
        it unchanged); the surface at t_mid from the base. Returns (state at te, surface at t_mid)."""
        T = self._ss_tables()
        n_r = torch.ceil(te / TAU_SUB)
        n_r = n_r + 1.0 - torch.remainder(n_r, 2.0)
        tau_r = te / n_r
        tr_state = self._lerp(T['TT'], (torch.log(tau_r) - T['l0']) / T['dl'])                 # (rows, n, h)
        m_r = torch.clamp(torch.floor(t_mid / tau_r), max=n_r - 1.0)
        t_loc = torch.clamp(t_mid - m_r * tau_r, min=1e-7)
        tr_mid = self._lerp(T['TS'], (torch.log(t_loc) - T['l0']) / T['dl'])                    # (rows, h)
        frac = t_loc / tau_r
        bc = phir[:, None] * bc_xi[None, :]
        y, y_mid = yb, torch.zeros_like(te)
        for s in range(n_slots):
            c_pin = y[:, :1]
            prod = self._branch(y - c_pin, phir)
            z = self._out(prod[:, None, :], tr_state, 1.0, bc) + c_pin
            shift = -3.0 * phir * tau_r - ((z - y) @ self.vm)
            ym = self._out(prod, tr_mid, phir, 0.5) + c_pin[:, 0] + frac * shift
            y_mid = torch.where(m_r == s, ym, y_mid)
            y = torch.where((s < n_r)[:, None], z + shift[:, None], y)
        return y, y_mid

    # ------------------------------------------------------------------ cell
    PARAMS = (WINDOW + ('Ds_n', 'Ds_p', 'j0_n', 'j0_p', 'R_contact', 'alpha_De', 'beta_ce') + THERMAL + tuple(OPTIONAL)
              + tuple(k for k in GROUPS if k != 'R_contact') + ('L_e',) + tuple(D_LAW) + tuple(MS_OPTIONAL))
    STATE = ('y', 'j_n', 'j_p', 'q', 'kk', 'yb', 'te', 'pacc', 'phimax')
    # the buffers a run's set-up writes (simulate keeps them per (dt, temperature window) and restores them)
    RUN_KEYS = ('tau_s', 'dtr', 'tr_state', 'tr_half', 'gain', 'kap_n', 'kap_p', 'lam', 'E_full', 'E_half', 'Ms')

    def _workspace(self, B: int, K: int) -> dict:
        """The fixed buffers of a batch size B, for tests of up to K samples (grown, and its graphs
        dropped, when a longer test comes)."""
        ws = self._ws.get(B)
        if ws is not None and ws['K'] >= K:
            return ws
        dev, N, n, M = self.device, self.n_slices, self.n, self.e_lam_hat.numel()
        K = max(K, ws['K'] if ws else 0)
        z = lambda *s: torch.zeros(*s, device=dev)
        h = self._trunk(torch.zeros(1, device=dev), torch.ones(1, device=dev)).shape[-1]
        ws = dict(K=K, graphs={}, pool=torch.cuda.graph_pool_handle() if self.graphs else None,
                  **{k: z(B) for k in self.PARAMS},
                  sb=z(B), alpha=z(B), Qrow=z(2 * B * N),
                  y=z(2 * B * N, n), j_n=z(B, N), j_p=z(B, N), q=z(B, M), kk=torch.zeros(1, dtype=torch.long, device=dev),
                  tr_state=z(2 * B * N, n, h), tr_half=z(2 * B * N, h), tau_s=z(2 * B * N), gain=z(2 * B * N),
                  kap_n=z(B, N), kap_p=z(B, N), lam=z(B, M), E_full=z(B, M), E_half=z(B, M), Ms=z(B, 2 * N, 2 * N),
                  P=z(B, 2 * N, 2 * N), Pj=z(B, 2 * N, 2 * N), G=z(B, M, 2 * N), C=z(B, 2 * N, M), lam25=z(B, M), sg=z(B),
                  I=z(K), f=z(K), kf=z(K), ark=z(K), rTs=z(K), dT=z(K), qc=z(K), q0=z(B),
                  V=z(B, K), xs=z(B, K), ys=z(B, K),
                  yb=z(2 * B * N, n), te=z(2 * B * N), pacc=z(2 * B * N), dtr=z(2 * B * N), fc=z(K), phimax=z(2 * B * N))
        self._ws[B] = ws
        return ws

    def _step(self, ws, nsub: int, jump: bool):
        """One sample (module docstring, steps 1-4) on the workspace: reads the sample at ws['kk'],
        writes V, x_s, y_s there, updates the state in place and advances the counter."""
        B, N = ws['j_n'].shape
        BN = B * N
        m_mid = (nsub - 1) // 2
        kk = ws['kk']
        at = lambda name: ws[name].index_select(0, kk)                     # (1,)
        Ik, f, kf, ark, dT = at('I'), at('f'), at('kf'), at('ark'), at('dT')
        c0, tdf, tp = self.e_c0, self.e_tdf, self.e_tplus
        if self.electrolyte_T == 'set':
            rT = at('rTs') * ws['tort'] * torch.exp(ws['dEa_e'] * ark)
            Rc = ws['R_contact'] * torch.exp(ws['Ea_s'] * ark)
        else:
            rT = torch.exp(ws['Ea_e'] * ark) * ws['tort']
            Rc = ws['R_contact'] * torch.exp(ws['Ea_e'] * ark)
        kin = torch.exp(-ws['Ea_r'] * ark)
        # dEa_rp (MS_OPTIONAL, 0 unless fitted): the cathode's activation energy is Ea_r + dEa_rp, the anode's Ea_r
        j0n_k, j0p_k = ws['j0_n'] * kin, ws['j0_p'] * kin * torch.exp(-ws['dEa_rp'] * ark)
        dfac = (kf * ((1.0 - tp) * tdf) * ws['sg'])[:, None]
        tgt_n, tgt_p = Ik / self.ms_aLA['n'], -Ik / self.ms_aLA['p']
        gain, lam = ws['gain'], ws['lam']
        P, Cm, Gm = ws['P'], ws['C'], ws['G']                                # per candidate (_set_maps, _group_maps)
        Pn, Pp, Ppn = P[:, :N, :N], P[:, N:, N:], P[:, N:, :N]
        q = ws['q']
        mv = lambda A_, v: (A_ @ v[:, :, None])[..., 0]                     # batched matrix-vector

        def salt(qq):
            c = torch.clamp(c0 + mv(Cm, qq), min=1.0)
            return c[:, :N], c[:, N:]

        def j0(j0k, s, c):
            fs = torch.sqrt(torch.clamp(s * (1 - s), min=1e-12)) / 0.5 if J0_STOICH else 1.0
            return j0k[:, None] * fs * torch.sqrt(c / c0)

        # 1. predictor to the end of the step (or the midpoint) at the guessed split
        if jump and JUMP_START == 'uniform':
            jg_n, jg_p = ws['j_n'] * 0.0 + tgt_n, ws['j_p'] * 0.0 + tgt_p
        else:
            jg_n = ws['j_n'] - self._avg(ws['j_n'], 'n', True) + tgt_n
            jg_p = ws['j_p'] - self._avg(ws['j_p'], 'p', True) + tgt_p
        phig = torch.cat([jg_n.reshape(-1), jg_p.reshape(-1)]) * gain
        if self.superstep:                                                   # the particles from their base (_ss_tables)
            dtr = ws['dtr']
            te1 = ws['te'] + dtr
            yg = self._advance_ss(ws['yb'], (ws['pacc'] + phig * dtr) / te1, te1, te1 - 0.5 * dtr, self.xi2h, nsub)[0][:, -1]
            kap = gain * self._h_t(te1) * dtr / te1
            kap_n, kap_p = kap[:BN].view(B, N), kap[BN:].view(B, N)
        elif self.split == 'end':
            yg = self._advance(ws['y'], phig, ws['tr_state'], ws['tr_half'], self.xi2h, ws['tau_s'], nsub, m_mid)[0][:, -1]
        else:
            yg = self._advance(ws['y'], phig, ws['tr_state'], ws['tr_half'], self.xi2h, ws['tau_s'], m_mid + 1, m_mid)[1]
        if not self.superstep:
            kap_n, kap_p = self._kap(ws)
        jg = torch.cat([jg_n, jg_p], 1)
        cb = c0 + mv(Cm, q * ws['E_full' if self.split == 'end' else 'E_half'])   # the salt there: cb + Ms j
        c = torch.clamp(cb + mv(ws['Ms'], jg), min=1.0)
        cn, cp = c[:, :N], c[:, N:]
        dn, dp = dfac * torch.log(cn / c0), dfac * torch.log(cp / c0)
        xg, yg_ = yg[:BN].view(B, N), yg[BN:].view(B, N)
        # 2. the split at the end of the step (or the midpoint), implicit in the particles and the salt
        it = NEWTON_JUMP if jump else NEWTON
        ph_n = self._avg(self.U_n(xg) + dn + torch.asinh(jg_n / (2 * j0(j0n_k, xg, cn))) / f[:, None], 'n')
        ph_p = self._avg(self.U_p(yg_) + dp + torch.asinh(jg_p / (2 * j0(j0p_k, yg_, cp))) / f[:, None], 'p')
        j, _ = self._newton(jg, torch.stack([ph_n, ph_p], 1), torch.cat([xg, yg_], 1),
                            torch.cat([kap_n, kap_p], 1),
                            torch.cat([j0n_k[:, None].expand(B, N), j0p_k[:, None].expand(B, N)], 1),
                            cb, ws['Ms'], ws['Pj'], rT, dfac, (tgt_n, tgt_p), f, it)
        j_n, j_p = j[:, :N].contiguous(), j[:, N:].contiguous()
        off = rT[:, None] * mv(Ppn, j_n)                                      # the anode's share at the cathode
        # 3. particles and salt over the step at the split's currents
        phir = torch.cat([j_n.reshape(-1), j_p.reshape(-1)]) * gain
        if self.superstep:
            pacc1 = ws['pacc'] + phir * dtr
            phib = pacc1 / te1
            y, y_mid = self._advance_ss(ws['yb'], phib, te1, te1 - 0.5 * dtr, self.xi2h, nsub)
            commit = (te1 >= DTAU_MIN * (1.0 - 1e-6)) | (at('fc') > 0.5)
            ws['yb'].copy_(torch.where(commit[:, None], y, ws['yb']))
            ws['te'].copy_(torch.where(commit, torch.zeros_like(te1), te1))
            ws['pacc'].copy_(torch.where(commit, torch.zeros_like(pacc1), pacc1))
            ws['phimax'].copy_(torch.maximum(ws['phimax'], phib.abs()))
        else:
            y, y_mid = self._advance(ws['y'], phir, ws['tr_state'], ws['tr_half'], self.xi2h, ws['tau_s'], nsub, m_mid)
        q_inf = -mv(Gm, torch.cat([j_n, j_p], 1)) / lam
        cn, cp = salt(q_inf + (q - q_inf) * ws['E_half'])
        q_new = q_inf + (q - q_inf) * ws['E_full']
        # 4. voltage at the midpoint
        xs = torch.clamp(y_mid[:BN].view(B, N), 1e-6, 1 - 1e-6)
        ys = torch.clamp(y_mid[BN:].view(B, N), 1e-6, 1 - 1e-6)
        pe_n = rT[:, None] * mv(Pn, j_n) + dfac * torch.log(cn / c0)
        pe_p = rT[:, None] * mv(Pp, j_p) + off + dfac * torch.log(cp / c0)
        ps_n = self._avg(self.U_n(xs) + pe_n + torch.asinh(j_n / (2 * j0(j0n_k, xs, cn))) / f[:, None], 'n')
        ps_p = self._avg(self.U_p(ys) + pe_p + torch.asinh(j_p / (2 * j0(j0p_k, ys, cp))) / f[:, None], 'p')
        if self.entropic == 'cell':
            ent = dT * (ws['dUdT0'] + ws['dUdT_n'] * (self._avg(xs, 'n') - 0.5) + ws['dUdT_p'] * (self._avg(ys, 'p') - 0.5))
        elif self.entropic == 'knots':                                       # on the bulk anode stoichiometry
            xb = ws['x100'] - (ws['q0'] + at('qc')) / ws['Q_n']
            kn = self._t_knots
            hat = torch.clamp(1.0 - torch.abs(torch.clamp(xb, kn[0], kn[-1])[:, None] - kn[None, :]) / self._t_h, min=0.0)
            ent = dT * (hat * torch.stack([ws['dUdTk%d' % i] for i in range(len(T_KNOTS))], 1)).sum(1)
        else:
            ent = ws['s_ocv'] * dT * (self._avg(self.dU_p(ys), 'p') - self._avg(self.dU_n(xs), 'n'))
        Vk = ps_p - ps_n - Ik * Rc + ent
        ws['V'].index_copy_(1, kk, Vk[:, None])
        ws['xs'].index_copy_(1, kk, self._avg(xs, 'n')[:, None])
        ws['ys'].index_copy_(1, kk, self._avg(ys, 'p')[:, None])
        ws['y'].copy_(y)
        ws['j_n'].copy_(j_n)
        ws['j_p'].copy_(j_p)
        q.copy_(q_new)
        kk.add_(1)

    def _replay(self, ws, nsub: int, jump: bool):
        """Run one sample: replay its CUDA graph (captured at first use, after a warm-up whose effect on
        the state is undone), or step eagerly."""
        if not self.graphs:
            return self._step(ws, nsub, jump)
        g = ws['graphs'].get((nsub, jump))
        if g is None:
            keep = {k: ws[k].clone() for k in self.STATE + ('V', 'xs', 'ys')}
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._step(ws, nsub, jump)
            torch.cuda.current_stream().wait_stream(s)
            for k, v in keep.items():
                ws[k].copy_(v)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=ws['pool']):
                self._step(ws, nsub, jump)
            ws['graphs'][(nsub, jump)] = g
        g.replay()

    @torch.no_grad()
    def simulate(self, g: dict, I, dt: float, T=None, T_ref: float = T_REF, q_start: float = 0.0,
                 electrolyte: bool = True, parts: bool = False):
        """Terminal voltage (B, K) at the sample midpoints, as BatchEM.simulate (electrolyte is ignored)."""
        if abs(T_ref - T_REF) > 1e-9:
            raise ValueError('multislice: T_ref must be T_REF')
        dev, N = self.device, self.n_slices
        I = np.asarray(I, dtype=float)
        K = len(I)
        Tn = np.full(K, T_ref) if T is None else np.broadcast_to(np.asarray(T, dtype=float), (K,)).copy()
        dt = np.broadcast_to(np.asarray(dt, dtype=float), (K,)).copy()
        g = {k: v for k, v in g.items()
             if k in WINDOW + self.dynamic + THERMAL + tuple(OPTIONAL) + tuple(D_LAW) + tuple(MS_OPTIONAL)}
        missing = [k for k in WINDOW + self.dynamic if k not in g]
        if missing:
            raise KeyError('simulate (multislice): missing %s' % missing)
        B = max(int(np.size(v)) for v in g.values())
        ws = self._workspace(B, K)
        defaults = {**{k: 0.0 for k in THERMAL}, 's_ocv': 1.0, **OPTIONAL, **D_LAW, **MS_OPTIONAL,  # unused: placeholders
                    **{k: 1.0 for k in self.PARAMS
                       if k not in THERMAL and k not in OPTIONAL and k not in D_LAW and k not in MS_OPTIONAL}}
        for k in self.PARAMS:
            v = g.get(k, defaults.get(k))
            ws[k].copy_(torch.as_tensor(np.broadcast_to(np.asarray(v, dtype=np.float64), (B,)).copy(),
                                        dtype=torch.float32, device=dev))
        geo = self.geo
        if self.geometry == 'groups':                                           # the groups as pm's internal Ds, j0
            for k in ('n', 'p'):
                ws['Ds_%s' % k].copy_(geo['R_%s' % k] ** 2 / ws['theta_%s' % k])
                ws['j0_%s' % k].copy_(ws['I0_%s' % k] / self.ms_aLA[k])
            ws['alpha_De'].fill_(1.0)
            ws['beta_ce'].fill_(1.0)
            ws['tort'].fill_(1.0)
        ws['sb'].copy_(torch.sqrt(ws['beta_ce']))
        ws['alpha'].copy_(ws['alpha_De'] / ws['tort'])
        (self._group_maps if self.geometry == 'groups' else self._set_maps)(ws)
        ws['Qrow'].copy_(torch.cat([ws['Q_n'].repeat_interleave(N), ws['Q_p'].repeat_interleave(N)]))
        qs = torch.as_tensor(np.broadcast_to(np.asarray(q_start, dtype=np.float64), (B,)).copy(),
                             dtype=torch.float32, device=dev)
        # particles: rows [anode slices of every candidate (B*N), cathode slices (B*N)], slice-major per candidate
        ws['y'].copy_(torch.cat([(ws['x100'] - qs / ws['Q_n'])[:, None].expand(B, N).reshape(-1),
                                 (ws['y100'] + qs / ws['Q_p'])[:, None].expand(B, N).reshape(-1)])[:, None]
                      .expand(-1, self.n))
        for k in ('j_n', 'j_p', 'q', 'kk', 'V', 'xs', 'ys', 'te', 'pacc', 'phimax'):
            ws[k].zero_()
        ws['yb'].copy_(ws['y'])
        # per-sample inputs
        rTs = self.electrolyte_factors(Tn)[1] if self.electrolyte_T == 'set' else np.ones(K)
        # kf scales the concentration overpotential (dfac in _step): with tdf_T on, chi(c_e0, T) / chi(c_e0, T_REF)
        # enters it per sample (off: chi at T_REF, the original model)
        kf = 2.0 * R_GAS * Tn / F_FARADAY
        if getattr(self, 'tdf_T', False):
            kf = kf * self.tdf_factor(Tn)
        qc = (np.r_[0.0, np.cumsum(I * dt)[:-1]] + 0.5 * I * dt) / 3600.0      # [Ah] at the sample midpoints
        ws['q0'].copy_(qs)
        for k, v in (('I', I), ('f', F_FARADAY / (2.0 * R_GAS * Tn)), ('kf', kf),
                     ('ark', (1.0 / Tn - 1.0 / T_ref) / R_GAS), ('rTs', rTs), ('dT', Tn - physics.T_REF), ('qc', qc)):
            ws[k][:K].copy_(torch.as_tensor(v, dtype=torch.float32, device=dev))
        aLA = torch.cat([torch.full((B * N,), self.ms_aLA['n'], device=dev),
                         torch.full((B * N,), self.ms_aLA['p'], device=dev)])
        tol = 0.05 * max(float(np.max(np.abs(I))), 1e-9)
        jumps = np.r_[True, np.abs(np.diff(I)) > tol]
        # runs of constant dt and temperature. A run's set-up depends on (dt, its temperature) only and is
        # computed once per distinct pair. T_per_window=False (default): the particles see each run's own mean
        # temperature (pairs repeat only at an exactly constant T). True: the mean of the temperature window
        # the run lies in, so an adaptive grid (two steps, many runs) needs two set-ups per window; on a grid
        # without changes of dt both are the same.
        runs = []
        if self.T_per_window:
            for a0, b0 in windows(np.zeros(K), Tn, dT=self.dT_window):
                Tw = float(Tn[a0:b0].mean())
                runs += [(a0 + a, a0 + b, Tw) for a, b in windows(np.zeros(b0 - a0), np.zeros(b0 - a0), dt=dt[a0:b0])]
        else:
            runs = [(a, b, float(Tn[a:b].mean())) for a, b in windows(np.zeros(K), Tn, dT=self.dT_window, dt=dt)]
        if self.superstep:                  # superstep: commit before a change of current and at the end of a run
            fc = np.r_[jumps[1:], True].astype(float)
            fc[[b - 1 for _, b, _ in runs]] = 1.0
            ws['fc'][:K].copy_(torch.as_tensor(fc, dtype=torch.float32, device=dev))
        setups = {}
        for a, b, Tw in runs:
            key = (float(dt[a]), Tw)
            if key in setups:                                                   # the pair's set-up, restored
                nsub, saved = setups[key]
                for k, v in saved.items():
                    ws[k].copy_(v)
                for k in range(a, b):
                    self._replay(ws, nsub, bool(jumps[k]))
                continue
            arr = (1.0 / Tw - 1.0 / T_ref) / R_GAS
            th = torch.cat([(geo['R_n'] ** 2 / ws['Ds_n'] * torch.exp(ws['Ea_Dn'] * arr)).repeat_interleave(N),
                            (geo['R_p'] ** 2 / ws['Ds_p'] * torch.exp(ws['Ea_Dp'] * arr)).repeat_interleave(N)])
            dtk = float(dt[a])
            dtau = dtk / th
            nsub = self._nsub(dtau, dtk)
            tau_s = dtau / nsub
            ws['tau_s'].copy_(tau_s)
            ws['dtr'].copy_(dtau)
            ws['tr_state'].copy_(self._trunk(tau_s[:, None].expand(-1, self.n), self.xi.expand(len(th), self.n)))
            ws['tr_half'].copy_(self._trunk(0.5 * tau_s, torch.ones_like(tau_s)))
            self._on_window(ws, tau_s)
            gain = aLA * th / (3.0 * ws['Qrow'] * 3600.0)                       # Phi per unit j
            ws['gain'].copy_(gain)
            kap = gain * self.h(dtau if self.split == 'end' else 0.5 * dtau)   # surface per unit j where the split balances
            ws['kap_n'].copy_(kap[:B * N].view(B, N))
            ws['kap_p'].copy_(kap[B * N:].view(B, N))
            if self.electrolyte_T == 'set':
                de_T = torch.full((B,), float(self.electrolyte_factors(Tw)[0][0]), device=dev) * torch.exp(-ws['dEa_De'] * arr)
            else:
                de_T = torch.exp(-ws['Ea_e'] * arr)
            lam = ws['lam25'] * de_T[:, None]                                   # (B, M) < 0
            ws['lam'].copy_(lam)
            ws['E_full'].copy_(torch.exp(lam * dtk))
            ws['E_half'].copy_(torch.exp(lam * 0.5 * dtk))
            Esp = ws['E_full' if self.split == 'end' else 'E_half']             # the salt where the split balances, per unit j
            ws['Ms'].copy_((ws['C'] * (-(1.0 - Esp) / lam)[:, None, :]) @ ws['G'])
            if any(r[2] == Tw and float(dt[r[0]]) == key[0] for r in runs if r[0] > a):   # needed again
                setups[key] = (nsub, {k: ws[k].clone() for k in self.RUN_KEYS if k in ws})
            for k in range(a, b):
                self._replay(ws, nsub, bool(jumps[k]))
        V = ws['V'][:, :K].clone()
        # the largest |Phi| (flux per particle, mean since its base) each candidate reached: the network is
        # trained for |Phi| <= 3 (superstep only)
        self.last_phi_max = torch.maximum(ws['phimax'][:B * N].view(B, N).amax(1), ws['phimax'][B * N:].view(B, N).amax(1))
        if parts:
            return {'V': V, 'x_surf': ws['xs'][:, :K].clone(), 'y_surf': ws['ys'][:, :K].clone(), 'phi_max': self.last_phi_max}
        return V
