"""
ms_fv.py
========
Verification twin of the multi-slice model: BatchMS (srcs/ms_model.py) with every particle advanced by a
finite-volume discretisation instead of the MIONet. Everything else is shared, line for line: the groups,
the electrolyte maps and modes, the implicit current split, the sub-steps, the voltage at the midpoint.
The difference between the two is therefore the surrogate's error alone, within the model.

Particle: vertex-centred finite volumes on n_fine equally spaced radii r_i in [0, 1] (the surface is the
last node), control volumes V_i = (r_{i+1/2}^3 - r_{i-1/2}^3) / 3, face fluxes r_{i+1/2}^2 (s_{i+1} - s_i)
/ dr, the flux -Phi entering the surface volume. Conservative by construction (d<s>/dtau = -3 Phi). Over a
sub-step of constant flux the linear system ds/dtau = A s + b Phi is integrated exactly by the matrix
exponential of the augmented matrix [[A, b], [0, 0]], so the only discretisation error is the radial one.

    em = BatchMSFV(run, pm, ..., n_fine=101)      # same arguments as BatchMS (CUDA graphs too)

The propagators of every row are computed once per run of constant step and temperature (_on_window,
outside the graphs) into the workspace, so a fit can run on the twin as fast as on the network.
One sub-step per sample (_nsub): every sub-step is exact at the sample's constant Phi and D factor, and the
midpoint is dtau / 2 for any odd count, so BatchMS's TAU_SUB rule (the network's horizon, set by the
batch's smallest theta) only added kernels; same V up to float32 rounding (checked). Each step
pins the particle's mean to the exact charge balance (_pin): the float32 round trip through the propagators
otherwise drifts the mean by ~1e-4 to 1e-3 in stoichiometry over 10^4 samples.

Cathode D_s(y) (d_law=True; groups Dp_drop, Dp_shape, srcs/ms_model.D_LAW): D_p = D_p(y100) f(y),
log10 f = -Dp_drop u^Dp_shape, u = max(y - y100, 0) / (y_end - y100) (d_window, default (0.40, 0.95): the
step-1 conventional cathode window; 0 at the top of charge, 1 at the end of the
window), with y the particle's MEAN stoichiometry at the start of each step, the diffusivity then constant
over the step (as a network fed one D per step would see it). The surface stoichiometry is not used: D(y_surf)
feeds back on the surface within the step (a falling D saturates the surface, which lowers D) and its
discrete solution depends on the sub-step (and through a shared sub-step count, on the batch). The step is
ds/dtau = f A s + b Phi, integrated exactly in A's eigenbasis (A = V diag(lam) V^-1, real: diag(vol) A is
symmetric): a(h) = exp(f lam h) a0 + h phi1(f lam h) (V^-1 b) Phi, phi1(z) = (e^z - 1) / z, so any number of
sub-steps gives the same result. The split's surface response (kap) is recomputed the same way (_kap).
Surface weight (group Dp_wsurf, default 0): the cathode's y is y_mean + Dp_wsurf (y_surf - y_mean), both at
the start of the sample (the surface lagged by one sample, so the step stays linear and exact, independent of
the sub-steps). 0 is the mean-stoichiometry law above; towards 1 a particle saturated at its surface (the end of
a discharge) keeps a low D through the following rest, which relaxes over hours in NMC811, while the mean alone
sets it back to the bulk value at once.
Anode D_s(x) (same switch; group Dn_slope): D_n = D_n(0.5) 10^(Dn_slope (x - 0.5)), x the mean
stoichiometry, theta_n at x = 0.5; Dn_slope = 0 is the constant D_n.
"""
from __future__ import annotations

import numpy as np
import torch

from srcs.ms_model import BatchMS


class BatchMSFV(BatchMS):
    RUN_KEYS = BatchMS.RUN_KEYS + ('fvE', 'fvg', 'fvEh', 'fvgh')

    def __init__(self, *args, n_fine: int = 101, d_law: bool = False, d_window=(0.40, 0.95), **kw):
        kw['superstep'] = False                  # exact propagators: any step is fine
        super().__init__(*args, **kw)
        self.d_law = bool(d_law)
        self.d_y0, self.d_span = float(d_window[0]), float(d_window[1]) - float(d_window[0])
        n = int(n_fine)
        r = np.linspace(0.0, 1.0, n)
        dr = r[1] - r[0]
        face = 0.5 * (r[:-1] + r[1:])
        lo = np.r_[0.0, face]
        hi = np.r_[face, 1.0]
        vol = (hi ** 3 - lo ** 3) / 3.0
        A = np.zeros((n, n))
        for i, rf in enumerate(face):           # face between nodes i and i + 1
            k = rf ** 2 / dr
            A[i, i] -= k / vol[i]
            A[i, i + 1] += k / vol[i]
            A[i + 1, i + 1] -= k / vol[i + 1]
            A[i + 1, i] += k / vol[i + 1]
        b = np.zeros(n)
        b[-1] = -1.0 / vol[-1]                  # -ds/dr|_1 = Phi leaves through the surface
        M = np.zeros((n + 1, n + 1))
        M[:n, :n], M[:n, n] = A, b
        self._fv_M = torch.tensor(M, dtype=torch.float64, device=self.device)
        # eigenbasis of A (d_law): S = W^1/2 A W^-1/2 symmetric, W = diag(vol)
        sw = np.sqrt(vol)
        lam, Q = np.linalg.eigh(sw[:, None] * A / sw[None, :])
        lam[np.argmax(lam)] = 0.0                                          # the conserved mode, exactly
        Vm, Vi = Q / sw[:, None], Q.T * sw[None, :]
        f32 = lambda z: torch.tensor(z, dtype=torch.float32, device=self.device)
        self._eg = dict(lam=f32(lam), VT=f32(Vm.T), ViT=f32(Vi.T), bt=f32(Vi @ b), vs=f32(Vm[-1]))
        self.n = n
        self.xi = torch.tensor(r, dtype=torch.float32, device=self.device)
        self.xi2h = 0.5 * self.xi ** 2
        self.vm = torch.tensor(3.0 * vol, dtype=torch.float32, device=self.device)   # mean = y @ vm
        self._ws = {}

    def _nsub(self, dtau, dtk):
        """One sub-step: the propagators are exact over the whole sample (module docstring)."""
        return 1

    def _trunk(self, tau, xi):                  # no network: a placeholder of the right shape
        return torch.zeros(*tau.shape, 1, device=self.device)

    def _prop(self, h):
        """(E, g) of every row for sub-steps h (rows,): s(h) = E s(0) + g Phi (one matrix exponential per
        distinct h)."""
        u, inv = torch.unique(h.double(), return_inverse=True)
        P = torch.linalg.matrix_exp(u[:, None, None] * self._fv_M[None])                # (U, n + 1, n + 1)
        return P[:, :self.n, :self.n].float()[inv], P[:, :self.n, self.n].float()[inv]

    def _dfac(self, y):
        """D_s factor f of every row from its profile y (rows, n): the mean stoichiometry, moved towards the
        surface by fv_wsurf (anode rows: fv_drop 0, fv_wsurf 0; cathode rows: fv_slope 0)."""
        w = self._fvbuf
        ym = y @ self.vm
        ys = ym + w['fv_wsurf'] * (y[:, -1] - ym)
        u = torch.clamp((ys - self.d_y0) / self.d_span, min=0.0)
        return torch.pow(10.0, -w['fv_drop'] * torch.pow(u + 1e-12, w['fv_shape']) + w['fv_slope'] * (ys - 0.5))

    def _modal(self, a, phir, h, fd):
        """Modal coefficients after a sub-step h (rows,) at factor fd from a (rows, n)."""
        e = self._eg
        z = e['lam'][None, :] * (fd * h)[:, None]
        p1 = torch.where(z.abs() < 1e-6, 1.0 + 0.5 * z, torch.expm1(z) / torch.where(z.abs() < 1e-6, torch.ones_like(z), z))
        return torch.exp(z) * a + (h * phir)[:, None] * p1 * e['bt'][None, :]

    def _kap(self, ws):
        if not self.d_law:
            return ws['kap_n'], ws['kap_p']
        B, N = ws['kap_n'].shape
        fd = self._dfac(ws['y'])
        dtau = ws['dtr']                                   # the whole step
        z = torch.zeros_like(ws['y'])
        resp = (self._modal(z, torch.ones_like(dtau), dtau, fd) * self._eg['vs'][None, :]).sum(1)   # surface per unit Phi
        kap = -ws['gain'] * resp
        return kap[:B * N].view(B, N), kap[B * N:].view(B, N)

    def _on_window(self, ws, tau_s):
        """The propagators of the run's sub-step and half sub-step into fixed workspace buffers (read by
        _advance inside the graphs)."""
        E, g = self._prop(tau_s)
        Eh, gh = self._prop(0.5 * tau_s)
        for k, v in (('fvE', E), ('fvg', g), ('fvEh', Eh[:, -1, :].contiguous()), ('fvgh', gh[:, -1].contiguous())):
            if k not in ws:
                ws[k] = torch.empty_like(v)
            ws[k].copy_(v)
        if self.d_law:
            B = ws['Dp_drop'].shape[0]
            N = self.n_slices
            zero = torch.zeros(B * N, device=self.device)
            for k, v in (('fv_drop', torch.cat([zero, ws['Dp_drop'].repeat_interleave(N)])),
                         ('fv_shape', torch.cat([zero + 1.0, ws['Dp_shape'].repeat_interleave(N)])),
                         ('fv_slope', torch.cat([ws['Dn_slope'].repeat_interleave(N), zero])),
                         ('fv_wsurf', torch.cat([zero, ws['Dp_wsurf'].repeat_interleave(N)]))):
                if k not in ws:
                    ws[k] = torch.empty_like(v)
                ws[k].copy_(v)
        self._fvbuf = ws

    def _advance(self, y, phir, tr_state, tr_half, bc_xi, tau_s, n_steps, m_mid):
        if self.d_law:
            e = self._eg
            y_mid = None
            fd = self._dfac(y)                                 # D at the sample's initial stoichiometry
            for s in range(n_steps):
                a = y @ e['ViT']
                if s == m_mid:
                    y_mid = (self._modal(a, phir, 0.5 * tau_s, fd) * e['vs'][None, :]).sum(1)
                y = self._pin(y, self._modal(a, phir, tau_s, fd) @ e['VT'], phir, tau_s)
            return y, y_mid
        w = self._fvbuf
        E, g, Eh, gh = w['fvE'], w['fvg'], w['fvEh'], w['fvgh']
        y_mid = None
        for s in range(n_steps):
            if s == m_mid:
                y_mid = (Eh * y).sum(1) + gh * phir
            y = self._pin(y, (E @ y[:, :, None])[..., 0] + g * phir[:, None], phir, tau_s)
        return y, y_mid

    def _pin(self, y0, y1, phir, h):
        """y1 with its mean set to the exact d<s>/dtau = -3 Phi from y0: the conserved mode is the uniform
        vector, so a uniform shift removes the float32 round-trip drift of the charge (module docstring)."""
        return y1 + ((y0 @ self.vm) - 3.0 * phir * h - (y1 @ self.vm))[:, None]
