"""
identifiability.py
==================
Local practical identifiability of the fitted groups at the optimum, with the same batched model:
the sensitivity of the voltage to each parameter in log space, by central differences, all 2d + 1
perturbed sets of a test simulated in ONE batch.

    S_t[i, j] = dV_i / d ln(p_j)       [V], masked samples i of test t
    F = sum_t S_t^T S_t                 Fisher matrix up to the noise variance
    corr = D^-1/2 F^-1 D^-1/2           parameter correlation
    rel_std_j = sigma sqrt(F^-1_jj)     sigma = the pooled residual RMSE. Assumes white, independent
                                        residuals; model error is neither, so these are LOWER bounds
    collinearity index gamma            Brun, Reichert & Kuensch, Water Resour. Res. 37 (2001) 1015:
                                        1 / sqrt(min eig(S_n^T S_n)), S_n = S with unit-norm columns,
                                        each test weighted 1/sqrt(N_t); gamma > 10-15 marks a set whose
                                        parameters can compensate each other
    eigen-directions of F               the stiffest and the sloppiest combination of ln p

A local analysis complements, and does not replace, the spread of the optimum over seeds (study.py
reports both): a flat valley shows up as seeds landing far apart at the same cost.
"""
from __future__ import annotations

import numpy as np

from srcs.identification import add_bias, relative, test_groups


def analyse(em, tests, names, x, fixed, h: float = 0.05, sigma_V: float | None = None) -> dict:
    """sigma_V: the noise level of the lower bounds [V]; default the pooled residual RMSE at x (at an
    optimum that is the honest choice; at the TRUE parameters of a plant the residual is model-form
    error, so pass the sensor noise instead)."""
    names, x = list(names), np.asarray(x, float)
    d = len(names)
    X = np.tile(x, (2 * d + 1, 1))
    for j in range(d):
        X[1 + 2 * j, j] *= np.exp(+h)
        X[2 + 2 * j, j] *= np.exp(-h)
    S_blocks, S_w, rss, n_tot = [], [], 0.0, 0
    for t in tests:
        g, q = test_groups(t, {**fixed, **{n: X[:, i] for i, n in enumerate(names)}}, em.ocp)
        V = relative(add_bias(em.simulate(g, t['I'], t['dt'], T=t['T'], q_start=q).double().cpu().numpy(), t, g, q), t)[:, t['mask']]
        r0 = V[0] - relative(t['V'], t)[t['mask']]
        S = np.stack([(V[1 + 2 * j] - V[2 + 2 * j]) / (2 * h) for j in range(d)], axis=1)   # (N_t, d)
        S_blocks.append(S)
        S_w.append(S / np.sqrt(len(r0)))
        rss += float(np.sum(r0 ** 2))
        n_tot += len(r0)
    S_all = np.vstack(S_blocks)
    Sw_all = np.vstack(S_w)
    # a group at zero (a weight on its lower bound 0, say) is not moved by a step in ln p: its column is
    # zero. Such groups are reported as insensitive and left out of the matrices (NaN rows and columns)
    nrm = np.linalg.norm(Sw_all, axis=0)
    on = np.isfinite(nrm) & (nrm > 0) & np.all(np.isfinite(S_all), axis=0)
    S, Sw, k = S_all[:, on], Sw_all[:, on], int(on.sum())
    F = S.T @ S
    Fi = np.linalg.pinv(F)
    resid = np.sqrt(rss / max(n_tot - d, 1))
    sigma = resid if sigma_V is None else float(sigma_V)
    sd = np.sqrt(np.clip(np.diag(Fi), 0, None))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = Fi / np.outer(sd, sd)
    Sn = Sw / np.linalg.norm(Sw, axis=0, keepdims=True)
    cos = Sn.T @ Sn
    ev_n = np.linalg.eigvalsh(cos)
    lam, vec = np.linalg.eigh(F)

    def full(M):                       # back to all d groups, NaN for the insensitive ones
        out = np.full((d, d), np.nan)
        out[np.ix_(on, on)] = M
        return out

    def vfull(v, fill=np.nan):
        out = np.full(d, fill)
        out[on] = v
        return out
    c_off = corr - np.eye(k)
    return dict(
        names=names, x=x.tolist(), step_log=h, sigma_mV=1e3 * sigma, residual_mV=1e3 * resid, n_samples=n_tot,
        insensitive=[n for n, o in zip(names, on) if not o],
        rel_std_lower_bound=dict(zip(names, vfull(sigma * sd).tolist())),
        sensitivity_mV=dict(zip(names, (1e3 * np.sqrt(np.mean(np.nan_to_num(S_all) ** 2, axis=0))).tolist())),
        correlation=full(corr).tolist(),
        max_abs_correlation=float(np.nanmax(np.abs(c_off))) if k > 1 else 0.0,
        collinearity_index=float(1.0 / np.sqrt(max(ev_n.min(), 1e-300))),
        # cosines between the normalised sensitivity vectors: the collinearity index of any subset of the
        # groups is 1 / sqrt(min eig(cosine[subset, subset])) (Brun et al.), with no new simulation
        cosine=full(cos).tolist(),
        condition_number=float(lam.max() / max(lam.min(), 1e-300)),
        stiffest=dict(zip(names, vfull(vec[:, -1], 0.0).tolist())),
        sloppiest=dict(zip(names, vfull(vec[:, 0], 0.0).tolist())))
