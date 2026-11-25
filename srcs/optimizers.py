"""
optimizers.py
=============
Global optimisers for the parameter identification, behind one interface:

    res = optimize(method, cost, lb, ub, budget, seed=0, **options)

cost(P) -> (P,) costs for a (P, d) array of physical parameter sets, evaluated as ONE batch (on the
GPU in this study). Every method searches in log space between the bounds lb, ub - linearly for a
parameter whose lower bound is <= 0 (a signed one, e.g. an entropic coefficient) - and is given the
same number of cost evaluations (`budget`), so the comparison is fair by construction.
Returns dict(x=best parameters, f=best cost, history=[best cost after each batch], n_eval, wall_s).

    swarm       the proposed method: an elitist, smoothed cross-entropy swarm. Iteration 0 samples the box
                uniformly; each later iteration samples a Gaussian whose mean and covariance move by
                `smooth` towards those of the elite (the best `elite` fraction of the new candidates AND
                of the previous elite), and keeps the best set found. All candidates of an iteration are
                independent, so an iteration is one batched call.
                The original swarm (elite 5 % of the new candidates only, covariance contracted by 0.7^2,
                no smoothing; smooth=1, keep=False, shrink=0.7, elite=0.05) collapses within ~5
                iterations: on the step-A Mohtat HPPC it stopped at 0.9-1.3 mV against 0.05 mV at the
                truth, the same with 4x the budget.
    cmaes       CMA-ES (Hansen 2016), (mu/mu_w, lambda) with rank-one and rank-mu updates and cumulative
                step-size adaptation, started at the box centre with sigma = 0.3 of the box; candidates
                outside the box are clipped for the cost and penalised for the ranking
    ga          genetic algorithm: tournament selection, SBX crossover, polynomial mutation, elitism
    pso         particle swarm: inertia 0.9 -> 0.4, cognitive and social weights 1.5
    de          differential evolution, rand/1/bin, F = 0.7, CR = 0.9
    multistart  L-BFGS-B from uniform random starts, forward-difference gradients evaluated as one
                batch of d + 1 sets; restarts until the budget is spent

Least squares on the residual vector instead of a scalar cost (a smooth model, e.g. the finite-volume
particles):

    res = lm_multistart(residuals, lb, ub, n_starts=32, seed=0)

batched multi-start Levenberg-Marquardt (lm_multistart's docstring).

Several seeds at once:

    results = optimize_seeds(method, cost, lb, ub, budget, seeds=[0, 1, 2, 3], **options)

runs one optimiser per seed in its own thread and merges their requests: whenever every running
optimiser is waiting for its next batch, the batches are stacked and evaluated as ONE cost call. S
seeds therefore take the same number of sequential GPU calls as one; each seed still sees exactly the
costs it would see alone (the candidates of a batch are independent rows) and keeps its own budget.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from scipy.optimize import minimize


class _Counted:
    """Wraps the cost: unit-cube input (log space, linear where lb <= 0), evaluation count, best-so-far
    history."""

    def __init__(self, cost, lb, ub, budget):
        self.cost, self.budget = cost, int(budget)
        lb, ub = np.asarray(lb, float), np.asarray(ub, float)
        self.log = lb > 0
        self.lo = np.where(self.log, np.log(np.where(self.log, lb, 1.0)), lb)
        self.hi = np.where(self.log, np.log(np.where(self.log, ub, 1.0)), ub)
        self.n_eval, self.best_f, self.best_x, self.history = 0, np.inf, None, []

    def phys(self, U):
        X = self.lo + np.clip(U, 0.0, 1.0) * (self.hi - self.lo)
        return np.where(self.log, np.exp(X), X)

    def __call__(self, U):
        U = np.atleast_2d(U)
        U = U[:max(0, self.budget - self.n_eval)]
        if len(U) == 0:
            return np.full(0, np.inf)
        X = self.phys(U)
        f = np.asarray(self.cost(X), dtype=float)
        f = np.where(np.isfinite(f), f, np.inf)
        self.n_eval += len(U)
        i = int(np.argmin(f))
        if f[i] < self.best_f:
            self.best_f, self.best_x = float(f[i]), X[i].copy()
        self.history.append((self.n_eval, self.best_f))
        return f

    @property
    def spent(self):
        return self.n_eval >= self.budget


def swarm(C, d, rng, pop=1000, elite=0.1, smooth=0.7, keep=True, shrink=1.0):
    k = max(10, int(elite * pop))
    U = rng.random((pop, d))
    f = C(U)
    o = np.argsort(f)[:k]
    EU, Ef = U[o], f[o]
    mu, cov = EU.mean(axis=0), np.cov(EU, rowvar=False)
    while not C.spent:
        U = np.clip(rng.multivariate_normal(mu, cov * shrink ** 2 + 1e-10 * np.eye(d), size=pop), 0.0, 1.0)
        f = C(U)
        PU, Pf = (np.vstack([EU, U[:len(f)]]), np.concatenate([Ef, f])) if keep else (U[:len(f)], f)
        o = np.argsort(Pf)[:k]
        EU, Ef = PU[o], Pf[o]
        mu = smooth * EU.mean(axis=0) + (1.0 - smooth) * mu
        cov = smooth * np.cov(EU, rowvar=False) + (1.0 - smooth) * cov


def cmaes(C, d, rng, pop=200, sigma0=0.3, x0=None):
    """x0: the start in the unit cube (default its centre), e.g. a warm start for a cost profile."""
    lam, mu = pop, pop // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
    cs = (mueff + 2) / (d + mueff + 5)
    c1 = 2 / ((d + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
    chi = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d))
    m, sigma = (np.full(d, 0.5) if x0 is None else np.clip(np.asarray(x0, float), 0.0, 1.0)), sigma0
    Cm, pc, ps, g = np.eye(d), np.zeros(d), np.zeros(d), 0
    while not C.spent:
        ev, B = np.linalg.eigh(Cm)
        D = np.sqrt(np.maximum(ev, 1e-20))
        Y = (rng.standard_normal((lam, d)) * D) @ B.T
        X = m + sigma * Y
        f = C(np.clip(X, 0.0, 1.0))
        if len(f) < lam:
            break
        f = f + 1e3 * np.sum((X - np.clip(X, 0.0, 1.0)) ** 2, axis=1)       # outside the box
        o = np.argsort(f)[:mu]
        yw = w @ Y[o]
        m = m + sigma * yw
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ yw) / D))
        g += 1
        hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * g)) / chi < 1.4 + 2 / (d + 1)
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * yw
        Cm = ((1 - c1 - cmu) * Cm + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * Cm)
              + cmu * (Y[o].T * w) @ Y[o])
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi - 1))


def ga(C, d, rng, pop=200, eta_c=15.0, eta_m=20.0, p_m=None, n_elite=2):
    p_m = 1.0 / d if p_m is None else p_m
    U = rng.random((pop, d))
    f = C(U)
    while not C.spent:
        def pick():
            a, b = rng.integers(0, pop, 2)
            return U[a] if f[a] < f[b] else U[b]
        kids = []
        while len(kids) < pop - n_elite:
            p1, p2 = pick(), pick()
            u = rng.random(d)
            beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)))
            kids += [0.5 * ((1 + beta) * p1 + (1 - beta) * p2), 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)]
        K = np.clip(np.array(kids[:pop - n_elite]), 0, 1)
        m = rng.random(K.shape) < p_m
        u = rng.random(K.shape)
        delta = np.where(u < 0.5, (2 * u) ** (1 / (eta_m + 1)) - 1, 1 - (2 * (1 - u)) ** (1 / (eta_m + 1)))
        K = np.clip(K + m * delta, 0, 1)
        fk = C(K)
        if len(fk) < len(K):
            break
        e = np.argsort(f)[:n_elite]
        U, f = np.vstack([U[e], K]), np.concatenate([f[e], fk])


def pso(C, d, rng, pop=200, c1=1.5, c2=1.5, w0=0.9, w1=0.4):
    X = rng.random((pop, d))
    Vel = 0.1 * (rng.random((pop, d)) - 0.5)
    f = C(X)
    P, fp = X.copy(), f.copy()
    n_it = max(1, C.budget // pop - 1)
    it = 0
    while not C.spent:
        g = P[np.argmin(fp)]
        w = w0 - (w0 - w1) * min(it / n_it, 1.0)
        Vel = w * Vel + c1 * rng.random((pop, d)) * (P - X) + c2 * rng.random((pop, d)) * (g - X)
        X = np.clip(X + Vel, 0, 1)
        f = C(X)
        if len(f) < pop:
            break
        better = f < fp
        P[better], fp[better] = X[better], f[better]
        it += 1


def de(C, d, rng, pop=200, F=0.7, CR=0.9):
    U = rng.random((pop, d))
    f = C(U)
    while not C.spent:
        idx = np.array([rng.choice(np.delete(np.arange(pop), i), 3, replace=False) for i in range(pop)])
        M = np.clip(U[idx[:, 0]] + F * (U[idx[:, 1]] - U[idx[:, 2]]), 0, 1)
        cross = rng.random((pop, d)) < CR
        cross[np.arange(pop), rng.integers(0, d, pop)] = True
        T = np.where(cross, M, U)
        ft = C(T)
        if len(ft) < pop:
            break
        better = ft < f
        U[better], f[better] = T[better], ft[better]


def multistart(C, d, rng, h=1e-3, maxiter=50):
    while not C.spent:
        def fun(u):
            pts = np.vstack([u] + [u + h * e for e in np.eye(d)])
            f = C(pts)
            if len(f) < d + 1:
                raise StopIteration
            return f[0], (f[1:] - f[0]) / h
        try:
            minimize(fun, rng.random(d), jac=True, method='L-BFGS-B', bounds=[(0.0, 1.0 - h)] * d,
                     options={'maxiter': maxiter})
        except StopIteration:
            break


METHODS = {'swarm': swarm, 'cmaes': cmaes, 'ga': ga, 'pso': pso, 'de': de, 'multistart': multistart}


def optimize(method, cost, lb, ub, budget, seed=0, **options):
    rng = np.random.default_rng(seed)
    C = _Counted(cost, lb, ub, budget)
    t0 = time.time()
    METHODS[method](C, len(lb), rng, **options)
    return dict(method=method, x=C.best_x, f=C.best_f, history=C.history, n_eval=C.n_eval,
                wall_s=time.time() - t0, seed=seed, options=options)


class _Broker:
    """Merges the cost requests of several optimiser threads into one call per round."""

    def __init__(self, cost, n):
        self.cost, self.active = cost, n
        self.cv = threading.Condition()
        self.waiting, self.results = {}, {}
        self.error, self.n_calls, self.rows = None, 0, 0

    def _flush(self):                                     # called with the lock held
        ids = list(self.waiting)
        X = np.vstack([self.waiting[i] for i in ids])
        try:
            f = np.asarray(self.cost(X), dtype=float)
        except BaseException as e:                        # surfaces in every waiting thread
            self.error, f = e, np.full(len(X), np.inf)
        self.n_calls += 1
        self.rows += len(X)
        a = 0
        for i in ids:
            n = len(self.waiting[i])
            self.results[i] = f[a:a + n]
            a += n
        self.waiting.clear()
        self.cv.notify_all()

    def request(self, i, X):
        with self.cv:
            self.waiting[i] = np.atleast_2d(X)
            if len(self.waiting) == self.active:
                self._flush()
            while i not in self.results:
                self.cv.wait()
            if self.error is not None:
                raise self.error
            return self.results.pop(i)

    def leave(self):
        with self.cv:
            self.active -= 1
            if self.active and len(self.waiting) == self.active:
                self._flush()


def optimize_seeds(method, cost, lb, ub, budget, seeds, **options):
    """One optimisation per seed, their batches merged into shared cost calls (see the module docstring).
    Returns the list of optimize() results, each with n_shared_calls (the sequential calls of the whole
    group) and wall_s (the group's wall time)."""
    seeds = list(seeds)
    if len(seeds) == 1:
        r = optimize(method, cost, lb, ub, budget, seed=seeds[0], **options)
        r['n_shared_calls'] = len(r['history'])
        return [r]
    broker = _Broker(cost, len(seeds))
    out, errors = [None] * len(seeds), []

    def run(k, s):
        try:
            out[k] = optimize(method, lambda X: broker.request(k, X), lb, ub, budget, seed=s, **options)
        except BaseException as e:
            errors.append(e)
        finally:
            broker.leave()

    t0 = time.time()
    threads = [threading.Thread(target=run, args=(k, s), daemon=True) for k, s in enumerate(seeds)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    for r in out:
        r['n_shared_calls'], r['wall_s'] = broker.n_calls, time.time() - t0
    return out


def lm_multistart(residuals, lb, ub, n_starts=32, seed=0, h=1e-3, max_iter=40, prune=((4, 16), (8, 8), (12, 4)),
                  lam0=1e-2, ladder=(0.1, 1.0, 10.0, 100.0), tol=1e-4, log=print, x0=None, checkpoint=None,
                  resume=None, select=None, names=None, respawn=None):
    """Batched multi-start Levenberg-Marquardt in the same unit cube as the other methods (log space
    between the bounds, linear where lb <= 0). residuals(X) -> (P, M) for a (P, d) array of physical sets,
    evaluated as ONE batch. Starts: a scrambled Sobol set. Each iteration takes two batched calls: the
    forward-difference Jacobian of every active start (d points each, step h), then len(ladder) trial steps
    per start, (J'J + lam diag(J'J)) delta = -J'r with lam = lam_s * ladder; the best trial is accepted
    if it lowers sum r^2 (lam_s := its lam), else lam_s *= 100. A start stops when an accepted step lowers
    sum r^2 by less than tol (relative) twice, or lam_s > 1e8; prune ((iteration, n), ...) keeps the n best
    starts at that iteration. Returns dict(x, f (sum r^2), u, J (at x, unit-cube columns), starts
    [(x, f, iterations)], history [(n_eval, best f)], n_eval, wall_s). x0: (k, d) physical sets taken as
    the first k starts (clipped into the bounds), the rest Sobol.
    checkpoint(state): called after every iteration with the search state (dict of arrays: iteration, U, lam,
    active, small, its, F, n_eval, history), enough to continue the search after a crash; resume: such a
    state, from which the search continues at the next iteration (the residuals of the starts are
    recomputed in one call, which is not counted in n_eval).
    select: None (every parameter free throughout, the default) or a dict for a selection INSIDE the search,
    per start, from the Jacobian the search computes anyway (no Morris, no extra simulation):
        every       check every k iterations (default 3): at a check the Jacobian has all d columns, between
                    checks only the free ones (the frozen parameters cost no simulation)
        S_mV        a parameter with S_j < S_mV (default 1) is frozen. S_j = |J_j| d u_j / d ln p_j: the RMS
                    voltage change [mV] per unit change of ln p_j over the weighted residuals (for a signed,
                    linearly mapped parameter, per unit relative change of p_j)
        gamma_max   then, while the collinearity index (Brun et al.) of the free set exceeds gamma_max (15),
                    the parameter whose freezing lowers it most is frozen, among those with S_j <= protect_mV
                    (20) and removals that lower the index by at least min_gain (0.05); ties within 5 % go
                    to the smaller S_j (the rule of scripts/select_groups.py)
    A frozen parameter keeps its current value. Every check starts again from all d parameters, so a
    parameter frozen early is released when it has become sensitive and independent (no path lock-in).
    With select, a parameter on a bound whose gradient pushes it outward is also left out of that step
    (projected step) instead of being clipped. names: the parameter names, for the log.
    respawn: None (the default: pruned starts stop) or {"frac": 1.0, "k": 2.0, "umin": 0.003, "umax": 0.05}: at every
    prune that fraction of the pruned starts restarts around the best start, perturbed in the unit cube by k times
    its Fisher standard deviation (from its last Jacobian and residual), clipped to [umin, umax]. On a GPU the
    width of the batch costs little against its length in time, so the freed places explore the valley of the best
    start for free (cliffs of the cost, collinear directions)."""
    from scipy.stats import qmc
    t0 = time.time()
    C = _Counted(lambda X: np.zeros(len(X)), lb, ub, 0)                    # its unit-cube map only
    d = len(lb)
    U = qmc.Sobol(d, scramble=True, seed=seed).random(n_starts) * (1.0 - h)
    if x0 is not None:
        X0 = np.clip(np.atleast_2d(np.asarray(x0, float))[:n_starts], np.asarray(lb, float), np.asarray(ub, float))
        Z = np.where(C.log, np.log(np.where(C.log, X0, 1.0)), X0)
        U[:len(X0)] = np.clip((Z - C.lo) / (C.hi - C.lo), 0.0, 1.0 - h)
    it0 = 0
    if resume is not None:
        U = np.array(resume['U'], float)
        if U.shape != (n_starts, d):
            raise ValueError('resume state has %s starts x groups, the search %s' % (U.shape, (n_starts, d)))
    R = residuals(C.phys(U)).astype(np.float64)
    n_eval = n_starts
    F = (R ** 2).sum(1)
    lam = np.full(n_starts, lam0)
    active = np.ones(n_starts, bool)
    small = np.zeros(n_starts, int)
    its = np.zeros(n_starts, int)
    J_last = [None] * n_starts
    free = np.ones((n_starts, d), bool)
    sel = dict(dict(every=3, S_mV=1.0, gamma_max=15.0, protect_mV=20.0, min_gain=0.05), **select) if select else None
    names = list(names) if names is not None else ['p%d' % j for j in range(d)]
    select_log = []
    history = [(n_eval, float(F.min()))]
    if resume is not None:
        it0, n_eval = int(resume['iteration']), int(resume['n_eval'])
        lam, active = np.array(resume['lam'], float), np.array(resume['active'], bool)
        small, its = np.array(resume['small'], int), np.array(resume['its'], int)
        if 'free' in resume:
            free = np.array(resume['free'], bool)
        history = [tuple(h_) for h_ in np.asarray(resume['history']).tolist()]
        if log:
            log('lm resumed after iteration %d: %d active, best sum r^2 %.4g (saved %.4g)'
                % (it0, int(active.sum()), F.min(), float(np.min(resume['F']))))
    rng = np.random.default_rng(seed + 7919)
    for it in range(it0 + 1, max_iter + 1):
        for at, keep in prune:
            if it == at:
                idx = np.flatnonzero(active)
                if len(idx) > keep:
                    gone = idx[np.argsort(F[idx])[keep:]]
                    active[gone] = False
                    if respawn:          # restart part of the pruned starts around the best point
                        b = int(np.argmin(F))
                        n_new = int(round(float(respawn.get('frac', 1.0)) * len(gone)))
                        sd = np.full(d, float(respawn.get('umax', 0.05)))
                        if J_last[b] is not None:
                            try:
                                cov = np.linalg.pinv(J_last[b].T @ J_last[b]) * F[b] / max(R.shape[1] - d, 1)
                                sd = np.sqrt(np.clip(np.diag(cov), 0.0, None)) * float(respawn.get('k', 2.0))
                            except np.linalg.LinAlgError:
                                pass
                        sd = np.clip(sd, float(respawn.get('umin', 0.003)), float(respawn.get('umax', 0.05)))
                        new = gone[:n_new]
                        if len(new):
                            U[new] = np.clip(U[b] + rng.standard_normal((len(new), d)) * sd, 0.0, 1.0 - h)
                            R[new] = residuals(C.phys(U[new])).astype(np.float64)
                            n_eval += len(new)
                            F[new] = (R[new] ** 2).sum(1)
                            lam[new], small[new], its[new], active[new] = lam0, 0, 0, True
                            J_last_b = J_last[b]
                            for s in new:
                                J_last[s] = J_last_b
                            if log:
                                log('lm respawn at iteration %d: %d of the %d pruned starts restarted around start %d '
                                    '(sum r^2 %.4g; their best %.4g), unit-cube sd %.3g..%.3g'
                                    % (it, len(new), len(gone), b, F[b], F[new].min(), sd.min(), sd.max()))
        A = np.flatnonzero(active)
        if not len(A):
            break
        check = sel is not None and it % int(sel['every']) == 0
        cols = {s: (np.arange(d) if (sel is None or check) else np.flatnonzero(free[s])) for s in A}
        P = np.vstack([U[s] + h * np.eye(d)[cols[s]] for s in A])
        Rp = residuals(C.phys(P)).astype(np.float64)
        n_eval += len(P)
        trials, owner, lams = [], [], []
        best_s = A[np.argmin(F[A])]
        k0 = 0
        for s in A:
            c = cols[s]
            J = np.zeros((R.shape[1], d))
            J[:, c] = (Rp[k0:k0 + len(c)] - R[s]).T / h                              # (M, d), frozen columns 0
            k0 += len(c)
            J_last[s] = J
            if check:
                free[s], rec = _select_free(J, C.phys(U[s:s + 1])[0], C, sel, names)
                rec.update(iteration=it, start=int(s), sum_r2=float(F[s]))
                select_log.append(rec)
                if log and s == best_s:
                    log('lm selection at iteration %d (start %d): %d of %d free, index %.1f, frozen %s'
                        % (it, s, int(free[s].sum()), d, rec['gamma'], rec['frozen']))
            on = free[s].copy()
            if sel is not None:          # projected step: a parameter on a bound pushed outward stays there
                g_all = J.T @ R[s]
                on &= ~(((U[s] <= 1e-9) & (g_all > 0)) | ((U[s] >= 1.0 - h - 1e-9) & (g_all < 0)))
            idx = np.flatnonzero(on)
            if not len(idx):
                continue
            Jf = J[:, idx]
            H, g = Jf.T @ Jf, Jf.T @ R[s]
            Dg = np.diag(np.diag(H)) + 1e-12 * np.eye(len(idx))
            for f in ladder:
                try:
                    dl = np.linalg.solve(H + lam[s] * f * Dg, -g)
                except np.linalg.LinAlgError:
                    continue
                delta = np.zeros(d)
                delta[idx] = dl
                trials.append(np.clip(U[s] + delta, 0.0, 1.0 - h))
                owner.append(s)
                lams.append(lam[s] * f)
        if not trials:
            break
        Rt = residuals(C.phys(np.array(trials))).astype(np.float64)
        n_eval += len(trials)
        Ft = (Rt ** 2).sum(1)
        owner, lams = np.array(owner), np.array(lams)
        for s in A:
            k = np.flatnonzero(owner == s)
            if not len(k):
                lam[s] *= 100.0
                continue
            b = k[np.argmin(Ft[k])]
            its[s] += 1
            if Ft[b] < F[s]:
                rel = (F[s] - Ft[b]) / max(F[s], 1e-30)
                small[s] = small[s] + 1 if rel < tol else 0
                U[s], R[s], F[s], lam[s] = trials[b], Rt[b], Ft[b], max(lams[b], 1e-9)
            else:
                lam[s] *= 100.0
            if small[s] >= 2 or lam[s] > 1e8:
                active[s] = False
        history.append((n_eval, float(F.min())))
        if log:
            log('lm iteration %d: %d active, best sum r^2 %.4g (start %d), median of the active %.4g, %d evaluations, %.0f s'
                % (it, int(active.sum()), F.min(), int(np.argmin(F)), np.median(F[A]), n_eval, time.time() - t0))
        if checkpoint is not None:
            checkpoint(dict(iteration=it, U=U.copy(), lam=lam.copy(), active=active.copy(), small=small.copy(),
                            its=its.copy(), F=F.copy(), n_eval=n_eval, history=np.array(history, float),
                            free=free.copy()))
    b = int(np.argmin(F))
    return dict(x=C.phys(U[b:b + 1])[0], f=float(F[b]), u=U[b], J=J_last[b],
                starts=[(C.phys(U[s:s + 1])[0], float(F[s]), int(its[s])) for s in range(n_starts)],
                history=history, n_eval=n_eval, wall_s=time.time() - t0,
                free=[names[j] for j in np.flatnonzero(free[b])], select_log=select_log)


def _select_free(J, x, C, sel, names):
    """The free set of one start from its full residual Jacobian J (M, d) in the unit cube, at the physical
    point x: the rule of scripts/select_groups.py on the weighted residuals (see lm_multistart, select)."""
    d = J.shape[1]
    span = C.hi - C.lo
    # d u / d ln p; a linearly mapped parameter (lb <= 0) per relative change of p, but at least 10 % of its
    # range (at p = 0 a relative change is no change: the parameter would be frozen there for good)
    dlnp = np.where(C.log, 1.0 / span, np.maximum(np.abs(x), 0.1 * span) / span)
    nrm = np.linalg.norm(J, axis=0)
    Sj = nrm * dlnp                                                 # [mV] per unit ln p (sum r^2 is a mean square)
    keep = [j for j in range(d) if Sj[j] >= sel['S_mV'] and nrm[j] > 0]
    why = {names[j]: 'S %.2f mV' % Sj[j] for j in range(d) if j not in keep}
    Jn = J / np.where(nrm > 0, nrm, 1.0)
    cos = Jn.T @ Jn

    def gam(ix):
        if len(ix) < 2:
            return 1.0
        return float(1.0 / np.sqrt(max(np.linalg.eigvalsh(cos[np.ix_(ix, ix)]).min(), 1e-300)))
    g_now = gam(keep)
    while g_now > sel['gamma_max'] and len(keep) > 1:
        trial = {j: gam([m for m in keep if m != j]) for j in keep}
        useful = [j for j in keep if trial[j] <= (1 - sel['min_gain']) * g_now]
        pool = [j for j in useful if Sj[j] <= sel['protect_mV']] or useful or list(keep)
        best = min(trial[j] for j in pool)
        drop = min([j for j in pool if trial[j] <= 1.05 * best], key=lambda j: Sj[j])
        why[names[drop]] = 'collinear %.3g -> %.3g (S %.2f mV)' % (g_now, trial[drop], Sj[drop])
        keep.remove(drop)
        g_now = trial[drop]
    free = np.zeros(d, bool)
    free[keep] = True
    return free, dict(gamma=g_now, S_mV={names[j]: float(Sj[j]) for j in range(d)}, frozen=why)
