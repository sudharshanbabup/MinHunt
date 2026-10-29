"""Hint-ladder plans, their costs, the shortest-path planner and the dual solver."""
import itertools
import numpy as np


def enumerate_plans(K):
    """All escalation plans: increasing level sequences that end at K-1."""
    plans = []
    for r in range(K):
        for sub in itertools.combinations(range(K - 1), r):
            plans.append(list(sub) + [K - 1])
    return plans


def plan_costs(plans, K, w=None):
    """T[p,n] = dialogue turns, L[p,n] = leakage, C[p,n] = consumed level,
    when plan p meets a learner whose need is n.  w[l,n] is the leakage weight
    of consuming level l >= n (default l - n)."""
    if w is None:
        w = np.maximum(np.arange(K)[:, None] - np.arange(K)[None, :], 0).astype(float)
    P = len(plans)
    T = np.zeros((P, K)); L = np.zeros((P, K)); C = np.zeros((P, K), dtype=int)
    for p, plan in enumerate(plans):
        for n in range(K):
            for j, l in enumerate(plan):
                if l >= n:                       # first offered level that covers the need
                    T[p, n] = j + 1; L[p, n] = w[l, n]; C[p, n] = l
                    break
    return T, L, C


def dp_plan(pi, lam, w=None):
    """Shortest-path planner (Proposition 1).  pi: need posterior (K,).
    Returns (optimal plan, optimal Lagrangian cost) in O(K^2)."""
    K = len(pi)
    if w is None:
        w = np.maximum(np.arange(K)[:, None] - np.arange(K)[None, :], 0).astype(float)
    tail = np.concatenate([[1.0], 1.0 - np.cumsum(pi)])     # tail[a+1] = P(N > a), a = -1..K-1
    V = np.full(K + 1, np.inf); nxt = np.zeros(K + 1, dtype=int)
    V[K] = 0.0                                              # node index a+1; node K = level K-1 reached
    for a in range(K - 2, -2, -1):                          # current level a (-1 = nothing offered yet)
        for b in range(a + 1, K):
            edge = lam * tail[a + 1] + sum(pi[n] * w[b, n] for n in range(a + 1, b + 1))
            val = edge + V[b + 1]
            if val < V[a + 1] - 1e-15:
                V[a + 1] = val; nxt[a + 1] = b
    plan, a = [], -1
    while a < K - 1:
        a = nxt[a + 1]; plan.append(int(a))
    return plan, V[0]


class Frontier:
    """Lagrangian policy family {plan(x; lam)}.

    choose(lam) returns one plan index per event.  Metric matrices are given per
    event and plan, M[i, p] (e.g. turns of plan p on event i)."""

    def __init__(self, choose):
        self.choose = choose

    def mean(self, lam, M):
        return M[np.arange(M.shape[0]), self.choose(lam)].mean()

    def calibrate(self, tau, Tm, hi=64.0, iters=60):
        """Dual solution on calibration data: lam_lo (largest price whose mean
        turns exceed tau), lam_hi (smallest price meeting tau) and the weight
        rho on lam_lo that makes the mixture meet tau with equality."""
        if self.mean(0.0, Tm) <= tau:
            return 0.0, 0.0, 0.0
        if self.mean(hi, Tm) > tau:        # budget unreachable for this family
            return hi, hi, 0.0
        lo = 0.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.mean(mid, Tm) > tau: lo = mid
            else: hi = mid
        t_lo, t_hi = self.mean(lo, Tm), self.mean(hi, Tm)
        rho = 0.0 if t_lo <= t_hi else (tau - t_hi) / (t_lo - t_hi)
        return lo, hi, float(np.clip(rho, 0, 1))

    def evaluate(self, cal, mats):
        """Exact expectation (over the mixing coin) of each metric in mats."""
        lo, hi, rho = cal
        return {k: float(rho * self.mean(lo, M) + (1 - rho) * self.mean(hi, M)) for k, M in mats.items()}
