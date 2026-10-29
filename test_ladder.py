"""Audit tests for the paper (version_11): Proposition 1 (shortest path planner), eqs. (3)-(4),
the budget mixture of Sec. IV-B, the cost bound of Sec. IV-C, and extra checks."""
import numpy as np
from ladder import enumerate_plans, plan_costs, dp_plan

rng = np.random.default_rng(0)
for K in [2, 3, 4, 5, 6]:
    plans = enumerate_plans(K); T, L, C = plan_costs(plans, K)
    assert len(plans) == 2 ** (K - 1)
    for _ in range(300):
        pi = rng.dirichlet(np.ones(K) * rng.uniform(0.2, 2)); lam = rng.uniform(0, 3)
        w = np.triu(rng.uniform(0, 2, (K, K)), 1).T          # w[l,n] for l>n, zero diagonal
        T, L, C = plan_costs(plans, K, w)
        brute = (lam * T + L) @ pi
        plan, v = dp_plan(pi, lam, w)
        assert abs(v - brute.min()) < 1e-10, (K, v, brute.min())
        p = plans.index(plan); assert abs(brute[p] - v) < 1e-10
        # closed forms (eqs. for expected turns and leakage)
        tail = lambda a: 1.0 if a < 0 else 1 - pi[:a + 1].sum()
        prev = [-1] + plan[:-1]
        ET = sum(tail(a) for a in prev)
        EL = sum(pi[n] * w[b, n] for a, b in zip(prev, plan) for n in range(a + 1, b + 1))
        assert abs(ET - T[p] @ pi) < 1e-10 and abs(EL - L[p] @ pi) < 1e-10
        # cost bound of Sec. IV-C: plug-in plan from a perturbed posterior
        pih = rng.dirichlet(np.ones(K) * 2)
        c = lam * T + L
        ph = (c @ pih).argmin()
        gap = c[ph] @ pi - (c @ pi).min()
        bound = np.abs(pi - pih).sum() * (c.max() - c.min())
        assert gap <= bound + 1e-10
        # sharper: Delta_lambda bound used in the paper
        wmax = w.max(); assert c.max() - c.min() <= (K - 1) * lam + wmax + 1e-12
print("all ladder audits passed")

# ---- Odds rule for linear leakage (Sec. IV-A, text after Proposition 1)
for K in [3, 4, 5, 6]:
    plans = enumerate_plans(K); T, L, C = plan_costs(plans, K)
    for _ in range(400):
        pi = rng.dirichlet(np.ones(K)); lam = rng.uniform(0, 3)
        plan, _ = dp_plan(pi, lam)
        tail = lambda a: 1.0 if a < 0 else 1 - pi[:a + 1].sum()
        seq = [-1] + plan
        for a, c in zip(seq[:-1], seq[1:]):                 # levels skipped between consecutive a < c
            for b in range(a + 1, c):
                assert (c - b) * pi[a + 1:b + 1].sum() <= lam * tail(b) + 1e-9
        for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):   # offered level b with neighbours a < b < c
            assert (c - b) * pi[a + 1:b + 1].sum() >= lam * tail(b) - 1e-9
print("odds rule audit passed")

# ---- Sec. IV-B: two-policy mixture attains the LP optimum of the budgeted problem
from scipy.optimize import linprog
from ladder import Frontier
K = 4; plans = enumerate_plans(K); T, L, C = plan_costs(plans, K); P = len(plans)
for trial in range(20):
    n = 60; PI = rng.dirichlet(np.ones(K) * 0.7, n)         # true posteriors of n equally likely contexts
    ET, EL = PI @ T.T, PI @ L.T                              # expected turns / leakage per context and plan
    tau = rng.uniform(1.1, 1.0 + (PI @ np.arange(K)).mean() - 0.05)
    A_eq = np.kron(np.eye(n), np.ones((1, P)))
    lp = linprog(EL.ravel() / n, A_ub=[ET.ravel() / n], b_ub=[tau], A_eq=A_eq, b_eq=np.ones(n), bounds=(0, 1))
    fr = Frontier(lambda lam: (lam * ET + EL).argmin(1))
    cal = fr.calibrate(tau, ET)
    r = fr.evaluate(cal, {"turns": ET, "leak": EL})
    assert abs(r["turns"] - tau) < 1e-6 and abs(r["leak"] - lp.fun) < 1e-6, (r, lp.fun, tau)
    # turns of the Lagrangian policy are non-increasing in the price (true posterior)
    tt = [fr.mean(l, ET) for l in np.linspace(0, 5, 200)]; assert all(np.diff(tt) <= 1e-12)
print("duality audit passed (mixture == LP optimum)")

# ---- Sec. IV-C cost bound with Pinsker (regret <= Delta * sqrt(2 KL)) incl. the Pinsker step, exact expectation over contexts
for trial in range(200):
    K = int(rng.integers(2, 7)); plans = enumerate_plans(K); T, L, C = plan_costs(plans, K); lam = rng.uniform(0, 3)
    n = 30; PI = rng.dirichlet(np.ones(K), n); PH = rng.dirichlet(np.ones(K) * 3, n)
    c = lam * T + L
    ph = (PH @ c.T).argmin(1); true_cost = PI @ c.T
    regret = (true_cost[np.arange(n), ph] - true_cost.min(1)).mean()
    kl = (PI * np.log(PI / PH)).sum(1).mean()                # = excess cross entropy
    assert regret <= ((K - 1) * lam + L.max()) * np.sqrt(2 * kl) + 1e-12
print("plug-in regret audit passed")

# ---- Re-planning (Sec. IV-A, text after Proposition 1): the tail of an optimal plan is optimal for the truncated posterior
for _ in range(500):
    K = int(rng.integers(3, 7)); pi = rng.dirichlet(np.ones(K)); lam = rng.uniform(0, 3)
    plans = enumerate_plans(K); T, L, C = plan_costs(plans, K)
    plan, _ = dp_plan(pi, lam)
    for j, a in enumerate(plan[:-1]):
        if pi[a + 1:].sum() < 1e-9: continue
        post = np.where(np.arange(K) > a, pi, 0.0); post /= post.sum()
        cost = (lam * T + L) @ post
        tail_plan = plan[j + 1:]                              # remaining levels after an unresolved turn at a
        feas = [i for i, p in enumerate(plans) if p[0] > a]   # plans that only use levels above a
        assert abs(cost[plans.index(tail_plan)] - cost[feas].min()) < 1e-10
print("re-planning audit passed")

# ---- Value of telemetry (Sec. IV-C): refining the context never raises the optimal leakage
from scipy.optimize import linprog
def lp_opt(PI, wts, tau, T, L):
    n, P = len(PI), T.shape[0]
    ET, EL = PI @ T.T, PI @ L.T
    A_eq = np.kron(np.eye(n), np.ones((1, P)))
    r = linprog((EL * wts[:, None]).ravel(), A_ub=[(ET * wts[:, None]).ravel()], b_ub=[tau],
                A_eq=A_eq, b_eq=np.ones(n), bounds=(0, 1))
    return r.fun
K = 4; plans = enumerate_plans(K); T, L, C = plan_costs(plans, K)
for trial in range(40):
    n = 24; fine = rng.dirichlet(np.ones(K) * 0.6, n); wf = np.full(n, 1.0 / n)
    groups = rng.integers(0, 4, n)                               # coarse context = function of the fine one
    coarse = np.array([fine[groups == g].mean(0) for g in range(4) if (groups == g).any()])
    wc = np.array([(groups == g).mean() for g in range(4) if (groups == g).any()])
    pooled = (fine * wf[:, None]).sum(0, keepdims=True)          # no context at all
    tau = rng.uniform(1.1, 1.0 + (pooled @ np.arange(K))[0] - 0.05)
    a, b, c = lp_opt(fine, wf, tau, T, L), lp_opt(coarse, wc, tau, T, L), lp_opt(pooled, np.ones(1), tau, T, L)
    assert a <= b + 1e-9 <= c + 2e-9, (a, b, c)
print("value of context audit passed (fine <= coarse <= pooled)")

# ---- Extra check, not stated in the paper: tail form of the priced cost and the tail calibration bound
for _ in range(3000):
    K = int(rng.integers(2, 7)); plans = enumerate_plans(K); T, L, C = plan_costs(plans, K)
    pi = rng.dirichlet(np.ones(K)); lam = rng.uniform(0, 3)
    Fb = lambda q, a: 1 - q[:a + 1].sum()
    EN = pi @ np.arange(K)
    for p, pl in enumerate(plans):
        tail = lam + pl[0] - EN + sum((lam + pl[j] - pl[j - 1]) * Fb(pi, pl[j - 1]) for j in range(1, len(pl)))
        assert abs(tail - (lam * T[p] + L[p]) @ pi) < 1e-10
    pih = rng.dirichlet(np.ones(K) * 2)
    eps = max(abs(Fb(pi, a) - Fb(pih, a)) for a in range(K - 1)) if K > 1 else 0.0
    c = lam * T + L
    ph = (c @ pih).argmin()
    assert c[ph] @ pi - (c @ pi).min() <= 2 * eps * (K - 1) * (1 + lam) + 1e-10
print("tail form and tail calibration audit passed")

# ---- Extra check, not stated in the paper: likelihood ratio dominance never lowers the first hint
viol = 0
for _ in range(20000):
    K = int(rng.integers(3, 7)); lam = rng.uniform(0, 3)
    a = rng.dirichlet(np.ones(K)); r = np.sort(rng.uniform(0.2, 5, K)); b = a * r; b /= b.sum()
    viol += dp_plan(b, lam)[0][0] < dp_plan(a, lam)[0][0]
assert viol == 0, viol
print("monotone first hint check passed (20000 random pairs, 0 violations)")
