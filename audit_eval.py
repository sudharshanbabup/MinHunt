"""Independent end-to-end audit of the evaluation path (second code audit).

1. Re-evaluates a calibrated MinHint policy by *simulating every dialogue turn
   by turn* with the shortest-path planner and re-planning on the truncated
   posterior, and compares with the vectorised numbers used in the paper.
2. Label-shuffle control: with needs permuted in training, the need network
   must not beat the marginal distribution (guards against feature leakage
   through the pipeline itself).
"""
import numpy as np
import experiment as E
from data import LOADERS, split, K
from ladder import dp_plan, Frontier

ds, seed, tau = "ct", 0, 1.3
ev, feats = LOADERS[ds](); parts = split(ev, seed)
Ns = [p.level.values.astype(np.int64) for p in parts]; X = E.standardise(parts, feats)
net, _ = E.train_need_net(X[0], Ns[0], X[1], Ns[1], 60, seed)
P = [E.posterior(net, x) for x in X]
fr = [Frontier(E.plugin_choose(p)) for p in P]
Tm = [E.T[:, n].T for n in Ns]; Lm = [E.L[:, n].T for n in Ns]
lo, hi, rho = fr[1].calibrate(tau, Tm[1])
vec = fr[2].evaluate((lo, hi, rho), {"turns": Tm[2], "leak": Lm[2]})

def simulate(pi, n, lam):
    """Play one dialogue: offer levels until one covers the need; re-plan each turn."""
    a, turns, post = -1, 0, pi.copy()
    while True:
        plan, _ = dp_plan(post, lam)
        b = a + 1 + plan[0] if a >= 0 else plan[0]       # plan is over the truncated ladder (levels > a)
        turns += 1
        if b >= n: return turns, b - n
        a = b; post = pi[a + 1:] / pi[a + 1:].sum()

# The bisection end points lo < hi straddle the price at which one event flips its
# plan (an exact tie); to compare the two code paths away from that tie we shift
# the prices by 1e-6 and check both paths event by event.
for lam in (lo - 1e-6, hi + 1e-6, 0.3, 2.5):
    sim = np.array([simulate(P[2][i], Ns[2][i], lam) for i in range(len(Ns[2]))])
    idx = fr[2].choose(lam); ar = np.arange(len(idx))
    assert np.array_equal(sim[:, 0], Tm[2][ar, idx]) and np.array_equal(sim[:, 1], Lm[2][ar, idx]), lam
    print("price %.4f: simulated turns %.4f leak %.4f == vectorised" % (lam, sim[:, 0].mean(), sim[:, 1].mean()))
print("calibrated policy on test learners:", vec, "(budget %.2f)" % tau)

rng = np.random.default_rng(0); Nsh = rng.permutation(Ns[0])
net2, _ = E.train_need_net(X[0], Nsh, X[1], Ns[1], 60, seed)
P2 = E.posterior(net2, X[2]); prior = np.bincount(Ns[0], minlength=K) / len(Ns[0])
nll = lambda Pm: -np.log(Pm[np.arange(len(Ns[2])), Ns[2]]).mean()
print("shuffled-label test NLL %.4f vs marginal %.4f" % (nll(P2), nll(np.tile(prior, (len(Ns[2]), 1)))))
assert nll(P2) > nll(np.tile(prior, (len(Ns[2]), 1))) - 0.01
print("evaluation audit passed")
