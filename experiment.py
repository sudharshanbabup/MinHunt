"""MinHint experiments on real tutoring logs.

Usage: python experiment.py <assist|ct> [--epochs 400] [--seed_from a --seeds b]
Writes results/<dataset>_part<a>.json (seeds a..b-1); merge with merge.py
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from data import LOADERS, MASTERY_FEATS, split, K
from ladder import enumerate_plans, plan_costs, Frontier

torch.set_num_threads(1)
PLANS = enumerate_plans(K)
T, L, C = plan_costs(PLANS, K)
B = ((C == K - 1) & (np.arange(K)[None, :] < K - 1)).astype(float)   # unneeded bottom-out
LADDER, DIRECT = PLANS.index(list(range(K))), PLANS.index([K - 1])
GREEDY = [PLANS.index(list(range(n, K))) for n in range(K)]          # start at predicted need
LAM_GRID = np.concatenate([[0.0], np.geomspace(0.02, 16, 60)])


LR, WD, DROP = 5e-4, 1e-3, 0.2


def mlp(d, out, h=64):
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(DROP),
                         nn.Linear(h, h), nn.ReLU(), nn.Dropout(DROP), nn.Linear(h, out))


def standardise(parts, feats):
    mu, sd = parts[0][feats].values.mean(0), parts[0][feats].values.std(0) + 1e-8
    return [((p[feats].values - mu) / sd).astype(np.float32) for p in parts]


def train_need_net(Xtr, Ntr, Xva, Nva, epochs, seed, bs=512):
    """Need posterior q_theta(n|x): cross-entropy, Adam, cosine schedule, best-val checkpoint."""
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    net = mlp(Xtr.shape[1], K)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, Nt, Xv, Nv = map(torch.as_tensor, (Xtr, Ntr, Xva, Nva))
    ce = nn.CrossEntropyLoss(); hist = {"train": [], "val": []}; best = (1e9, None)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]; opt.zero_grad(); ce(net(Xt[idx]), Nt[idx]).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            ltr, lva = ce(net(Xt), Nt).item(), ce(net(Xv), Nv).item()
        hist["train"].append(ltr); hist["val"].append(lva)
        if lva < best[0]:
            best = (lva, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    return net, hist


def posterior(net, X):
    with torch.no_grad():
        return torch.softmax(net(torch.as_tensor(X)), 1).numpy().astype(np.float64)


def train_plan_regressor(Xtr, Ntr, Xva, Nva, epochs, seed, bs=512):
    """Unstructured baseline: regress the turns and the leakage of every plan
    separately (2 * 2^(K-1) outputs, squared loss) and pick argmin of the
    predicted Lagrangian.  It ignores the ladder structure of the costs."""
    torch.manual_seed(seed + 1000); g = torch.Generator().manual_seed(seed + 1000)
    P = len(PLANS); net = mlp(Xtr.shape[1], 2 * P)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Y = torch.as_tensor(np.hstack([T.T, L.T]), dtype=torch.float32)       # [need, 2P] targets
    Xt, Nt, Xv, Nv = map(torch.as_tensor, (Xtr, Ntr, Xva, Nva))
    mse = nn.MSELoss(); hist = {"val": []}; best = (1e9, None)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]; opt.zero_grad(); mse(net(Xt[idx]), Y[Nt[idx]]).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad():
            lv = mse(net(Xv), Y[Nv]).item()
        hist["val"].append(lv)
        if lv < best[0]:
            best = (lv, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()

    def choose_factory(X):
        with torch.no_grad():
            out = net(torch.as_tensor(X)).numpy().astype(np.float64)
        That, Lhat = out[:, :P], out[:, P:]
        return lambda lam: (lam * That + Lhat).argmin(1)
    return choose_factory, hist


def plugin_choose(P):
    """Lagrangian-optimal plan under posterior P (equals the shortest-path planner; see test_ladder)."""
    PT, PL = P @ T.T, P @ L.T
    return lambda lam: (lam * PT + PL).argmin(1)


def mats_for(N, G):
    idx = np.arange(len(N))
    return {"turns": T[:, N].T, "leak": L[:, N].T, "bottom": B[:, N].T,
            "succ": G[idx[:, None], C[:, N].T]}


def frontier_points(fr, mats):
    return [[float(fr.mean(l, mats["turns"])), float(fr.mean(l, mats["leak"]))] for l in LAM_GRID]


def lower_hull(points, grid):
    """Leakage of the best time-sharing mixture of the family at each turn budget."""
    pts = sorted(set(map(tuple, points)))
    out = []
    for t in grid:
        best = np.inf
        for (t1, l1) in pts:
            if t1 <= t + 1e-12: best = min(best, l1)
            for (t2, l2) in pts:
                if t1 < t < t2: best = min(best, l1 + (l2 - l1) * (t - t1) / (t2 - t1))
        out.append(float(best) if np.isfinite(best) else None)
    return out


def dual_ascent_trace(choose, Tm, tau, lam0, iters, seed, bs=256, eta0=0.5):
    """Stochastic projected dual subgradient ascent on the calibration set."""
    rng = np.random.default_rng(seed); lam = lam0; tr = []
    for k in range(1, iters + 1):
        tr.append(lam)
        idx = rng.integers(0, Tm.shape[0], bs)
        g = Tm[idx, choose(lam)[idx]].mean() - tau          # subgradient of the dual function
        lam = max(0.0, lam + eta0 / np.sqrt(k) * g)
    return tr


def run(ds, epochs, seeds, taus, sigmas, noise_seeds, seed_from=0):
    ev, feats = LOADERS[ds]()
    tele = [f for f in feats if f not in MASTERY_FEATS]
    res = {"dataset": ds, "n_events": len(ev), "n_learners": int(ev.u.nunique()), "n_skills": int(ev.skill.nunique()),
           "need_dist": (np.bincount(ev.level, minlength=K) / len(ev)).tolist(), "feats": feats,
           "taus": list(taus), "runs": [], "noise": [], "lam_grid": LAM_GRID.tolist()}
    for seed in range(seed_from, seeds):
        t0 = time.time()
        parts = split(ev, seed)
        Ns = [p.level.values.astype(np.int64) for p in parts]
        Xf = standardise(parts, feats); Xm = standardise(parts, MASTERY_FEATS)
        # projected learning model g(x, consumed level): next first-attempt success
        def gfeat(X, lev): return np.hstack([X, np.eye(K)[lev]])
        ok = [p.nxt.notna().values for p in parts]
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(gfeat(Xf[0][ok[0]], Ns[0][ok[0]]), parts[0].nxt.values[ok[0]].astype(int))
        g_auc = roc_auc_score(parts[2].nxt.values[ok[2]].astype(int), lr.predict_proba(gfeat(Xf[2][ok[2]], Ns[2][ok[2]]))[:, 1])
        G = [np.stack([lr.predict_proba(gfeat(X, np.full(len(X), l)))[:, 1] for l in range(K)], 1) for X in Xf]
        mats = [mats_for(N, g) for N, g in zip(Ns, G)]      # train / val / test
        # ---- models
        net, h_full = train_need_net(Xf[0], Ns[0], Xf[1], Ns[1], epochs, seed)
        netm, h_mast = train_need_net(Xm[0], Ns[0], Xm[1], Ns[1], epochs, seed)
        pol, h_pol = train_plan_regressor(Xf[0], Ns[0], Xf[1], Ns[1], epochs, seed)
        prior = np.bincount(Ns[0], minlength=K) / len(Ns[0])
        Pf = [posterior(net, X) for X in Xf]; Pm = [posterior(netm, X) for X in Xm]
        Pp = [np.tile(prior, (len(N), 1)) for N in Ns]
        nll = lambda P, N: float(-np.log(P[np.arange(len(N)), N] + 1e-12).mean())
        fams = {"minhint": [Frontier(plugin_choose(P)) for P in Pf],
                "mastery": [Frontier(plugin_choose(P)) for P in Pm],
                "prior": [Frontier(plugin_choose(P)) for P in Pp],
                "planreg": [Frontier(pol(X)) for X in Xf]}
        run_ = {"seed": seed, "sizes": [len(n) for n in Ns], "g_auc": float(g_auc),
                "curves": {"full_train": h_full["train"], "full_val": h_full["val"],
                           "mast_train": h_mast["train"], "mast_val": h_mast["val"], "planreg_val": h_pol["val"]},
                "nll_test": {"minhint": nll(Pf[2], Ns[2]), "mastery": nll(Pm[2], Ns[2]), "prior": nll(Pp[2], Ns[2])},
                "acc_test": {"minhint": float((Pf[2].argmax(1) == Ns[2]).mean()), "mastery": float((Pm[2].argmax(1) == Ns[2]).mean()),
                             "prior": float((prior.argmax() == Ns[2]).mean())},
                "frontier": {}, "at_tau": {}, "fixed": {}, "hull": {}}
        tgrid = np.round(np.arange(1.0, 2.61, 0.05), 2)
        run_["tgrid"] = tgrid.tolist()
        for nm, fr in fams.items():
            pts = frontier_points(fr[2], mats[2])
            run_["frontier"][nm] = pts
            run_["hull"][nm] = lower_hull(pts, tgrid)
            run_["at_tau"][nm] = {}
            for tau in taus:
                cal = fr[1].calibrate(tau, mats[1]["turns"])            # price fitted on validation learners
                r = fr[2].evaluate(cal, mats[2]); r["lam"] = cal[1]; r["rho"] = cal[2]
                run_["at_tau"][nm][str(tau)] = r
        te = mats[2]; n = len(Ns[2]); ar = np.arange(n)
        fixed = {"ladder": np.full(n, LADDER), "direct": np.full(n, DIRECT),
                 "greedy": np.array(GREEDY)[Pf[2].argmax(1)], "oracle": np.array(GREEDY)[Ns[2]]}
        for i, pl in enumerate(PLANS): fixed["plan_" + "".join(map(str, pl))] = np.full(n, i)
        for nm, idx in fixed.items():
            run_["fixed"][nm] = {k: float(M[ar, idx].mean()) for k, M in te.items()}
        # ---- dual ascent trace (seed 0 only)
        if seed == 0:
            tau0 = taus[len(taus) // 2]; ch = plugin_choose(Pf[1]); Tm = mats[1]["turns"]
            lo, hi, rho = fams["minhint"][1].calibrate(tau0, Tm)
            res["dual"] = {"tau": tau0, "lam_star": hi,
                           "traces": {str(l0): dual_ascent_trace(ch, Tm, tau0, l0, 3000, 0) for l0 in (0.0, 2.0)}}
        res["runs"].append(run_)
        print(ds, "seed", seed, "nll", {k: round(v, 4) for k, v in run_["nll_test"].items()}, "g_auc", round(g_auc, 3),
              "t=%.0fs" % (time.time() - t0), flush=True)
        # ---- telemetry noise study (matched train/test corruption of telemetry features)
        if noise_seeds and seed % 2 == 0:               # every other split
            ti = [feats.index(f) for f in tele]; tau = taus[len(taus) // 2]
            for sg in sigmas:
                rng = np.random.default_rng(100 + seed)
                Xn = []
                for X in Xf:
                    Y = X.copy(); Y[:, ti] = (Y[:, ti] + sg * rng.standard_normal((len(Y), len(ti)))) / np.sqrt(1 + sg ** 2)
                    Xn.append(Y.astype(np.float32))
                nn_, _ = train_need_net(Xn[0], Ns[0], Xn[1], Ns[1], epochs, seed)
                Pn = [posterior(nn_, X) for X in Xn]
                fr = [Frontier(plugin_choose(P)) for P in Pn]
                r = fr[2].evaluate(fr[1].calibrate(tau, mats[1]["turns"]), mats[2])
                r.update(sigma=sg, seed=seed, nll=nll(Pn[2], Ns[2]), tau=tau)
                res["noise"].append(r)
            print("  noise done t=%.0fs" % (time.time() - t0), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{ds}_part{seed_from}.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=10); ap.add_argument("--seed_from", type=int, default=0); ap.add_argument("--noise_seeds", type=int, default=1, help="0 disables the telemetry-noise study")
    ap.add_argument("--taus", type=float, nargs="+", default=[1.2, 1.3, 1.5])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    a = ap.parse_args()
    run(a.ds, a.epochs, a.seeds, a.taus, a.sigmas, a.noise_seeds, a.seed_from)
