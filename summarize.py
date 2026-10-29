import json, sys, numpy as np
def ms(v): return "%.3f±%.3f" % (np.mean(v), np.std(v))
for ds in sys.argv[1:]:
    r = json.load(open(f"results/{ds}.json")); R = r["runs"]
    print("==", ds, r["n_events"], r["n_learners"], r["n_skills"], np.round(r["need_dist"], 3), "sizes", R[0]["sizes"])
    for k in ["full_train", "full_val", "mast_val", "planreg_val"]:
        c = np.mean([x["curves"][k] for x in R], 0); print(" curve", k, np.round(c[[0, 4, 9, 24, 49, 99, 199, -1]], 4), "argmin", c.argmin())
    print(" nll", {k: ms([x["nll_test"][k] for x in R]) for k in R[0]["nll_test"]}, "acc", {k: ms([x["acc_test"][k] for x in R]) for k in R[0]["acc_test"]}, "gauc", ms([x["g_auc"] for x in R]))
    for nm in ["ladder", "direct", "greedy", "oracle"]:
        print(" ", nm, {k: ms([x["fixed"][nm][k] for x in R]) for k in ["turns", "leak", "bottom", "succ"]})
    for tau in r["taus"]:
        for nm in R[0]["at_tau"]:
            print("  tau", tau, nm, {k: ms([x["at_tau"][nm][str(tau)][k] for x in R]) for k in ["turns", "leak", "bottom", "succ", "lam"]})
    tg = np.array(R[0]["tgrid"])
    for nm in R[0]["hull"]:
        H = np.array([[np.nan if v is None else v for v in x["hull"][nm]] for x in R]); print("  hull", nm, np.round(np.nanmean(H, 0)[::3], 3))
    print("  tgrid", tg[::3])
    if r["noise"]:
        for sg in sorted(set(x["sigma"] for x in r["noise"])):
            xs = [x for x in r["noise"] if x["sigma"] == sg]; print("  noise", sg, ms([x["leak"] for x in xs]), ms([x["turns"] for x in xs]), ms([x["nll"] for x in xs]))
    if "dual" in r: print("  dual lam*", r["dual"]["lam_star"], {k: np.round(np.array(v)[[0, 10, 100, 500, 1000, 2999]], 3) for k, v in r["dual"]["traces"].items()})
