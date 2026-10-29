"""Turn results/*.json into panel JSONs for matlab_fig_1 and a LaTeX table."""
import json, os, numpy as np
from scipy import stats
from data import LOADERS, K

OUT = "../matlab/data"; os.makedirs(OUT, exist_ok=True)
BLUE, VERM, GREEN, PURP, GREY = [0, .447, .698], [.835, .369, 0], [0, .620, .451], [.482, .196, .580], [.35, .35, .35]
PAL = {"minhint": dict(label="MinHint (proposed)", color=BLUE, marker="o", line="-", proposed=True),
       "mastery": dict(label="Mastery only", color=VERM, marker="s", line="--"),
       "planreg": dict(label="Plan regression", color=GREEN, marker="d", line="-."),
       "prior": dict(label="Time sharing", color=PURP, marker="^", line=":"),
       "greedy": dict(label="Classify then ladder", color=GREY, marker="p", line="none")}
os.makedirs("../matlab", exist_ok=True)
json.dump(PAL, open("../matlab/palette.json", "w"), indent=1)
NAMES = {"assist": "ASSISTments", "ct": "Cognitive Tutor"}
R = {ds: json.load(open(f"results/{ds}.json")) for ds in NAMES}
TAU = "1.3"
dump = lambda nm, d: json.dump(dict(d, height_in=1.85), open(f"{OUT}/{nm}.json", "w"), indent=1)   # all panels share the trio layout, 1.85 in tall
mean = lambda ds, f: np.mean([f(r) for r in R[ds]["runs"]], 0)

for tag, ds in zip("ab", NAMES):
    runs = R[ds]["runs"]; ep = list(range(1, len(runs[0]["curves"]["full_train"]) + 1))
    cur = {k: mean(ds, lambda r: r["curves"][k]) for k in runs[0]["curves"]}
    lo = min(cur["full_train"].min(), cur["full_val"].min()); hi = cur["mast_val"][4]
    dump(f"fig2{tag}_training_{ds}", {
        "name": f"fig2{tag}_training_{ds}", "layout": "trio", "xlabel": "Epoch", "ylabel": "Cross entropy (nats)",
        "xscale": "log", "xlim": [1, len(ep)], "xticks": [1, 10, 100, len(ep)], "xticklabels": ["1", "10", "100", str(len(ep))],
        "ylim": [float(np.floor(lo * 50) / 50 - 0.02), float(np.ceil(cur["mast_val"][0] * 20) / 20)], "ytickformat": "%.2f",
        "series": [
            {"label": "Full, training", "x": ep, "y": cur["full_train"].tolist(), "marker": "none", "color": BLUE, "line": "-", "proposed": True},
            {"label": "Full, validation", "x": ep, "y": cur["full_val"].tolist(), "marker": "none", "color": GREEN, "line": "--"},
            {"label": "Mastery, validation", "x": ep, "y": cur["mast_val"].tolist(), "marker": "none", "color": VERM, "line": "-."},
            {"kind": "hline", "y": float(mean(ds, lambda r: r["nll_test"]["prior"])), "line": ":", "label": "Marginal"}]})
    # frontier: leakage of each family at matched turn budgets (test learners)
    tg = np.array(runs[0]["tgrid"]); lad = mean(ds, lambda r: r["fixed"]["ladder"]["turns"])
    Hm = np.array([[np.nan if v is None else v for v in r["hull"]["minhint"]] for r in runs]).mean(0)
    sel = (tg >= 1.05) & (np.nan_to_num(Hm, nan=0.0) >= 0.03)       # log axis: stop before leakage reaches zero
    ser = []
    for nm in ["minhint", "mastery", "planreg", "prior"]:
        H = np.array([[np.nan if v is None else v for v in r["hull"][nm]] for r in runs])
        ok = sel & ~np.isnan(H).any(0)
        ser.append(dict(PAL[nm], x=tg[ok].tolist(), y=H[:, ok].mean(0).tolist()))
    g = [mean(ds, lambda r: r["fixed"]["greedy"][k]) for k in ("turns", "leak")]
    ser.append(dict(PAL["greedy"], x=[float(g[0])], y=[float(g[1])]))
    xmax = float(tg[sel].max() + 0.05)
    dump(f"fig3{tag}_frontier_{ds}", {"name": f"fig3{tag}_frontier_{ds}", "layout": "trio",
         "xlabel": "Mean turns per failed attempt", "ylabel": "Mean leakage (hint levels)",
         "xlim": [1.0, xmax], "yscale": "log", "ylim": [0.0004, 3], "yticks": [0.001, 0.01, 0.1, 1],
         "yticklabels": ["0.001", "0.01", "0.1", "1"], "legend": "southwest", "lock_ylim": True, "series": ser})

# dual ascent trace (ASSISTments, split 0)
d = R["assist"]["dual"]; it = list(range(1, len(d["traces"]["0.0"]) + 1)); step = 10
dump("fig2c_dual", {"name": "fig2c_dual", "layout": "trio", "xlabel": "Dual iteration {\\itk}", "ylabel": "Turn price {\\it\\lambda}",
     "xscale": "log", "xlim": [1, len(it)], "xticks": [1, 10, 100, 1000], "xticklabels": ["1", "10", "100", "1000"],
     "series": [{"label": "{\\it\\lambda}_1 = 0", "x": it, "y": d["traces"]["0.0"], "marker": "none", "color": BLUE, "line": "-", "proposed": True},
                {"label": "{\\it\\lambda}_1 = 2", "x": it, "y": d["traces"]["2.0"], "marker": "none", "color": VERM, "line": "--"},
                {"kind": "hline", "y": d["lam_star"], "line": ":", "label": "Bisection"}]})

# telemetry-noise study (ASSISTments)
ds = "assist"; runs = R[ds]["runs"]; nz = R[ds]["noise"]; seeds = sorted(set(x["seed"] for x in nz))
sig = [0.0] + sorted(set(x["sigma"] for x in nz))
base = [r["at_tau"]["minhint"][TAU]["leak"] for r in runs if r["seed"] in seeds]
ys = [np.mean(base)] + [np.mean([x["leak"] for x in nz if x["sigma"] == s]) for s in sig[1:]]
es = [np.std(base)] + [np.std([x["leak"] for x in nz if x["sigma"] == s]) for s in sig[1:]]
ref = {nm: float(np.mean([r["at_tau"][nm][TAU]["leak"] for r in runs if r["seed"] in seeds])) for nm in ("mastery", "prior")}
dump("fig3c_noise", {"name": "fig3c_noise", "layout": "trio", "xlabel": "Telemetry noise level {\\it\\sigma}", "ylabel": "Mean leakage (hint levels)",
     "xlim": [-0.3, len(sig) - 0.7], "ylim": [0.28, 0.44], "xticks": list(range(len(sig))), "xticklabels": ["%g" % s for s in sig], "ytickformat": "%.2f",
     "series": [dict(PAL["minhint"], kind="errorbar", x=list(range(len(sig))), y=[float(v) for v in ys], err=[float(v) for v in es]),
                {"kind": "hline", "y": ref["mastery"], "line": "--", "color": VERM, "label": "Mastery only"}]})

# ---------------- numbers for the paper
S = {}
def ms(v): return "%.3f $\\pm$ %.3f" % (np.mean(v), np.std(v))
rows = [("Direct answer", "fixed", "direct"), ("Full ladder", "fixed", "ladder"), ("Classify then ladder", "fixed", "greedy"),
        ("Time sharing", "tau", "prior"), ("Mastery only", "tau", "mastery"), ("Plan regression", "tau", "planreg"),
        ("MinHint (proposed)", "tau", "minhint"), ("Oracle", "fixed", "oracle")]
tex = []
for lab, kind, key in rows:
    cells = []
    for ds in NAMES:
        get = (lambda r, m: r["fixed"][key][m]) if kind == "fixed" else (lambda r, m: r["at_tau"][key][TAU][m])
        for m in ("turns", "leak", "bottom"):
            v = [get(r, m) * (100 if m == "bottom" else 1) for r in R[ds]["runs"]]
            cells.append(("%.1f" if m == "bottom" else "%.3f") % np.mean(v) + (" {\\scriptsize$\\pm$ " + ("%.1f" if m == "bottom" else "%.3f") % np.std(v) + "}" if m == "leak" else ""))
            S[f"{ds}.{key}.{m}"] = [float(np.mean(v)), float(np.std(v))]
        S[f"{ds}.{key}.succ"] = float(np.mean([get(r, "succ") for r in R[ds]["runs"]]))
    tex.append(("\\textbf{%s}" % lab if key == "minhint" else lab) + " & " + " & ".join(cells) + " \\\\")
os.makedirs("results", exist_ok=True)
open("results/table_main.tex", "w").write("\n".join(tex) + "\n")
for ds in NAMES:
    runs = R[ds]["runs"]
    S[f"{ds}.meta"] = {k: R[ds][k] for k in ("n_events", "n_learners", "n_skills", "need_dist")}
    S[f"{ds}.sizes"] = np.mean([r["sizes"] for r in runs], 0).tolist()
    for k in ("minhint", "mastery", "prior"):
        S[f"{ds}.nll.{k}"] = [float(np.mean([r["nll_test"][k] for r in runs])), float(np.std([r["nll_test"][k] for r in runs]))]
        S[f"{ds}.acc.{k}"] = float(np.mean([r["acc_test"][k] for r in runs]))
    S[f"{ds}.g_auc"] = float(np.mean([r["g_auc"] for r in runs]))
    for tau in R[ds]["taus"]:
        a = np.array([r["at_tau"]["minhint"][str(tau)]["leak"] for r in runs])
        for other in ("prior", "mastery", "planreg"):
            b = np.array([r["at_tau"][other][str(tau)]["leak"] for r in runs])
            S[f"{ds}.tau{tau}.vs_{other}"] = {"minhint": float(a.mean()), "other": float(b.mean()), "rel_reduction_pct": float(100 * (1 - a.mean() / b.mean())),
                                              "wins": int((a < b).sum()), "n": len(a), "paired_t_p": float(stats.ttest_rel(a, b).pvalue),
                                              "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue)}
        S[f"{ds}.tau{tau}.turns"] = {nm: float(np.mean([r["at_tau"][nm][str(tau)]["turns"] for r in runs])) for nm in ("minhint", "mastery", "prior", "planreg")}
        S[f"{ds}.tau{tau}.lam"] = float(np.mean([r["at_tau"]["minhint"][str(tau)]["lam"] for r in runs]))
    # turns needed by each family to reach the leakage of the greedy classifier (equal-leak comparison)
    S[f"{ds}.dual"] = {"lam_star": R[ds]["dual"]["lam_star"], "last": {k: float(np.mean(v[-200:])) for k, v in R[ds]["dual"]["traces"].items()}}
    ev, _ = LOADERS[ds]()
    S[f"{ds}.next_success_by_level"] = ev.dropna(subset=["nxt"]).groupby("level").nxt.mean().round(3).tolist()
S["noise"] = {"sigma": sig, "leak": [float(v) for v in ys], "std": [float(v) for v in es], "ref": ref, "seeds": seeds}
json.dump(S, open("results/summary.json", "w"), indent=1)
print(open("results/table_main.tex").read())

# ---------------- LaTeX macros (every number quoted in the paper text)
from experiment import lower_hull
M = {}
for P_, ds in (("AS", "assist"), ("CT", "ct")):
    runs = R[ds]["runs"]; meta = S[f"{ds}.meta"]
    M[P_ + "events"] = "{:,}".format(meta["n_events"]).replace(",", "{,}")
    M[P_ + "learners"] = "{:,}".format(meta["n_learners"]).replace(",", "{,}"); M[P_ + "skills"] = str(meta["n_skills"])
    for c_, v in zip("ABCD", meta["need_dist"]): M[P_ + "need" + c_] = "%.1f" % (100 * v)
    for nm, k in (("Full", "minhint"), ("Mast", "mastery"), ("Prior", "prior")): M[P_ + "nll" + nm] = "%.3f" % S[f"{ds}.nll.{k}"][0]
    M[P_ + "accFull"] = "%.1f" % (100 * S[f"{ds}.acc.minhint"])
    M[P_ + "ladT"] = "%.2f" % S[f"{ds}.ladder.turns"][0]; M[P_ + "dirL"] = "%.2f" % S[f"{ds}.direct.leak"][0]
    M[P_ + "dirB"] = "%.0f" % S[f"{ds}.direct.bottom"][0]
    for nm, k in (("Prior", "prior"), ("Mast", "mastery")):
        c = S[f"{ds}.tau{TAU}.vs_{k}"]; M[P_ + "red" + nm] = "%.0f" % c["rel_reduction_pct"]
        M[P_ + "wins" + nm] = {10: "ten", 9: "nine", 8: "eight", 7: "seven", 6: "six"}.get(c["wins"], str(c["wins"]))
        M[P_ + "p" + nm] = ("=%.3f" % c["paired_t_p"]) if c["paired_t_p"] >= 0.001 else "<0.001"
    c = S[f"{ds}.tau{TAU}.vs_planreg"]; M[P_ + "redPR"] = "%.0f" % (100 * (c["other"] / c["minhint"] - 1))
    S[f"{ds}.planreg_wins_for_minhint"] = c["wins"]
    g_at = [lower_hull(r["frontier"]["minhint"], [r["fixed"]["greedy"]["turns"]])[0] for r in runs]
    M[P_ + "greedyMH"] = "%.3f" % np.mean(g_at); M[P_ + "greedyL"] = "%.3f" % S[f"{ds}.greedy.leak"][0]
    S[f"{ds}.greedy_cmp"] = {"minhint_at_greedy_turns": float(np.mean(g_at)), "wins": int(sum(a < r["fixed"]["greedy"]["leak"] for a, r in zip(g_at, runs)))}
    M[P_ + "nsZero"] = "%.1f" % (100 * S[f"{ds}.next_success_by_level"][0]); M[P_ + "nsThree"] = "%.1f" % (100 * S[f"{ds}.next_success_by_level"][3])
    M[P_ + "lamStar"] = "%.2f" % S[f"{ds}.dual"]["lam_star"]
dev = max(abs(S[f"{ds}.tau{TAU}.turns"][k] - float(TAU)) for ds in NAMES for k in ("minhint", "mastery", "prior", "planreg"))
M["maxTurnDev"] = "%.2f" % (np.ceil(dev * 100) / 100)
M["noisePrior"] = "%.3f" % S["noise"]["ref"]["prior"]; M["noiseMast"] = "%.3f" % S["noise"]["ref"]["mastery"]
# Cost bound of Sec. IV-C in data: leakage versus test cross entropy across the noise runs (and their clean runs)
_L = [x["leak"] for x in nz] + [r["at_tau"]["minhint"][TAU]["leak"] for r in R["assist"]["runs"] if r["seed"] in seeds]
_N = [x["nll"] for x in nz] + [r["nll_test"]["minhint"] for r in R["assist"]["runs"] if r["seed"] in seeds]
_rho = stats.pearsonr(_N, _L)[0]; _slope = np.polyfit(_N, _L, 1)[0]
S["noise_ce_vs_leak"] = {"n": len(_L), "pearson_r": float(_rho), "slope_levels_per_nat": float(_slope)}
M["noiseCorr"] = "%.2f" % _rho; M["noiseSlope"] = "%.1f" % _slope; M["noiseRuns"] = str(len(_L))
M["noiseZero"] = "%.3f" % S["noise"]["leak"][0]; M["noiseFour"] = "%.3f" % S["noise"]["leak"][-1]
os.makedirs("../paper", exist_ok=True)
open("../paper/numbers.tex", "w").write("% generated by code/export_figs.py -- do not edit\n" + "".join("\\newcommand{\\%s}{%s}\n" % kv for kv in M.items()))
open("../paper/numbers.tex", "a").write("\\newcommand{\\MainTableRows}{%\n" + open("results/table_main.tex").read() + "}\n")
json.dump(S, open("results/summary.json", "w"), indent=1)
print(M)
