"""Event construction for the hint-ladder experiments.

Each dataset is turned into a table of *decision events*: a learner has just
submitted an incorrect first attempt, and the tutor must decide how much help
to give.  The logged depth of help the learner ended up consuming is the
assistance need N in {0,...,K-1} (K = 4).  All features are computed from
information available *before* the help decision.
"""
import os
import numpy as np
import pandas as pd

K = 4  # number of hint levels: 0 = reflective question ... 3 = bottom-out


def _history(df, u, k):
    """Causal (strictly past) per-learner and per-skill history features."""
    g = df.groupby([u, k], sort=False)
    df["n"] = g.cumcount()                                   # past opportunities on the skill
    df["cs"] = g["correct"].cumsum() - df["correct"]         # past first-attempt successes
    df["m"] = (df["cs"] + 1.0) / (df["n"] + 2.0)             # Laplace-smoothed mastery estimate
    df["fail"] = 1 - df["correct"]
    gu = df.groupby(u, sort=False)
    df["un"] = gu.cumcount()                                 # past items of the learner
    df["uf"] = gu["fail"].cumsum() - df["fail"]              # past failures of the learner
    df["ulev"] = gu["level"].cumsum() - df["level"]          # past consumed help of the learner
    df["ufr"] = (df["uf"] + 1.0) / (df["un"] + 2.0)          # learner failure rate
    df["uhr"] = (df["ulev"] + 0.5) / ((K - 1) * (df["uf"] + 1.0))  # learner mean past need (0..1)
    df["pneed"] = g["level"].shift(1).fillna(0.0)            # help consumed on previous item of skill
    df["nxt"] = g["correct"].shift(-1)                       # next first-attempt outcome on the skill
    df["logn"] = np.log1p(df["n"])
    df["logun"] = np.log1p(df["un"])
    return df


def load_assist(path="data/as.csv"):
    if not os.path.exists(path) and os.path.exists("as.csv"):
        path = "as.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run 'python download_data.py' to download."
        )
    a = pd.read_csv(path, encoding="latin1", low_memory=False)
    scaffolded = set(a[a.original == 0].assistment_id)       # items that also offer scaffolding help
    a = a[(a.original == 1) & (a.tutor_mode == "tutor")].dropna(subset=["skill_name"])
    a = a.drop_duplicates("order_id").sort_values("order_id").reset_index(drop=True)
    hc = a.hint_count.fillna(0).values
    ht = a.hint_total.fillna(0).values
    bottom = (a.bottom_hint.fillna(0).values == 1) | ((ht > 0) & (hc >= ht))
    frac = np.divide(hc, np.maximum(ht, 1))
    lev = np.where(hc == 0, 0, np.where(bottom, 3, np.where(frac <= 0.5, 1, 2)))
    a["level"] = lev * (1 - a.correct.values)                # help consumed (0 on a correct first attempt)
    a = _history(a, "user_id", "skill_name")
    a["lt"] = np.log1p(a.ms_first_response.clip(0, 6e5) / 1000.0)   # latency of the first response
    a["ht"] = a.hint_total.clip(0, 8)
    a["u"] = a.user_id.astype(str)
    a["skill"] = a.skill_name.astype(str)
    # decision events: wrong first *attempt* on an item that offers a full 4-level ladder
    # (hints are then the only help channel, so the logged hint depth is the help consumed)
    ev = a[(a.correct == 0) & (a.first_action == 0) & (a.hint_total >= 3)
           & (~a.assistment_id.isin(scaffolded))].copy()
    feats = ["m", "logn", "sd", "lt", "ufr", "uhr", "pneed", "logun", "ht"]
    return ev, feats


def load_ct(path="data/ct.csv"):
    if not os.path.exists(path) and os.path.exists("ct.csv"):
        path = "ct.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Run 'python download_data.py' to download."
        )
    c = pd.read_csv(path, encoding="latin1")
    c = c.sort_values(["Anon Student Id", "Row"]).reset_index(drop=True)
    c["correct"] = c["Correct First Attempt"].astype(int)
    c["level"] = np.minimum(c.Hints.fillna(0).astype(int), K - 1) * (1 - c.correct)
    c = _history(c, "Anon Student Id", "KC(Default)")
    t = (pd.to_datetime(c["First Transaction Time"]) - pd.to_datetime(c["Step Start Time"])).dt.total_seconds()
    c["lt"] = np.log1p(t.clip(0, 600).fillna(0))
    c["u"] = c["Anon Student Id"].astype(str)
    c["skill"] = c["KC(Default)"].astype(str)
    ev = c[(c.correct == 0) & (c.Incorrects >= 1)].copy()    # at least one wrong submission
    feats = ["m", "logn", "sd", "lt", "ufr", "uhr", "pneed", "logun"]
    return ev, feats


MASTERY_FEATS = ["m", "logn", "sd"]      # what a knowledge-tracing-only tutor sees
LOADERS = {"assist": load_assist, "ct": load_ct}


def split(ev, seed, frac=(0.70, 0.15, 0.15)):
    """Learner-level split; skill difficulty 'sd' is estimated on train only."""
    us = np.array(sorted(ev.u.unique()))
    rng = np.random.default_rng(seed)
    us = us[rng.permutation(len(us))]
    a, b = int(frac[0] * len(us)), int((frac[0] + frac[1]) * len(us))
    parts = [set(us[:a]), set(us[a:b]), set(us[b:])]
    tr = ev[ev.u.isin(parts[0])]
    glob = (tr.level > 0).mean()
    grp = tr.groupby("skill")["level"].agg(lambda v: ((v > 0).sum() + 5 * glob) / (len(v) + 5))
    out = []
    for p in parts:
        d = ev[ev.u.isin(p)].copy()
        d["sd"] = d.skill.map(grp).fillna(glob).values
        out.append(d)
    return out


if __name__ == "__main__":
    for nm, f in LOADERS.items():
        ev, feats = f()
        print(nm, "events", len(ev), "learners", ev.u.nunique(), "skills", ev.skill.nunique(),
              "need dist", np.round(np.bincount(ev.level, minlength=K) / len(ev), 3),
              "next-obs", ev.nxt.notna().mean().round(3))
