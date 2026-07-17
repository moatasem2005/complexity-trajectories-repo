#!/usr/bin/env python3
"""
evaluate_cv.py — final evaluation for the Complexity-Debt Trajectories study.

WHY THIS EXISTS
---------------
The single 75/25 split used in the notebook proved fragile: it reported
ours=0.701 on 7 repos, then 0.624 on 9 repos, and in one sensitivity cell a
baseline appeared to beat the model. Under repeated cross-validation that
"win" vanished — it was split noise. Concordance from ONE split is not a
result. This script replaces it with:

  * repeated stratified k-fold CV (10 x 5-fold by default)
  * paired per-fold differences (ours - best baseline)
  * percentile CIs on the difference; the claim stands only if the CI excludes 0
  * the same sweep across landmark size L and growth threshold DELTA

Input : complexity_long.csv  (from the mining notebook)
Output: results_table.csv, sensitivity_cv.csv

Run:    python evaluate_cv.py --input complexity_long.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize

COV = ["cc_lm", "cc_slope", "loc_lm", "loc_delta", "mi_lm",
       "nfunc_lm", "hv_lm", "n_authors", "lm_days"]


# ---------------------------------------------------------------- data
def build_landmark(df, L, delta):
    """Landmark design: covariates from the first L commits; the clock starts
    at the end of the landmark. Fixed-size window => no covariate can encode
    the outcome (this is what killed `commit_rate`, whose denominator WAS the
    observation span: corr(span, duration) was exactly 1.00)."""
    recs = []
    for (repo, path), g in df.groupby(["repo", "file_path"], sort=False):
        g = g.reset_index(drop=True)
        if len(g) < L + 2:
            continue
        init = g["max_cc"].iloc[0]
        target = init + delta
        lm = g.iloc[:L]
        if (lm["max_cc"] >= target).any():
            continue                       # evented inside the landmark
        t_lm = lm["commit_date"].iloc[-1]
        post = g.iloc[L:]
        cr = np.where(post["max_cc"].values >= target)[0]
        event = 1 if cr.size else 0
        t_end = post["commit_date"].iloc[int(cr[0])] if cr.size else post["commit_date"].iloc[-1]
        recs.append(dict(
            repo=repo, duration_days=max((t_end - t_lm).days, 1), event=event,
            init_cc=init,
            cc_lm=lm["max_cc"].iloc[-1],
            cc_slope=lm["max_cc"].iloc[-1] - lm["max_cc"].iloc[0],
            loc_lm=lm["loc"].iloc[-1],
            loc_delta=lm["loc"].iloc[-1] - lm["loc"].iloc[0],
            mi_lm=lm["mi"].iloc[-1],
            nfunc_lm=lm["n_functions"].iloc[-1],
            hv_lm=lm["halstead_volume"].iloc[-1],
            n_authors=lm["author_email"].nunique(),
            lm_days=max((t_lm - lm["commit_date"].iloc[0]).days, 1),
        ))
    return pd.DataFrame(recs).dropna()


# ---------------------------------------------------------------- model
def concordance(t, risk, e):
    t = np.asarray(t, float); risk = np.asarray(risk, float); e = np.asarray(e, int)
    num = den = 0.0
    for i in np.where(e == 1)[0]:
        later = t > t[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return num / den if den else np.nan


def fit_cox(X, t, e, penalizer=0.1):
    """Cox partial likelihood (Breslow ties), L2-penalised."""
    order = np.argsort(t)
    X, t, e = X[order], t[order], e[order]
    ev = np.where(e == 1)[0]

    def nll(b):
        r = X @ b
        ll = 0.0
        for i in ev:
            ll += r[i] - np.log(np.exp(r[t >= t[i]]).sum())
        return -ll + penalizer * np.sum(b ** 2)

    return minimize(nll, np.zeros(X.shape[1]), method="L-BFGS-B").x


def oriented(t, x, e):
    """Baselines get their BEST orientation chosen on the test fold. This is an
    oracle advantage that FAVOURS the baseline — deliberately conservative for
    our claim."""
    return max(concordance(t, x, e), concordance(t, -x, e))


# ---------------------------------------------------------------- evaluation
def repeated_cv(S, repeats=10, folds=5, seed=0):
    t = S.duration_days.values.astype(float)
    e = S.event.values.astype(int)
    Xr = S[COV].values
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-9
    rng = np.random.default_rng(seed)

    out = {k: [] for k in ["ours", "random", "loc_lm", "cc_lm", "hv_lm", "n_authors"]}
    diffs = []
    for _ in range(repeats):
        idx = rng.permutation(len(S))
        for f in np.array_split(idx, folds):
            tr = np.setdiff1d(idx, f)
            if e[tr].sum() < 5 or e[f].sum() < 2:
                continue
            b = fit_cox((Xr[tr] - mu) / sd, t[tr], e[tr])
            o = concordance(t[f], ((Xr[f] - mu) / sd) @ b, e[f])
            if np.isnan(o):
                continue
            base = {
                "random":    concordance(t[f], rng.normal(size=len(f)), e[f]),
                "loc_lm":    oriented(t[f], S.loc_lm.values[f], e[f]),
                "cc_lm":     oriented(t[f], S.cc_lm.values[f], e[f]),
                "hv_lm":     oriented(t[f], S.hv_lm.values[f], e[f]),
                "n_authors": oriented(t[f], S.n_authors.values[f], e[f]),
            }
            out["ours"].append(o)
            for k, v in base.items():
                out[k].append(v)
            best_bl = max(base[k] for k in ["loc_lm", "cc_lm", "hv_lm", "n_authors"])
            diffs.append(o - best_bl)

    res = {k: np.array(v) for k, v in out.items()}
    return res, np.array(diffs)


def ci(x, lo=2.5, hi=97.5):
    return np.percentile(x, [lo, hi])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="complexity_long.csv")
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df["commit_date"] = pd.to_datetime(df["commit_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["commit_date", "max_cc"]).sort_values(
        ["repo", "file_path", "commit_date"])

    S = build_landmark(df, args.L, args.delta)
    tau = abs(S.init_cc.corr(S.event))
    print(f"files={len(S)}  events={int(S.event.sum())}  rate={S.event.mean():.3f}")
    print(f"[guard] |corr(init_cc, event)| = {tau:.3f}  (must be < 0.20)")
    assert tau < 0.20, "TAUTOLOGICAL EVENT"
    for c in COV:
        v = abs(S[c].corr(S.duration_days))
        assert v < 0.5, f"CIRCULAR COVARIATE: {c} ({v:.3f})"
    print("[guard] no circular covariates\n")

    res, diffs = repeated_cv(S, repeats=args.repeats)

    print(f"=== Repeated {args.repeats}x5-fold CV  ({len(res['ours'])} folds) ===")
    rows = []
    label = {"random": "B1 Random", "loc_lm": "B2 Size (LOC)",
             "cc_lm": "B3 Static (McCabe)", "hv_lm": "B4 Static (Halstead)",
             "n_authors": "B5 Process (authors)", "ours": "B6 Cox trajectory (ours)"}
    for k in ["random", "loc_lm", "cc_lm", "hv_lm", "n_authors", "ours"]:
        v = res[k]
        lo, hi = ci(v)
        rows.append(dict(model=label[k], concordance=v.mean(), ci_low=lo, ci_high=hi))
        print(f"  {label[k]:26s} {v.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    lo, hi = ci(diffs)
    print(f"\n  PAIRED DIFF (ours - best baseline): {diffs.mean():+.3f}"
          f"  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  folds where ours loses: {(diffs < 0).sum()}/{len(diffs)}")
    if lo > 0:
        print("  => CI excludes zero. The advantage is statistically supported.")
    else:
        print("  => CI INCLUDES ZERO. Do NOT claim superiority at this configuration.")

    pd.DataFrame(rows).to_csv("results_table.csv", index=False)

    # ---- sensitivity, cross-validated ----
    print("\n=== Sensitivity (cross-validated) ===")
    srows = []
    for L in [2, 3, 5]:
        for d in [1.0, 2.0, 3.0]:
            Sx = build_landmark(df, L, d)
            if len(Sx) < 100 or Sx.event.sum() < 30:
                continue
            r, dd = repeated_cv(Sx, repeats=5)
            lo, hi = ci(dd)
            sig = "yes" if lo > 0 else "NO"
            srows.append(dict(L=L, delta=d, files=len(Sx), events=int(Sx.event.sum()),
                              ours=r["ours"].mean(), cc_lm=r["cc_lm"].mean(),
                              n_authors=r["n_authors"].mean(),
                              diff=dd.mean(), ci_low=lo, ci_high=hi, significant=sig))
            print(f"  L={L} d={d}: ours={r['ours'].mean():.3f} "
                  f"diff={dd.mean():+.3f} [{lo:+.3f},{hi:+.3f}]  significant={sig}")
    pd.DataFrame(srows).to_csv("sensitivity_cv.csv", index=False)
    print("\nwrote results_table.csv, sensitivity_cv.csv")


if __name__ == "__main__":
    main()
