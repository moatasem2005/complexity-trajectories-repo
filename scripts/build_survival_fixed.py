"""
build_survival_fixed.py
-----------------------
Corrects two defects found in the first pipeline run:

  BUG 1 (critical): post-outcome leakage.
      `ai_exposure` and `churn` were averaged/counted over a file's ENTIRE history,
      including commits that occurred AFTER the file crossed the complexity
      threshold. 94.3% of event files were affected. Covariates must only use
      information available strictly BEFORE the event time.

  BUG 2 (design): `init_cc` mechanically dominates the hazard (files that start
      near the threshold cross sooner), swamping the exposure signal. Handled by
      (a) left-truncating files that already exceed the threshold at birth, and
      (b) offering a time-varying (counting-process) format that is the statistically
      correct way to model exposure that changes over a file's life.

Drop these functions into the Colab notebook in place of the original
`build_survival` (cell 6).
"""

import numpy as np
import pandas as pd

EVENT_METRIC = "max_cc"
CC_THRESHOLD = 10.0


# --------------------------------------------------------------------------
# Baseline (time-fixed) survival table -- leakage-free
# --------------------------------------------------------------------------
def build_survival_fixed(df, metric=EVENT_METRIC, threshold=CC_THRESHOLD):
    """One row per file. All covariates use ONLY commits up to (and including)
    the event commit; for censored files, the whole observed history."""
    df = df.sort_values(["repo", "file_path", "commit_date"])
    recs = []
    for (repo, path), g in df.groupby(["repo", "file_path"], sort=False):
        g = g.reset_index(drop=True)

        # Left-truncation: a file already at/over threshold on its first commit
        # contributes no information about TIME-to-threshold. Exclude it.
        first_val = g[metric].dropna()
        if first_val.empty:
            continue
        if first_val.iloc[0] >= threshold:
            continue

        crossed = np.where(g[metric].values >= threshold)[0]
        if crossed.size:
            cut = int(crossed[0])          # index of the event commit
            event = 1
        else:
            cut = len(g) - 1               # last observed commit
            event = 0

        # >>> covariates computed on the pre-event window ONLY <<<
        window = g.iloc[: cut + 1]
        t0 = g["commit_date"].iloc[0]
        dur = (g["commit_date"].iloc[cut] - t0).days

        # a file whose event lands on its first commit has zero exposure time
        if dur <= 0:
            dur = 1

        recs.append(
            dict(
                repo=repo,
                file_path=path,
                duration_days=int(dur),
                event=int(event),
                init_cc=float(window[metric].dropna().iloc[0]),
                init_loc=float(window["loc"].dropna().iloc[0])
                if window["loc"].notna().any() else np.nan,
                ai_exposure=float(window["ai_flag"].mean()),      # pre-event only
                churn=int(len(window)),                           # pre-event only
            )
        )
    out = pd.DataFrame(recs)
    for c in ["init_cc", "init_loc"]:
        out[c] = out[c].fillna(out[c].median())
    return out


# --------------------------------------------------------------------------
# Time-varying (counting-process) format -- the statistically correct model
# --------------------------------------------------------------------------
def build_counting_process(df, metric=EVENT_METRIC, threshold=CC_THRESHOLD):
    """One row per (file, commit-interval): [start, stop). Exposure is allowed to
    change over the file's life, so AI adoption is modelled as a time-varying
    covariate rather than a single averaged number. Feed to
    lifelines.CoxTimeVaryingFitter."""
    df = df.sort_values(["repo", "file_path", "commit_date"])
    rows = []
    for (repo, path), g in df.groupby(["repo", "file_path"], sort=False):
        g = g.reset_index(drop=True)
        vals = g[metric].dropna()
        if vals.empty or vals.iloc[0] >= threshold:
            continue                                  # left-truncate

        crossed = np.where(g[metric].values >= threshold)[0]
        cut = int(crossed[0]) if crossed.size else len(g) - 1
        event = 1 if crossed.size else 0
        t0 = g["commit_date"].iloc[0]

        cum_ai = 0
        prev_stop = 0
        fid = f"{repo}::{path}"
        for i in range(cut + 1):
            stop = (g["commit_date"].iloc[i] - t0).days
            if stop <= prev_stop:
                stop = prev_stop + 1                  # enforce strictly increasing
            cum_ai += int(g["ai_flag"].iloc[i])
            rows.append(
                dict(
                    id=fid, repo=repo, file_path=path,
                    start=prev_stop, stop=stop,
                    event=int(event and i == cut),    # event only on the last interval
                    ai_cum=cum_ai,                    # cumulative AI-flagged edits so far
                    ai_rate=cum_ai / (i + 1),         # exposure to date
                    commits_to_date=i + 1,
                    cc_to_date=float(g[metric].iloc[i]),
                )
            )
            prev_stop = stop
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fitting helpers
# --------------------------------------------------------------------------
def fit_timevarying(cp_df):
    """Cox model with time-varying exposure. Note: do NOT include cc_to_date as a
    covariate -- it is on the causal path to the event and would be a collider."""
    from lifelines import CoxTimeVaryingFitter
    ctv = CoxTimeVaryingFitter(penalizer=0.1)
    cols = ["id", "start", "stop", "event", "ai_rate", "commits_to_date"]
    ctv.fit(cp_df[cols], id_col="id", event_col="event",
            start_col="start", stop_col="stop")
    return ctv


def sanity_check(surv_fixed, df_long, metric=EVENT_METRIC, threshold=CC_THRESHOLD):
    """Assert no post-outcome leakage remains."""
    df = df_long.sort_values(["repo", "file_path", "commit_date"])
    bad = 0
    for (repo, path), g in df.groupby(["repo", "file_path"], sort=False):
        row = surv_fixed[(surv_fixed.repo == repo) & (surv_fixed.file_path == path)]
        if row.empty:
            continue
        g = g.reset_index(drop=True)
        crossed = np.where(g[metric].values >= threshold)[0]
        if crossed.size:
            expected_churn = int(crossed[0]) + 1
            if int(row.iloc[0].churn) != expected_churn:
                bad += 1
    print(f"[sanity] files with leaked covariates: {bad} (must be 0)")
    return bad == 0
