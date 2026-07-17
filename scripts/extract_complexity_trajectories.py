#!/usr/bin/env python3
"""
extract_complexity_trajectories.py
-----------------------------------
Mines per-file complexity time series from one or more Git repositories for the
"Complexity-Debt Trajectories" study.

For every commit (chronological order) and every modified Python file, it records
the complexity of the file *after* the change, so you can later reconstruct a
per-file trajectory and fit growth / survival (time-to-threshold) models.

Dependencies:
    pip install pydriller radon

Usage:
    # local clone
    python extract_complexity_trajectories.py --repo /path/to/repo --out out.csv
    # remote (PyDriller clones to a temp dir)
    python extract_complexity_trajectories.py --repo https://github.com/org/name --out out.csv
    # several repos into one file
    python extract_complexity_trajectories.py --repo r1 --repo r2 --out out.csv

Output columns (one row per modified Python file per commit):
    repo, commit_hash, commit_date, author_email, ai_flag,
    file_path, loc, total_cc, max_cc, n_functions, mi, halstead_volume
"""

import argparse
import csv
import re
import sys

from pydriller import Repository
from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit

# Heuristic AI-authorship signal from commit-message trailers. Treat as NOISY:
# prefer a known AI-adoption date per repo for the main covariate.
AI_PATTERN = re.compile(
    r"(co-authored-by:[^\n]*(copilot|cursor|codeium|tabnine|aider))"
    r"|(assisted-by:)"
    r"|(generated (with|by)[^\n]*(copilot|chatgpt|claude|gpt|llm))",
    re.IGNORECASE,
)


def ai_signal(message: str) -> int:
    return 1 if (message and AI_PATTERN.search(message)) else 0


def safe_metrics(source: str):
    """Compute complexity metrics defensively (historic versions often don't parse)."""
    loc = total_cc = max_cc = n_func = mi = hv = None
    if not source:
        return loc, total_cc, max_cc, n_func, mi, hv
    try:
        loc = source.count("\n") + 1
    except Exception:
        pass
    try:
        blocks = cc_visit(source)
        if blocks:
            ccs = [b.complexity for b in blocks]
            total_cc, max_cc, n_func = sum(ccs), max(ccs), len(ccs)
    except Exception:
        pass  # SyntaxError on older snapshots is expected and skipped
    try:
        mi = mi_visit(source, multi=True)
    except Exception:
        pass
    try:
        hv = h_visit(source).total.volume
    except Exception:
        pass
    return loc, total_cc, max_cc, n_func, mi, hv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", action="append", required=True,
                    help="Repo path or URL (repeatable).")
    ap.add_argument("--out", required=True, help="Output CSV path.")
    ap.add_argument("--ext", default=".py", help="File extension to analyse.")
    args = ap.parse_args()

    header = ["repo", "commit_hash", "commit_date", "author_email", "ai_flag",
              "file_path", "loc", "total_cc", "max_cc", "n_functions",
              "mi", "halstead_volume"]

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for repo in args.repo:
            repo_name = repo.rstrip("/").split("/")[-1].replace(".git", "")
            n_commits = 0
            print(f"[+] Mining {repo_name} ...", file=sys.stderr)
            for commit in Repository(repo).traverse_commits():
                flag = ai_signal(commit.msg)
                for mf in commit.modified_files:
                    path = mf.new_path or mf.old_path
                    if not path or not path.endswith(args.ext):
                        continue
                    src = mf.source_code  # content AFTER the change; None on deletion
                    if src is None:
                        continue
                    loc, tcc, mcc, nf, mi, hv = safe_metrics(src)
                    writer.writerow([
                        repo_name, commit.hash, commit.committer_date.isoformat(),
                        commit.author.email, flag, path, loc, tcc, mcc, nf, mi, hv,
                    ])
                n_commits += 1
                if n_commits % 200 == 0:
                    print(f"    {repo_name}: {n_commits} commits", file=sys.stderr)
            print(f"[+] Done {repo_name}: {n_commits} commits", file=sys.stderr)

    print(f"[+] Written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
