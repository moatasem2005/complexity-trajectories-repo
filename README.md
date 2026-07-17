# Early Development Dynamics Predict Complexity Accumulation

A landmark survival analysis of file-level cyclomatic-complexity growth across
fourteen Python open-source projects (2005–2026).

> **Status:** manuscript in preparation for submission to *Information and
> Software Technology* (Elsevier). This repository is the full replication
> package: mining pipeline, guard-checked landmark dataset, evaluation code,
> all figures/tables, and the manuscript itself.

## TL;DR

- **Question:** given a source file's first few commits, can we forecast
  whether — and how soon — it will accumulate substantially more cyclomatic
  complexity?
- **Method:** a *relative* growth event (max CC ≥ init + δ) combined with a
  *landmark* design (all predictors measured only in the first `L` commits),
  analyzed with penalized Cox proportional-hazards models and evaluated by
  repeated cross-validation, leave-one-project-out, and an `L × δ` sensitivity
  sweep. Two automated guards reject the pipeline if the event definition is
  tautological or predictors are circular with the outcome.
- **Result:** early development *process* dynamics (commit tempo, author
  count, early growth) predict complexity accumulation
  (concordance = 0.663, 95% CI 0.626–0.704), clearly beating size,
  static-complexity, coupling, and process baselines reimplemented on the same
  data (+0.086, 95% CI +0.023 to +0.131). Initial static complexity is
  near-chance (0.541). Two disciplined negative results: neither structural
  predictors nor gradient boosting improve the model further. Cross-project
  transfer (leave-one-project-out) is honest but moderate (0.599).

## Repository layout

```
.


│    complexity_debt_trajectories.ipynb   # end-to-end pipeline (mining → guards → models → figures)
├── scripts/
│   ├── extract_complexity_trajectories.py   # PyDriller/Radon/AST mining
│   ├── build_survival_fixed.py              # landmark table construction + guards
│   └── evaluate_cv.py                       # repeated CV, LORO, sensitivity sweep
├── data/
│   ├── complexity_long_final.csv            # raw file-commit observations (36,076 rows)
│   ├── survival_landmark_final.csv          # landmark survival table (1,291 rows)
│   ├── table2_results.csv                   # main model-vs-baseline results
│   ├── table3_leave_one_repo_out.csv        # LORO results per project
│   └── table4_sensitivity.csv               # L × δ sensitivity grid
├── LICENSE
├── .gitignore
└── README.md
```

## Reproducing the pipeline

The notebook `complexity_debt_trajectories.ipynb` is self-contained
and regenerates every table and figure in the manuscript from scratch:

1. **Mining** — clones/traverses each repository with PyDriller, computes
   per-commit metrics with Radon + a custom AST walker, and maintains a
   time-ordered import graph for coupling.
2. **Landmark construction** — builds the survival table with a relative
   growth event and predictors measured only in the first `L` commits.
3. **Guards** — halts automatically if the event-tautology check
   (`|corr(initial CC, event)| < 0.20`) or the predictor-circularity check
   (`max |corr(predictor, duration)| < 0.50`) fails.
4. **Modeling** — fits a penalized, repository-stratified Cox model and a
   gradient-boosted Cox model; reimplements five baselines (size, static
   complexity, Maintainability Index, coupling, process).
5. **Evaluation** — repeated 10×5-fold cross-validation with paired
   bootstrap confidence intervals, leave-one-project-out validation, and an
   `L ∈ {2,3,5} × δ ∈ {1,2,3}` sensitivity sweep.
6. **Output** — every table (`data/table*.csv`) and figure (`figures/*.png`)
   used in the manuscript.

```bash
pip install pydriller radon lifelines scikit-survival pandas numpy scipy matplotlib
jupyter nbconvert --to notebook --execute notebooks/complexity_debt_trajectories.ipynb
```

Runtime on the full 14-project corpus is several hours (network-bound mining
dominates); the pre-mined CSVs in `data/` let you skip straight to modeling
and evaluation if you only want to reproduce the statistical results.

## Key numbers (verified against `data/`)

| Quantity | Value |
|---|---|
| Repositories | 14 |
| File-commit observations | 36,076 |
| Distinct files mined | 4,233 |
| Landmark population | 1,291 files (387 events, 904 censored) |
| Event rate | 30.0% |
| Primary concordance (Cox, repeated CV) | 0.663 (95% CI 0.626–0.704) |
| Best baseline (process / author count) | 0.563 |
| Paired gain over best baseline | +0.086 (95% CI +0.023 to +0.131) |
| Leave-one-project-out mean concordance | 0.599 vs. 0.588 (McCabe) |

## Citation

A formal citation will be added once the manuscript is accepted. In the
meantime, please cite this repository directly if you build on it.

## License

Code and data are released under the MIT License (see `LICENSE`). 
