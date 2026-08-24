# where everything is computed

every number in the dissertation, and the exact function that produces it.

## the scores

| number in the report | file | function |
|---|---|---|
| macro F1 (every one of them) | `src/rasff/evaluation/metrics.py` | `macro_f1()` |
| the 95% CI on a macro F1 | `src/rasff/evaluation/metrics.py` | `bootstrap_ci()` |
| delta between two models + its CI | `src/rasff/evaluation/metrics.py` | `paired_bootstrap()` |
| accuracy, balanced accuracy | `src/rasff/evaluation/metrics.py` | `point_metrics()` |
| "significantly better/worse" | `src/rasff/evaluation/metrics.py` | `verdict()` |

nothing else in the codebase computes a score. if a number appears in the
report, it came out of one of those five functions.

## how a score gets made

the whole fit-select-score sequence is one function:

`src/rasff/evaluation/experiment.py` → `fit_score()`

it does four things in order:

1. `select_hyperparameters()` fits each grid setting on **train**, scores on **validation**
2. keeps the best setting by validation macro F1
3. refits that setting on **train + validation**
4. calls `.predict()` on **test**, once

the test window is read on the line marked `# first and only time the test
window is touched`. that is the only place in the codebase where test data
reaches a model. everything else works on train and validation.

## the ranking numbers

| number | file | function |
|---|---|---|
| probability a consignment is serious | `evaluation/prioritisation.py` | `serious_scores()` |
| recall at 5/10/20/30/50% budget | `evaluation/prioritisation.py` | `detection_curve()` |
| lift vs random, % of ceiling | `evaluation/prioritisation.py` | `detection_curve()` |
| worst-origin heuristic baseline | `evaluation/prioritisation.py` | `origin_risk_scores()` |
| the 10% headline (1.47x, 91%) | `evaluation/prioritisation.py` | `summarise()` |

## the ablations

| number | file | function |
|---|---|---|
| tier ladder, gain per rung | `evaluation/ablation.py` | `tier_ablation()` |
| notification_type = +0.111 | `evaluation/ablation.py` | `tier_ablation()`, row 2 |
| text vs structured (RQ2) | `evaluation/ablation.py` | `text_ablation()` |
| DistilBERT vs structured | `evaluation/ablation.py` | `add_external_comparison()` |
| % importance per source column | `evaluation/ablation.py` | `feature_importance_by_source()` |
| text vs BOTH baselines | `scripts/reconcile_baseline.py` | run it directly |

## the dataset numbers

| number | file | function |
|---|---|---|
| 19,890 rows read, 12 repaired | `data/loading.py` | `read_raw_csv()` |
| 15,331 analysed, label counts | `data/labels.py` | `select_window()` |
| hazard coverage, duplicate subjects | `data/cleaning.py` | `clean()` |
| the 2023 cut evidence | `data/labels.py` | `regime_tables()` |
| 10,731 / 2,300 / 2,300 split | `evaluation/splits.py` | `temporal_split()` |
| proof the split is not leaking | `evaluation/splits.py` | `verify_temporal_order()` |

## the dashboard

`app/main.py`. it does not fit or evaluate anything, it loads the saved
pipeline and calls `predict_proba`.

| what you see on screen | function |
|---|---|
| the score | `predict_proba` on the loaded pipeline, column `SI` |
| "riskier than N% of arrivals" | `rank_of()` against the held-out window |
| RELEASE / REVIEW / INSPECT | `bands()` then `verdict_for()` |
| the red band width | `bands()`, exactly `budget` wide. same top-k rule as `detection_curve()` |
| the basis panel bars | `drivers()` |
| "% serious on record" | `history()`, only shown when n >= 30 |

`drivers()` explains ONE consignment. `feature_importance_by_source()` in
`evaluation/ablation.py` is global across training. they answer different
questions, do not quote one for the other.

## which fields go into which model

`src/rasff/config.py`, the `FeatureTier` block. TIER1 through TIER5, each
listing its columns and when that information becomes available. that is the
only place feature membership is defined.

## reading order

if you have ten minutes and want to know what this does:

1. `config.py` — the tiers, the label schemes, the seed
2. `evaluation/metrics.py` — how anything gets scored
3. `evaluation/experiment.py` → `fit_score()` — the protocol
4. `run_experiment.py` — the 12 steps, in order

everything else is called from those.
