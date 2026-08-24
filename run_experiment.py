#!/usr/bin/env python
"""run the whole project end to end.

    python run_experiment.py

reads the RASFF file from data/, does everything, and writes every result into
results/. takes about 10 to 15 minutes on a normal laptop.

this is the only thing you need to run. every number in my dissertation comes
out of it, and nothing in results/ is edited by hand afterwards. if my report
and results/ ever disagree, results/ is the one that is right.

    --quick     a faster, rougher version. for when I am developing.
    --csv PATH  point it at a different file.
    --seed N    change the random seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from rasff.config import (  # noqa: E402
    DEPLOYMENT_TIER,
    FEATURE_SETS,
    LABEL_ARMS,
    LABEL_SCHEMES,
    MODELS_DIR,
    PREDS_DIR,
    RAW_CSV,
    RESULTS_DIR,
    SEED,
    YEAR_MIN,
    ensure_output_dirs,
)
from rasff.data.cleaning import clean  # noqa: E402
from rasff.data.labels import (  # noqa: E402
    apply_arm,
    assign_labels,
    regime_tables,
    select_window,
)
from rasff.data.loading import load_raw  # noqa: E402
from rasff.evaluation.ablation import (  # noqa: E402
    feature_importance_by_source,
    text_ablation,
    tier_ablation,
)
from rasff.evaluation.experiment import (  # noqa: E402
    best_model_for,
    fit_score,
    headline_table,
    run_grid,
)
from rasff.evaluation.prioritisation import (  # noqa: E402
    detection_curve,
    origin_risk_scores,
    serious_scores,
    summarise,
)
from rasff.evaluation.splits import describe_split, make_splits  # noqa: E402
from rasff.models.zoo import (  # noqa: E402
    DEPLOYMENT_MODEL,
    HAS_LIGHTGBM,
    available_models,
)
from rasff.reporting.tables import write_json, write_summary, write_table  # noqa: E402

HEADLINE_SPLIT = "temporal"
HEADLINE_ARM = "binary_serious"


def parse_args() -> argparse.Namespace:
    """read the command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--year-min", type=int, default=YEAR_MIN)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="200 resamples, temporal split only.",
    )
    return parser.parse_args()


def step(number: int, message: str) -> None:
    """print a heading so you can see where the run has got to."""
    print(f"\n[{number}] {message}")


def main() -> int:
    """the whole pipeline, in twelve steps. read the step() lines to follow it."""
    args = parse_args()
    started = time.time()
    ensure_output_dirs()

    n_bootstrap = 200 if args.quick else 1000
    split_names = ["temporal"] if args.quick else ["temporal", "random"]

    print("=" * 72)
    print("RASFF import risk triage, full pipeline")
    print("=" * 72)
    print(f"  export     : {args.csv}")
    print(f"  seed       : {args.seed}")
    print(f"  bootstrap  : {n_bootstrap}")
    print(f"  models     : {', '.join(available_models())}")
    if not HAS_LIGHTGBM:
        print("  note       : lightgbm not installed, that comparison is skipped")

    # ---------------------------------------------------------------- load
    step(1, "loading and cleaning the export")
    raw, load_counts = load_raw(args.csv)
    print(f"    {load_counts}")

    frame, clean_diagnostics = clean(raw)
    print(f"    {clean_diagnostics}")
    if clean_diagnostics["day_month_ambiguous"]:
        print(
            "    WARNING: no day value above 12 in the column, so day-first vs "
            "month-first cannot be settled from the data. confirm the export format."
        )

    frame = assign_labels(frame, scheme="baseline")

    # ------------------------------------------------------- regime tables
    step(2, "labelling regime by year, the evidence for the window cut")
    regime = regime_tables(frame)
    for name, table in regime.items():
        write_table(table.reset_index(), f"regime_{name}")
    print(regime["label_by_year"].to_string())

    analysis, window_counts = select_window(frame, args.year_min)
    print(f"\n    {window_counts}")
    write_json({**load_counts, **clean_diagnostics, **window_counts}, "dataset_summary")

    # --------------------------------------------------------------- split
    step(3, "building splits")
    splits = make_splits(analysis, split_names, seed=args.seed)
    split_info = [describe_split(s) for s in splits.values()]
    for info in split_info:
        print(f"    {info}")
    write_table(pd.DataFrame(split_info), "splits")

    # ---------------------------------------------------------------- grid
    step(4, "running the experiment grid")
    results_table, results = run_grid(
        splits=splits,
        arms=LABEL_ARMS,
        feature_sets=FEATURE_SETS,
        model_names=available_models(),
        n_bootstrap=n_bootstrap,
        seed=args.seed,
    )
    write_table(results_table, "results_all")
    print(f"    {len(results_table)} experiment cells")

    # ------------------------------------------------------------ headline
    step(5, "headline results")
    headlines = {}
    for arm in LABEL_ARMS:
        table = headline_table(results_table, HEADLINE_SPLIT, arm)
        headlines[arm] = table
        write_table(table, f"headline_{HEADLINE_SPLIT}_{arm}")
        print(f"\n  --- temporal split, {arm} ---")
        print(table.to_string(index=False))

    # ------------------------------------------------------------ ablation
    step(6, "tier ablation, what each layer of information buys")
    tiers = tier_ablation(
        results, results_table, HEADLINE_SPLIT, HEADLINE_ARM, n_bootstrap, args.seed
    )
    write_table(tiers, "ablation_tiers")
    print(tiers.to_string(index=False))

    step(7, "text ablation, does free text beat structured metadata")
    texts = pd.concat(
        [
            text_ablation(
                results, results_table, HEADLINE_SPLIT, arm, DEPLOYMENT_TIER,
                n_bootstrap, args.seed,
            )
            for arm in LABEL_ARMS
        ],
        ignore_index=True,
    )
    write_table(texts, "ablation_text")
    print(texts.to_string(index=False))

    # -------------------------------------------------- label sensitivity
    step(8, "label scheme sensitivity")
    sensitivity = []
    for scheme in LABEL_SCHEMES:
        relabelled = assign_labels(frame, scheme=scheme)
        subset, _ = select_window(relabelled, args.year_min)
        scheme_splits = make_splits(subset, ["temporal"], seed=args.seed)
        for feature_set in (DEPLOYMENT_TIER, "text"):
            result = fit_score(
                scheme_splits["temporal"],
                HEADLINE_ARM,
                feature_set,
                DEPLOYMENT_MODEL,
                n_bootstrap,
                args.seed,
            )
            if result is None:
                continue
            sensitivity.append(
                {
                    "scheme": scheme,
                    "serious_pct": round(
                        float((subset["label"] == "serious").mean()) * 100, 1
                    ),
                    "feature_set": feature_set,
                    "macro_f1": result.macro_f1,
                    "ci_lo": round(result.macro_f1_lo, 3),
                    "ci_hi": round(result.macro_f1_hi, 3),
                }
            )
    sensitivity_table = pd.DataFrame(sensitivity)
    write_table(sensitivity_table, "label_sensitivity")
    print(sensitivity_table.to_string(index=False))

    # -------------------------------------------------------- deployment
    step(9, "fitting and saving the deployment model")
    deployment = fit_score(
        splits[HEADLINE_SPLIT],
        HEADLINE_ARM,
        DEPLOYMENT_TIER,
        DEPLOYMENT_MODEL,
        n_bootstrap,
        args.seed,
    )
    if deployment is None:
        print("    deployment model failed to fit")
        return 1

    model_path = MODELS_DIR / "deployment_model.joblib"
    joblib.dump(deployment.fitted, model_path)

    # the artefact says what it is. a model file that cannot tell you which
    # tier and which estimator it holds is how a lightgbm file ends up loaded
    # where a random forest was expected. do this before demo day, not on it.
    write_json(
        {
            "estimator": DEPLOYMENT_MODEL,
            "feature_tier": DEPLOYMENT_TIER,
            "columns": list(
                deployment.fitted.named_steps["features"].transformers_[0][2]
            ),
            "arm": HEADLINE_ARM,
            "classes": list(deployment.fitted.classes_),
            "macro_f1": deployment.macro_f1,
            "macro_f1_ci": [
                round(deployment.macro_f1_lo, 3),
                round(deployment.macro_f1_hi, 3),
            ],
            "seed": args.seed,
            "trained_through": str(splits[HEADLINE_SPLIT].val["date"].max().date()),
        },
        "deployment_model_card",
        directory=MODELS_DIR,
    )
    print(f"    saved {model_path.name} ({DEPLOYMENT_MODEL} on {DEPLOYMENT_TIER})")

    importance = feature_importance_by_source(deployment.fitted, DEPLOYMENT_TIER)
    write_table(importance, "feature_importance")
    print(importance.to_string(index=False))

    # ---------------------------------------------------- prioritisation
    step(10, "prioritisation under an inspection budget")
    split = splits[HEADLINE_SPLIT]
    full_train = pd.concat([split.train, split.val], ignore_index=True)
    y_train = apply_arm(full_train["label"], HEADLINE_ARM)
    y_test = apply_arm(split.test["label"], HEADLINE_ARM)

    curve = detection_curve(
        y_test=y_test,
        model_scores=serious_scores(deployment.fitted, split.test),
        heuristic_scores=origin_risk_scores(full_train, y_train, split.test),
        seed=args.seed,
    )
    write_table(curve, "prioritisation")
    print(curve.to_string(index=False))
    print(f"\n    at 10% budget: {summarise(curve, '10%')}")

    # ------------------------------------------------ test window for app
    step(11, "exporting the held-out test window for the dashboard")
    # the dashboard has to score percentiles and base rates against rows the
    # model never trained on. writing this file explicitly is what stops it
    # falling back to the full export.
    test_export = PREDS_DIR / "test_window.csv"
    split.test.assign(date=lambda d: d["date"].astype(str)).to_csv(
        test_export, index=False
    )
    print(f"    {len(split.test)} rows -> {test_export.name}")

    # ------------------------------------------------------------ summary
    step(12, "writing the run summary")
    run_info = {
        "seed": args.seed,
        "year_min": args.year_min,
        "n_bootstrap": n_bootstrap,
        "deployment_model": DEPLOYMENT_MODEL,
        "deployment_tier": DEPLOYMENT_TIER,
        "lightgbm_available": HAS_LIGHTGBM,
        "dataset": {**load_counts, **window_counts},
    }
    summary_path = write_summary(
        run_info,
        {
            "headline, temporal split, binary": headlines["binary_serious"],
            "headline, temporal split, three-class": headlines["three_class"],
            "tier ablation, information availability": tiers,
            "text ablation, RQ2": texts,
            "label scheme sensitivity, RQ3": sensitivity_table,
            "prioritisation under budget, RQ4": curve,
            "feature importance by source column": importance,
            "splits": pd.DataFrame(split_info),
        },
        RESULTS_DIR / "SUMMARY.md",
    )

    print("\n" + "=" * 72)
    print(f"done in {time.time() - started:.0f}s. see {summary_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
