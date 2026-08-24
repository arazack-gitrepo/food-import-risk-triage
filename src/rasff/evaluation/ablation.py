"""what is each piece of information actually worth?

two separate questions get answered here and my report keeps them apart.

tier_ablation climbs the ladder one step at a time and measures what each
extra field buys. this is the answer to "is your model just reading the
answer off the form?" if the score only jumps once notification_type is
added, then the number I should honestly be quoting is the one from the step
below.

text_ablation asks whether the written description adds anything over the
tick-box fields. it compares against tier1, because comparing against a
version that already knows the lab result would be comparing against the
wrong thing.
"""

from __future__ import annotations

import pandas as pd

from rasff.config import DEPLOYMENT_TIER, FEATURE_TIERS, N_BOOTSTRAP, SEED
from rasff.evaluation.experiment import ExperimentResult, best_model_for
from rasff.evaluation.metrics import paired_bootstrap, verdict

TIER_ORDER = [
    "tier1_declaration",
    "tier2_plus_reporter",
    "tier3_plus_notification_type",
    "tier4_plus_hazard",
    "tier5_post_hoc",
]


def _lookup(
    results: dict[tuple, ExperimentResult],
    table: pd.DataFrame,
    split: str,
    arm: str,
    feature_set: str,
) -> ExperimentResult | None:
    """find the best model for one particular setup, or None if there isn't one."""
    model = best_model_for(table, split, arm, feature_set)
    if model is None:
        return None
    return results.get((split, arm, feature_set, model))


def tier_ablation(
    results: dict[tuple, ExperimentResult],
    table: pd.DataFrame,
    split: str,
    arm: str,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """how much each extra field is worth, one step at a time."""
    rows = []
    for lower, upper in zip(TIER_ORDER, TIER_ORDER[1:]):
        base = _lookup(results, table, split, arm, lower)
        step = _lookup(results, table, split, arm, upper)
        if base is None or step is None:
            continue

        comparison = paired_bootstrap(
            base.y_true, step.y_pred, base.y_pred, n_boot=n_bootstrap, seed=seed
        )
        rows.append(
            {
                "split": split,
                "arm": arm,
                "step": f"{upper} - {lower}",
                "adds": ", ".join(
                    sorted(
                        set(FEATURE_TIERS[upper].columns)
                        - set(FEATURE_TIERS[lower].columns)
                    )
                ),
                "available_at": FEATURE_TIERS[upper].available_at,
                "macro_f1_lower": base.macro_f1,
                "macro_f1_upper": step.macro_f1,
                "delta": round(comparison["delta"], 3),
                "ci_lo": round(comparison["ci_lo"], 3),
                "ci_hi": round(comparison["ci_hi"], 3),
                "p_no_gain": round(comparison["p_no_gain"], 3),
                "verdict": verdict(comparison["ci_lo"], comparison["ci_hi"]),
            }
        )
    return pd.DataFrame(rows)


def text_ablation(
    results: dict[tuple, ExperimentResult],
    table: pd.DataFrame,
    split: str,
    arm: str,
    baseline_tier: str = DEPLOYMENT_TIER,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> pd.DataFrame:
    """does the written description beat, or add to, the tick-box fields?"""
    base = _lookup(results, table, split, arm, baseline_tier)
    if base is None:
        return pd.DataFrame()

    rows = []
    for feature_set in ("text", "hybrid"):
        candidate = _lookup(results, table, split, arm, feature_set)
        if candidate is None:
            continue

        comparison = paired_bootstrap(
            base.y_true, candidate.y_pred, base.y_pred, n_boot=n_bootstrap, seed=seed
        )
        rows.append(
            {
                "split": split,
                "arm": arm,
                "comparison": f"{feature_set} - {baseline_tier}",
                "model": candidate.model,
                "baseline_model": base.model,
                "macro_f1_baseline": base.macro_f1,
                "macro_f1_candidate": candidate.macro_f1,
                "delta": round(comparison["delta"], 3),
                "ci_lo": round(comparison["ci_lo"], 3),
                "ci_hi": round(comparison["ci_hi"], 3),
                "p_no_gain": round(comparison["p_no_gain"], 3),
                "verdict": verdict(comparison["ci_lo"], comparison["ci_hi"]),
            }
        )
    return pd.DataFrame(rows)


def add_external_comparison(
    results: dict[tuple, ExperimentResult],
    table: pd.DataFrame,
    split: str,
    arm: str,
    y_true,
    y_pred,
    label: str,
    baseline_tier: str = DEPLOYMENT_TIER,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict:
    """score a model that was trained somewhere else.

    DistilBERT needs a GPU so it runs separately in Colab and comes back as a
    file of predictions. this puts it through exactly the same comparison as
    everything else, so it is a fair fight.
    """
    base = _lookup(results, table, split, arm, baseline_tier)
    if base is None:
        raise ValueError(f"no baseline result for {split}/{arm}/{baseline_tier}")

    comparison = paired_bootstrap(
        y_true, y_pred, base.y_pred, n_boot=n_bootstrap, seed=seed
    )
    return {
        "split": split,
        "arm": arm,
        "comparison": f"{label} - {baseline_tier}",
        "model": label,
        "baseline_model": base.model,
        "delta": round(comparison["delta"], 3),
        "ci_lo": round(comparison["ci_lo"], 3),
        "ci_hi": round(comparison["ci_hi"], 3),
        "p_no_gain": round(comparison["p_no_gain"], 3),
        "verdict": verdict(comparison["ci_lo"], comparison["ci_hi"]),
    }


def feature_importance_by_source(fitted_pipeline, tier_name: str) -> pd.DataFrame:
    """which fields is the model actually leaning on?

    a single field like origin country gets split into hundreds of yes/no
    columns internally, one per country, so the raw numbers mean nothing on
    their own. this adds them back up per original field, which is what lets
    me say something like "notification_type accounts for 48% of it".
    """
    transformer = fitted_pipeline.named_steps["features"]
    classifier = fitted_pipeline.named_steps["classifier"]

    names = transformer.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_).sum(axis=0)
    else:
        raise TypeError(f"{type(classifier).__name__} exposes no feature importances.")

    frame = pd.DataFrame({"feature": names, "importance": importances})
    frame["source"] = frame["feature"].str.replace("^cat__", "", regex=True)

    # longest name first, or "product_type" swallows "product_category".
    for column in sorted(FEATURE_TIERS[tier_name].columns, key=len, reverse=True):
        mask = frame["source"].str.startswith(column + "_")
        frame.loc[mask, "source"] = column

    total = frame["importance"].sum()
    by_source = (
        frame.groupby("source")["importance"].sum().sort_values(ascending=False)
        / total
        * 100
    ).round(1)

    return by_source.rename("pct_importance").to_frame().reset_index()
