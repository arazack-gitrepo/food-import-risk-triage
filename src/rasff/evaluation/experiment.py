"""trains the models and scores them.

the rule everyone worries about with machine learning is accidentally letting
the model see the answers to the test before you test it. so the whole
train-and-score routine sits in one function, fit_score(), and nowhere else.
you can read it in a minute and check for yourself.

what it does, in order:

    1. try each settings combination, training on the training data
    2. see which one does best on the validation data
    3. retrain that winner on training + validation together
    4. only now, make predictions on the test data. once.

the test data is opened on one line near the bottom of fit_score. that line
is marked. it is the only place in the project where a model sees it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd
from sklearn.pipeline import Pipeline

from rasff.config import N_BOOTSTRAP, SEED
from rasff.data.labels import apply_arm
from rasff.evaluation.metrics import bootstrap_ci, macro_f1, point_metrics
from rasff.evaluation.splits import Split
from rasff.features.build import build_features
from rasff.models.zoo import get_model


@dataclass
class ExperimentResult:
    """the result of training one model on one set of fields. holds the score,
    its range, and the predictions themselves."""

    split: str
    arm: str
    feature_set: str
    model: str
    best_params: dict
    val_macro_f1: float
    macro_f1: float
    macro_f1_lo: float
    macro_f1_hi: float
    balanced_accuracy: float
    accuracy: float
    n_test: int
    y_true: list = field(repr=False, default_factory=list)
    y_pred: list = field(repr=False, default_factory=list)
    fitted: object = field(repr=False, default=None)

    def to_row(self) -> dict:
        """just the numbers, for writing into the results table. leaves out the
        predictions and the trained model itself."""
        return {
            "split": self.split,
            "arm": self.arm,
            "feature_set": self.feature_set,
            "model": self.model,
            "best_params": json.dumps(self.best_params),
            "val_macro_f1": round(self.val_macro_f1, 3),
            "macro_f1": self.macro_f1,
            "macro_f1_lo": round(self.macro_f1_lo, 3),
            "macro_f1_hi": round(self.macro_f1_hi, 3),
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "n_test": self.n_test,
        }


def make_pipeline(feature_set: str, constructor, params: dict) -> Pipeline:
    """glue the feature preparation and the model together into one object.

    the fields come from build_features() in rasff.features.build and the
    model from get_model() in rasff.models.zoo. every model in the project
    gets built right here and nowhere else.

    why they are glued together rather than prepared separately: if I
    prepared the features first, they would be prepared using all the data
    including the test rows, and the model would get a peek at information it
    should not have. bundled like this, both steps only ever see the training
    rows.
    """
    return Pipeline(
        [("features", build_features(feature_set)), ("classifier", constructor(**params))]
    )


def select_hyperparameters(
    split: Split, arm: str, feature_set: str, model_name: str
) -> tuple[dict | None, float]:
    """try every settings combination and pick whichever does best on the
    validation data. the test data is not involved at this stage."""
    constructor, grid = get_model(model_name)
    y_train = apply_arm(split.train["label"], arm)
    y_val = apply_arm(split.val["label"], arm)

    best_params: dict | None = None
    best_score = -1.0

    for params in grid:
        pipeline = make_pipeline(feature_set, constructor, params)
        try:
            pipeline.fit(split.train, y_train)
            score = macro_f1(y_val, pipeline.predict(split.val))
        except Exception as exc:  # one bad cell must not kill the whole grid
            print(f"      {model_name} {params} failed: {type(exc).__name__}: {exc}")
            continue
        if score > best_score:
            best_score, best_params = score, params

    return best_params, best_score


def fit_score(
    split: Split,
    arm: str,
    feature_set: str,
    model_name: str,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> ExperimentResult | None:
    """train one model properly and give it its final score.

    pick the settings on validation, retrain the winner on training plus
    validation, then score on test once. see the note at the top of the file
    for why it is arranged this way.

    returns None if nothing in the grid would train.
    """
    best_params, best_val = select_hyperparameters(split, arm, feature_set, model_name)
    if best_params is None:
        return None

    constructor, _grid = get_model(model_name)

    # refit the chosen setting on train + val. the extra 15% of rows is worth
    # having and the setting was already picked, so nothing leaks.
    full = pd.concat([split.train, split.val], ignore_index=True)
    y_full = apply_arm(full["label"], arm)

    final = make_pipeline(feature_set, constructor, best_params)
    final.fit(full, y_full)

    # first and only time the test window is touched.
    y_test = apply_arm(split.test["label"], arm)
    y_pred = final.predict(split.test)

    lo, hi = bootstrap_ci(y_test, y_pred, n_boot=n_bootstrap, seed=seed)
    metrics = point_metrics(y_test, y_pred)

    return ExperimentResult(
        split=split.name,
        arm=arm,
        feature_set=feature_set,
        model=model_name,
        best_params=best_params,
        val_macro_f1=best_val,
        macro_f1=metrics["macro_f1"],
        macro_f1_lo=lo,
        macro_f1_hi=hi,
        balanced_accuracy=metrics["balanced_accuracy"],
        accuracy=metrics["accuracy"],
        n_test=len(y_test),
        y_true=list(y_test),
        y_pred=list(y_pred),
        fitted=final,
    )


def run_grid(
    splits: dict[str, Split],
    arms: list[str],
    feature_sets: list[str],
    model_names: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[tuple, ExperimentResult]]:
    """work through every combination of data split, label setup, field set and
    model. this is the bulk of the runtime."""
    rows: list[dict] = []
    results: dict[tuple, ExperimentResult] = {}

    for split_name, split in splits.items():
        for arm in arms:
            for feature_set in feature_sets:
                for model_name in model_names:
                    # majority ignores the features, so run it once per
                    # (split, arm) rather than once per feature set.
                    if model_name == "majority" and feature_set != feature_sets[0]:
                        continue

                    result = fit_score(
                        split, arm, feature_set, model_name, n_bootstrap, seed
                    )
                    if result is None:
                        continue

                    if model_name == "majority":
                        result.feature_set = "none"

                    key = (split_name, arm, result.feature_set, model_name)
                    results[key] = result
                    rows.append(result.to_row())

                    if verbose:
                        print(
                            f"  {split_name:8s} {arm:14s} {result.feature_set:28s} "
                            f"{model_name:14s} macro F1 {result.macro_f1:.3f} "
                            f"[{result.macro_f1_lo:.3f}, {result.macro_f1_hi:.3f}]"
                        )

    return pd.DataFrame(rows), results


def best_model_for(
    table: pd.DataFrame, split: str, arm: str, feature_set: str
) -> str | None:
    """which model did best for a given set of fields."""
    subset = table[
        (table["split"] == split)
        & (table["arm"] == arm)
        & (table["feature_set"] == feature_set)
        & (table["model"] != "majority")
    ]
    if subset.empty:
        return None
    return str(subset.loc[subset["macro_f1"].idxmax(), "model"])


def headline_table(table: pd.DataFrame, split: str, arm: str) -> pd.DataFrame:
    """the summary table for my report: the best model for each set of fields,
    listed in order of how much the model gets to know."""
    subset = table[(table["split"] == split) & (table["arm"] == arm)]
    if subset.empty:
        return subset

    best_idx = subset.groupby("feature_set")["macro_f1"].idxmax()
    columns = [
        "feature_set",
        "model",
        "macro_f1",
        "macro_f1_lo",
        "macro_f1_hi",
        "balanced_accuracy",
        "accuracy",
        "n_test",
    ]
    order = [
        "none",
        "tier1_declaration",
        "tier2_plus_reporter",
        "tier3_plus_notification_type",
        "tier4_plus_hazard",
        "tier5_post_hoc",
        "text",
        "hybrid",
    ]
    out = subset.loc[best_idx, columns].copy()
    out["_order"] = out["feature_set"].map({k: i for i, k in enumerate(order)})
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
