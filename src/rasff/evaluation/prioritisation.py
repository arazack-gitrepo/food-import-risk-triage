"""does the model actually help an inspector?

a score out of one tells you how good the predictions are. it does not answer
the question an inspector actually has, which is: I can open 10% of the
shipments arriving this week, so which 10%?

that is a ranking problem, so it gets measured differently. I sort every
shipment by how risky the model thinks it is, take the top 10%, and count how
many genuinely serious ones I caught. then I compare that against:

    random          picking shipments blindly, so no model at all
    worst-origin    the rule of thumb inspectors already use, which is to
                    lean on countries with a bad track record
    perfect         a hypothetical inspector who already knew the answers

that last one is the bit people forget. if I can only open 10% of shipments,
I cannot possibly catch more than 10% of them. so when serious cases are 16%
of the total, even a perfect inspector only catches 16%. catching 14.7% looks
terrible on its own and looks very good next to a ceiling of 16.2%. always
show both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rasff.config import POSITIVE_CLASS, SEED

N_RANDOM_TRIALS = 2000


def serious_scores(fitted_pipeline, frame: pd.DataFrame) -> np.ndarray:
    """how likely the model thinks each shipment is to be serious, 0 to 1.

    this number is what everything gets sorted by.
    """
    classes = list(fitted_pipeline.classes_)
    if POSITIVE_CLASS not in classes:
        raise ValueError(f"fitted model has no {POSITIVE_CLASS!r} class; found {classes}.")
    return fitted_pipeline.predict_proba(frame)[:, classes.index(POSITIVE_CLASS)]


def origin_risk_scores(
    train_frame: pd.DataFrame,
    train_labels: np.ndarray,
    test_frame: pd.DataFrame,
    min_count: int = 5,
) -> np.ndarray:
    """the rule of thumb I am competing against: how often has this country's
    food been a problem before?

    countries I have seen fewer than min_count times get the overall average
    instead of their own rate. otherwise a country that appeared once, with
    one bad shipment, would shoot to the top of the queue on a sample of one.
    """
    history = (
        train_frame.assign(is_serious=(train_labels == POSITIVE_CLASS).astype(int))
        .groupby("origin_country")["is_serious"]
        .agg(["mean", "count"])
    )
    reliable = history[history["count"] >= min_count]["mean"]
    return test_frame["origin_country"].map(reliable).fillna(reliable.mean()).to_numpy()


def detection_curve(
    y_test: np.ndarray,
    model_scores: np.ndarray,
    heuristic_scores: np.ndarray | None = None,
    budgets: list[float] | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """how many serious shipments get caught, at each level of how many you can
    afford to open."""
    budgets = budgets or [0.05, 0.10, 0.20, 0.30, 0.50]

    is_serious = (np.asarray(y_test) == POSITIVE_CLASS).astype(int)
    n = len(is_serious)
    n_serious = int(is_serious.sum())
    if n_serious == 0:
        raise ValueError("no serious cases in the test window, nothing to rank for.")

    rng = np.random.default_rng(seed)
    rows = []

    for budget in budgets:
        k = max(1, int(round(n * budget)))

        caught_model = int(is_serious[np.argsort(-model_scores)[:k]].sum())
        caught_perfect = min(k, n_serious)
        caught_random = float(
            np.mean(
                [is_serious[rng.permutation(n)[:k]].sum() for _ in range(N_RANDOM_TRIALS)]
            )
        )

        row = {
            "budget": f"{int(budget * 100)}%",
            "n_inspected": k,
            "model": round(caught_model / n_serious, 3),
            "random": round(caught_random / n_serious, 3),
            "perfect": round(caught_perfect / n_serious, 3),
            "lift_vs_random": round(caught_model / caught_random, 2)
            if caught_random > 0
            else float("nan"),
            "pct_of_ceiling": round(caught_model / caught_perfect * 100, 1),
        }

        if heuristic_scores is not None:
            caught_heuristic = int(is_serious[np.argsort(-heuristic_scores)[:k]].sum())
            row["worst_origin"] = round(caught_heuristic / n_serious, 3)

        rows.append(row)

    columns = [
        "budget",
        "n_inspected",
        "model",
        "random",
        "worst_origin",
        "perfect",
        "lift_vs_random",
        "pct_of_ceiling",
    ]
    frame = pd.DataFrame(rows)
    return frame[[c for c in columns if c in frame.columns]]


def summarise(curve: pd.DataFrame, budget: str = "10%") -> dict:
    """pull out the numbers for one budget level, for quoting in the report."""
    row = curve[curve["budget"] == budget]
    if row.empty:
        raise ValueError(f"budget {budget!r} is not in the curve.")
    record = row.iloc[0]
    return {
        "budget": budget,
        "recall": float(record["model"]),
        "ceiling": float(record["perfect"]),
        "pct_of_ceiling": float(record["pct_of_ceiling"]),
        "lift_vs_random": float(record["lift_vs_random"]),
    }
