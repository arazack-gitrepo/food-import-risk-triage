#!/usr/bin/env python
"""text vs structured, against BOTH baselines.

the first version of the report says text adds nothing over "structured
metadata". that comparison used tier4, which contains hazard_category. this
runs the same paired bootstrap against tier4 and against tier1 so I can state
both numbers and explain why they disagree.

    python scripts/reconcile_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from rasff.config import RAW_CSV, SEED, YEAR_MIN  # noqa: E402
from rasff.data.cleaning import clean  # noqa: E402
from rasff.data.labels import apply_arm, assign_labels, select_window  # noqa: E402
from rasff.data.loading import load_raw  # noqa: E402
from rasff.evaluation.experiment import fit_score  # noqa: E402
from rasff.evaluation.metrics import paired_bootstrap, verdict  # noqa: E402
from rasff.evaluation.prioritisation import (  # noqa: E402
    detection_curve,
    origin_risk_scores,
    serious_scores,
)
from rasff.evaluation.splits import make_splits  # noqa: E402
from rasff.reporting.tables import write_table  # noqa: E402

ARM = "binary_serious"
N_BOOT = 1000
BASELINES = ["tier1_declaration", "tier4_plus_hazard"]
CANDIDATES = ["text", "hybrid"]


def main() -> int:
    """score text against both baselines and print the two answers side by side."""
    raw, _ = load_raw(RAW_CSV)
    frame, _ = clean(raw)
    frame = assign_labels(frame)
    analysis, _ = select_window(frame, YEAR_MIN)
    split = make_splits(analysis, ["temporal"], seed=SEED)["temporal"]

    fitted: dict[str, object] = {}
    print("fitting each feature set: random_forest, temporal split, binary arm\n")
    for name in BASELINES + CANDIDATES:
        result = fit_score(split, ARM, name, "random_forest", N_BOOT, SEED)
        fitted[name] = result
        print(
            f"  {name:28s} macro F1 {result.macro_f1:.3f} "
            f"[{result.macro_f1_lo:.3f}, {result.macro_f1_hi:.3f}]"
        )

    rows = []
    for baseline in BASELINES:
        base = fitted[baseline]
        for candidate in CANDIDATES:
            other = fitted[candidate]
            comparison = paired_bootstrap(
                base.y_true, other.y_pred, base.y_pred, n_boot=N_BOOT, seed=SEED
            )
            rows.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "macro_f1_baseline": base.macro_f1,
                    "macro_f1_candidate": other.macro_f1,
                    "delta": round(comparison["delta"], 3),
                    "ci_lo": round(comparison["ci_lo"], 3),
                    "ci_hi": round(comparison["ci_hi"], 3),
                    "p_no_gain": round(comparison["p_no_gain"], 3),
                    "verdict": verdict(comparison["ci_lo"], comparison["ci_hi"]),
                }
            )

    table = pd.DataFrame(rows)
    write_table(table, "reconciliation_text_vs_baselines")
    print("\ntext and hybrid against each baseline:\n")
    print(table.to_string(index=False))

    # both tiers, so I can quote the deployable figure not just the best one.
    full_train = pd.concat([split.train, split.val], ignore_index=True)
    y_train = apply_arm(full_train["label"], ARM)
    y_test = apply_arm(split.test["label"], ARM)
    heuristic = origin_risk_scores(full_train, y_train, split.test)

    curves = []
    for name in BASELINES:
        curve = detection_curve(
            y_test=y_test,
            model_scores=serious_scores(fitted[name].fitted, split.test),
            heuristic_scores=heuristic,
            seed=SEED,
        )
        curve.insert(0, "feature_set", name)
        curves.append(curve)

    combined = pd.concat(curves, ignore_index=True)
    write_table(combined, "prioritisation_by_tier")
    print("\nprioritisation under an inspection budget:\n")
    print(combined.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
