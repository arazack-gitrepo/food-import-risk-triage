"""deciding what counts as "serious" - the thing the model is trying to predict.

the EU records a risk decision in words, and I have to turn that into
something a model can learn. that happens in two steps on purpose.

assign_labels decides which words mean serious. that is a judgement call, so
which set of rules to use gets passed in rather than hardcoded, and I test
all three.

apply_arm then either keeps three categories or squashes them into a simple
serious / not serious, depending on which experiment is running.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rasff.config import LABEL_SCHEMES, NEGATIVE_CLASS, POSITIVE_CLASS


class LabelMappingError(RuntimeError):
    """raised when the file contains a risk wording I have not accounted for."""


def assign_labels(frame: pd.DataFrame, scheme: str = "baseline") -> pd.DataFrame:
    """label every shipment serious or not, using the chosen set of rules.

    if the file contains a wording I have not seen before, this stops and
    tells me which one. the alternative is those rows quietly becoming blanks
    and disappearing later without anyone noticing.
    """
    if scheme not in LABEL_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}. options: {list(LABEL_SCHEMES)}")

    out = frame.copy()
    out["label"] = out["risk_decision"].map(LABEL_SCHEMES[scheme])

    unmapped = sorted(out.loc[out["label"].isna(), "risk_decision"].unique())
    if unmapped:
        raise LabelMappingError(
            f"risk_decision values with no entry in scheme {scheme!r}: {unmapped}. "
            "add them to LABEL_SCHEMES in config.py."
        )
    return out


def select_window(
    frame: pd.DataFrame, year_min: int, drop_undecided: bool = True
) -> tuple[pd.DataFrame, dict]:
    """keep only 2023 onwards, and drop the "undecided" rows.

    "undecided" is a leftover from the old pre-2023 system, not the EU saying
    they were unsure. treating it as a real third answer would be reading
    something into it that isn't there. it gets excluded here and looked at
    separately.
    """
    out = frame[frame["year"] >= year_min].copy()
    n_before = len(out)

    n_undecided = int((out["label"] == "undecided").sum())
    if drop_undecided:
        out = out[out["label"] != "undecided"].copy()

    out = out.sort_values("date").reset_index(drop=True)

    counts = {
        "rows_in_window": n_before,
        "undecided_dropped": n_undecided if drop_undecided else 0,
        "rows_analysed": len(out),
        "label_counts": out["label"].value_counts().to_dict(),
    }
    return out, counts


def apply_arm(labels, arm: str) -> np.ndarray:
    """either keep three categories or squash down to serious / not serious.

    the two-way version is my headline, because the three-way one has only a
    couple of examples in its smallest category, which is not enough to
    measure anything reliably.
    """
    series = pd.Series(labels).astype(str)
    if arm == "binary_serious":
        return series.where(series == POSITIVE_CLASS, NEGATIVE_CLASS).to_numpy()
    if arm == "three_class":
        return series.to_numpy()
    raise ValueError(f"unknown arm {arm!r}. options: ['binary_serious', 'three_class']")


def regime_tables(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """break the labels down year by year.

    this is my evidence for only using 2023 onwards. if the proportions jump
    sharply between one year and the next, the EU changed its rules, and a
    model trained across that line is learning two systems at once.
    """
    raw_by_year = (
        pd.crosstab(frame["year"], frame["risk_decision"], normalize="index")
        .mul(100)
        .round(1)
    )
    label_by_year = (
        pd.crosstab(frame["year"], frame["label"], normalize="index").mul(100).round(1)
    )
    return {"raw_by_year": raw_by_year, "label_by_year": label_by_year}
