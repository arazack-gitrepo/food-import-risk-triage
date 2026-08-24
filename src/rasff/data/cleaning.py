"""tidying the raw records into something usable.

one job per function, and none of them change the data they are given, they
return a new copy. that way I can stop after any step and look at what the
data looks like at that point.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from rasff.config import CATEGORICAL_COLUMNS, TEXT_COLUMN

# the hazard field looks like "aflatoxins (B1 = 12 ug/kg) {mycotoxins}".
# the bit in curly brackets is the general category, which is what I use.
# the text before it names the exact substance, and there are far too many
# different ones for a model to learn anything useful from.
_BRACE = re.compile(r"\{([^}]*)\}")


class DateParseError(RuntimeError):
    """raised when the dates come out as something impossible."""


def clean_text_field(value) -> str:
    """tidy a text value: lowercase, trim, single spaces. blanks become empty."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def extract_hazard_category(value) -> float | str:
    """pull the category out of the curly brackets. blank if there isn't one."""
    if pd.isna(value):
        return np.nan
    matches = _BRACE.findall(str(value))
    if not matches:
        return np.nan
    return matches[-1].strip().lower()


def add_hazard_category(frame: pd.DataFrame) -> pd.DataFrame:
    """work out the hazard category for every row.

    about 26% of rows do not have one. I am NOT deleting those rows. having no
    hazard listed tells you something in itself, and throwing away a quarter
    of my data to tidy up one column would cost far more than the column is
    worth. they get marked "unknown" and the model can make of that what it
    will.
    """
    out = frame.copy()
    out["hazard_category"] = out["hazard_substance"].apply(extract_hazard_category)
    return out


def hazard_coverage(frame: pd.DataFrame) -> dict[str, float]:
    """how many rows I managed to get a hazard category out of."""
    blank = (
        frame["hazard_substance"].fillna("").astype(str).str.strip() == ""
    ).mean()
    return {
        "hazards_blank_pct": round(float(blank) * 100, 1),
        "category_coverage_pct": round(
            float(frame["hazard_category"].notna().mean()) * 100, 1
        ),
        "n_categories": int(frame["hazard_category"].nunique()),
    }


def parse_dates(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """read the dates, throwing out anything unreadable.

    one thing to watch: 03/04/2024 could be the 3rd of April or the 4th of
    March. if no value anywhere in the column is above 12, there is no way to
    tell which from the data itself, and I have to check the export format by
    hand. the flag this returns warns me when that is the case.
    """
    out = frame.copy()
    out["date"] = pd.to_datetime(
        out["date"], errors="coerce", dayfirst=True, format="mixed"
    )
    n_bad = int(out["date"].isna().sum())
    out = out[out["date"].notna()].copy()

    years = out["date"].dt.year
    if not years.between(2000, 2030).all():
        raise DateParseError(
            f"dates outside 2000-2030 (min {years.min()}, max {years.max()}). "
            "check the export format."
        )

    diagnostics = {
        "unparseable_dropped": n_bad,
        "day_month_ambiguous": bool(
            out["date"].dt.day.max() <= 12 and out["date"].dt.month.max() <= 12
        ),
        "date_min": str(out["date"].min().date()),
        "date_max": str(out["date"].max().date()),
    }
    return out, diagnostics


def normalise_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
    """tidy every category column and mark blanks as "unknown"."""
    out = frame.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(clean_text_field).replace("", "unknown")
    if "hazard_category" in out.columns:
        out["hazard_category"] = out["hazard_category"].fillna("unknown")
    return out


def add_text_column(frame: pd.DataFrame) -> pd.DataFrame:
    """the description line, which is all the text-based models get to see."""
    out = frame.copy()
    out[TEXT_COLUMN] = out["subject"].apply(clean_text_field)
    return out


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """run all the tidying steps in order, oldest shipment first."""
    out = add_hazard_category(frame)
    coverage = hazard_coverage(out)

    out, date_diagnostics = parse_dates(out)
    out = add_text_column(out)
    out = normalise_categoricals(out)

    duplicate_pct = round(float(out[TEXT_COLUMN].duplicated().mean()) * 100, 1)
    out = out.sort_values("date").reset_index(drop=True)
    out["year"] = out["date"].dt.year

    diagnostics = {
        **coverage,
        **date_diagnostics,
        "duplicate_subject_pct": duplicate_pct,
        "rows": len(out),
    }
    return out, diagnostics
