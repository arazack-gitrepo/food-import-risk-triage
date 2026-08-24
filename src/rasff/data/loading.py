"""reading the RASFF file off disk and working out which column is which.

the export is not clean. some rows are broken and the column names change
between versions of the portal, so this file deals with both before anything
else touches the data.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

from rasff.config import COLUMN_OVERRIDES, REQUIRED_COLUMNS, SCHEMA_CANDIDATES

_DATE_START = re.compile(r"^\s*\d{1,2}[-/]\d{1,2}[-/]\d{4}")


class SchemaError(RuntimeError):
    """raised when the file is missing a column the project cannot run without."""


def normalise_headers(columns) -> list[str]:
    """tidy up column names so they are all lowercase with underscores."""
    cleaned = []
    for col in columns:
        text = str(col).strip().lower()
        cleaned.append(re.sub(r"[^a-z0-9]+", "_", text).strip("_"))
    return cleaned


def _repair_row(row: list[str], width: int) -> list[str] | None:
    """fix a row that got split into too many pieces.

    the file is comma-separated, but some descriptions contain commas of their
    own, which fools the reader into thinking the row has extra columns. this
    stitches the description back together. returns None if the repair does
    not look right.
    """
    extra = len(row) - width
    merged = row[:3] + [",".join(row[3 : 4 + extra])] + row[4 + extra :]
    # only accept the repair if the date field still looks like a date,
    # otherwise this happily produces well-shaped nonsense.
    if len(merged) == width and _DATE_START.match(merged[4]):
        return merged
    return None


def read_raw_csv(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """read the file, fixing broken rows as it goes.

    12 rows in my data need fixing. I repair rather than delete them because
    the rows that break are the ones with long detailed descriptions, so
    throwing them away would leave me with a dataset of unusually short
    entries.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. download the RASFF Window export and put it there. "
            "README has the portal filters I used."
        )

    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    header, body = rows[0], rows[1:]
    width = len(header)

    kept: list[list[str]] = []
    repaired = 0
    dropped = 0

    for row in body:
        if len(row) == width:
            kept.append(row)
        elif len(row) > width:
            fixed = _repair_row(row, width)
            if fixed is None:
                dropped += 1
            else:
                kept.append(fixed)
                repaired += 1
        else:
            dropped += 1

    frame = pd.DataFrame(kept, columns=normalise_headers(header))

    deduplicated = 0
    if "reference" in frame.columns:
        duplicates = int(frame["reference"].duplicated().sum())
        if duplicates:
            frame = frame.drop_duplicates(subset="reference", keep="first")
            frame = frame.reset_index(drop=True)
            deduplicated = duplicates

    counts = {
        "rows_in_file": len(body),
        "repaired": repaired,
        "dropped": dropped,
        "deduplicated": deduplicated,
        "rows_loaded": len(frame),
    }
    return frame, counts


def detect_schema(frame: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """work out which column in this file corresponds to which field.

    the portal renames things between versions, so this checks each possible
    name and reports what it found and what is missing.
    """
    found: dict[str, str] = {}
    missing: list[str] = []

    for canonical, candidates in SCHEMA_CANDIDATES.items():
        override = COLUMN_OVERRIDES.get(canonical)
        if override and override in frame.columns:
            found[canonical] = override
            continue
        hit = next((c for c in candidates if c in frame.columns), None)
        if hit:
            found[canonical] = hit
        else:
            missing.append(canonical)

    return found, missing


def apply_schema(frame: pd.DataFrame, found: dict[str, str]) -> pd.DataFrame:
    """rename everything to the names the rest of the project expects.

    if something essential is missing this stops the run immediately, rather
    than letting a model get trained on the wrong columns and finding out
    later.
    """
    absent = [c for c in REQUIRED_COLUMNS if c not in found]
    if absent:
        raise SchemaError(
            f"required columns not in the export: {absent}. "
            "add the real column name to COLUMN_OVERRIDES in config.py."
        )
    rename = {actual: canonical for canonical, actual in found.items()}
    return frame.rename(columns=rename)[list(found)].copy()


def load_raw(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """read the file and sort out its columns, in one go."""
    frame, counts = read_raw_csv(path)
    found, _missing = detect_schema(frame)
    return apply_schema(frame, found), counts
