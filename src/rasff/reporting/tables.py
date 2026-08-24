"""saving the results to files.

every number in my dissertation comes from a file written by this, produced by
one run of run_experiment.py. nothing is typed in by hand, so nothing can
drift out of date without me noticing.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from rasff.config import TABLES_DIR


def write_table(frame: pd.DataFrame, name: str, directory: Path = TABLES_DIR) -> Path:
    """save a table as a csv."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def write_json(payload: dict, name: str, directory: Path = TABLES_DIR) -> Path:
    """save a set of values as json."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _section(title: str, frame: pd.DataFrame) -> list[str]:
    """format one table as a markdown section."""
    if frame is None or frame.empty:
        return [f"## {title}", "", "_nothing produced for this section._", ""]
    return [f"## {title}", "", frame.to_markdown(index=False), ""]


def write_summary(run_info: dict, sections: dict[str, pd.DataFrame], path: Path) -> Path:
    """the readable summary of everything, which is what I write my report from."""
    lines = [
        "# RASFF import risk triage results",
        "",
        f"generated {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "produced by `python run_experiment.py`. do not edit this by hand, "
        "regenerate it.",
        "",
        "## run configuration",
        "",
        "```json",
        json.dumps(run_info, indent=2, default=str),
        "```",
        "",
    ]
    for title, frame in sections.items():
        lines += _section(title, frame)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
