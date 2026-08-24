"""all the settings live here.

every file path, every fixed number, and the definition of which fields go
into which model. no other file has a hardcoded path or a magic number in it,
so if you want to change how the project runs, you change it here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

RAW_CSV = DATA_DIR / "RASFF_window.csv"

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
PREDS_DIR = RESULTS_DIR / "predictions"

ALL_OUTPUT_DIRS = (RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, PREDS_DIR)

SEED = 42
N_BOOTSTRAP = 1000

# I only use 2023 onwards. the EU changed how they categorise risk during
# 2023, so including 2022 would mean training the model on two different sets
# of rules at the same time. the year-by-year breakdown that shows this is in
# results/tables/regime_label_by_year.csv.
YEAR_MIN = 2023

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test is whatever is left

# the RASFF portal has renamed its columns between versions, so instead of
# assuming a fixed layout I list every name a column might have and look for
# whichever one is actually there.
SCHEMA_CANDIDATES: dict[str, list[str]] = {
    "ref": ["reference", "ref", "notification_reference"],
    "date": ["date", "notification_date", "date_of_case"],
    "notification_type": ["classification", "notification_type", "type_of_notification"],
    "subject": ["subject", "title", "description"],
    "product_category": ["category", "product_category"],
    "product_type": ["type", "product_type"],
    "hazard_substance": ["hazards", "hazard", "substance"],
    "origin_country": ["origin", "country_of_origin", "origin_country"],
    "notifying_country": ["notifying_country", "notifying", "notified_by"],
    "distribution_status": ["distribution", "distribution_status"],
    "risk_decision": ["risk_decision", "risk", "decision"],
    "for_attention": ["forattention", "for_attention"],
    "for_follow_up": ["forfollowup", "for_follow_up"],
    "operator": ["operator"],
}

# some exports have both a "type" and a "product_type" column, and the list
# above would grab the wrong one. this pins it.
COLUMN_OVERRIDES: dict[str, str] = {"product_type": "type"}

REQUIRED_COLUMNS = [
    "date",
    "subject",
    "risk_decision",
    "hazard_substance",
    "product_category",
    "origin_country",
]

CATEGORICAL_COLUMNS = [
    "notification_type",
    "product_category",
    "product_type",
    "hazard_category",
    "origin_country",
    "notifying_country",
    "distribution_status",
    "risk_decision",
    "for_attention",
    "for_follow_up",
    "operator",
]

LABEL_MAP: dict[str, str] = {
    "serious": "serious",
    "potentially serious": "serious",
    "potential risk": "not_serious",
    "not serious": "not_serious",
    "no risk": "no_risk",
    "undecided": "undecided",
}

# "serious" is a judgement call, so I test whether my conclusions change
# depending on where the line is drawn. strict, lenient, and my default. all
# three get run and all three get reported.
LABEL_SCHEMES: dict[str, dict[str, str]] = {
    "baseline": LABEL_MAP,
    "conservative": {
        "serious": "serious",
        "potentially serious": "serious",
        "potential risk": "serious",
        "not serious": "not_serious",
        "no risk": "no_risk",
        "undecided": "undecided",
    },
    "strict": {
        "serious": "serious",
        "potentially serious": "not_serious",
        "potential risk": "not_serious",
        "not serious": "not_serious",
        "no risk": "no_risk",
        "undecided": "undecided",
    },
}

NEGATIVE_CLASS = "not_serious_or_no_risk"
POSITIVE_CLASS = "serious"

LABEL_ARMS = ["binary_serious", "three_class"]

# this is the core idea of the project.
#
# a shipment does not reveal everything about itself at once. some things are
# on the paperwork before anyone opens the box. some are only known after an
# official has looked at it and formed an opinion. some only after a lab test.
#
# so instead of throwing every field at the model, I built five versions, each
# knowing a bit more than the last. the jump in score between two versions
# tells me exactly what that extra knowledge was worth.
#
# tier3 is the interesting one. notification_type is basically an official
# saying how worried they already are, so a model using it is partly reading
# back an answer rather than working it out. tier4 adds the hazard, which
# normally comes from a lab. tier4 is where my headline 0.839 comes from, and
# tier1 is what is actually available at the border.


@dataclass(frozen=True)
class FeatureTier:
    """one step on the ladder: which fields, and when they become known."""

    name: str
    columns: list[str]
    available_at: str
    note: str = ""


TIER1 = FeatureTier(
    name="tier1_declaration",
    columns=["product_category", "product_type", "origin_country"],
    available_at="pre-inspection",
    note="what is on the shipping paperwork. any country could use this.",
)

TIER2 = FeatureTier(
    name="tier2_plus_reporter",
    columns=TIER1.columns + ["notifying_country"],
    available_at="pre-inspection (RASFF network only)",
    note="which EU country reported it. useless outside the EU system.",
)

TIER3 = FeatureTier(
    name="tier3_plus_notification_type",
    columns=TIER2.columns + ["notification_type"],
    available_at="at filing",
    note="how alarmed the official already was when they filed it.",
)

TIER4 = FeatureTier(
    name="tier4_plus_hazard",
    columns=TIER3.columns + ["hazard_category"],
    available_at="post-analysis",
    note="what the contaminant was. usually you need a lab test to know.",
)

TIER5 = FeatureTier(
    name="tier5_post_hoc",
    columns=TIER4.columns
    + ["distribution_status", "for_attention", "for_follow_up", "operator"],
    available_at="post-handling",
    note="filled in after the shipment was dealt with. shows the best case only.",
)

FEATURE_TIERS: dict[str, FeatureTier] = {
    t.name: t for t in (TIER1, TIER2, TIER3, TIER4, TIER5)
}

# the version I actually ship and quote as the usable number, because it is
# the only one that works before a shipment is opened.
DEPLOYMENT_TIER = TIER1.name

# what I called "structured" in my first draft. kept so my report can quote
# both numbers and explain why they differ.
LEGACY_STRUCTURED_TIER = TIER4.name

TEXT_COLUMN = "text"

FEATURE_SETS = list(FEATURE_TIERS) + ["text", "hybrid"]


@dataclass
class ExperimentConfig:
    """the settings for a single run, so I can change one thing without
    editing the file."""

    csv_path: Path = RAW_CSV
    year_min: int = YEAR_MIN
    seed: int = SEED
    n_bootstrap: int = N_BOOTSTRAP
    arms: list[str] = field(default_factory=lambda: list(LABEL_ARMS))
    feature_sets: list[str] = field(default_factory=lambda: list(FEATURE_SETS))
    splits: list[str] = field(default_factory=lambda: ["temporal", "random"])
    budgets: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30, 0.50])


def ensure_output_dirs() -> None:
    """create the output folders if they are not there. safe to run repeatedly."""
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
