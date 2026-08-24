"""the list of models I try, and the settings tested for each.

experiment.py picks models from here. nothing else creates one.

the settings lists are deliberately short. my question is which INFORMATION
helps, not which knob setting does, and searching hard over a few thousand
rows would just find patterns in the validation data that do not really exist.

"majority" is not filler. it is a model that ignores everything and always
guesses the most common answer. it scores 0.382, and every other number in my
report should be read against that. a model that cannot beat it has learned
nothing.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from rasff.config import SEED

# lightgbm is only for comparison, not what I ship, so if it is not installed
# the project just skips it and carries on rather than refusing to run.
try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False


ModelSpec = tuple[Callable[..., Any], list[dict]]


def _majority(**_kwargs):
    """ignores everything and guesses the most common answer. the bar to beat."""
    return DummyClassifier(strategy="most_frequent")


def _logreg(**kwargs):
    """a simple starting point. weighted so the rarer class still counts."""
    return LogisticRegression(max_iter=2000, class_weight="balanced", **kwargs)


def _linear_svm(**kwargs):
    """usually the best on text. cannot give a probability, only a yes/no, so
    it is left out of the ranking analysis."""
    return LinearSVC(class_weight="balanced", max_iter=5000, **kwargs)


def _random_forest(**kwargs):
    """what I ship. copes well with lots of category columns and needs no
    fiddling to work."""
    return RandomForestClassifier(
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced_subsample",
        **kwargs,
    )


def _lightgbm(**kwargs):
    """for comparison only, never what I ship."""
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed")
    return lgb.LGBMClassifier(
        random_state=SEED, n_jobs=-1, verbose=-1, class_weight="balanced", **kwargs
    )


MODEL_ZOO: dict[str, ModelSpec] = {
    "majority": (_majority, [{}]),
    "logreg": (_logreg, [{"C": c} for c in (0.1, 1.0, 5.0, 20.0)]),
    "linear_svm": (_linear_svm, [{"C": c} for c in (0.05, 0.5, 1.0, 5.0)]),
    "random_forest": (
        _random_forest,
        [{"n_estimators": 200, "min_samples_leaf": m} for m in (2, 5)],
    ),
}

if HAS_LIGHTGBM:
    MODEL_ZOO["lightgbm"] = (
        _lightgbm,
        [
            {"n_estimators": 400, "learning_rate": lr, "num_leaves": leaves}
            for lr in (0.05, 0.1)
            for leaves in (15, 31)
        ],
    )

# what goes in the dashboard and what I quote. I picked random forest over
# lightgbm because it does just as well here and is easier to install
# anywhere. my whole argument is that a cheap simple model is good enough, so
# shipping a heavier one would undercut the point I am making.
DEPLOYMENT_MODEL = "random_forest"


def available_models() -> list[str]:
    """which models are actually available on this machine."""
    return list(MODEL_ZOO)


def get_model(name: str) -> ModelSpec:
    """fetch one model and the settings to try for it."""
    if name not in MODEL_ZOO:
        raise ValueError(f"unknown model {name!r}. available here: {available_models()}")
    return MODEL_ZOO[name]


def supports_probability(name: str) -> bool:
    """can this model give a probability rather than just a yes/no?

    the ranking analysis needs to sort shipments from most to least risky, so
    it needs a number, not a verdict. the SVM cannot give one, so it sits that
    part out.
    """
    return name not in {"linear_svm"}
