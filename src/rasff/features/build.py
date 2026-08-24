"""turning shipment records into numbers a model can learn from.

a model cannot read "Pakistan" or "aflatoxin", so every field has to be
converted into numbers first. that is all this file does.

called from make_pipeline() in experiment.py and nowhere else.

the important detail: this hands back an unprepared converter, not a prepared
one. it gets prepared later, inside the pipeline, using only the training
shipments. if it were prepared here on everything, it would have already seen
the test data.

which fields belong to which version of the model is decided in config.py.
this file only decides how to convert them.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from rasff.config import FEATURE_TIERS, TEXT_COLUMN


def make_one_hot() -> OneHotEncoder:
    """turn a category like origin country into yes/no columns, one per value.

    "ignore" matters in real use: if a shipment turns up from a country that
    never appeared in the training data, it still has to get a score rather
    than crashing the system.

    min_frequency=2 lumps together any value that only ever appeared once,
    rather than giving each of them their own column to overfit on.
    """
    return OneHotEncoder(
        handle_unknown="ignore", min_frequency=2, sparse_output=True
    )


def make_tfidf() -> TfidfVectorizer:
    """turn the written description into numbers.

    counts words and word pairs, weighting down ones that appear everywhere
    and so carry no information.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def tier_columns(name: str) -> list[str]:
    """which fields belong to one step on the ladder."""
    if name not in FEATURE_TIERS:
        raise ValueError(f"unknown tier {name!r}. options: {list(FEATURE_TIERS)}")
    return list(FEATURE_TIERS[name].columns)


def build_features(name: str) -> ColumnTransformer:
    """build the converter for one set of fields.

    accepts any tier name from config, plus "text" for the written
    description on its own, and "hybrid" for the paperwork fields and the
    description together.
    """
    if name in FEATURE_TIERS:
        return ColumnTransformer([("cat", make_one_hot(), tier_columns(name))])

    if name == "text":
        return ColumnTransformer([("txt", make_tfidf(), TEXT_COLUMN)])

    if name == "hybrid":
        from rasff.config import DEPLOYMENT_TIER

        return ColumnTransformer(
            [
                ("cat", make_one_hot(), tier_columns(DEPLOYMENT_TIER)),
                ("txt", make_tfidf(), TEXT_COLUMN),
            ]
        )

    raise ValueError(f"unknown feature set {name!r}.")


def feature_width(name: str, frame) -> int:
    """how many numeric columns this set of fields turns into."""
    return build_features(name).fit_transform(frame).shape[1]
