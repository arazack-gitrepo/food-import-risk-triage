"""dividing the data into training, validation and test.

I split by date, not at random. in real use the model is trained on the past
and applied to the future, so it should be tested that way: train on the
earliest shipments, test on the latest.

most published work on this dataset splits randomly instead, which quietly
makes the results look better because the model gets to see shipments from
the same period it is being tested on. I run the random split too, purely so
I can report how much difference it makes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from rasff.config import SEED, TRAIN_FRAC, VAL_FRAC


class SplitIntegrityError(RuntimeError):
    """a temporal split came out in the wrong chronological order."""


@dataclass(frozen=True)
class Split:
    """one way of dividing the data into three groups."""

    name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        """how many shipments ended up in each group."""
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }

    def as_tuple(self):
        """the three groups, for unpacking."""
        return self.train, self.val, self.test


def temporal_split(
    frame: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC
) -> Split:
    """split by date. oldest shipments to train on, newest to test on."""
    ordered = frame.sort_values("date").reset_index(drop=True)
    n = len(ordered)
    cut_train = int(n * train_frac)
    cut_val = int(n * (train_frac + val_frac))

    split = Split(
        name="temporal",
        train=ordered.iloc[:cut_train].copy(),
        val=ordered.iloc[cut_train:cut_val].copy(),
        test=ordered.iloc[cut_val:].copy(),
    )
    verify_temporal_order(split)
    return split


def random_split(
    frame: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
) -> Split:
    """split at random instead of by date. only used to show how much the
    honest split costs me. never the headline number."""
    train, rest = train_test_split(
        frame, train_size=train_frac, stratify=frame["label"], random_state=seed
    )
    relative = val_frac / (1 - train_frac)
    val, test = train_test_split(
        rest, train_size=relative, stratify=rest["label"], random_state=seed
    )
    return Split(name="random", train=train.copy(), val=val.copy(), test=test.copy())


def verify_temporal_order(split: Split) -> None:
    """double-check the dates really are in order.

    if the data arrived unsorted for any reason, the model would end up
    trained on future shipments and tested on past ones, and nothing would
    warn me. this stops the run if that happens.
    """
    if not (
        split.train["date"].max()
        <= split.val["date"].min()
        <= split.val["date"].max()
        <= split.test["date"].min()
    ):
        raise SplitIntegrityError(
            "temporal split is not in chronological order. the frame was "
            "probably not sorted by date before splitting."
        )


def make_splits(
    frame: pd.DataFrame, names: list[str], seed: int = SEED
) -> dict[str, Split]:
    """build whichever splits were asked for."""
    builders = {
        "temporal": lambda: temporal_split(frame),
        "random": lambda: random_split(frame, seed=seed),
    }
    unknown = [n for n in names if n not in builders]
    if unknown:
        raise ValueError(f"unknown split(s) {unknown}. options: {list(builders)}")
    return {name: builders[name]() for name in names}


def describe_split(split: Split) -> dict:
    """the row counts and date ranges, for the run log."""
    info = {"split": split.name, **split.sizes()}
    if split.name == "temporal":
        info["train_ends"] = str(split.train["date"].max().date())
        info["test_starts"] = str(split.test["date"].min().date())
    info["test_label_counts"] = split.test["label"].value_counts().to_dict()
    return info
