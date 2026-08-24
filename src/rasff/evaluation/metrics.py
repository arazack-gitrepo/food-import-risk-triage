"""every score in the project is worked out here. nothing else calculates one.

if you want to know how a number in my report was produced, it came out of
one of these functions.

why I score with macro F1 instead of accuracy
    most shipments are not serious, so a model that just says "not serious"
    to everything still gets 61.7% accuracy while being completely useless.
    macro F1 gives the rare class the same weight as the common one, so you
    cannot hide a bad model behind the easy cases.

why every number has a range attached
    I only have 2,300 shipments to test on. if two models differ by a point
    or two, that gap could easily be luck of the draw. so instead of quoting
    one number I resample the test set a thousand times and report the range
    the score falls in. if two models' ranges overlap, I cannot claim one is
    better.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from rasff.config import N_BOOTSTRAP, SEED


def macro_f1(y_true, y_pred) -> float:
    """the main score. if the model never predicts a class, that class gets 0
    rather than crashing."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def point_metrics(y_true, y_pred) -> dict[str, float]:
    """the three headline numbers for one set of predictions."""
    return {
        "macro_f1": round(macro_f1(y_true, y_pred), 3),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 3),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
    }


def _encode(y_true, *predictions) -> tuple[int, np.ndarray, list[np.ndarray]]:
    """swap the text labels for numbers once up front.

    the resampling loop runs thousands of times, and comparing numbers is far
    faster than comparing strings.
    """
    arrays = [np.asarray(y_true)] + [np.asarray(p) for p in predictions]
    labels = np.unique(np.concatenate(arrays))
    index = {label: i for i, label in enumerate(labels)}

    def encode(values):
        """swap each label for its number."""
        return np.fromiter((index[v] for v in values), dtype=np.int64, count=len(values))

    return len(labels), encode(y_true), [encode(p) for p in predictions]


def _macro_f1_from_cells(cells: np.ndarray, k: int) -> float:
    """work out macro F1 directly from a counted-up confusion matrix.

    this is the fast path used inside the resampling loop.
    """
    matrix = np.bincount(cells, minlength=k * k).reshape(k, k)
    true_positives = np.diag(matrix).astype(float)
    false_positives = matrix.sum(axis=0) - true_positives
    false_negatives = matrix.sum(axis=1) - true_positives
    denominator = 2 * true_positives + false_positives + false_negatives
    per_class = np.divide(
        2 * true_positives, denominator, out=np.zeros(k), where=denominator > 0
    )
    return float(per_class.mean())


def bootstrap_ci(
    y_true, y_pred, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float]:
    """how much the score would wobble if I had drawn a different test set.

    I take the 2,300 test rows, draw 2,300 of them at random with repeats,
    score that, and do it a thousand times. the middle 95% of those scores is
    the range I report.

    the obvious way is to call sklearn's f1_score a thousand times, but that
    took 3.4 seconds per model on my test set, about 4 minutes across the
    whole grid for no reason. counting up the confusion matrix myself is
    roughly 85x faster and I checked it gives identical answers.
    """
    k, true_encoded, (pred_encoded,) = _encode(y_true, y_pred)
    cells = true_encoded * k + pred_encoded
    rng = np.random.default_rng(seed)
    n = len(true_encoded)

    scores = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n, n)
        scores[b] = _macro_f1_from_cells(cells[draw], k)

    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def paired_bootstrap(
    y_true, pred_a, pred_b, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> dict[str, float]:
    """is model A actually better than model B, or is the gap just noise?

    the trick is that both models get scored on the SAME resampled rows every
    time. that matters: if I gave each model its own separate range and they
    happened to overlap, that would not prove they are equivalent, because it
    ignores that both were tested on identical shipments. comparing them on
    the same draws measures the gap itself.

    if the returned range includes zero, I cannot claim A beats B.
    p_no_gain is how often A failed to beat B across the thousand draws.
    """
    k, true_encoded, (a_encoded, b_encoded) = _encode(y_true, pred_a, pred_b)
    cells_a = true_encoded * k + a_encoded
    cells_b = true_encoded * k + b_encoded

    rng = np.random.default_rng(seed)
    n = len(true_encoded)

    deltas = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, n, n)
        deltas[b] = _macro_f1_from_cells(cells_a[draw], k) - _macro_f1_from_cells(
            cells_b[draw], k
        )

    observed = _macro_f1_from_cells(cells_a, k) - _macro_f1_from_cells(cells_b, k)
    return {
        "delta": float(observed),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_no_gain": float((deltas <= 0).mean()),
    }


def verdict(ci_lo: float, ci_hi: float) -> str:
    """turn a range into a plain yes/no/cannot-tell."""
    if ci_hi < 0:
        return "significantly worse"
    if ci_lo > 0:
        return "significantly better"
    return "indistinguishable (interval spans zero)"


def confusion(y_true, y_pred) -> tuple[np.ndarray, list]:
    """which classes got confused with which, plus the row and column order."""
    labels = sorted(set(np.asarray(y_true)) | set(np.asarray(y_pred)))
    return confusion_matrix(y_true, y_pred, labels=labels), labels
