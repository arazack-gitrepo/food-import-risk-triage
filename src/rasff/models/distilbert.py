"""DistilBERT: the language model arm. the only bit that needs a GPU.

DistilBERT is a small pre-trained language model. the idea is to see whether
something that genuinely understands the written description does better than
just counting words.

why the code is here and not in the Colab notebook
    it is a model like any other, so it sits with the other models. the
    notebook just calls it. one copy of the code, not two that drift apart.

why it is not in the main model list
    everything in zoo.py is a scikit-learn model that slots straight into the
    normal pipeline. this one needs its own training loop and a GPU, so it
    would not fit.

how it still gets compared fairly
    it hands its predictions back as a plain list, and those go through
    exactly the same comparison as every other model. same test shipments,
    same measure.

torch and transformers are deliberately not in requirements.txt, and this file
is only loaded when asked for, so everything else runs without them installed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from rasff.config import SEED, TEXT_COLUMN
from rasff.data.labels import apply_arm
from rasff.evaluation.metrics import bootstrap_ci, macro_f1, point_metrics
from rasff.evaluation.splits import Split

MODEL_NAME = "distilbert-base-uncased"

# 64 tokens covers a RASFF subject line with room to spare. I checked the
# length distribution before picking it, going longer just pads.
MAX_LENGTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5


@dataclass
class DistilBertResult:
    """the outcome of one DistilBERT run."""

    arm: str
    macro_f1: float
    macro_f1_lo: float
    macro_f1_hi: float
    balanced_accuracy: float
    accuracy: float
    best_epoch: int
    n_test: int
    y_true: np.ndarray
    y_pred: np.ndarray

    def to_row(self) -> dict:
        """just the numbers, in the same shape as every other model's results."""
        return {
            "split": "temporal",
            "arm": self.arm,
            "feature_set": "text",
            "model": "distilbert",
            "best_params": f"best_epoch={self.best_epoch}",
            "macro_f1": self.macro_f1,
            "macro_f1_lo": round(self.macro_f1_lo, 3),
            "macro_f1_hi": round(self.macro_f1_hi, 3),
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "n_test": self.n_test,
        }


def _require_torch():
    """load the deep learning libraries, with a clear message if they are missing."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        return torch, AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "the DistilBERT arm needs torch and transformers. they are not in "
            "requirements.txt on purpose, the rest of the pipeline does not "
            "need them. install with: pip install torch transformers"
        ) from exc


def run_distilbert(
    split: Split,
    arm: str,
    seed: int = SEED,
    epochs: int = EPOCHS,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> DistilBertResult:
    """train DistilBERT on the descriptions, then score it once on the test set.

    same rules as everywhere else in the project: decide when to stop using
    the validation data, then look at the test data once at the end.

    the rare class gets weighted up to match what the other models do. without
    that, one side would be handling the imbalance and the other would not,
    and the comparison would not mean anything.
    """
    torch, AutoTokenizer, AutoModelForSequenceClassification = _require_torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print("device:", device)
        if device.type == "cpu":
            print("  no GPU. this takes ~30 min instead of ~3.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    y_train = apply_arm(split.train["label"], arm)
    y_val = apply_arm(split.val["label"], arm)
    y_test = apply_arm(split.test["label"], arm)

    classes = sorted(set(y_train))
    to_index = {c: i for i, c in enumerate(classes)}

    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

    class SubjectDataset(Dataset):
        """the descriptions chopped into tokens, with their answers."""

        def __init__(self, texts, labels):
            self.encoded = tokeniser(
                list(texts),
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.encoded.items()}
            item["labels"] = self.labels[i]
            return item

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_train = DataLoader(
        SubjectDataset(split.train[TEXT_COLUMN], [to_index[c] for c in y_train]),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    loader_val = DataLoader(
        SubjectDataset(split.val[TEXT_COLUMN], [to_index[c] for c in y_val]),
        batch_size=64,
    )
    loader_test = DataLoader(
        SubjectDataset(split.test[TEXT_COLUMN], [to_index[c] for c in y_test]),
        batch_size=64,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(classes)
    ).to(device)

    counts = np.array([(y_train == c).sum() for c in classes], dtype=float)
    weights = torch.tensor(
        len(y_train) / (len(classes) * counts), dtype=torch.float32, device=device
    )
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(loader_train) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser, int(0.1 * total_steps), total_steps
    )

    @torch.no_grad()
    def predict(loader):
        """run the model over a batch of shipments and return its raw outputs."""
        model.eval()
        chunks = []
        for batch in loader:
            batch.pop("labels")
            moved = {k: v.to(device) for k, v in batch.items()}
            chunks.append(model(**moved).logits.float().cpu())
        return torch.cat(chunks).numpy()

    best_f1, best_state, best_epoch = -1.0, None, -1

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in loader_train:
            labels = batch.pop("labels").to(device)
            moved = {k: v.to(device) for k, v in batch.items()}
            loss = loss_fn(model(**moved).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad()
            running += loss.item()

        val_pred = np.array(classes)[predict(loader_val).argmax(1)]
        val_f1 = macro_f1(y_val, val_pred)
        if verbose:
            print(
                f"  epoch {epoch}  loss {running / len(loader_train):.4f}  "
                f"val macro F1 {val_f1:.4f}"
            )

        # keep the best version rather than whatever the last pass happened to
        # leave behind. otherwise the result depends on where I stopped training,
        # which is arbitrary.
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if verbose:
        print(f"  best epoch {best_epoch} (val {best_f1:.4f}), restoring it")
    model.load_state_dict(best_state)

    # first and only time the test window is touched.
    y_pred = np.array(classes)[predict(loader_test).argmax(1)]

    lo, hi = bootstrap_ci(y_test, y_pred, n_boot=n_bootstrap, seed=seed)
    metrics = point_metrics(y_test, y_pred)

    return DistilBertResult(
        arm=arm,
        macro_f1=metrics["macro_f1"],
        macro_f1_lo=lo,
        macro_f1_hi=hi,
        balanced_accuracy=metrics["balanced_accuracy"],
        accuracy=metrics["accuracy"],
        best_epoch=best_epoch,
        n_test=len(y_test),
        y_true=y_test,
        y_pred=y_pred,
    )
