"""Evaluation metrics for the failure-mode classifier.

Pure functions: arrays in, DataFrames or dicts out. No MLflow, no database and
no knowledge of how folds are built, so the same code scores the out-of-fold
predictions during training and the sealed test set during evaluation.
Class codes follow ai4i.model.CLASSES; the predicted class is the argmax of
the probabilities.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ai4i.model import CLASSES

LABEL_CODES = tuple(range(len(CLASSES)))


def _check_proba_shape(y_true: np.ndarray, proba: np.ndarray) -> None:
    """Raise ValueError unless proba has one row per label and one column per class."""
    expected = (len(y_true), len(CLASSES))
    if proba.shape != expected:
        raise ValueError(f"proba shape {proba.shape}, expected {expected}")


def per_class_metrics(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Return per-class precision, recall, F1, one-vs-rest ROC-AUC and support.

    The predicted class is the argmax of proba. Precision is 0 for a class that
    is never predicted (zero_division=0). Scoring is unweighted. Rows are the
    class names in CLASSES order.
    """
    _check_proba_shape(y_true, proba)
    y_pred = proba.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_CODES,
        zero_division=0,
    )
    roc_auc = [roc_auc_score(y_true == k, proba[:, k]) for k in LABEL_CODES]

    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "support": support,
        },
        index=pd.Index(CLASSES, name="class"),
    )


def per_class_fold_std(
    y_true: np.ndarray,
    proba: np.ndarray,
    folds: Sequence[np.ndarray],
) -> pd.DataFrame:
    """Return the across-fold std of each per-class score.

    folds holds one array of row positions per fold (the validation rows).
    Each fold's slice of the out-of-fold predictions is scored separately, and
    the std uses ddof=0 to match cross_validate_pipeline.
    """
    frames = [
        per_class_metrics(y_true[idx], proba[idx]).drop(columns="support")
        for idx in folds
    ]
    stacked = np.stack([frame.to_numpy() for frame in frames])
    std = np.std(stacked, axis=0, ddof=0)
    return pd.DataFrame(std, index=frames[0].index, columns=frames[0].columns)


def confusion_matrices(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the raw and row-normalised confusion matrices.

    Rows are the true class and columns the predicted class, both in CLASSES
    order. In the row-normalised matrix each row sums to 1, so the diagonal is
    recall.
    """
    _check_proba_shape(y_true, proba)
    y_pred = proba.argmax(axis=1)
    raw = confusion_matrix(y_true, y_pred, labels=LABEL_CODES)
    norm = confusion_matrix(y_true, y_pred, labels=LABEL_CODES, normalize="true")
    index = pd.Index(CLASSES, name="true")
    columns = pd.Index(CLASSES, name="predicted")
    return (
        pd.DataFrame(raw, index=index, columns=columns),
        pd.DataFrame(norm, index=index, columns=columns),
    )


def to_metric_dict(table: pd.DataFrame, prefix: str, suffix: str = "") -> dict[str, float]:
    """Flatten a per-class table into MLflow metrics named prefix_metric_class[suffix].

    The support column is skipped because it is a row count, not a score.
    """
    scores = table.drop(columns="support", errors="ignore")
    return {
        f"{prefix}_{metric}_{cls}{suffix}": float(scores.loc[cls, metric])
        for cls in scores.index
        for metric in scores.columns
    }


def false_alarms(raw: pd.DataFrame) -> int:
    """Return how many truly healthy rows were predicted as any failure mode.

    This is the operator's needless-review count: the no_failure row of the raw
    confusion matrix minus its diagonal cell.
    """
    healthy = raw.loc["no_failure"]
    return int(healthy.sum() - healthy["no_failure"])


def confusion_to_dict(matrix: pd.DataFrame) -> dict:
    """Return a confusion matrix as a JSON-ready dict that keeps its orientation."""
    return {
        "rows": "true",
        "columns": "predicted",
        "labels": list(matrix.index),
        "values": matrix.to_numpy().tolist(),
    }


def misclassified_rows(
    index: pd.Index,
    y_true: np.ndarray,
    proba: np.ndarray,
) -> pd.DataFrame:
    """Return one row per misclassified sample, indexed by its row id (udi).

    Columns: true and predicted class names, the predicted probability, the
    runner-up class (second-highest probability, never the predicted class)
    and its probability, and the probability given to the true class.
    """
    _check_proba_shape(y_true, proba)
    rows = np.arange(len(proba))
    predicted = proba.argmax(axis=1)
    masked = proba.copy()
    masked[rows, predicted] = -np.inf
    runner_up = masked.argmax(axis=1)
    wrong = predicted != y_true
    names = np.asarray(CLASSES)
    return pd.DataFrame(
        {
            "true": names[y_true][wrong],
            "predicted": names[predicted][wrong],
            "p_predicted": proba[rows, predicted][wrong],
            "runner_up": names[runner_up][wrong],
            "p_runner_up": proba[rows, runner_up][wrong],
            "p_true": proba[rows, y_true][wrong],
        },
        index=index[wrong],
    )