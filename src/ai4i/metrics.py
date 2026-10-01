from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from ai4i.model import CLASSES

LABEL_CODES = tuple(range(len(CLASSES)))

def _check_proba_shape(proba: np.ndarray, y_true: np.ndarray):
    if proba.shape != (len(y_true), len(CLASSES)):
        raise ValueError(f"proba shape {proba.shape}, expected ({len(y_true)}, {len(CLASSES)})")

def per_class_metrics(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    _check_proba_shape(proba, y_true)
    y_pred = proba.argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_CODES, zero_division=0)
    roc_auc = [roc_auc_score(y_true == k, proba[:, k]) for k in LABEL_CODES]

    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "support": support,
        },
        index=pd.Index(CLASSES, name="class")
    )

def confusion_matrices(y_true: np.ndarray, proba: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    _check_proba_shape(proba, y_true)
    y_pred = proba.argmax(axis=1)
    raw = confusion_matrix(y_true, y_pred, labels=LABEL_CODES)
    norm = confusion_matrix(y_true, y_pred, labels=LABEL_CODES, normalize="true")
    index = pd.Index(CLASSES, name="true")
    columns = pd.Index(CLASSES, name="predicted")
    return (
        pd.DataFrame(raw, index=index, columns=columns),
        pd.DataFrame(norm, index=index, columns=columns),
    )

def per_class_fold_std(y_true: np.ndarray, proba: np.ndarray, folds: Sequence[np.ndarray]) -> pd.DataFrame:
    frames = [
        per_class_metrics(y_true[idx], proba[idx]).drop(columns="support")
        for idx in folds
    ]
    stacked = np.stack([frame.to_numpy() for frame in frames])
    std = np.std(stacked, axis=0, ddof=0)
    return pd.DataFrame(std, index=frames[0].index, columns=frames[0].columns)