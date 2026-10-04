"""Unit tests for ai4i.metrics on a hand-computed 10-row toy.

Every expected number was worked out by hand from Y and PRED, not copied
from the code's output, so a wrong formula cannot pass by agreeing with
itself. Class codes follow CLASSES: 0 no_failure, 1 TWF, 2 HDF, 3 PWF, 4 OSF.
"""

import numpy as np
import pandas as pd
import pytest

from ai4i.metrics import (
    confusion_matrices,
    confusion_to_dict,
    false_alarms,
    misclassified_rows,
    per_class_fold_std,
    per_class_metrics,
    to_metric_dict,
)
from ai4i.model import CLASSES

Y = np.array([0, 0, 0, 0, 1, 1, 2, 3, 4, 4])
PRED = np.array([0, 0, 0, 1, 1, 0, 2, 3, 4, 2])
ONE_HOT = np.eye(len(CLASSES))[PRED]

EXPECTED_RAW = [
    [3, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1],
]


def test_per_class_metrics_match_hand_computed_toy():
    """Precision, recall, F1, ROC-AUC and support per class, in CLASSES order."""
    table = per_class_metrics(Y, ONE_HOT)

    assert list(table.index) == list(CLASSES)
    assert table["precision"].tolist() == pytest.approx([0.75, 0.5, 0.5, 1.0, 1.0])
    assert table["recall"].tolist() == pytest.approx([0.75, 0.5, 1.0, 1.0, 0.5])
    assert table["f1"].tolist() == pytest.approx([0.75, 0.5, 2 / 3, 1.0, 2 / 3])
    # With 0/1 scores, ROC-AUC = (1 + recall - false positive rate) / 2.
    assert table["roc_auc"].tolist() == pytest.approx(
        [19 / 24, 11 / 16, 17 / 18, 1.0, 0.75]
    )
    assert table["support"].tolist() == [4, 2, 1, 1, 2]


def test_raw_confusion_matrix_is_true_by_predicted():
    """Rows are the true class, columns the predicted class, both in CLASSES order."""
    raw, _ = confusion_matrices(Y, ONE_HOT)

    assert raw.to_numpy().tolist() == EXPECTED_RAW
    assert raw.index.name == "true"
    assert raw.columns.name == "predicted"
    assert list(raw.index) == list(raw.columns) == list(CLASSES)


def test_normalised_confusion_diagonal_is_recall():
    """Each row of the normalised matrix sums to 1 and its diagonal is recall."""
    _, norm = confusion_matrices(Y, ONE_HOT)
    recall = per_class_metrics(Y, ONE_HOT)["recall"].to_numpy()

    np.testing.assert_allclose(norm.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.diag(norm.to_numpy()), recall)


def test_false_alarms_counts_healthy_rows_flagged_as_failures():
    """One healthy row (position 3) was predicted as TWF."""
    raw, _ = confusion_matrices(Y, ONE_HOT)

    assert false_alarms(raw) == 1


def test_confusion_to_dict_keeps_orientation_and_labels():
    """The JSON form records which axis is true and which is predicted."""
    raw, _ = confusion_matrices(Y, ONE_HOT)

    assert confusion_to_dict(raw) == {
        "rows": "true",
        "columns": "predicted",
        "labels": list(CLASSES),
        "values": EXPECTED_RAW,
    }


def test_to_metric_dict_names_every_score_and_skips_support():
    """Four scores per class, named prefix_metric_class[suffix], as plain floats."""
    table = per_class_metrics(Y, ONE_HOT)
    metrics = to_metric_dict(table, "oof")

    assert len(metrics) == 4 * len(CLASSES)
    assert metrics["oof_recall_TWF"] == pytest.approx(0.5)
    assert not any("support" in name for name in metrics)
    assert all(isinstance(value, float) for value in metrics.values())
    assert "cv_f1_HDF_std" in to_metric_dict(table, "cv", "_std")


def test_fold_std_uses_population_std_across_folds():
    """TWF recall is 1 in fold A and 0 in fold B: std 0.5 with ddof=0 (0.707 with ddof=1)."""
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    pred = np.array([0, 1, 2, 3, 4, 0, 0, 2, 3, 4])
    proba = np.eye(len(CLASSES))[pred]
    folds = [np.arange(0, 5), np.arange(5, 10)]

    std = per_class_fold_std(y, proba, folds)

    assert list(std.columns) == ["precision", "recall", "f1", "roc_auc"]
    assert std.loc["TWF", "recall"] == pytest.approx(0.5)
    assert std.loc["HDF", "recall"] == 0.0


@pytest.mark.parametrize(
    ("y", "proba"),
    [
        pytest.param(Y, ONE_HOT[:, :4], id="missing a class column"),
        pytest.param(Y[:9], ONE_HOT, id="one label short"),
    ],
)
def test_wrong_proba_shape_is_rejected(y, proba):
    """A misaligned probability matrix fails loudly instead of scoring garbage."""
    with pytest.raises(ValueError, match="proba shape"):
        per_class_metrics(y, proba)


def test_misclassified_rows_are_indexed_by_udi_with_runner_up():
    """Only wrong rows are reported; the runner-up is second-highest, never the predicted class."""
    rows = np.arange(len(Y))
    runner_up = np.where(PRED == Y, (PRED + 1) % len(CLASSES), Y)
    proba = np.full((len(Y), len(CLASSES)), 0.05)
    proba[rows, PRED] = 0.60
    proba[rows, runner_up] = 0.25
    udi = pd.Index(range(101, 111), name="udi")

    wrong = misclassified_rows(udi, Y, proba)

    assert list(wrong.index) == [104, 106, 110]
    assert wrong.index.name == "udi"
    assert wrong["true"].tolist() == ["no_failure", "TWF", "OSF"]
    assert wrong["predicted"].tolist() == ["TWF", "no_failure", "HDF"]
    assert wrong["runner_up"].tolist() == wrong["true"].tolist()
    assert wrong["p_predicted"].tolist() == pytest.approx([0.60] * 3)
    assert wrong["p_runner_up"].tolist() == pytest.approx([0.25] * 3)
    assert wrong["p_true"].tolist() == pytest.approx([0.25] * 3)