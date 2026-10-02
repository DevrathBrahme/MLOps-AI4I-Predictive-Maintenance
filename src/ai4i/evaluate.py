"""One-time evaluation of the champion run on the sealed test set."""

import argparse
import logging

import mlflow
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from ai4i.data import load_training_data
from ai4i.db import get_connection
from ai4i.metrics import (
    LABEL_CODES,
    confusion_matrices,
    confusion_to_dict,
    false_alarms,
    misclassified_rows,
    per_class_metrics,
    to_metric_dict,
)
from ai4i.train import EXPERIMENT_NAME, dataset_fingerprint, split_data


SEALED_TAG = "sealed_test_evaluated"
logger = logging.getLogger(__name__)


def ensure_not_evaluated() -> None:
    """Raise RuntimeError if any run in the experiment was already evaluated on the sealed test set."""
    done = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.{SEALED_TAG} = 'true'",
    )
    if not done.empty:
        raise RuntimeError(f"Sealed test set already evaluated on run(s) {done['run_id'].tolist()}")


def verify_fingerprint(run: mlflow.entities.Run, X_test: pd.DataFrame, y_test) -> None:
    """Raise RuntimeError unless the rebuilt test set matches the fingerprint logged at training."""
    expected = run.data.params["test_fingerprint"]
    actual = dataset_fingerprint(X_test, y_test)
    if actual != expected:
        raise RuntimeError(f"Test set changed since training: expected {expected}, got {actual}")


def compute_test_results(index: pd.Index, y_test, proba):
    """Compute everything to log for the sealed test set, before the run is reopened.
    Returns (metrics, matrices, errors, table): test_* metrics, confusion matrices keyed
    by artifact path, misclassified rows indexed by udi, and the per-class table."""
    y_pred = proba.argmax(axis=1)
    table = per_class_metrics(y_test, proba)
    raw, norm = confusion_matrices(y_test, proba)
    metrics = {
        **to_metric_dict(table, "test"),
        "test_roc_auc_macro": float(table["roc_auc"].mean()),
        "test_false_alarms": false_alarms(raw),
        "test_f1_macro": float(f1_score(y_test, y_pred, labels=LABEL_CODES, average="macro")),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }
    matrices = {
        "test/confusion_matrix_counts.json": confusion_to_dict(raw),
        "test/confusion_matrix_row_normalised.json": confusion_to_dict(norm),
    }
    errors = misclassified_rows(index, y_test, proba)
    return metrics, matrices, errors, table


def main() -> None:
    """Evaluate one run's logged model once on the sealed test set and log into that run.
    Refuses to run if the test set was already evaluated or has changed since training."""
    parser = argparse.ArgumentParser(description="Evaluate the champion run once on the sealed test set.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ensure_not_evaluated()
    run = mlflow.get_run(args.run_id)
    model_name = run.data.params["model_name"]

    with get_connection() as conn:
        df = load_training_data(conn)
    _, X_test, _, y_test = split_data(df)
    verify_fingerprint(run, X_test, y_test)

    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")
    proba = model.predict_proba(X_test)
    metrics, matrices, errors, table = compute_test_results(X_test.index, y_test, proba)

    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metrics(metrics)
        for path, matrix in matrices.items():
            mlflow.log_dict(matrix, path)
        mlflow.log_table(errors.reset_index(), artifact_file="test/errors.json")
        mlflow.set_tag(SEALED_TAG, "true")

    logger.info("Sealed test evaluation of %s (%s): macro-F1=%.3f, TWF recall=%.3f, false alarms=%d",
                model_name, args.run_id,
                metrics["test_f1_macro"],
                metrics["test_recall_TWF"],
                metrics["test_false_alarms"])
    logger.info("Per-class test metrics:\n%s",
                table.round(3).to_string())


if __name__ == "__main__":
    main()