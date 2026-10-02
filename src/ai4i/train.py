import argparse
import hashlib
import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from ai4i.data import FEATURE_COLUMNS, TARGET, load_training_data
from ai4i.db import get_connection
from ai4i.metrics import (
    confusion_matrices,
    confusion_to_dict,
    false_alarms,
    misclassified_rows,
    per_class_fold_std,
    per_class_metrics,
    to_metric_dict,
)
from ai4i.model import CLASSES, MODEL_NAMES, build_pipeline, encode_labels


TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
CV_SCORING = ("f1_macro", "balanced_accuracy")
EXPERIMENT_NAME = "ai4i-failure-classifier"
SKOPS_TRUSTED_TYPES = {
    "rf": ["sklearn.tree._tree.Tree"],
    "xgb": ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"],
}
logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split features and encoded labels into a stratified train/test holdout.

    Returns X_train, X_test, y_train, y_test. X keeps the udi index; y holds
    the integer codes defined by CLASSES.
    """
    X = df[list(FEATURE_COLUMNS)]
    y = encode_labels(df[TARGET])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def dataset_fingerprint(X: pd.DataFrame, y: np.ndarray) -> str:
    """Return a SHA-256 fingerprint of the rows of (X, y).

    Changes if any row is added or removed, or any value, label or index (udi)
    changes; independent of row order. Used to verify that the sealed test set
    rebuilt at evaluation time is the one held out at training time.
    """
    frame = X.assign(label=y)
    frame = frame.sort_index()
    row_hashes = pd.util.hash_pandas_object(frame, index=True)
    digest = hashlib.sha256()
    for row_hash in row_hashes:
        digest.update(row_hash.to_bytes(8, "big"))
    return digest.hexdigest()


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    """Return per-row weights n / (k * n_c), so every class contributes equally to the fit."""
    return compute_sample_weight("balanced", y)


def make_cv(random_state: int) -> StratifiedKFold:
    """Return the stratified K-fold splitter shared by every CV computation.

    An int random_state re-seeds on each split() call, so the folds are
    identical wherever this splitter is used.
    """
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)


def cv_test_folds(X: pd.DataFrame, y: np.ndarray, random_state: int = RANDOM_STATE) -> list[np.ndarray]:
    """Return the validation-row positions of each CV fold, in fold order."""
    return [val for train, val in make_cv(random_state).split(X, y)]


def cross_validate_pipeline(pipeline, X, y, sample_weight, random_state: int = RANDOM_STATE) -> dict:
    """Return the mean and std (ddof=0) across folds of each CV_SCORING metric.

    Weights are routed to the model step for fitting only; scoring is unweighted.
    """
    cv = make_cv(random_state)
    results = cross_validate(
        pipeline, X, y, cv=cv, scoring=list(CV_SCORING),
        params={"model__sample_weight": sample_weight},
    )
    metrics = {}
    for name in CV_SCORING:
        scores = results[f"test_{name}"]
        metrics[f"cv_{name}_mean"] = float(np.mean(scores))
        metrics[f"cv_{name}_std"] = float(np.std(scores))
    return metrics


def out_of_fold_proba(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    random_state: int,
) -> np.ndarray:
    """Return out-of-fold class probabilities, shape (n_rows, len(CLASSES)).

    Each row is predicted by the fold model that did not train on it; columns
    follow CLASSES. The pipeline is cloned per fold, so the one passed in stays
    unfitted. Weights affect fitting only.
    """
    return cross_val_predict(
        pipeline,
        X,
        y,
        cv=make_cv(random_state),
        method="predict_proba",
        params={"model__sample_weight": sample_weight},
    )


def log_cv_evaluation(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    weights: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Log the out-of-fold evaluation to the active MLflow run; return the per-class table.

    Logs pooled per-class metrics (oof_*), their fold std (cv_*_std), macro
    ROC-AUC, the false-alarm count, both confusion matrices and the
    misclassified rows indexed by udi.
    """
    proba = out_of_fold_proba(build_pipeline(model_name, random_state), X_train, y_train, weights, random_state)
    table = per_class_metrics(y_train, proba)
    spread = per_class_fold_std(y_train, proba, cv_test_folds(X_train, y_train, random_state))
    raw, norm = confusion_matrices(y_train, proba)

    mlflow.log_metrics(to_metric_dict(table, "oof"))
    mlflow.log_metrics(to_metric_dict(spread, "cv", "_std"))
    mlflow.log_metrics({
        "oof_roc_auc_macro": float(np.mean([table.loc[cls, "roc_auc"] for cls in table.index])),
        "oof_false_alarms": false_alarms(raw),
    })

    mlflow.log_dict(confusion_to_dict(raw), "cv/confusion_matrix_counts.json")
    mlflow.log_dict(confusion_to_dict(norm), "cv/confusion_matrix_row_normalized.json")
    errors = misclassified_rows(X_train.index, y_train, proba)
    mlflow.log_table(errors.reset_index(), artifact_file="cv/oof_errors.json")
    return table


def main() -> None:
    """Train one model, log its CV and out-of-fold evaluation, and save it to MLflow.

    The test set is only fingerprinted here, never evaluated.
    """
    parser = argparse.ArgumentParser(description="Train a failure-mode classifier and log it to MLflow.")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with get_connection() as conn:
        df = load_training_data(conn)
    X_train, _X_test, y_train, _y_test = split_data(df)
    weights = balanced_sample_weights(y_train)

    pipeline = build_pipeline(args.model)
    model_params = pipeline.named_steps["model"].get_params()
    hyperparams = {
        k: model_params[k]
        for k in ("n_estimators", "max_depth", "learning_rate")
        if model_params.get(k) is not None
    }

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=args.model):
        mlflow.log_params({
            "model_name": args.model,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cv_folds": CV_FOLDS,
            "n_train": len(X_train),
            **hyperparams,
            "sample_weighting": "balanced",
            "n_test": len(_X_test),
            "test_fingerprint": dataset_fingerprint(_X_test, _y_test),
        })
        mlflow.log_dict({"classes": list(CLASSES)}, "classes.json")

        cv_metrics = cross_validate_pipeline(build_pipeline(args.model), X_train, y_train, weights)
        mlflow.log_metrics(cv_metrics)
        oof_table = log_cv_evaluation(args.model, X_train, y_train, weights)

        pipeline.fit(X_train, y_train, model__sample_weight=weights)

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline, name="model",
            signature=signature, input_example=X_train.head(5),
            skops_trusted_types=SKOPS_TRUSTED_TYPES[args.model],
        )
        logger.info(
            "Trained %s: cv f1_macro=%.3f, cv balanced_accuracy=%.3f, oof TWF recall=%.3f",
            args.model, cv_metrics["cv_f1_macro_mean"], cv_metrics["cv_balanced_accuracy_mean"],
            oof_table.loc["TWF", "recall"],
        )


if __name__ == "__main__":
    main()