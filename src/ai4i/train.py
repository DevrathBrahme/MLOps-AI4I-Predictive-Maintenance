import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from ai4i.data import FEATURE_COLUMNS, TARGET
from ai4i.model import encode_labels

import argparse
import logging

import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_validate

from ai4i.db import get_connection
from ai4i.data import load_training_data
from ai4i.model import CLASSES, MODEL_NAMES, build_pipeline

TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    X = df[list(FEATURE_COLUMNS)]
    y = encode_labels(df[TARGET])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    return compute_sample_weight("balanced", y)


logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "ai4i-failure-classifier"
SKOPS_TRUSTED_TYPES = {
    "rf": ["sklearn.tree._tree.Tree"],
    "xgb": ['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier'],   # fill in from the error message the first time you train xgb
}
CV_FOLDS = 5
CV_SCORING = ("f1_macro", "balanced_accuracy")


def cross_validate_pipeline(pipeline, X, y, sample_weight, random_state: int = RANDOM_STATE) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)
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


def main() -> None:
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
            "sample_weighting": "balanced"            
        })
        mlflow.log_dict({"classes": list(CLASSES)}, "classes.json")

        cv_metrics = cross_validate_pipeline(build_pipeline(args.model), X_train, y_train, weights)
        mlflow.log_metrics(cv_metrics)

        pipeline.fit(X_train, y_train, model__sample_weight=weights)                       

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline, name="model",
            signature=signature, input_example=X_train.head(5),
            skops_trusted_types=SKOPS_TRUSTED_TYPES[args.model],
        )
        logger.info(
            "Trained %s: cv f1_macro=%.3f, cv balanced_accuracy=%.3f",
            args.model, cv_metrics["cv_f1_macro_mean"], cv_metrics["cv_balanced_accuracy_mean"],
        )                         


if __name__ == "__main__":
    main()