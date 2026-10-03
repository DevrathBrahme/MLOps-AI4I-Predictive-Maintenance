"""Choose the elevated-tier threshold out of fold and tag the champion with it."""

import argparse
import logging

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from ai4i.data import load_training_data
from ai4i.db import get_connection
from ai4i.evaluate import verify_fingerprint
from ai4i.metrics import confusion_matrices, confusion_to_dict
from ai4i.model import CLASSES, MODEL_NAMES, build_pipeline
from ai4i.registry import REGISTERED_MODEL_NAME, resolve_champion
from ai4i.risk import (
    NO_FAILURE,
    THRESHOLD_TAG,
    TIERS,
    assign_tiers,
    choose_threshold,
)
from ai4i.train import (
    RANDOM_STATE,
    TEST_SIZE,
    balanced_sample_weights,
    out_of_fold_proba,
    split_data,
)

# D19a, pre-registered: committed before this script first ran. Never tune it.
BUDGET_FRACTION = 0.02
OOF_MATRIX_ARTIFACT = "cv/confusion_matrix_counts.json"

logger = logging.getLogger(__name__)


def check_reproduction(run_id: str, y_train: np.ndarray, proba: np.ndarray) -> None:
    """Raise unless these OOF predictions reproduce the confusion matrix logged in M5.

    Guards against choosing a threshold for a different model configuration,
    split or seed than the one the champion run was trained and scored with.
    """
    logged = mlflow.artifacts.load_dict(f"runs:/{run_id}/{OOF_MATRIX_ARTIFACT}")
    raw, _ = confusion_matrices(y_train, proba)
    reproduced = confusion_to_dict(raw)
    if (
        reproduced["labels"] != logged["labels"]
        or reproduced["values"] != logged["values"]
    ):
        raise RuntimeError(
            f"OOF predictions do not reproduce run {run_id}'s logged matrix: "
            f"got {reproduced['values']}, logged {logged['values']}"
        )


def tier_report(y_train: np.ndarray, tiers: np.ndarray) -> pd.DataFrame:
    """Return a tier x true-class count table (README material)."""
    table = pd.crosstab(
        pd.Series(tiers, name="tier"),
        pd.Series(np.asarray(CLASSES)[y_train], name="true"),
    )
    return table.reindex(index=list(TIERS), columns=list(CLASSES), fill_value=0)


def main() -> None:
    """Recompute OOF probabilities, choose the threshold, tag the champion version."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Choose the elevated-tier threshold for the champion model."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
        help="model family of the champion run (rebuilds its pipeline)",
    )
    args = parser.parse_args()

    client = MlflowClient()
    version = resolve_champion(client)
    run = client.get_run(version.run_id)
    logger.info(
        "Champion: %s version %s from run %s",
        REGISTERED_MODEL_NAME, version.version, run.info.run_id,
    )

    with get_connection() as conn:
        df = load_training_data(conn)
    X_train, X_test, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_STATE)
    verify_fingerprint(run, X_test, y_test)
    y = np.asarray(y_train)

    weights = balanced_sample_weights(y_train)
    proba = out_of_fold_proba(
        build_pipeline(args.model), X_train, y_train, weights, RANDOM_STATE
    )
    check_reproduction(run.info.run_id, y, proba)
    logger.info("OOF predictions reproduce the logged M5 confusion matrix")

    threshold = choose_threshold(proba, y, BUDGET_FRACTION)
    tiers = assign_tiers(proba, threshold)
    flagged = tiers != "low"
    healthy = y == NO_FAILURE
    twf = y == CLASSES.index("TWF")
    healthy_flagged = int((flagged & healthy).sum())
    twf_flagged = int((flagged & twf).sum())

    logger.info("Elevated threshold %r (budget fraction %r)", threshold, BUDGET_FRACTION)
    logger.info(
        "Flagged healthy rows: %d of %d; TWF flagged: %d of %d",
        healthy_flagged, int(healthy.sum()), twf_flagged, int(twf.sum()),
    )
    logger.info("Tier report (OOF, training rows):\n%s", tier_report(y, tiers))

    tags = {
        THRESHOLD_TAG: repr(threshold),
        "budget_fraction": repr(BUDGET_FRACTION),
        "threshold_rule": (
            "lowest t with high+elevated flagging <= floor(budget_fraction x "
            "healthy OOF training rows); elevated if 1 - P(no_failure) > t"
        ),
        "oof_healthy_flagged": str(healthy_flagged),
        "oof_twf_flagged": str(twf_flagged),
    }
    for key, value in tags.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, key, value)
    logger.info(
        "Tagged %s version %s with %s",
        REGISTERED_MODEL_NAME, version.version, sorted(tags),
    )


if __name__ == "__main__":
    main()