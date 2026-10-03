"""Register a sealed-test-evaluated run's model and point the champion alias at it."""

import argparse
import logging

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from ai4i.evaluate import SEALED_TAG

REGISTERED_MODEL_NAME = "ai4i-failure-classifier"
CHAMPION_ALIAS = "champion"
MODEL_ARTIFACT_PATH = "model"

logger = logging.getLogger(__name__)


def find_version_for_run(client: MlflowClient, run_id: str) -> ModelVersion | None:
    """Return the registered version created from run_id, or None if there is none.

    Raises RuntimeError if the run was registered more than once, because then
    it is ambiguous which version should be served.
    """
    versions = client.search_model_versions(
        filter_string=f"name = '{REGISTERED_MODEL_NAME}' and run_id = '{run_id}'"
    )
    if not versions:
        return None
    if len(versions) > 1:
        numbers = sorted(int(v.version) for v in versions)
        raise RuntimeError(
            f"Run {run_id} is registered as several versions of "
            f"{REGISTERED_MODEL_NAME}: {numbers}; resolve this by hand"
        )
    return versions[0]


def register_champion(client: MlflowClient, run_id: str) -> ModelVersion:
    """Register run_id's model once and point the champion alias at that version.

    Refuses runs without the sealed-test tag: only the artifact that was scored
    on the sealed test set may be served. Safe to rerun: a run that is already
    registered is reused instead of creating a duplicate version.
    """
    run = client.get_run(run_id)
    if run.data.tags.get(SEALED_TAG) != "true":
        raise RuntimeError(
            f"Run {run_id} has no {SEALED_TAG}=true tag; only the run scored "
            "on the sealed test set can become the champion"
        )
    version = find_version_for_run(client, run_id)
    if version is None:
        # Uses the global tracking URI, same as MlflowClient(): both read
        # MLFLOW_TRACKING_URI from the environment.
        version = mlflow.register_model(
            f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}", REGISTERED_MODEL_NAME
        )
        logger.info(
            "Registered %s version %s from run %s",
            REGISTERED_MODEL_NAME, version.version, run_id,
        )
    else:
        logger.info(
            "Run %s is already registered as %s version %s",
            run_id, REGISTERED_MODEL_NAME, version.version,
        )
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version.version
    )
    logger.info("Alias %s -> version %s", CHAMPION_ALIAS, version.version)
    return version


def resolve_champion(client: MlflowClient) -> ModelVersion:
    """Return the model version the champion alias currently points to.

    Raises MlflowException if the alias does not exist, so the API fails at
    startup instead of serving without a champion.
    """
    return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)


def main() -> None:
    """Parse --run-id, register that run's model and set the champion alias."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Register a sealed-test-evaluated run as the champion model."
    )
    parser.add_argument("--run-id", required=True, help="MLflow run id to register")
    args = parser.parse_args()
    register_champion(MlflowClient(), args.run_id)


if __name__ == "__main__":
    main()