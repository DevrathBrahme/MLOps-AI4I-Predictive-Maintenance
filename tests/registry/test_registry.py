"""Registry tests against a private, throwaway MLflow store (never the real server).

Metadata lives in the per-test SQLite store from tests/conftest.py; artifacts go
to an experiment created inside tmp_path, and the working directory moves there
too, so no MLflow default can write into the repository.
"""

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.dummy import DummyClassifier

from ai4i.evaluate import SEALED_TAG
from ai4i.registry import (
    MODEL_ARTIFACT_PATH,
    REGISTERED_MODEL_NAME,
    find_version_for_run,
    register_champion,
    resolve_champion,
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    """An MlflowClient on the private store, with the working directory in tmp_path."""
    monkeypatch.chdir(tmp_path)
    return MlflowClient()


@pytest.fixture
def experiment_id(client, tmp_path):
    """A throwaway experiment whose artifacts live in tmp_path/artifacts."""
    return client.create_experiment(
        "registry-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )


@pytest.fixture
def log_run(experiment_id):
    """Return a function that logs a tiny model in a new run, sealed by default."""

    def log(sealed=True):
        model = DummyClassifier(strategy="prior").fit(
            np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 1, 0, 1])
        )
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.sklearn.log_model(sk_model=model, name=MODEL_ARTIFACT_PATH)
            if sealed:
                mlflow.set_tag(SEALED_TAG, "true")
        return run.info.run_id

    return log


def test_unsealed_run_is_refused_and_nothing_is_registered(client, log_run):
    """Only a run scored on the sealed test set may become the champion."""
    run_id = log_run(sealed=False)

    with pytest.raises(RuntimeError, match=SEALED_TAG):
        register_champion(client, run_id)

    assert len(client.search_registered_models()) == 0


def test_sealed_run_becomes_the_champion(client, log_run):
    """Registration creates version 1 and points the alias at it."""
    run_id = log_run()

    version = register_champion(client, run_id)
    champion = resolve_champion(client)

    assert int(version.version) == 1
    assert champion.version == version.version
    assert champion.run_id == run_id


def test_registering_the_same_run_again_reuses_its_version(client, log_run):
    """Idempotent: a rerun never creates a duplicate version."""
    run_id = log_run()

    first = register_champion(client, run_id)
    second = register_champion(client, run_id)

    assert second.version == first.version
    assert len(client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")) == 1


def test_alias_moves_to_a_newly_registered_run(client, log_run):
    """Registering a newer sealed run moves the champion to version 2."""
    register_champion(client, log_run())
    newer = log_run()

    register_champion(client, newer)
    champion = resolve_champion(client)

    assert int(champion.version) == 2
    assert champion.run_id == newer


def test_resolving_without_a_champion_raises(client):
    """No alias means an exception, which is what makes the API fail fast at startup."""
    with pytest.raises(MlflowException):
        resolve_champion(client)


def test_a_run_registered_twice_is_ambiguous(client, log_run):
    """If a run somehow has two versions, find_version_for_run refuses to guess."""
    run_id = log_run()
    for _ in range(2):
        mlflow.register_model(f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}", REGISTERED_MODEL_NAME)

    with pytest.raises(RuntimeError, match="several versions"):
        find_version_for_run(client, run_id)


def test_model_artifacts_land_in_tmp_path(client, log_run, tmp_path):
    """The logged model's files are inside this test's tmp_path, nowhere else."""
    register_champion(client, log_run())

    assert list((tmp_path / "artifacts").rglob("MLmodel"))