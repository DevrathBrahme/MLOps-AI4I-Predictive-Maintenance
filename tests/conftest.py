"""Suite-wide pytest configuration for the AI4I tests.

Two guards apply to every test in the suite:

1. The folder decides the ``integration`` marker. Every test collected from
   ``tests/integration/`` is marked automatically, so ``-m "not integration"``
   can never run a test that needs Postgres just because its author forgot
   the decorator.
2. No test can reach the real MLflow server. Each test gets a private, empty
   SQLite store for tracking and registry metadata inside its own
   ``tmp_path``. Artifacts are not covered by this guard: they follow the
   experiment's artifact location (default ``./mlruns`` in the current
   directory), so any test that logs artifacts must create its own
   experiment inside ``tmp_path``.
"""

from pathlib import Path

import pytest

INTEGRATION_DIR = Path(__file__).parent / "integration"


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Mark every test under tests/integration/ as ``integration``.

    ``tryfirst`` makes this run before pytest's own ``-m`` deselection,
    so the marker added here is visible to ``-m "not integration"``.
    """
    for item in items:
        if item.path.is_relative_to(INTEGRATION_DIR):
            item.add_marker(pytest.mark.integration)


@pytest.fixture(autouse=True)
def private_mlflow_store(tmp_path, monkeypatch):
    """Point MLflow tracking and registry metadata at a per-test SQLite store.

    Metadata only: artifacts go wherever the experiment's artifact location
    says, so tests that log artifacts create their own experiment in
    ``tmp_path``. This works because ai4i never calls
    ``mlflow.set_tracking_uri``: MLflow reads ``MLFLOW_TRACKING_URI`` at call
    time, and the registry falls back to the tracking URI when
    ``MLFLOW_REGISTRY_URI`` is unset.

    Returns:
        The tracking URI, for tests that assert on it.
    """
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.delenv("MLFLOW_REGISTRY_URI", raising=False)
    return uri