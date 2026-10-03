"""Tests for the suite's own safety nets in tests/conftest.py.

A guard that silently stops working is worse than no guard: the suite
would keep passing while writing to the real MLflow server.
"""

import os

import mlflow


def test_mlflow_resolves_to_this_tests_private_store(tmp_path, private_mlflow_store):
    """Tracking and registry calls both resolve to the per-test SQLite store."""
    expected = f"sqlite:///{tmp_path / 'mlflow.db'}"

    assert private_mlflow_store == expected
    assert os.environ["MLFLOW_TRACKING_URI"] == expected
    assert mlflow.get_tracking_uri() == expected
    assert mlflow.get_registry_uri() == expected