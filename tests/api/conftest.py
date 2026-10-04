"""Fixtures for the API tests: a stub champion and a fake database.

The real lifespan (which loads the champion from MLflow) never runs, and no
test can reach Postgres: every client comes with get_connection and
log_prediction replaced by an in-memory fake.
"""

import datetime
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai4i import api


class StubPipeline:
    """Stands in for the champion: fixed probabilities and a record of every input."""

    def __init__(self, row):
        self.row = np.asarray(row, dtype=float)
        self.inputs = []

    def predict_proba(self, features):
        """Return the fixed row for a one-row frame and remember what was scored."""
        self.inputs.append(features.copy())
        return np.array([self.row])


class FakeConnection:
    """Connection that is also its own cursor: the parts of psycopg the API uses."""

    def __init__(self, database):
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def cursor(self):
        """Return self, so `conn.cursor()` works as a context manager too."""
        return self

    def execute(self, query):
        """Record the query instead of running it."""
        self.database.executed.append(query)


class FakeDatabase:
    """Replaces get_connection and log_prediction; remembers what would have been logged."""

    def __init__(self):
        self.request_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        self.created_at = datetime.datetime(2026, 10, 4, 3, 30, tzinfo=datetime.UTC)
        self.records = []
        self.executed = []
        self.connect_error = None
        self.log_error = None

    def connect(self):
        """Stand-in for get_connection: raise connect_error if set."""
        if self.connect_error is not None:
            raise self.connect_error
        return FakeConnection(self)

    def log(self, conn, record):
        """Stand-in for log_prediction: raise log_error if set, else return the row's id and time."""
        if self.log_error is not None:
            raise self.log_error
        self.records.append(record)
        return self.request_id, self.created_at


def _refuse_to_load():
    """Fail loudly if anything runs the lifespan during the API tests."""
    raise AssertionError("the lifespan ran: API tests must not load a model from MLflow")


@pytest.fixture
def db(monkeypatch):
    """Patch the API's database calls with a fresh in-memory fake."""
    fake = FakeDatabase()
    monkeypatch.setattr(api, "get_connection", fake.connect)
    monkeypatch.setattr(api, "log_prediction", fake.log)
    return fake


@pytest.fixture
def serve(monkeypatch):
    """Return a function that installs a stub champion serving one probability row."""

    def install(row, threshold=0.2):
        served = api.ServedModel(
            name="test-model",
            version=7,
            threshold=threshold,
            pipeline=StubPipeline(row),
        )
        monkeypatch.setattr(api.app.state, "served", served, raising=False)
        return served

    return install


@pytest.fixture
def make_client(db, monkeypatch):
    """Return a TestClient factory; requiring `db` means no client can reach Postgres."""
    monkeypatch.setattr(api, "load_champion", _refuse_to_load)

    def build(**kwargs):
        return TestClient(api.app, **kwargs)

    return build


@pytest.fixture
def client(make_client):
    """A TestClient used without `with`, so the lifespan never runs."""
    return make_client()