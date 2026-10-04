"""API tests for POST /predict with a stub champion and a fake database.

Covers each tier, log-then-respond (the response carries the logged row's id
and time), the logged record's keys, and the 422 / 503 / 500 paths.
"""

import datetime
import uuid

import psycopg
import pytest

from ai4i.data import FEATURE_COLUMNS
from ai4i.model import CLASSES
from ai4i.prediction_log import LOG_COLUMNS, PROBA_COLUMNS
from ai4i.schemas import Recommendation, SensorReading

EXAMPLE = SensorReading.model_config["json_schema_extra"]["examples"][0]

HIGH = (0.15, 0.70, 0.10, 0.03, 0.02)        # top class TWF
ELEVATED = (0.75, 0.20, 0.03, 0.01, 0.01)    # no_failure on top, score 0.25 > 0.2
LOW = (0.95, 0.03, 0.01, 0.005, 0.005)       # score 0.05


@pytest.mark.parametrize(
    ("row", "tier", "predicted", "runner_up"),
    [
        pytest.param(HIGH, "high", "TWF", "no_failure", id="high"),
        pytest.param(ELEVATED, "elevated", "no_failure", "TWF", id="elevated"),
        pytest.param(LOW, "low", "no_failure", "TWF", id="low"),
    ],
)
def test_predict_returns_the_tiered_recommendation(
    client, db, serve, row, tier, predicted, runner_up
):
    """Tier, flag, top two modes, score and probabilities follow the served row."""
    serve(row)

    response = client.post("/predict", json=EXAMPLE)

    assert response.status_code == 200
    body = response.json()
    assert body["risk_tier"] == tier
    assert body["flagged"] is (tier != "low")
    assert body["predicted_mode"] == predicted
    assert body["runner_up_mode"] == runner_up
    assert body["confidence_score"] == pytest.approx(1 - row[0])
    assert body["class_probabilities"] == pytest.approx(dict(zip(CLASSES, row)))


def test_response_has_exactly_the_contract_fields(client, db, serve):
    """What goes over the wire is the 13-field contract, nothing more."""
    serve(HIGH)

    body = client.post("/predict", json=EXAMPLE).json()

    assert set(body) == set(Recommendation.model_fields)


def test_response_carries_the_logged_rows_id_and_time(client, db, serve):
    """Log-then-respond: request_id and created_at come from the log, not the API."""
    served = serve(HIGH)

    body = client.post("/predict", json=EXAMPLE).json()

    assert len(db.records) == 1
    assert uuid.UUID(body["request_id"]) == db.request_id
    assert datetime.datetime.fromisoformat(body["created_at"]) == db.created_at
    assert body["model_name"] == served.name
    assert body["model_version"] == served.version


def test_logged_record_has_exactly_the_log_columns_and_matches_the_response(
    client, db, serve
):
    """The record has exactly LOG_COLUMNS as keys and agrees with what the operator sees."""
    served = serve(ELEVATED)

    body = client.post("/predict", json=EXAMPLE).json()
    record = db.records[0]

    assert set(record) == set(LOG_COLUMNS)
    assert {name: record[name] for name in EXAMPLE} == EXAMPLE
    assert record["elevated_threshold"] == served.threshold
    for field in (
        "model_name", "model_version", "risk_tier",
        "predicted_mode", "runner_up_mode", "confidence_score",
    ):
        assert record[field] == body[field]
    assert [record[column] for column in PROBA_COLUMNS] == pytest.approx(list(ELEVATED))
    physics = body["physics_features"]
    assert {name: record[name] for name in physics} == pytest.approx(physics)


def test_model_receives_one_row_of_training_features(client, db, serve):
    """The request is turned into exactly the 9 training columns before scoring."""
    served = serve(LOW)

    client.post("/predict", json=EXAMPLE)

    (features,) = served.pipeline.inputs
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert len(features) == 1


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"action": "stop_machine"}, id="action field"),
        pytest.param({"rotational_speed_rpm": "1551"}, id="rpm as string"),
        pytest.param({"torque_nm": -1.0}, id="negative torque"),
    ],
)
def test_invalid_reading_gets_422_and_is_never_scored_or_logged(client, db, serve, change):
    """Validation happens before the model or the database is touched."""
    served = serve(HIGH)

    response = client.post("/predict", json={**EXAMPLE, **change})

    assert response.status_code == 422
    assert served.pipeline.inputs == []
    assert db.records == []


def test_unreachable_database_gives_503_and_no_recommendation(client, db, serve):
    """The model scored the reading, but without a log row the operator gets nothing."""
    served = serve(HIGH)
    db.connect_error = psycopg.OperationalError("connection refused")

    response = client.post("/predict", json=EXAMPLE)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Recommendation could not be logged, so none is returned"
    }
    assert len(served.pipeline.inputs) == 1
    assert db.records == []


def test_rejected_log_row_is_a_server_error_not_an_outage(make_client, db, serve):
    """A CHECK violation is our bug (500), not the database being down (503)."""
    serve(HIGH)
    db.log_error = psycopg.errors.CheckViolation("violates check constraint")
    client = make_client(raise_server_exceptions=False)

    response = client.post("/predict", json=EXAMPLE)

    assert response.status_code == 500
    assert "risk_tier" not in response.text