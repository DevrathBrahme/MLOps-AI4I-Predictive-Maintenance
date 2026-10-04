"""predictions_log against the real table: columns, constraints, rollback.

Every insert happens inside the pg_conn fixture's transaction, which is
rolled back, so the real log never gains a row. Constraint assertions check
the constraint's name, which proves which rule fired.
"""

import uuid

import psycopg
import pytest
from psycopg import sql

from ai4i.prediction_log import LOG_COLUMNS, TABLE, log_prediction

THRESHOLD = 0.0626075267791748

VALID = {
    "model_name": "test-model",
    "model_version": 1,
    "type": "M",
    "air_temp_k": 298.1,
    "process_temp_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
    "power_w": 6951.59,
    "temp_diff_k": 10.5,
    "strain_min_nm": 0.0,
    "p_no_failure": 0.95,
    "p_twf": 0.03,
    "p_hdf": 0.01,
    "p_pwf": 0.005,
    "p_osf": 0.005,
    "predicted_mode": "no_failure",
    "runner_up_mode": "TWF",
    "confidence_score": 0.05,
    "elevated_threshold": THRESHOLD,
    "risk_tier": "low",
}


def row_count(conn):
    """Return the number of rows in predictions_log as this transaction sees it."""
    query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(TABLE))
    return conn.execute(query).fetchone()[0]


def test_log_columns_are_the_table_columns_except_db_assigned_ones(pg_conn):
    """LOG_COLUMNS plus request_id and created_at are exactly the table's 23 columns."""
    rows = pg_conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s",
        (TABLE,),
    ).fetchall()
    columns = {name for (name,) in rows}

    assert len(columns) == 23
    assert columns - {"request_id", "created_at"} == set(LOG_COLUMNS)


def test_valid_record_is_logged_and_the_rollback_removes_it(pg_conn):
    """The database assigns id and time, stores the threshold exactly, and rollback undoes it."""
    before = row_count(pg_conn)

    request_id, created_at = log_prediction(pg_conn, VALID)

    assert isinstance(request_id, uuid.UUID)
    assert created_at.tzinfo is not None
    assert row_count(pg_conn) == before + 1
    stored = pg_conn.execute(
        sql.SQL("SELECT elevated_threshold, risk_tier FROM {} WHERE request_id = %s").format(
            sql.Identifier(TABLE)
        ),
        (request_id,),
    ).fetchone()
    assert stored == (THRESHOLD, "low")

    pg_conn.rollback()

    assert row_count(pg_conn) == before


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({}, id="low"),
        pytest.param({"confidence_score": 0.5, "risk_tier": "elevated"}, id="elevated"),
        pytest.param(
            {"predicted_mode": "TWF", "runner_up_mode": "no_failure", "risk_tier": "high"},
            id="high",
        ),
    ],
)
def test_each_tier_that_follows_the_rule_is_accepted(pg_conn, changes):
    """The CHECK constraint accepts every tier the rule can produce."""
    request_id, _ = log_prediction(pg_conn, {**VALID, **changes})

    assert isinstance(request_id, uuid.UUID)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("model_version", 0),
        ("type", "X"),
        ("air_temp_k", 0.0),
        ("rotational_speed_rpm", 0),
        ("torque_nm", -1.0),
        ("tool_wear_min", -1),
        ("p_twf", 1.5),
        ("runner_up_mode", "RNF"),
        ("elevated_threshold", 1.5),
    ],
)
def test_column_check_rejects_an_out_of_range_value(pg_conn, column, value):
    """Each column CHECK fires on its own column, named <table>_<column>_check."""
    with pytest.raises(psycopg.errors.CheckViolation) as excinfo:
        log_prediction(pg_conn, {**VALID, column: value})

    assert excinfo.value.diag.constraint_name == f"{TABLE}_{column}_check"


def test_runner_up_equal_to_prediction_is_rejected(pg_conn):
    """The named runner_up_differs constraint backs the response validator."""
    with pytest.raises(psycopg.errors.CheckViolation) as excinfo:
        log_prediction(pg_conn, {**VALID, "runner_up_mode": "no_failure"})

    assert excinfo.value.diag.constraint_name == "runner_up_differs"


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param(
            {"predicted_mode": "TWF", "runner_up_mode": "no_failure"},
            id="failure on top but low",
        ),
        pytest.param({"confidence_score": 0.5}, id="above threshold but low"),
        pytest.param({"risk_tier": "elevated"}, id="below threshold but elevated"),
        pytest.param(
            {"confidence_score": THRESHOLD, "risk_tier": "elevated"},
            id="exactly at threshold but elevated",
        ),
    ],
)
def test_tier_that_breaks_the_rule_is_rejected(pg_conn, changes):
    """The full tier rule, with the threshold, is enforced by the database."""
    with pytest.raises(psycopg.errors.CheckViolation) as excinfo:
        log_prediction(pg_conn, {**VALID, **changes})

    assert excinfo.value.diag.constraint_name == "tier_matches_rule"


def test_missing_value_is_rejected_by_not_null(pg_conn):
    """A None value cannot slip through as a NULL in the audit log."""
    with pytest.raises(psycopg.errors.NotNullViolation) as excinfo:
        log_prediction(pg_conn, {**VALID, "confidence_score": None})

    assert excinfo.value.diag.column_name == "confidence_score"