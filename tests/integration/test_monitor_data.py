"""Integration tests for the monitoring data layer (ai4i.monitor) against Postgres.

Rows are inserted inside the pg_conn transaction and rolled back. Every row
inserted in one transaction gets the same created_at (now() is the
transaction's start time), which makes the cutoff exactly testable.
"""

import datetime

import pytest

from ai4i.data import FEATURE_COLUMNS
from ai4i.monitor import load_current, load_reference, tier_counts
from ai4i.prediction_log import log_prediction

RECORD = {
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
    "elevated_threshold": 0.0626075267791748,
    "risk_tier": "low",
}
ELEVATED = {"confidence_score": 0.5, "risk_tier": "elevated"}
HIGH = {"predicted_mode": "TWF", "runner_up_mode": "no_failure", "risk_tier": "high"}


def transaction_start(conn):
    """Return now() for this transaction: the created_at of every row it inserts."""
    return conn.execute("SELECT now()").fetchone()[0]


def test_reference_is_the_training_split_of_the_model_inputs(pg_conn):
    """7,992 rows (the training split, never the sealed test rows), nine columns, no gaps."""
    reference = load_reference(pg_conn)

    assert len(reference) == 7992
    assert list(reference.columns) == list(FEATURE_COLUMNS)
    assert not reference.isna().any().any()


def test_current_holds_only_requests_logged_since_the_cutoff(pg_conn):
    """Rows from before the cutoff (the real checkpoint rows) are left out."""
    cutoff = transaction_start(pg_conn)
    for changes in ({}, ELEVATED, HIGH):
        log_prediction(pg_conn, {**RECORD, **changes})

    current = load_current(pg_conn, since=cutoff)

    assert len(current) == 3
    assert list(current.columns) == [*FEATURE_COLUMNS, "risk_tier", "predicted_mode"]
    assert len(load_current(pg_conn, since=cutoff + datetime.timedelta(seconds=1))) == 0


def test_tier_counts_cover_every_tier(pg_conn):
    """The GROUP BY result is complete: two low, one elevated, one high."""
    cutoff = transaction_start(pg_conn)
    for changes in ({}, {}, ELEVATED, HIGH):
        log_prediction(pg_conn, {**RECORD, **changes})

    assert tier_counts(pg_conn, since=cutoff) == {"low": 2, "elevated": 1, "high": 1}


def test_tier_counts_report_zero_for_an_empty_window(pg_conn):
    """A window with no requests gives zeros, not missing keys."""
    future = transaction_start(pg_conn) + datetime.timedelta(seconds=1)

    assert tier_counts(pg_conn, since=future) == {"low": 0, "elevated": 0, "high": 0}


@pytest.mark.parametrize("load", [load_current, tier_counts])
def test_naive_cutoff_is_rejected(pg_conn, load):
    """A cutoff without a time zone would be read in the session's zone, so it's refused."""
    with pytest.raises(ValueError, match="timezone"):
        load(pg_conn, since=datetime.datetime(2026, 10, 4))