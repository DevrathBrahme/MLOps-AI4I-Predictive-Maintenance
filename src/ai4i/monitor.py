"""Drift and workload monitoring for the recommendation service (M10).

Compares what the API has been asked to score with what the model was trained
on: Evidently for feature drift, SQL for the operator's workload.

Reference: the training split of the nine model inputs, rebuilt exactly as
training built it (split_data on the sensor_features view), so the sealed test
rows never become "normal". Current: the same columns from predictions_log for
requests logged at or after a cutoff, so earlier rows (checkpoints, demos) are
excluded without deleting anything from the audit log.
"""

import datetime

import pandas as pd
import psycopg
from psycopg import sql

from ai4i.data import FEATURE_COLUMNS, load_training_data
from ai4i.prediction_log import TABLE as LOG_TABLE
from ai4i.risk import TIERS
from ai4i.train import split_data

CURRENT_COLUMNS = (*FEATURE_COLUMNS, "risk_tier", "predicted_mode")


def _require_aware(since: datetime.datetime) -> None:
    """Raise ValueError for a naive datetime: created_at is TIMESTAMPTZ."""
    if since.tzinfo is None:
        raise ValueError(
            "since must be timezone-aware (created_at is TIMESTAMPTZ); "
            "got a naive datetime"
        )


def load_reference(conn: psycopg.Connection) -> pd.DataFrame:
    """Return the drift reference: the training split's nine model inputs (7,992 rows)."""
    X_train, _X_test, _y_train, _y_test = split_data(load_training_data(conn))
    return X_train[list(FEATURE_COLUMNS)].reset_index(drop=True)


def load_current(conn: psycopg.Connection, since: datetime.datetime) -> pd.DataFrame:
    """Return the model inputs, tier and mode of every request logged at or after since."""
    _require_aware(since)
    query = sql.SQL(
        "SELECT {columns} FROM {table} WHERE created_at >= %s ORDER BY created_at"
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in CURRENT_COLUMNS),
        table=sql.Identifier(LOG_TABLE),
    )
    rows = conn.execute(query, (since,)).fetchall()
    return pd.DataFrame(rows, columns=list(CURRENT_COLUMNS))


def tier_counts(conn: psycopg.Connection, since: datetime.datetime) -> dict[str, int]:
    """Return how many requests logged at or after since fell in each risk tier."""
    _require_aware(since)
    query = sql.SQL(
        "SELECT risk_tier, count(*) FROM {table} WHERE created_at >= %s GROUP BY risk_tier"
    ).format(table=sql.Identifier(LOG_TABLE))
    counts = dict(conn.execute(query, (since,)).fetchall())
    return {tier: int(counts.get(tier, 0)) for tier in TIERS}