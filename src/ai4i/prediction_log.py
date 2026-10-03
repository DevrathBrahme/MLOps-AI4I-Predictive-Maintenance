import datetime
import uuid
from collections.abc import Mapping

import psycopg
from psycopg import sql

from ai4i.data import FEATURE_COLUMNS
from ai4i.model import CLASSES

TABLE = "predictions_log"
PROBA_COLUMNS = tuple(f"p_{name.lower()}" for name in CLASSES)
LOG_COLUMNS = (
    "model_name",
    "model_version",
    *FEATURE_COLUMNS,
    *PROBA_COLUMNS,
    "predicted_mode",
    "runner_up_mode",
    "confidence_score",
    "elevated_threshold",
    "risk_tier",
)


def log_prediction(
    conn: psycopg.Connection, record: Mapping[str, object]
) -> tuple[uuid.UUID, datetime.datetime]:
    """Insert one recommendation and return its database-assigned id and timestamp.

    record must have exactly the LOG_COLUMNS keys, with plain Python values
    (float, int, str), not numpy scalars. Does not commit: the caller's
    `with get_connection() as conn:` block commits on success and rolls back
    on error.
    """
    missing = sorted(set(LOG_COLUMNS) - set(record))
    unexpected = sorted(set(record) - set(LOG_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            f"Prediction record has missing keys {missing} "
            f"and unexpected keys {unexpected}"
        )
    query = sql.SQL(
        "INSERT INTO {table} ({columns}) VALUES ({values}) "
        "RETURNING request_id, created_at"
    ).format(
        table=sql.Identifier(TABLE),
        columns=sql.SQL(", ").join(sql.Identifier(c) for c in LOG_COLUMNS),
        values=sql.SQL(", ").join(sql.Placeholder(c) for c in LOG_COLUMNS),
    )
    with conn.cursor() as cur:
        cur.execute(query, record)
        request_id, created_at = cur.fetchone()
    return request_id, created_at