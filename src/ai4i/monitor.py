"""Drift and workload monitoring for the recommendation service (M10).

Compares what the API has been asked to score with what the model was trained
on: Evidently for feature drift, SQL for the operator's workload.

Reference: the training split of the nine model inputs, rebuilt exactly as
training built it (split_data on the sensor_features view), so the sealed test
rows never become "normal". Current: the same columns from predictions_log for
requests logged at or after a cutoff, so earlier rows (checkpoints, demos) are
excluded without deleting anything from the audit log.

Rules fixed before any drift scenario was run (D58, D59): Evidently's default
per-column method and threshold, and an alert if ANY column drifts (not
Evidently's default "half the columns" dataset rule, which passes a real
two-column sensor fault); no drift verdict at all for a window under MIN_ROWS
requests, where the default thresholds raise false alarms on healthy data.
The any-column rule is also configured into Evidently's own dataset test
(DRIFT_SHARE), so the HTML report an operator opens says what the alert says.

    python -m ai4i.monitor --since 2026-10-05T00:00:00+05:30
"""

import argparse
import datetime
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import mlflow
import pandas as pd
import psycopg
from evidently import DataDefinition, Dataset, Report, Run
from evidently.presets import DataDriftPreset
from psycopg import sql

from ai4i.data import FEATURE_COLUMNS, load_training_data
from ai4i.db import get_connection
from ai4i.prediction_log import TABLE as LOG_TABLE
from ai4i.risk import TIERS
from ai4i.train import split_data

CURRENT_COLUMNS = (*FEATURE_COLUMNS, "risk_tier", "predicted_mode")
CATEGORICAL_COLUMNS = ("type",)
NUMERICAL_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS)
MIN_ROWS = 1000
# Evidently's dataset test fails when the share of drifted columns reaches
# drift_share; one column in nine is enough (D58). Its default, 0.5, would
# headline a two-column sensor fault as "Dataset Drift is NOT detected".
DRIFT_SHARE = 1 / len(FEATURE_COLUMNS)
EXPERIMENT_NAME = "ai4i-monitoring"
INSUFFICIENT_DATA = "insufficient_data"

# Tier shares of the champion's out-of-fold predictions on the 7,992 training
# rows (M6): what a window drawn from the training distribution should show.
# Labels are unknown in production, so the 2% healthy-row budget itself can't
# be checked live; the total flagged share (4.9%) is its observable proxy.
EXPECTED_TIER_SHARES = {"low": 7601 / 7992, "elevated": 120 / 7992, "high": 271 / 7992}
EXPECTED_FLAGGED_SHARE = EXPECTED_TIER_SHARES["elevated"] + EXPECTED_TIER_SHARES["high"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColumnDrift:
    """Evidently's verdict for one model input."""

    column: str
    method: str
    threshold: float
    score: float
    drifted: bool


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


def tier_shares(counts: Mapping[str, int]) -> dict[str, float]:
    """Return each tier's share of the window, plus flagged (elevated + high).

    Raises ValueError for an empty window, where shares are undefined.
    """
    total = sum(counts[tier] for tier in TIERS)
    if total == 0:
        raise ValueError("No requests in the window: tier shares are undefined")
    shares = {tier: counts[tier] / total for tier in TIERS}
    shares["flagged"] = shares["elevated"] + shares["high"]
    return shares


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[Run, list[ColumnDrift]]:
    """Run Evidently's data-drift preset on the nine model inputs.

    Returns the snapshot (for the HTML report) and one ColumnDrift per input,
    read from Evidently's own per-column tests, so the drift rule is
    Evidently's, not a re-implementation. Raises RuntimeError if Evidently's
    dataset test and the any-column verdict disagree.
    """
    definition = DataDefinition(
        numerical_columns=list(NUMERICAL_COLUMNS),
        categorical_columns=list(CATEGORICAL_COLUMNS),
    )
    columns = list(FEATURE_COLUMNS)
    snapshot = Report([DataDriftPreset(drift_share=DRIFT_SHARE)], include_tests=True).run(
        current_data=Dataset.from_pandas(
            current[columns].reset_index(drop=True), data_definition=definition
        ),
        reference_data=Dataset.from_pandas(
            reference[columns].reset_index(drop=True), data_definition=definition
        ),
    )
    snapshot_json = json.loads(snapshot.json())
    verdicts = column_verdicts(snapshot_json)
    if dataset_drift_detected(snapshot_json) != (summarize(verdicts) == "drift"):
        raise RuntimeError("Evidently's dataset test disagrees with the any-column verdict")
    return snapshot, verdicts


def column_verdicts(snapshot: Mapping) -> list[ColumnDrift]:
    """Return one ColumnDrift per model input from a snapshot's JSON form.

    Reads the per-column drift tests and the matching metric values; the
    dataset-level test has no column and is skipped here (see
    dataset_drift_detected). Raises ValueError on a test status other than
    SUCCESS or FAIL, or if the tests don't cover exactly FEATURE_COLUMNS, so
    an Evidently upgrade that changes the preset fails loudly.
    """
    scores = {metric["id"]: metric["value"] for metric in snapshot["metrics"]}
    verdicts = []
    for test in snapshot["tests"]:
        params = test["metric_config"]["params"]
        if "column" not in params:
            continue
        if test["status"] not in ("SUCCESS", "FAIL"):
            raise ValueError(f"Drift test for {params['column']!r} has status {test['status']!r}")
        verdicts.append(
            ColumnDrift(
                column=params["column"],
                method=params["method"],
                threshold=float(params["threshold"]),
                score=float(scores[test["metric_config"]["metric_id"]]),
                drifted=test["status"] == "FAIL",
            )
        )
    covered = sorted(v.column for v in verdicts)
    if covered != sorted(FEATURE_COLUMNS):
        raise ValueError(f"Drift tests cover {covered}, expected {sorted(FEATURE_COLUMNS)}")
    return verdicts


def dataset_drift_detected(snapshot: Mapping) -> bool:
    """Return Evidently's dataset-level verdict from a snapshot's JSON form.

    Raises ValueError unless there is exactly one dataset-level test and its
    status is SUCCESS or FAIL.
    """
    tests = [t for t in snapshot["tests"] if "column" not in t["metric_config"]["params"]]
    if len(tests) != 1 or tests[0]["status"] not in ("SUCCESS", "FAIL"):
        raise ValueError(f"Expected one passed or failed dataset drift test, got {tests}")
    return tests[0]["status"] == "FAIL"


def summarize(verdicts: Sequence[ColumnDrift]) -> str:
    """Return "drift" if any column drifted, else "no_drift" (D58)."""
    return "drift" if any(v.drifted for v in verdicts) else "no_drift"


def log_window(
    since: datetime.datetime,
    n_reference: int,
    n_current: int,
    counts: Mapping[str, int],
    snapshot: Run | None,
    verdicts: Sequence[ColumnDrift],
) -> str:
    """Log one monitoring window to the active MLflow run and return its verdict.

    Workload shares are logged for any non-empty window. Drift scores, the
    HTML report and the snapshot JSON only for windows of at least MIN_ROWS;
    smaller windows get the verdict "insufficient_data".
    """
    mlflow.log_params({
        "since": since.isoformat(),
        "n_reference": n_reference,
        "n_current": n_current,
        "min_rows": MIN_ROWS,
    })
    if sum(counts[tier] for tier in TIERS):
        shares = tier_shares(counts)
        mlflow.log_metrics({f"share_{name}": value for name, value in shares.items()})
        mlflow.log_metric("flagged_vs_expected", shares["flagged"] / EXPECTED_FLAGGED_SHARE)
    if n_current < MIN_ROWS or snapshot is None:
        mlflow.set_tag("verdict", INSUFFICIENT_DATA)
        return INSUFFICIENT_DATA
    verdict = summarize(verdicts)
    mlflow.log_metrics({f"drift_score_{v.column}": v.score for v in verdicts})
    mlflow.log_metric("drifted_columns", sum(v.drifted for v in verdicts))
    mlflow.set_tags({
        "verdict": verdict,
        "drifted": ",".join(v.column for v in verdicts if v.drifted) or "none",
    })
    mlflow.log_text(snapshot.get_html_str(as_iframe=False), "drift_report.html")
    mlflow.log_text(snapshot.json(), "drift_snapshot.json")
    return verdict


def parse_since(value: str) -> datetime.datetime:
    """Parse an ISO 8601 timestamp that includes a UTC offset (an argparse type)."""
    since = datetime.datetime.fromisoformat(value)
    if since.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"{value!r} has no UTC offset; add one, e.g. {value}+05:30"
        )
    return since


def main() -> None:
    """Report drift and workload for every request logged since --since."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Compare the requests logged since a cutoff with the training data."
    )
    parser.add_argument(
        "--since", type=parse_since, required=True,
        help="ISO 8601 cutoff with a UTC offset, e.g. 2026-10-05T00:00:00+05:30",
    )
    args = parser.parse_args()

    with get_connection() as conn:
        reference = load_reference(conn)
        current = load_current(conn, args.since)
        counts = tier_counts(conn, args.since)

    snapshot, verdicts = None, []
    if len(current) >= MIN_ROWS:
        snapshot, verdicts = drift_report(reference, current)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"since {args.since.isoformat()}") as run:
        verdict = log_window(
            args.since, len(reference), len(current), counts, snapshot, verdicts
        )

    logger.info("Window since %s: %d requests, tiers %s", args.since.isoformat(), len(current), counts)
    if sum(counts.values()):
        shares = tier_shares(counts)
        logger.info(
            "Flagged share %.2f%% (expected %.2f%% from training), elevated %.2f%%",
            100 * shares["flagged"], 100 * EXPECTED_FLAGGED_SHARE, 100 * shares["elevated"],
        )
    for v in verdicts:
        logger.info(
            "%-5s %-20s %s = %.3f (threshold %.2f)",
            "DRIFT" if v.drifted else "ok", v.column, v.method, v.score, v.threshold,
        )
    logger.info("Verdict: %s (MLflow run %s)", verdict, run.info.run_id)


if __name__ == "__main__":
    main()