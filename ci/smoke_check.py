"""End-to-end smoke check for a freshly bootstrapped stack (run by CI).

Runs inside the trainer container after the D33 bootstrap (load, train,
evaluate, register, thresholds, API):

    docker compose run --rm -T -v "$PWD/ci:/ci:ro" trainer python /ci/smoke_check.py

Part 1 checks that the champion trained from scratch reproduces the real
champion (run 1bb0e5f920ee487dae15903c021ae2c2, registry version 1). The
tolerance was fixed before CI's numbers were seen: the test-set fingerprint,
n_test and every count must match exactly; float scores and the review
threshold must agree within a relative 1e-6, and the output says whether they
matched bit for bit.

Part 2 checks the API end to end: /health, three known readings (low, high
PWF, high TWF), each response's request_id present in predictions_log with the
same tier, and an invalid request rejected with 422 and never logged.

Every check is reported, and a skipped provenance check counts as a failure
(CI always forwards GIT_COMMIT); the script exits 1 if any check failed.
"""

import json
import math
import os
import sys
import urllib.error
import urllib.request
import uuid

from mlflow import MlflowClient
from psycopg import sql

from ai4i.db import get_connection
from ai4i.evaluate import SEALED_TAG
from ai4i.features import RAW_COLUMNS
from ai4i.ingest import TABLE as READINGS_TABLE
from ai4i.prediction_log import TABLE as LOG_TABLE
from ai4i.registry import resolve_champion
from ai4i.risk import THRESHOLD_TAG
from ai4i.train import GIT_COMMIT_ENV, GIT_COMMIT_TAG

API_URL = "http://api:8000"
REL_TOL = 1e-6

# Reference values: the real champion (run 1bb0e5f920ee487dae15903c021ae2c2,
# registry version 1), read from the real MLflow on 2026-10-04.
EXPECTED_PARAMS = {
    "n_test": "1999",
    "test_fingerprint": "a972745e70d97fd79402ff27b58c98791b23c4040d8e1e718466d41718b2eead",
}
EXPECTED_COUNTS = {"oof_false_alarms": 43, "test_false_alarms": 7}
EXPECTED_SCORES = {
    "cv_f1_macro_mean": 0.7885764421635706,
    "oof_recall_TWF": 0.10810810810810811,
    "test_f1_macro": 0.8056111511968765,
}
EXPECTED_VERSION_TAGS = {
    "budget_fraction": "0.02",
    "oof_healthy_flagged": "154",
    "oof_twf_flagged": "13",
}
EXPECTED_THRESHOLD = 0.0626075267791748

# Known readings and the recommendation the champion gives each (M6/M7 checks).
EXPECTED_RECOMMENDATIONS = {
    1: ("low", "no_failure"),
    51: ("high", "PWF"),
    78: ("high", "TWF"),
}


class Report:
    """Collects check results so that every failure is reported, not only the first."""

    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail):
        """Print one PASS/FAIL line and remember the failures."""
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            self.failures.append(name)

    def check_close(self, name, actual, expected):
        """Pass within REL_TOL, and say whether the two values are bit-identical."""
        if actual is None:
            self.check(name, False, "missing")
            return
        ok = math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=0.0)
        how = "bit-identical" if actual == expected else f"differs by {actual - expected:.3e}"
        self.check(name, ok, f"{actual!r} vs {expected!r} ({how})")


def check_reproduction(report):
    """Compare the freshly registered champion with the real one; return its threshold."""
    client = MlflowClient()
    champion = resolve_champion(client)
    run = client.get_run(champion.run_id)

    report.check("champion version", int(champion.version) == 1, f"version {champion.version}")
    report.check(
        "sealed-test tag",
        run.data.tags.get(SEALED_TAG) == "true",
        f"{SEALED_TAG}={run.data.tags.get(SEALED_TAG)!r}",
    )
    for name, expected in EXPECTED_PARAMS.items():
        actual = run.data.params.get(name)
        report.check(f"param {name}", actual == expected, repr(actual))
    for name, expected in EXPECTED_COUNTS.items():
        actual = run.data.metrics.get(name)
        report.check(f"metric {name}", actual == expected, f"{actual!r} vs {expected!r}")
    for name, expected in EXPECTED_SCORES.items():
        report.check_close(f"metric {name}", run.data.metrics.get(name), expected)
    for name, expected in EXPECTED_VERSION_TAGS.items():
        actual = champion.tags.get(name)
        report.check(f"version tag {name}", actual == expected, repr(actual))

    expected_commit = os.environ.get(GIT_COMMIT_ENV, "")
    if expected_commit:
        actual_commit = run.data.tags.get(GIT_COMMIT_TAG)
        report.check("git commit tag", actual_commit == expected_commit, repr(actual_commit))
    else:
        report.check("git commit tag", False, f"{GIT_COMMIT_ENV} not set: CI must forward it")

    tag = champion.tags.get(THRESHOLD_TAG)
    threshold = float(tag) if tag is not None else None
    report.check_close(f"version tag {THRESHOLD_TAG}", threshold, EXPECTED_THRESHOLD)
    return threshold


def call(method, path, payload=None):
    """Send one request to the API and return (status code, parsed JSON body)."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        f"{API_URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def read_reading(conn, udi):
    """Return one reading from sensor_readings as an API request payload."""
    query = sql.SQL("SELECT {columns} FROM {table} WHERE udi = %s").format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in RAW_COLUMNS),
        table=sql.Identifier(READINGS_TABLE),
    )
    return dict(zip(RAW_COLUMNS, conn.execute(query, (udi,)).fetchone()))


def logged_row(conn, request_id):
    """Return (risk_tier, predicted_mode, model_version, elevated_threshold) or None."""
    query = sql.SQL(
        "SELECT risk_tier, predicted_mode, model_version, elevated_threshold "
        "FROM {} WHERE request_id = %s"
    ).format(sql.Identifier(LOG_TABLE))
    return conn.execute(query, (uuid.UUID(request_id),)).fetchone()


def log_count(conn):
    """Return the number of rows in predictions_log."""
    query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(LOG_TABLE))
    return conn.execute(query).fetchone()[0]


def check_api(report, threshold):
    """Check /health, three known recommendations, their log rows, and a 422."""
    status, body = call("GET", "/health")
    report.check(
        "GET /health",
        status == 200 and body.get("model_version") == 1 and body.get("elevated_threshold") == threshold,
        f"{status} {body}",
    )

    with get_connection() as conn:
        payload = None
        for udi, expected in EXPECTED_RECOMMENDATIONS.items():
            payload = read_reading(conn, udi)
            status, body = call("POST", "/predict", payload)
            if status != 200:
                report.check(f"POST /predict udi {udi}", False, f"{status} {body}")
                continue
            got = (body["risk_tier"], body["predicted_mode"])
            report.check(
                f"POST /predict udi {udi}",
                got == expected,
                f"{got}, p={body['predicted_probability']:.3f}",
            )
            row = logged_row(conn, body["request_id"])
            report.check(
                f"logged before returned, udi {udi}",
                row == (body["risk_tier"], body["predicted_mode"], body["model_version"], threshold),
                repr(row),
            )

        before = log_count(conn)
        status, _ = call("POST", "/predict", {**payload, "action": "stop_machine"})
        after = log_count(conn)
        report.check(
            "invalid request rejected and not logged",
            status == 422 and after == before,
            f"status {status}, log rows {before} -> {after}",
        )


def main():
    """Run both parts and exit non-zero if any check failed."""
    report = Report()
    print("== Part 1: reproduction of the real champion ==")
    threshold = check_reproduction(report)
    print("== Part 2: API end to end ==")
    check_api(report, threshold)
    if report.failures:
        print(f"{len(report.failures)} check(s) failed: {', '.join(report.failures)}")
        sys.exit(1)
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()