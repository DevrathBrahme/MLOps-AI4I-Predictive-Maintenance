"""Drift scenario: replay held-out readings through the API, then check what monitoring reports.

Runs inside the monitor container of a THROWAWAY stack, after the D33
bootstrap with the API up (CI's smoke job, or a local -p ai4i-monitor-demo
project), never against the real stack:

    docker compose run --rm -T -v "$PWD/ci:/ci:ro" monitor python /ci/drift_scenario.py

Two phases, each its own monitoring window:

    baseline   the 1,999 sealed test readings, as recorded
    air_fault  the same readings with the air temperature sensor reading 2 K
               high (a calibration fault; process temperature is untouched)

For each phase the script takes a cutoff from the database clock (the clock
that stamps created_at), sends every reading to POST /predict as ordinary
traffic, runs the monitor CLI exactly as an operator would
(python -m ai4i.monitor --since <cutoff>), and checks both the API's answers
and the monitoring run in MLflow against EXPECTED.

EXPECTED was fixed before the scenario first ran against a stack (D62). It
was derived offline from the same rows: the champion reproduced from scratch
(bit-identical to registry version 1) scoring them, and Evidently comparing
them with the training split. The stack must reproduce it through every
layer it adds: request validation, the Python physics features,
predictions_log, the SQL window, Evidently in the monitor image, and MLflow.
Counts, verdicts and drifted columns must match exactly; drift scores within
a relative 1e-6 (the M9 tolerance).

Refuses to start if the ai4i-monitoring experiment already has runs: the
real stack has a monitoring history, a freshly bootstrapped one has none.
"""

import subprocess
import sys
from collections import Counter
from dataclasses import dataclass

import mlflow
import pandas as pd

from ai4i.data import load_training_data
from ai4i.db import get_connection
from ai4i.features import RAW_COLUMNS
from ai4i.model import CLASSES
from ai4i.monitor import EXPERIMENT_NAME
from ai4i.risk import TIERS
from ai4i.train import split_data
from smoke_check import Report, call

N_READINGS = 1999
AIR_SHIFT_K = 2.0
# JSON types the API's SensorReading accepts in strict mode.
PAYLOAD_TYPES = {
    "type": str,
    "air_temp_k": float,
    "process_temp_k": float,
    "rotational_speed_rpm": int,
    "torque_nm": float,
    "tool_wear_min": int,
}

# Drift scores of the columns the air fault does not touch (identical in both phases).
UNTOUCHED_SCORES = {
    "process_temp_k": 0.029589566266810658,
    "rotational_speed_rpm": 0.03546877107568145,
    "torque_nm": 0.026943546062286435,
    "tool_wear_min": 0.022346936596647386,
    "power_w": 0.027767151092861493,
    "strain_min_nm": 0.021470287160672525,
    "type": 0.01023315633104959,
}


@dataclass(frozen=True)
class Scenario:
    """One phase of traffic and everything the stack must report for it."""

    name: str
    air_shift_k: float
    tiers: dict[str, int]
    modes: dict[str, int]
    verdict: str
    drifted: str
    scores: dict[str, float]


SCENARIOS = (
    Scenario(
        name="baseline",
        air_shift_k=0.0,
        tiers={"low": 1895, "elevated": 39, "high": 65},
        modes={"no_failure": 1934, "TWF": 6, "HDF": 23, "PWF": 20, "OSF": 16},
        verdict="no_drift",
        drifted="none",
        scores={
            "air_temp_k": 0.0192550556441214,
            "temp_diff_k": 0.011096147491275365,
            **UNTOUCHED_SCORES,
        },
    ),
    Scenario(
        name="air_fault",
        air_shift_k=AIR_SHIFT_K,
        tiers={"low": 1766, "elevated": 38, "high": 195},
        modes={"no_failure": 1804, "TWF": 12, "HDF": 147, "PWF": 20, "OSF": 16},
        verdict="drift",
        drifted="air_temp_k,temp_diff_k",
        scores={
            "air_temp_k": 1.0060140452621853,
            "temp_diff_k": 2.0013455926327666,
            **UNTOUCHED_SCORES,
        },
    ),
)


def require_fresh_monitoring_history():
    """Exit before sending any traffic unless the monitoring experiment is empty."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], max_results=1, output_format="list"
    )
    if runs:
        sys.exit(
            f"Refusing to run: {EXPERIMENT_NAME!r} already has monitoring runs, so this "
            "is not a freshly bootstrapped stack. Simulated traffic must never reach "
            "the real predictions_log; use a throwaway project (-p ai4i-monitor-demo)."
        )


def load_readings() -> pd.DataFrame:
    """Return the raw inputs of the 1,999 sealed test readings, via SQL and split_data."""
    with get_connection() as conn:
        _X_train, X_test, _y_train, _y_test = split_data(load_training_data(conn))
    return X_test[list(RAW_COLUMNS)]


def payloads(readings: pd.DataFrame, air_shift_k: float):
    """Yield one /predict request body per reading, with the air sensor offset applied."""
    shifted = readings.assign(air_temp_k=readings["air_temp_k"] + air_shift_k)
    for row in shifted.itertuples(index=False):
        yield {column: PAYLOAD_TYPES[column](value) for column, value in zip(RAW_COLUMNS, row)}


def db_clock():
    """Return the database's current time: the clock that stamps created_at.

    clock_timestamp(), not now(): now() is frozen at the start of the transaction.
    """
    with get_connection() as conn:
        return conn.execute("SELECT clock_timestamp()").fetchone()[0]


def replay(readings: pd.DataFrame, air_shift_k: float):
    """Send every reading to the API; return tier counts, mode counts and failed requests."""
    tiers, modes, failures = Counter(), Counter(), []
    for payload in payloads(readings, air_shift_k):
        status, body = call("POST", "/predict", payload)
        if status != 200:
            failures.append((status, body))
            continue
        tiers[body["risk_tier"]] += 1
        modes[body["predicted_mode"]] += 1
    return (
        {tier: tiers[tier] for tier in TIERS},
        {mode: modes[mode] for mode in CLASSES},
        failures,
    )


def monitoring_run(since):
    """Run the monitor CLI for the window since the cutoff and return its MLflow run."""
    subprocess.run(
        [sys.executable, "-m", "ai4i.monitor", "--since", since.isoformat()], check=True
    )
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"params.since = '{since.isoformat()}'",
        output_format="list",
    )
    if len(runs) != 1:
        raise RuntimeError(f"Expected one monitoring run since {since.isoformat()}, found {len(runs)}")
    return runs[0]


def check_scenario(report: Report, scenario: Scenario, readings: pd.DataFrame):
    """Replay one phase and check the API's answers and the monitoring run."""
    print(f"== {scenario.name}: {len(readings)} readings, air sensor offset {scenario.air_shift_k:+.1f} K ==")
    since = db_clock()
    tiers, modes, failures = replay(readings, scenario.air_shift_k)
    prefix = scenario.name

    report.check(f"{prefix}: every request accepted", not failures, f"{len(failures)} failed {failures[:3]}")
    report.check(f"{prefix}: API tiers", tiers == scenario.tiers, f"{tiers} vs {scenario.tiers}")
    report.check(f"{prefix}: API modes", modes == scenario.modes, f"{modes} vs {scenario.modes}")

    run = monitoring_run(since)
    params, metrics, tags = run.data.params, run.data.metrics, run.data.tags
    report.check(f"{prefix}: window size", params.get("n_current") == str(N_READINGS), repr(params.get("n_current")))
    report.check(f"{prefix}: verdict", tags.get("verdict") == scenario.verdict, repr(tags.get("verdict")))
    report.check(f"{prefix}: drifted columns", tags.get("drifted") == scenario.drifted, repr(tags.get("drifted")))
    for column, expected in scenario.scores.items():
        report.check_close(f"{prefix}: drift score {column}", metrics.get(f"drift_score_{column}"), expected)
    for tier, count in scenario.tiers.items():
        report.check_close(f"{prefix}: logged share {tier}", metrics.get(f"share_{tier}"), count / N_READINGS)
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)}
    report.check(f"{prefix}: HTML report logged", "drift_report.html" in artifacts, sorted(artifacts))


def main():
    """Run both phases in order and exit non-zero if any check failed."""
    require_fresh_monitoring_history()
    readings = load_readings()
    report = Report()
    report.check("sealed test readings", len(readings) == N_READINGS, f"{len(readings)} rows")
    for scenario in SCENARIOS:
        check_scenario(report, scenario, readings)
    if report.failures:
        print(f"{len(report.failures)} check(s) failed: {', '.join(report.failures)}")
        sys.exit(1)
    print("Drift scenario reproduced exactly.")


if __name__ == "__main__":
    main()