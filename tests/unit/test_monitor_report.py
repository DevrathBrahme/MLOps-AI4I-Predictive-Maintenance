"""Unit tests for the drift and workload logic in ai4i.monitor (no database).

Readings are synthetic: AI4I-like distributions with the physics features
computed by ai4i.features, so the three engineered columns move with their
inputs exactly as they would in production. The drift rule under test was
fixed before any scenario was run (D58): Evidently's default per-column
method and threshold, alert if any column drifts, with Evidently's own
dataset test configured to agree.
"""

import argparse
import datetime
import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance

from ai4i.data import FEATURE_COLUMNS
from ai4i.features import build_features
from ai4i.monitor import (
    EXPECTED_FLAGGED_SHARE,
    EXPECTED_TIER_SHARES,
    INSUFFICIENT_DATA,
    MIN_ROWS,
    NUMERICAL_COLUMNS,
    ColumnDrift,
    column_verdicts,
    dataset_drift_detected,
    drift_report,
    log_window,
    parse_since,
    summarize,
    tier_shares,
)

SINCE = datetime.datetime(2026, 10, 5, tzinfo=datetime.timezone.utc)
COUNTS = {"low": 1900, "elevated": 40, "high": 60}


HEALTHY_MIX = (0.6, 0.3, 0.1)


def readings(n: int, seed: int, air_shift_k: float = 0.0, type_mix=HEALTHY_MIX) -> pd.DataFrame:
    """Return n AI4I-like model inputs.

    air_shift_k simulates a miscalibrated air sensor; type_mix changes the
    L/M/H product mix, which moves the type column and nothing else.
    """
    rng = np.random.default_rng(seed)
    air = rng.normal(300.0, 2.0, n)
    raw = pd.DataFrame({
        "type": rng.choice(["L", "M", "H"], size=n, p=list(type_mix)),
        "air_temp_k": air + air_shift_k,
        "process_temp_k": air + 10.0 + rng.normal(0.0, 1.0, n),
        "rotational_speed_rpm": rng.normal(1540, 180, n).round().astype(int),
        "torque_nm": rng.normal(40.0, 10.0, n).clip(min=0),
        "tool_wear_min": rng.integers(0, 254, n),
    })
    return build_features(raw)


@pytest.fixture(scope="module")
def reference():
    return readings(4000, seed=1)


@pytest.fixture(scope="module")
def healthy(reference):
    """Report on a fresh draw from the reference distribution."""
    return drift_report(reference, readings(MIN_ROWS, seed=2))


@pytest.fixture(scope="module")
def air_fault(reference):
    """Report on a fresh draw with the air temperature sensor reading 2 K high."""
    return drift_report(reference, readings(MIN_ROWS, seed=2, air_shift_k=2.0))


def by_column(verdicts):
    return {v.column: v for v in verdicts}


def test_healthy_window_has_no_drift_in_any_column(healthy):
    """Same distribution, fresh sample: no column drifts, so no alert."""
    _snapshot, verdicts = healthy
    assert sorted(v.column for v in verdicts) == sorted(FEATURE_COLUMNS)
    assert [v.column for v in verdicts if v.drifted] == []
    assert summarize(verdicts) == "no_drift"


def test_methods_and_thresholds_are_the_preregistered_defaults(healthy):
    """Pins D58: an Evidently upgrade that changes the defaults fails here first."""
    _snapshot, verdicts = healthy
    for v in verdicts:
        expected = (
            "Jensen-Shannon distance" if v.column == "type" else "Wasserstein distance (normed)"
        )
        assert (v.method, v.threshold) == (expected, 0.1), v.column


@pytest.mark.parametrize("column", NUMERICAL_COLUMNS)
def test_numerical_scores_match_an_independent_normed_wasserstein(reference, air_fault, column):
    """Oracle: Evidently's score is SciPy's Wasserstein distance over the reference std."""
    current = readings(MIN_ROWS, seed=2, air_shift_k=2.0)
    expected = wasserstein_distance(reference[column], current[column]) / np.std(reference[column])
    _snapshot, verdicts = air_fault
    assert by_column(verdicts)[column].score == pytest.approx(expected, rel=1e-9)


def test_air_sensor_fault_flags_exactly_air_temperature_and_temperature_difference(air_fault):
    """+2 K on air moves air_temp_k and temp_diff_k (process - air) and nothing else."""
    _snapshot, verdicts = air_fault
    assert sorted(v.column for v in verdicts if v.drifted) == ["air_temp_k", "temp_diff_k"]


def test_evidentlys_dataset_verdict_agrees_with_the_any_column_rule(healthy, air_fault):
    """D58: the report's dataset headline must say what the alert says.

    With Evidently's default drift_share (0.5), the 2-of-9 air fault would be
    headlined "Dataset Drift is NOT detected" while the alert fires.
    """
    for (snapshot, verdicts), expected in ((healthy, False), (air_fault, True)):
        assert dataset_drift_detected(json.loads(snapshot.json())) is expected
        assert (summarize(verdicts) == "drift") is expected
        assert ("Dataset Drift is detected" in snapshot.get_html_str(as_iframe=False)) is expected


def test_a_single_drifted_column_is_enough(reference):
    """The boundary of D58: a product-mix change moves only `type`, and that alone alerts."""
    snapshot, verdicts = drift_report(
        reference, readings(MIN_ROWS, seed=2, type_mix=(0.3, 0.5, 0.2))
    )
    assert [v.column for v in verdicts if v.drifted] == ["type"]
    assert summarize(verdicts) == "drift"
    assert dataset_drift_detected(json.loads(snapshot.json())) is True


def snapshot_json(status: str = "FAIL", columns=FEATURE_COLUMNS, dataset_status: str = "FAIL") -> dict:
    """A minimal snapshot in Evidently's JSON shape: one dataset test, one test per column."""
    metrics = [{"id": "dataset", "value": {"count": 1.0, "share": 0.11}}]
    tests = [{
        "status": dataset_status,
        "metric_config": {"metric_id": "dataset", "params": {"drift_share": 0.11}},
    }]
    for i, column in enumerate(columns):
        metrics.append({"id": f"m{i}", "value": 0.5 if i == 0 else 0.01})
        tests.append({
            "status": status if i == 0 else "SUCCESS",
            "metric_config": {
                "metric_id": f"m{i}",
                "params": {"column": column, "method": "some method", "threshold": 0.1},
            },
        })
    return {"metrics": metrics, "tests": tests}


def test_column_verdicts_reads_each_column_test_with_its_own_metric_value():
    verdicts = by_column(column_verdicts(snapshot_json()))
    first = FEATURE_COLUMNS[0]
    assert verdicts[first] == ColumnDrift(first, "some method", 0.1, 0.5, True)
    assert [c for c, v in verdicts.items() if v.drifted] == [first]
    assert verdicts[FEATURE_COLUMNS[1]].score == 0.01


@pytest.mark.parametrize("status", ["ERROR", "WARNING", "SKIPPED"])
def test_column_verdicts_rejects_a_status_that_is_neither_pass_nor_fail(status):
    """An errored drift test must not be read as "no drift"."""
    with pytest.raises(ValueError, match=status):
        column_verdicts(snapshot_json(status=status))


def test_column_verdicts_rejects_tests_that_miss_a_model_input():
    with pytest.raises(ValueError, match="expected"):
        column_verdicts(snapshot_json(columns=FEATURE_COLUMNS[:-1]))


@pytest.mark.parametrize(("status", "expected"), [("FAIL", True), ("SUCCESS", False)])
def test_dataset_drift_detected_reads_the_dataset_test(status, expected):
    assert dataset_drift_detected(snapshot_json(dataset_status=status)) is expected


def test_dataset_drift_detected_rejects_an_errored_dataset_test():
    with pytest.raises(ValueError, match="dataset drift test"):
        dataset_drift_detected(snapshot_json(dataset_status="ERROR"))


def test_tier_shares_add_flagged_as_elevated_plus_high():
    shares = tier_shares(COUNTS)
    assert shares == pytest.approx(
        {"low": 0.95, "elevated": 0.02, "high": 0.03, "flagged": 0.05}
    )


def test_tier_shares_of_an_empty_window_are_undefined():
    with pytest.raises(ValueError, match="No requests"):
        tier_shares({"low": 0, "elevated": 0, "high": 0})


def test_expected_shares_are_the_m6_out_of_fold_tiers():
    """Low + elevated + high cover the 7,992 training rows; 391 of them were flagged."""
    assert sum(EXPECTED_TIER_SHARES.values()) == pytest.approx(1.0)
    assert EXPECTED_FLAGGED_SHARE == pytest.approx(391 / 7992)


@pytest.fixture
def monitoring_run(tmp_path):
    """An active MLflow run whose artifacts land in tmp_path (private store from conftest)."""
    experiment_id = mlflow.create_experiment(
        "monitor-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    with mlflow.start_run(experiment_id=experiment_id) as run:
        yield run.info.run_id


def finished(run_id):
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    artifacts = sorted(a.path for a in client.list_artifacts(run_id))
    return run.data, artifacts


def test_small_window_logs_workload_but_no_drift_verdict(monitoring_run):
    """D59: under MIN_ROWS there is no drift report, only tier shares."""
    verdict = log_window(SINCE, 4000, 11, {"low": 9, "elevated": 1, "high": 1}, None, [])
    mlflow.end_run()
    data, artifacts = finished(monitoring_run)
    assert verdict == INSUFFICIENT_DATA
    assert data.tags["verdict"] == INSUFFICIENT_DATA
    assert data.params["n_current"] == "11"
    assert data.metrics["share_flagged"] == pytest.approx(2 / 11)
    assert not any(name.startswith("drift_score_") for name in data.metrics)
    assert artifacts == []


def test_empty_window_logs_parameters_only(monitoring_run):
    verdict = log_window(SINCE, 4000, 0, {"low": 0, "elevated": 0, "high": 0}, None, [])
    mlflow.end_run()
    data, artifacts = finished(monitoring_run)
    assert verdict == INSUFFICIENT_DATA
    assert data.metrics == {}
    assert artifacts == []


def test_full_window_logs_scores_verdict_and_report(monitoring_run, air_fault):
    snapshot, verdicts = air_fault
    verdict = log_window(SINCE, 4000, MIN_ROWS, COUNTS, snapshot, verdicts)
    mlflow.end_run()
    data, artifacts = finished(monitoring_run)
    assert verdict == "drift"
    assert data.tags["verdict"] == "drift"
    assert data.tags["drifted"] == "air_temp_k,temp_diff_k"
    assert data.metrics["drifted_columns"] == 2
    assert {f"drift_score_{c}" for c in FEATURE_COLUMNS} <= set(data.metrics)
    assert data.metrics["flagged_vs_expected"] == pytest.approx(0.05 / EXPECTED_FLAGGED_SHARE)
    assert artifacts == ["drift_report.html", "drift_snapshot.json"]


def test_since_must_carry_a_utc_offset():
    with pytest.raises(argparse.ArgumentTypeError, match="UTC offset"):
        parse_since("2026-10-05T00:00:00")
    assert parse_since("2026-10-05T00:00:00+05:30").utcoffset() == datetime.timedelta(
        hours=5, minutes=30
    )