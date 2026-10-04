"""Unit tests for ai4i.features on two hand-computed rows.

Bit-level agreement with the SQL view is checked against Postgres in the
integration suite; here the formulas, units and column handling are checked
against values worked out by hand.
"""

import math

import pandas as pd
import pytest

from ai4i.data import FEATURE_COLUMNS
from ai4i.features import RAW_COLUMNS, add_physics_features, build_features

PHYSICS_COLUMNS = ("power_w", "temp_diff_k", "strain_min_nm")


@pytest.fixture
def raw():
    """Two readings chosen so the physics features are easy to work out by hand."""
    return pd.DataFrame(
        {
            "type": ["L", "H"],
            "air_temp_k": [300.0, 298.0],
            "process_temp_k": [310.0, 306.5],
            "rotational_speed_rpm": [1500, 1200],
            "torque_nm": [40.0, 60.0],
            "tool_wear_min": [100, 200],
        }
    )


def test_physics_features_match_hand_computed_values(raw):
    """power = torque x angular speed in rad/s; 1500 rpm = 50*pi rad/s, 1200 rpm = 40*pi rad/s."""
    features = add_physics_features(raw)

    assert features["power_w"].tolist() == pytest.approx(
        [2000 * math.pi, 2400 * math.pi], rel=1e-12
    )
    assert features["temp_diff_k"].tolist() == [10.0, 8.5]
    assert features["strain_min_nm"].tolist() == [4000.0, 12000.0]


def test_add_physics_features_does_not_modify_its_input(raw):
    """The caller's frame keeps its columns; the features come back on a copy."""
    before = list(raw.columns)
    add_physics_features(raw)

    assert list(raw.columns) == before


def test_missing_raw_columns_are_all_listed(raw):
    """One error names every missing column, in RAW_COLUMNS order."""
    with pytest.raises(ValueError, match=r"\['type', 'torque_nm'\]"):
        add_physics_features(raw.drop(columns=["torque_nm", "type"]))


def test_build_features_returns_training_columns_in_order(raw):
    """Extra columns (ids, labels) are dropped; the 9 training columns come back in order."""
    features = build_features(raw.assign(udi=[1, 2], machine_failure=[False, True]))

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert set(FEATURE_COLUMNS) == set(RAW_COLUMNS) | set(PHYSICS_COLUMNS)


def test_build_features_ignores_input_column_order(raw):
    """The pipeline sees the same frame however the request's columns were ordered."""
    shuffled = raw[list(reversed(raw.columns))]

    pd.testing.assert_frame_equal(build_features(shuffled), build_features(raw))