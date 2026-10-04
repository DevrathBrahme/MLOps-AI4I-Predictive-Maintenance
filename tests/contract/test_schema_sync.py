"""The schemas stay in sync with the model, the risk tiers, the feature code,
the ingestion rules and the predictions_log columns.
"""

from typing import get_args

from ai4i.data import FEATURE_COLUMNS
from ai4i.features import RAW_COLUMNS
from ai4i.ingest import VALID_TYPES
from ai4i.model import CLASSES
from ai4i.prediction_log import PROBA_COLUMNS
from ai4i.risk import TIERS
from ai4i.schemas import (
    ClassProbabilities,
    ModeName,
    PhysicsFeatures,
    SensorReading,
    TierName,
    TierName,
)


def test_mode_and_tier_literals_match_model_and_risk():
    """The response's literals are the model's classes and the risk tiers, in order."""
    assert get_args(ModeName) == CLASSES
    assert get_args(TierName) == TIERS


def test_grade_literal_matches_ingestion_types():
    """The API accepts exactly the product grades the loader accepts."""
    grades = get_args(SensorReading.model_fields["type"].annotation)

    assert set(grades) == set(VALID_TYPES)


def test_reading_fields_are_the_raw_feature_inputs_in_order():
    """A reading's fields are exactly what build_features needs."""
    assert tuple(SensorReading.model_fields) == RAW_COLUMNS


def test_probability_fields_follow_classes():
    """The API builds ClassProbabilities by zipping CLASSES with the model's columns."""
    assert tuple(ClassProbabilities.model_fields) == CLASSES


def test_log_probability_columns_follow_classes():
    """The API zips PROBA_COLUMNS with the model's columns, so their order must match CLASSES."""
    assert [column.removeprefix("p_") for column in PROBA_COLUMNS] == [
        cls.lower() for cls in CLASSES
    ]


def test_physics_fields_are_model_features():
    """The API reads each physics field from the built feature frame by name."""
    assert set(PhysicsFeatures.model_fields) <= set(FEATURE_COLUMNS)