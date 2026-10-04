"""Contract tests for the SensorReading request schema.

Each rejection asserts the exact error location and type, so a payload that
fails for an unrelated reason cannot make the test pass.
"""

import pandas as pd
import pytest
from pydantic import ValidationError

from ai4i.data import FEATURE_COLUMNS
from ai4i.features import build_features
from ai4i.schemas import SensorReading

EXAMPLE = SensorReading.model_config["json_schema_extra"]["examples"][0]
MISSING = object()

REJECTIONS = [
    pytest.param("udi", 1, "extra_forbidden", id="extra field udi"),
    pytest.param("action", "stop_machine", "extra_forbidden", id="action field"),
    pytest.param("air_temp_k", MISSING, "missing", id="missing field"),
    pytest.param("type", "X", "literal_error", id="unknown grade"),
    pytest.param("rotational_speed_rpm", "1551", "int_type", id="rpm as string"),
    pytest.param("rotational_speed_rpm", 1551.0, "int_type", id="rpm as float"),
    pytest.param("rotational_speed_rpm", True, "int_type", id="rpm as bool"),
    pytest.param("air_temp_k", "298.1", "float_type", id="temperature as string"),
    pytest.param("air_temp_k", float("nan"), "finite_number", id="temperature NaN"),
    pytest.param("air_temp_k", float("inf"), "finite_number", id="temperature inf"),
    pytest.param("air_temp_k", 0.0, "greater_than", id="zero kelvin"),
    pytest.param("rotational_speed_rpm", 0, "greater_than", id="zero rpm"),
    pytest.param("torque_nm", -1.0, "greater_than_equal", id="negative torque"),
    pytest.param("tool_wear_min", -1, "greater_than_equal", id="negative tool wear"),
]


def test_docs_example_is_a_valid_reading():
    """The example shown in /docs validates and round-trips unchanged."""
    assert SensorReading.model_validate(EXAMPLE).model_dump() == EXAMPLE


def test_valid_reading_feeds_the_feature_builder():
    """A validated reading becomes exactly one row of the 9 training columns."""
    reading = SensorReading.model_validate(EXAMPLE)
    features = build_features(pd.DataFrame([reading.model_dump()]))

    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert len(features) == 1


@pytest.mark.parametrize(("field", "value", "error_type"), REJECTIONS)
def test_invalid_reading_is_rejected_on_that_field_only(field, value, error_type):
    """Exactly one error, on the expected field, of the expected type."""
    payload = dict(EXAMPLE)
    if value is MISSING:
        del payload[field]
    else:
        payload[field] = value

    with pytest.raises(ValidationError) as excinfo:
        SensorReading.model_validate(payload)

    errors = excinfo.value.errors()
    assert [(error["loc"], error["type"]) for error in errors] == [((field,), error_type)]