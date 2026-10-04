"""Contract tests for the Recommendation response: advisory only, by construction.

The response has a fixed set of fields, none of which can carry an action;
unknown fields are rejected at every level; and it cannot contradict the tier
rule it reports.
"""

import datetime
import uuid

import pytest
from pydantic import ValidationError

from ai4i.schemas import Recommendation

EXPECTED_FIELDS = {
    "request_id",
    "created_at",
    "model_name",
    "model_version",
    "flagged",
    "risk_tier",
    "confidence_score",
    "predicted_mode",
    "predicted_probability",
    "runner_up_mode",
    "runner_up_probability",
    "class_probabilities",
    "physics_features",
}
ACTION_WORDS = ("action", "command", "execute", "trigger", "shutdown", "stop", "halt", "dispatch")


@pytest.fixture
def payload():
    """A valid high-tier TWF recommendation; tests change copies of it."""
    return {
        "request_id": uuid.UUID(int=1),
        "created_at": datetime.datetime(2026, 10, 4, 3, 0, tzinfo=datetime.UTC),
        "model_name": "ai4i-failure-classifier",
        "model_version": 1,
        "flagged": True,
        "risk_tier": "high",
        "confidence_score": 0.9,
        "predicted_mode": "TWF",
        "predicted_probability": 0.7,
        "runner_up_mode": "no_failure",
        "runner_up_probability": 0.1,
        "class_probabilities": {
            "no_failure": 0.1, "TWF": 0.7, "HDF": 0.1, "PWF": 0.05, "OSF": 0.05,
        },
        "physics_features": {"power_w": 6951.6, "temp_diff_k": 10.5, "strain_min_nm": 0.0},
    }


def property_names(schema: dict) -> set[str]:
    """Return every property name in a JSON schema, including nested models."""
    names = set(schema.get("properties", {}))
    for definition in schema.get("$defs", {}).values():
        names |= set(definition.get("properties", {}))
    return names


def test_valid_recommendation_is_accepted(payload):
    """The fixture itself is a consistent response."""
    recommendation = Recommendation.model_validate(payload)

    assert recommendation.flagged is True
    assert recommendation.risk_tier == "high"


def test_response_fields_are_exactly_the_published_contract():
    """Adding or removing a field is a deliberate contract change, not an accident."""
    assert set(Recommendation.model_fields) == EXPECTED_FIELDS


def test_no_field_at_any_level_can_carry_an_action():
    """No property name in the response or its nested models suggests an action."""
    names = property_names(Recommendation.model_json_schema())

    assert len(names) == 13 + 5 + 3
    assert not [name for name in names if any(word in name.lower() for word in ACTION_WORDS)]


def test_every_model_in_the_response_forbids_extra_fields():
    """additionalProperties is false for the response and both nested models."""
    schema = Recommendation.model_json_schema()
    models = [schema, *schema["$defs"].values()]

    assert len(models) == 3
    assert all(model.get("additionalProperties") is False for model in models)


def test_extra_top_level_field_is_rejected(payload):
    """The response cannot be extended with an action at validation time either."""
    with pytest.raises(ValidationError) as excinfo:
        Recommendation.model_validate({**payload, "action": "stop_machine"})

    assert [(e["loc"], e["type"]) for e in excinfo.value.errors()] == [
        (("action",), "extra_forbidden")
    ]


def test_extra_nested_field_is_rejected(payload):
    """RNF is not a modelled class, so it cannot appear among the probabilities."""
    payload["class_probabilities"]["RNF"] = 0.0

    with pytest.raises(ValidationError) as excinfo:
        Recommendation.model_validate(payload)

    assert [(e["loc"], e["type"]) for e in excinfo.value.errors()] == [
        (("class_probabilities", "RNF"), "extra_forbidden")
    ]


@pytest.mark.parametrize(
    ("tier", "predicted", "runner_up"),
    [
        ("high", "TWF", "no_failure"),
        ("elevated", "no_failure", "TWF"),
        ("low", "no_failure", "TWF"),
    ],
)
def test_flagged_is_true_exactly_for_elevated_and_high(payload, tier, predicted, runner_up):
    """The matching flag is accepted; the opposite flag is rejected."""
    payload.update(risk_tier=tier, predicted_mode=predicted, runner_up_mode=runner_up)
    flagged = tier != "low"

    assert Recommendation.model_validate({**payload, "flagged": flagged}).flagged is flagged
    with pytest.raises(ValidationError, match="contradicts risk_tier"):
        Recommendation.model_validate({**payload, "flagged": not flagged})


@pytest.mark.parametrize(
    ("tier", "predicted", "runner_up"),
    [
        pytest.param("high", "no_failure", "TWF", id="high but top class healthy"),
        pytest.param("elevated", "HDF", "no_failure", id="elevated but top class a failure"),
        pytest.param("low", "TWF", "no_failure", id="low but top class a failure"),
    ],
)
def test_tier_must_agree_with_predicted_mode(payload, tier, predicted, runner_up):
    """high exactly when the top class is a failure mode (the half of the rule the response can prove)."""
    payload.update(
        risk_tier=tier,
        flagged=tier != "low",
        predicted_mode=predicted,
        runner_up_mode=runner_up,
    )

    with pytest.raises(ValidationError, match="contradicts predicted_mode"):
        Recommendation.model_validate(payload)


def test_runner_up_equal_to_prediction_is_rejected(payload):
    """The runner-up is the second choice, never a repeat of the first."""
    payload["runner_up_mode"] = payload["predicted_mode"]

    with pytest.raises(ValidationError, match="runner_up_mode equals predicted_mode"):
        Recommendation.model_validate(payload)


def test_unknown_mode_is_rejected(payload):
    """Only the five modelled classes can be predicted."""
    payload["predicted_mode"] = "RNF"

    with pytest.raises(ValidationError) as excinfo:
        Recommendation.model_validate(payload)

    assert [(e["loc"], e["type"]) for e in excinfo.value.errors()] == [
        (("predicted_mode",), "literal_error")
    ]


def test_confidence_score_above_one_is_rejected(payload):
    """The score is bounded to [0, 1]."""
    payload["confidence_score"] = 1.5

    with pytest.raises(ValidationError) as excinfo:
        Recommendation.model_validate(payload)

    assert [(e["loc"], e["type"]) for e in excinfo.value.errors()] == [
        (("confidence_score",), "less_than_equal")
    ]


def test_honesty_caveats_are_published_in_the_schema():
    """/docs tells operators the score is uncalibrated and the API never acts."""
    fields = Recommendation.model_fields

    assert "Uncalibrated" in fields["confidence_score"].description
    assert "never acts" in fields["flagged"].description