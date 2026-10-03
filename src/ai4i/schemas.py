"""Request and response schemas for the recommendation API (human in the loop)."""

import datetime
import uuid
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai4i.model import CLASSES
from ai4i.risk import TIERS

ModeName = Literal["no_failure", "TWF", "HDF", "PWF", "OSF"]
TierName = Literal["low", "elevated", "high"]

if get_args(ModeName) != CLASSES or get_args(TierName) != TIERS:
    raise RuntimeError("Schema literals are out of sync with CLASSES or TIERS")


class SensorReading(BaseModel):
    """One machine reading; field names match sensor_readings and RAW_COLUMNS."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "type": "M",
                    "air_temp_k": 298.1,
                    "process_temp_k": 308.6,
                    "rotational_speed_rpm": 1551,
                    "torque_nm": 42.8,
                    "tool_wear_min": 0,
                }
            ]
        },
    )

    type: Literal["L", "M", "H"] = Field(description="Product quality grade")
    air_temp_k: float = Field(gt=0, description="Air temperature, K")
    process_temp_k: float = Field(gt=0, description="Process temperature, K")
    rotational_speed_rpm: int = Field(gt=0, description="Rotational speed, rpm")
    torque_nm: float = Field(ge=0, description="Torque, N·m")
    tool_wear_min: int = Field(ge=0, description="Tool wear, min")


class ClassProbabilities(BaseModel):
    """Model probability per class: class-weighted, so a ranking, not a frequency."""

    model_config = ConfigDict(extra="forbid")

    no_failure: float = Field(ge=0, le=1)
    TWF: float = Field(ge=0, le=1)
    HDF: float = Field(ge=0, le=1)
    PWF: float = Field(ge=0, le=1)
    OSF: float = Field(ge=0, le=1)


class PhysicsFeatures(BaseModel):
    """Physics features computed from the reading, for checking against known limits."""

    model_config = ConfigDict(extra="forbid")

    power_w: float = Field(description="Mechanical power, W (torque x rpm x 2π/60)")
    temp_diff_k: float = Field(description="Process minus air temperature, K")
    strain_min_nm: float = Field(description="Tool wear x torque, min·N·m")


class Recommendation(BaseModel):
    """Advisory output for a human operator. Deliberately has no action field."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    request_id: uuid.UUID = Field(description="Id of the predictions_log row")
    created_at: datetime.datetime = Field(description="Time the row was logged (UTC)")
    model_name: str = Field(description="Registered model name")
    model_version: int = Field(description="Registered model version that scored this")
    flagged: bool = Field(
        description="True for the elevated or high tier: an operator should review. "
        "The API never acts on this itself."
    )
    risk_tier: TierName = Field(
        description="high: the model's top class is a failure mode. elevated: top "
        "class is no_failure but confidence_score is above the model version's "
        "threshold (2% healthy-row review budget). low: everything else."
    )
    confidence_score: float = Field(
        ge=0,
        le=1,
        description="1 - P(no_failure). Uncalibrated ranking score from a "
        "class-weighted model, not a failure frequency.",
    )
    predicted_mode: ModeName = Field(description="Most probable class")
    predicted_probability: float = Field(ge=0, le=1)
    runner_up_mode: ModeName = Field(description="Second most probable class")
    runner_up_probability: float = Field(ge=0, le=1)
    class_probabilities: ClassProbabilities
    physics_features: PhysicsFeatures

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Reject a flag/tier mismatch or a runner-up equal to the prediction."""
        if self.flagged != (self.risk_tier != "low"):
            raise ValueError(
                f"flagged={self.flagged} contradicts risk_tier={self.risk_tier!r}"
            )
        if self.runner_up_mode == self.predicted_mode:
            raise ValueError(
                f"runner_up_mode equals predicted_mode ({self.predicted_mode!r})"
            )
        return self