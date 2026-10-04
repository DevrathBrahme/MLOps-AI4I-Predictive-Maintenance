"""Physics-informed features in Python, matching the sensor_features view exactly."""

import numpy as np
import pandas as pd

from ai4i.data import FEATURE_COLUMNS

RAW_COLUMNS = (
    "type",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)


def add_physics_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of raw with power_w, temp_diff_k and strain_min_nm appended.

    Same formulas and operation order as db/init/002_feature_view.sql, so the
    values are bit-identical to the view. Raises ValueError listing every raw
    input column that is missing.
    """
    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw input columns: {missing}")
    return raw.assign(
        power_w=raw["torque_nm"] * ((raw["rotational_speed_rpm"] * 2 * np.pi) / 6),
        temp_diff_k=raw["process_temp_k"] - raw["air_temp_k"],
        strain_min_nm=raw["tool_wear_min"] * raw["torque_nm"],
    )


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Return the 9-column model input in FEATURE_COLUMNS order.

    Adds the physics features, then selects exactly the training columns, so
    extra input columns are dropped and the order matches the fitted pipeline.
    """
    return add_physics_features(raw)[list(FEATURE_COLUMNS)]