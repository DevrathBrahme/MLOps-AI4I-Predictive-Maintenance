import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from ai4i.data import FEATURE_COLUMNS

CLASSES = ("no_failure", "TWF", "HDF", "PWF", "OSF")
CATEGORICAL_COLUMNS = ("type",)
NUMERIC_COLUMNS = tuple(c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS)
MODEL_NAMES = ("rf", "xgb")


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Map class names to integer codes (index in CLASSES); raise on unknown labels."""
    mapping = {name: i for i, name in enumerate(CLASSES)}
    codes = labels.map(mapping)

    if codes.isna().any():
        unknown = labels[~labels.isin(CLASSES)].unique()
        raise ValueError(f"Unknown class labels: {list(unknown)}; expected one of {CLASSES}")

    return codes.to_numpy()


def decode_labels(codes) -> list[str]:
    return [CLASSES[i] for i in codes]


def build_pipeline(model_name: str, random_state: int = 42) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("type", OrdinalEncoder(categories=[["L", "M", "H"]]), list(CATEGORICAL_COLUMNS)),
            ("numeric", "passthrough", list(NUMERIC_COLUMNS)),
        ],
        verbose_feature_names_out=False,
    )
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1,
        )
    elif model_name == "xgb":
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            objective="multi:softprob", random_state=random_state, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name {model_name!r}; expected one of {MODEL_NAMES}")
    return Pipeline([("preprocess", preprocess), ("model", model)])