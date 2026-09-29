import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from ai4i.data import FEATURE_COLUMNS, TARGET
from ai4i.model import encode_labels

TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    X = df[list(FEATURE_COLUMNS)]
    y = encode_labels(df[TARGET])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    return compute_sample_weight("balanced", y)