"""Property tests for ai4i.train.dataset_fingerprint.

The digest itself is meaningless, so the tests check its properties: the
same for the same rows in any order, different after any change to a value,
a label, a udi or the set of rows.
"""

import numpy as np
import pandas as pd
import pytest

from ai4i.train import dataset_fingerprint

EXTRA_ROW = pd.DataFrame(
    {"type": ["L"], "torque_nm": [45.0]},
    index=pd.Index([4], name="udi"),
)


@pytest.fixture
def sample():
    """Three rows indexed by udi, with labels aligned by position."""
    X = pd.DataFrame(
        {"type": ["L", "M", "H"], "torque_nm": [40.0, 50.0, 60.0]},
        index=pd.Index([1, 2, 3], name="udi"),
    )
    y = np.array([0, 1, 2])
    return X, y


def test_fingerprint_is_a_stable_sha256_hex_digest(sample):
    """Equal inputs give equal 64-character lowercase hex digests."""
    X, y = sample
    digest = dataset_fingerprint(X, y)

    assert digest == dataset_fingerprint(X.copy(), y.copy())
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_fingerprint_ignores_row_order(sample):
    """Shuffling rows (labels move with them) does not change the digest."""
    X, y = sample
    order = [2, 0, 1]

    assert dataset_fingerprint(X.iloc[order], y[order]) == dataset_fingerprint(X, y)


MUTATIONS = {
    "value changed by one bit": lambda X, y: (
        X.assign(torque_nm=[40.0, np.nextafter(50.0, np.inf), 60.0]),
        y,
    ),
    "label changed": lambda X, y: (X, np.array([0, 1, 3])),
    "udi changed": lambda X, y: (X.rename(index={3: 4}), y),
    "row removed": lambda X, y: (X.iloc[:2], y[:2]),
    "row added": lambda X, y: (pd.concat([X, EXTRA_ROW]), np.append(y, 0)),
}


@pytest.mark.parametrize("mutate", MUTATIONS.values(), ids=list(MUTATIONS))
def test_any_change_changes_the_fingerprint(sample, mutate):
    """Any difference in the rows, however small, gives a different digest."""
    X, y = sample

    assert dataset_fingerprint(*mutate(X, y)) != dataset_fingerprint(X, y)