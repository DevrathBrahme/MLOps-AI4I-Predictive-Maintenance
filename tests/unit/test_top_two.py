"""Unit tests for ai4i.api.top_two, the runner-up rule behind the API response.

The runner-up must be the second-highest class and never the predicted one,
and it must agree with the rule in ai4i.metrics.misclassified_rows, which
produced the M5 evidence that the true mode is usually ranked second.
"""

import numpy as np
import pandas as pd

from ai4i.api import top_two
from ai4i.metrics import misclassified_rows
from ai4i.model import CLASSES

ROW = (0.10, 0.60, 0.20, 0.05, 0.05)


def test_returns_predicted_and_runner_up_codes():
    """TWF (code 1) is highest, HDF (code 2) second."""
    assert top_two(np.array(ROW)) == (1, 2)


def test_returns_plain_python_ints():
    """The signature promises int; np.int64 is not an int subclass."""
    predicted, runner_up = top_two(np.array(ROW))

    assert type(predicted) is int
    assert type(runner_up) is int


def test_tied_top_scores_still_give_two_different_classes():
    """With a tie, argmax takes the first; masking guarantees a different runner-up."""
    assert top_two(np.array([0.40, 0.40, 0.10, 0.05, 0.05])) == (0, 1)


def test_input_row_is_not_modified():
    """Masking happens on a copy; the caller's probabilities are untouched."""
    row = np.array(ROW)
    top_two(row)

    assert row.tolist() == list(ROW)


def test_runner_up_rule_matches_the_error_analysis():
    """The API's runner-up and the M5 error-analysis runner-up are the same rule."""
    rng = np.random.default_rng(42)
    proba = rng.dirichlet(np.ones(len(CLASSES)), size=50)
    y_wrong = (proba.argmax(axis=1) + 1) % len(CLASSES)  # every row misclassified

    report = misclassified_rows(pd.RangeIndex(len(proba)), y_wrong, proba)

    assert report["runner_up"].tolist() == [CLASSES[top_two(row)[1]] for row in proba]