"""Risk tiers for operator review: pure functions shared by thresholds and the API."""

import numpy as np

from ai4i.model import CLASSES

NO_FAILURE = CLASSES.index("no_failure")
TIERS = ("low", "elevated", "high")
THRESHOLD_TAG = "elevated_threshold"


def _check_columns(proba: np.ndarray) -> None:
    """Raise ValueError unless proba is 2-D with one column per class in CLASSES."""
    if proba.ndim != 2 or proba.shape[1] != len(CLASSES):
        raise ValueError(
            f"Expected probabilities of shape (n, {len(CLASSES)}), got {proba.shape}"
        )


def _predicts_failure(proba: np.ndarray) -> np.ndarray:
    """Return a boolean mask of rows whose argmax is a failure mode."""
    return proba.argmax(axis=1) != NO_FAILURE


def confidence_score(proba: np.ndarray) -> np.ndarray:
    """Return each row's failure score, 1 - P(no_failure).

    Uncalibrated: class weighting inflates rare-class probabilities, so this is
    a ranking score for operators, not a failure frequency.
    """
    _check_columns(proba)
    return 1 - proba[:, NO_FAILURE]


def assign_tiers(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Return one tier name per row.

    high: the argmax is a failure mode. elevated: the argmax is no_failure but
    the confidence score is strictly above threshold. low: everything else.
    """
    high = _predicts_failure(proba)
    elevated = ~high & (confidence_score(proba) > threshold)
    return np.select([high, elevated], ["high", "elevated"], default="low")


def choose_threshold(
    proba: np.ndarray, y_true: np.ndarray, budget_fraction: float
) -> float:
    """Return the lowest threshold keeping flagged healthy rows within budget.

    Flagged means high or elevated. The budget is floor(budget_fraction x number
    of healthy rows). The threshold is the (room + 1)-th largest score among
    healthy rows not already high, so with a strict > at most `room` of them
    become elevated (fewer if scores tie). Raises ValueError if the high tier
    alone exceeds the budget or if the budget leaves nothing to cut.
    """
    if len(y_true) != len(proba):
        raise ValueError(f"{len(y_true)} labels but {len(proba)} probability rows")
    healthy = y_true == NO_FAILURE
    high = _predicts_failure(proba)
    budget = int(np.floor(budget_fraction * healthy.sum()))
    healthy_high = int((healthy & high).sum())
    room = budget - healthy_high
    if room < 0:
        raise ValueError(
            f"High tier alone flags {healthy_high} healthy rows, "
            f"above the budget of {budget}"
        )
    candidates = np.sort(confidence_score(proba)[healthy & ~high])[::-1]
    if room >= len(candidates):
        raise ValueError(
            f"Budget of {budget} healthy rows would flag every candidate; "
            "no threshold is needed"
        )
    return float(candidates[room])