"""Unit tests for ai4i.risk on a six-row toy with a hand-worked threshold.

Rows 0-3 are healthy; row 3's argmax is TWF, so it is already high. Row 4 is
a real TWF failure whose argmax is no_failure, so only the elevated tier can
catch it. Row 5 is an HDF failure predicted as HDF.
"""

import numpy as np
import pytest

from ai4i.model import CLASSES
from ai4i.risk import NO_FAILURE, TIERS, assign_tiers, choose_threshold, confidence_score

PROBA = np.array(
    [
        [0.95, 0.05, 0.00, 0.00, 0.00],
        [0.80, 0.20, 0.00, 0.00, 0.00],
        [0.70, 0.30, 0.00, 0.00, 0.00],
        [0.40, 0.60, 0.00, 0.00, 0.00],
        [0.50, 0.10, 0.40, 0.00, 0.00],
        [0.10, 0.00, 0.90, 0.00, 0.00],
    ]
)
Y = np.array([0, 0, 0, 0, 1, 2])


def test_confidence_score_is_one_minus_p_no_failure():
    """The score is 1 - P(no_failure), whatever the other columns hold."""
    assert confidence_score(PROBA).tolist() == pytest.approx(
        [0.05, 0.20, 0.30, 0.60, 0.50, 0.90]
    )


@pytest.mark.parametrize(
    "proba",
    [
        pytest.param(np.ones((2, len(CLASSES) - 1)), id="missing a class column"),
        pytest.param(np.ones(len(CLASSES)), id="one-dimensional"),
    ],
)
def test_confidence_score_rejects_wrong_shape(proba):
    """Only an (n, 5) matrix is accepted."""
    with pytest.raises(ValueError, match="Expected probabilities of shape"):
        confidence_score(proba)


def test_threshold_is_the_score_just_past_the_remaining_room():
    """Budget floor(0.5 x 4) = 2; row 3 is healthy and already high, so room = 1.

    Healthy non-high scores sorted: 0.30, 0.20, 0.05, so the threshold is 0.20.
    """
    assert choose_threshold(PROBA, Y, budget_fraction=0.5) == pytest.approx(0.20)


def test_tiers_follow_the_rule():
    """high = argmax is a failure; elevated = argmax no_failure and score > t; else low."""
    threshold = choose_threshold(PROBA, Y, budget_fraction=0.5)
    tiers = assign_tiers(PROBA, threshold)

    assert tiers.tolist() == ["low", "low", "elevated", "high", "elevated", "high"]
    assert set(tiers.tolist()) <= set(TIERS)


def test_score_equal_to_threshold_stays_low():
    """Strict >: row 1 scores exactly the threshold and is not flagged."""
    threshold = choose_threshold(PROBA, Y, budget_fraction=0.5)

    assert confidence_score(PROBA)[1] == threshold
    assert assign_tiers(PROBA, threshold)[1] == "low"


def test_flagged_healthy_rows_stay_within_budget():
    """High plus elevated healthy rows never exceed floor(budget x healthy rows)."""
    threshold = choose_threshold(PROBA, Y, budget_fraction=0.5)
    tiers = assign_tiers(PROBA, threshold)
    healthy = Y == NO_FAILURE

    assert int((healthy & (tiers != "low")).sum()) <= int(np.floor(0.5 * healthy.sum()))


def test_budget_below_the_high_tier_is_rejected():
    """Budget floor(0.1 x 4) = 0, but row 3 is already high."""
    with pytest.raises(ValueError, match="High tier alone"):
        choose_threshold(PROBA, Y, budget_fraction=0.1)


def test_budget_covering_every_candidate_is_rejected():
    """Budget 4 leaves room 3, which covers all three healthy non-high rows."""
    with pytest.raises(ValueError, match="no threshold is needed"):
        choose_threshold(PROBA, Y, budget_fraction=1.0)


def test_label_and_probability_counts_must_match():
    """Five labels for six probability rows is a caller bug, not a threshold."""
    with pytest.raises(ValueError, match="labels but"):
        choose_threshold(PROBA, Y[:5], budget_fraction=0.5)