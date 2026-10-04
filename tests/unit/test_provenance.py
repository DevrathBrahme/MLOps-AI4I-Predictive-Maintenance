"""Unit tests for ai4i.train.provenance_tags: the git commit on container-trained runs."""

import pytest

from ai4i.train import GIT_COMMIT_ENV, GIT_COMMIT_TAG, provenance_tags

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_commit_from_the_environment_becomes_the_standard_mlflow_tag():
    """A valid hash is recorded under MLflow's own git commit tag."""
    assert provenance_tags({GIT_COMMIT_ENV: SHA}) == {GIT_COMMIT_TAG: SHA}


def test_surrounding_whitespace_is_ignored():
    """A trailing newline from `git rev-parse HEAD` doesn't break the hash."""
    assert provenance_tags({GIT_COMMIT_ENV: f" {SHA}\n"}) == {GIT_COMMIT_TAG: SHA}


@pytest.mark.parametrize(
    "environ",
    [
        pytest.param({}, id="unset"),
        pytest.param({GIT_COMMIT_ENV: ""}, id="empty"),
        pytest.param({GIT_COMMIT_ENV: "   "}, id="blank"),
    ],
)
def test_no_commit_adds_no_tag(environ):
    """Unset or blank (local container runs) leaves MLflow's own detection alone."""
    assert provenance_tags(environ) == {}


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("abc1234", id="short hash"),
        pytest.param(SHA.upper(), id="uppercase"),
        pytest.param(SHA + "0", id="too long"),
        pytest.param("main", id="branch name"),
    ],
)
def test_anything_but_a_full_hash_is_rejected(value):
    """A value that isn't a full commit hash stops training instead of being recorded."""
    with pytest.raises(ValueError, match=GIT_COMMIT_ENV):
        provenance_tags({GIT_COMMIT_ENV: value})