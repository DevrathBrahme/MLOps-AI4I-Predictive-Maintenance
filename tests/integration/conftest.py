"""Fixtures for the integration tests, which run against the Compose Postgres.

Every connection is rolled back and closed after its test, so nothing a test
writes is ever committed. If Postgres is unreachable the test fails with
instructions instead of being skipped.
"""

import psycopg
import pytest

from ai4i.db import get_connection


@pytest.fixture
def pg_conn():
    """Yield a connection to the Compose Postgres whose work is always rolled back.

    Deliberately not ``with get_connection() as conn:``, which commits on
    success.
    """
    try:
        conn = get_connection()
    except (psycopg.OperationalError, RuntimeError) as exc:
        pytest.fail(
            f"Postgres is unreachable: {exc}\n"
            "Start the stack (docker compose up -d) and source .env, "
            "or deselect these tests with -m 'not integration'.",
            pytrace=False,
        )
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()