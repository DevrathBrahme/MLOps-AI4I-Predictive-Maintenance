import os

import psycopg

REQUIRED_ENV_VARS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB"
)


def get_connection() -> psycopg.Connection:
    """Returns a PostgreSQL connection using environment variables."""

    missing = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]

    if missing:
        raise RuntimeError(f"Missing environment variables (source .env first): {', '.join(missing)}")

    return psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )