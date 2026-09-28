import logging

import pandas as pd
import psycopg
from psycopg import sql

from ai4i.db import get_connection

logger = logging.getLogger(__name__)


COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf"
}


FLAG_COLUMNS = ("machine_failure", "twf", "hdf", "pwf", "osf", "rnf")
VALID_TYPES = {"L", "M", "H"}
DEFAULT_CSV = "data/raw/ai4i2020.csv"
TABLE = "sensor_readings"


def read_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, encoding='utf-8-sig')


def transform(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMN_MAP) - set(df.columns)
    unexpected = set(df.columns) - set(COLUMN_MAP)
    if missing or unexpected:
        raise ValueError(f"Column mismatch. Missing: {missing}. Unexpected: {unexpected}.")
    return df.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]


def validate(df: pd.DataFrame) -> None:
    problems = []
    if df.isnull().any().any():
        problems.append("DataFrame contains null values")
    if not df["udi"].is_unique:
        problems.append("UDI values are not unique")
    if not df["type"].isin(VALID_TYPES).all():
        problems.append(f"Type values must be one of {VALID_TYPES}")
    if not df["product_id"].apply(lambda x: isinstance(x, str) and len(x) == 6).all():
        problems.append("Product ID values must be strings of length 6")
    for flag in FLAG_COLUMNS:
        if not df[flag].isin([0, 1]).all():
            problems.append(f"{flag} values must be either 0 or 1")
    if problems:
        raise ValueError(f"Validation failed with the following issues: {'; '.join(problems)}")


def convert_flags(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({col: "bool" for col in FLAG_COLUMNS})


def load(df: pd.DataFrame, conn: psycopg.Connection) -> int:
    cols = list(COLUMN_MAP.values())
    col_list = sql.SQL(", ").join(sql.Identifier(col) for col in cols)
    update_cols = [c for c in cols if c != "udi"]

    set_clause = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
        for c in update_cols
    )
    target_row = sql.SQL(", ").join(sql.Identifier(TABLE, c) for c in update_cols)
    excluded_row = sql.SQL(", ").join(
        sql.SQL("EXCLUDED.{}").format(sql.Identifier(c)) for c in update_cols
    )

    upsert = sql.SQL(
        "INSERT INTO {table} ({cols}) SELECT {cols} FROM staging "
        "ON CONFLICT (udi) DO UPDATE SET {set_clause} "
        "WHERE ({target_row}) IS DISTINCT FROM ({excluded_row})"
    ).format(
        table=sql.Identifier(TABLE),
        cols=col_list,
        set_clause=set_clause,
        target_row=target_row,
        excluded_row=excluded_row,
    )
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE TEMP TABLE staging (LIKE {}) ON COMMIT DROP").format(sql.Identifier(TABLE)))

        with cur.copy(sql.SQL("COPY staging ({}) FROM STDIN").format(col_list)) as copy:
            for row in df.itertuples(index=False):
                copy.write_row(row)
        cur.execute(upsert)
        return cur.rowcount


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = read_raw(DEFAULT_CSV)
    df = transform(df)
    validate(df)
    df = convert_flags(df)
    with get_connection() as conn:
        changed = load(df, conn)
    logger.info("Read %d rows; %d inserted or updated", len(df), changed)


if __name__ == "__main__":
    main()