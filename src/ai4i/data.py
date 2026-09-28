import pandas as pd
import psycopg
from psycopg import sql

VIEW = "sensor_features"
FEATURE_COLUMNS = (
    "type",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "power_w",
    "temp_diff_k",
    "strain_min_nm"
)
TARGET = "failure_type"


def load_training_data(conn: psycopg.Connection) -> pd.DataFrame:
    columns = [*FEATURE_COLUMNS, TARGET]
    query = sql.SQL("SELECT {cols} FROM {view} WHERE {target} IS NOT NULL ORDER BY udi").format(
        cols=sql.SQL(", ").join(sql.Identifier(col) for col in columns),
        view=sql.Identifier(VIEW),
        target=sql.Identifier(TARGET),
    )
    with conn.cursor() as cur:
        cur.execute(query)
        names = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=names)