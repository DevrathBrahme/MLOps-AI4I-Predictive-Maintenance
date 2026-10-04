"""Training-serving parity: Python features are bit-identical to the SQL view.

Training reads power_w, temp_diff_k and strain_min_nm from the sensor_features
view; the API computes them in Python. A last-bit difference could move a
reading across a physics threshold (e.g. temp_diff_k < 8.6), so equality is
exact, on every row.
"""

import numpy as np
import pandas as pd
from psycopg import sql

from ai4i.data import VIEW
from ai4i.features import RAW_COLUMNS, add_physics_features

PHYSICS_COLUMNS = ("power_w", "temp_diff_k", "strain_min_nm")


def test_python_features_are_bit_identical_to_the_sql_view(pg_conn):
    """All 10,000 readings, compared with exact equality, not a tolerance."""
    query = sql.SQL("SELECT {columns} FROM {view} ORDER BY udi").format(
        columns=sql.SQL(", ").join(
            sql.Identifier(name) for name in ("udi", *RAW_COLUMNS, *PHYSICS_COLUMNS)
        ),
        view=sql.Identifier(VIEW),
    )
    with pg_conn.cursor() as cur:
        cur.execute(query)
        names = [column.name for column in cur.description]
        from_sql = pd.DataFrame(cur.fetchall(), columns=names).set_index("udi")

    from_python = add_physics_features(from_sql[list(RAW_COLUMNS)])

    assert len(from_sql) == 10_000
    for column in PHYSICS_COLUMNS:
        np.testing.assert_array_equal(
            from_python[column].to_numpy(),
            from_sql[column].to_numpy(),
            err_msg=column,
        )