# db_utils.py
from sqlalchemy import create_engine
import pandas as pd

def read_sql_table(conn_str, table_name, limit=None):
    engine = create_engine(conn_str)
    qry = f"SELECT * FROM {table_name}"
    if limit:
        qry += f" LIMIT {int(limit)}"
    return pd.read_sql(qry, engine)

def test_connection(conn_str):
    engine = create_engine(conn_str)
    with engine.connect() as conn:
        return True
