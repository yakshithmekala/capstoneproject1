import os
import duckdb
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw_transactions.parquet')
DUCKDB_PATH = os.path.join(DATA_DIR, 'fraud_warehouse.duckdb')


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def validate_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates required column types and handles missing values.
    """
    required_cols = [
        'transaction_id', 'timestamp', 'user_id', 'amount',
        'merchant_category', 'device_type', 'location_lat',
        'location_long', 'ip_address', 'card_present', 'is_fraud'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in input data: {col}")
            
    # Clean data types
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    df['location_lat'] = pd.to_numeric(df['location_lat'], errors='coerce').fillna(0.0)
    df['location_long'] = pd.to_numeric(df['location_long'], errors='coerce').fillna(0.0)
    df['card_present'] = df['card_present'].astype(int)
    df['is_fraud'] = df['is_fraud'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['transaction_id']).reset_index(drop=True)
    return df


def save_raw_transactions(df: pd.DataFrame, parquet_path: str = RAW_DATA_PATH):
    """
    Persists validated raw dataset to Parquet and loads into DuckDB analytical engine.
    """
    ensure_data_dir()
    validated_df = validate_raw_schema(df)
    
    # Save to Parquet
    validated_df.to_parquet(parquet_path, index=False)
    
    # Ingest into DuckDB warehouse
    con = duckdb.connect(DUCKDB_PATH)
    con.execute("CREATE OR REPLACE TABLE raw_transactions AS SELECT * FROM read_parquet(?)", [parquet_path])
    count = con.execute("SELECT count(*) FROM raw_transactions").fetchone()[0]
    con.close()
    
    print(f"[Data Ingestion] Successfully ingested {count} transactions into DuckDB ({DUCKDB_PATH}) & Parquet ({parquet_path})")
    return parquet_path


def load_raw_from_duckdb() -> pd.DataFrame:
    """
    Loads raw transaction data from DuckDB warehouse.
    """
    con = duckdb.connect(DUCKDB_PATH)
    df = con.execute("SELECT * FROM raw_transactions ORDER BY timestamp ASC").df()
    con.close()
    return df
