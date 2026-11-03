import time
from src.data_pipeline.generator import generate_synthetic_transactions
from src.data_pipeline.ingestion import save_raw_transactions
from src.data_pipeline.feature_engineering import build_and_save_feature_store


def run_data_pipeline(num_records: int = 10000, fraud_rate: float = 0.035, random_seed: int = 42):
    """
    Runs full Data Engineering ETL pipeline:
    1. Generates synthetic raw financial transactions.
    2. Validates schema & ingests into raw DuckDB table & Parquet.
    3. Computes high-dimensional velocity, distance, and ratio features.
    4. Persists into DuckDB Feature Store.
    """
    t0 = time.time()
    print("=" * 60)
    print(f"🚀 [Pipeline] Starting Data Engineering Pipeline ({num_records} records)...")
    print("=" * 60)
    
    # Step 1: Synthetic Generation
    raw_df = generate_synthetic_transactions(num_records=num_records, fraud_rate=fraud_rate, random_seed=random_seed)
    
    # Step 2: Ingestion & DuckDB Raw persistence
    parquet_path = save_raw_transactions(raw_df)
    
    # Step 3: Feature Engineering & Feature Store persistence
    feature_df = build_and_save_feature_store(raw_df)
    
    elapsed = round(time.time() - t0, 2)
    print("=" * 60)
    print(f"✅ [Pipeline Complete] Finished in {elapsed}s. Features engineered: {feature_df.shape[1]}")
    print("=" * 60)
    return feature_df


if __name__ == '__main__':
    run_data_pipeline(num_records=5000)
