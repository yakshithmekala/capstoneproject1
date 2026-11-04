import pytest
from src.data_pipeline.pipeline import run_data_pipeline
from src.models.train_model import train_fraud_detection_model
from src.models.evaluate import evaluate_fraud_model


@pytest.fixture(scope="session", autouse=True)
def setup_test_pipeline():
    """
    Session-level fixture that executes ONCE before any test in the test suite runs.
    Generates test DuckDB warehouse, Parquet feature store, baseline distribution,
    and trained model artifacts so all API and unit tests have data available.
    """
    feat_df = run_data_pipeline(num_records=500, fraud_rate=0.04, random_seed=42)
    model, scaler, X_test, y_test, metadata = train_fraud_detection_model(feat_df)
    evaluate_fraud_model(
        model, scaler, X_test, y_test,
        threshold=metadata['best_threshold'],
        full_df=feat_df
    )
    return feat_df
