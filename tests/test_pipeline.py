import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from src.data_pipeline.generator import generate_synthetic_transactions
from src.data_pipeline.ingestion import validate_raw_schema
from src.data_pipeline.feature_engineering import compute_vectorized_features, haversine_distance
from src.data_pipeline.pipeline import run_data_pipeline
from src.models.train_model import train_fraud_detection_model, FEATURE_COLUMNS
from src.models.evaluate import evaluate_fraud_model
from src.explainability.explainer import explain_single_transaction
from src.explainability.nl_explainer import generate_natural_language_explanation
from src.api.main import app


@pytest.fixture(scope="module", autouse=True)
def setup_pipeline():
    """
    Run pipeline once to generate test DuckDB warehouse & trained model artifacts.
    """
    feat_df = run_data_pipeline(num_records=500, fraud_rate=0.04, random_seed=42)
    model, scaler, X_test, y_test, metadata = train_fraud_detection_model(feat_df)
    evaluate_fraud_model(model, scaler, X_test, y_test, threshold=metadata['best_threshold'])


def test_generator_schema():
    df = generate_synthetic_transactions(num_records=100, random_seed=42)
    assert len(df) == 100
    assert 'transaction_id' in df.columns
    assert 'amount' in df.columns
    assert 'is_fraud' in df.columns
    assert df['is_fraud'].isin([0, 1]).all()


def test_haversine_distance():
    dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    assert 3900 <= dist <= 4000


def test_feature_engineering():
    raw_df = generate_synthetic_transactions(num_records=100, random_seed=42)
    feat_df = compute_vectorized_features(raw_df)
    
    for col in FEATURE_COLUMNS:
        assert col in feat_df.columns
        assert not feat_df[col].isnull().any()
        
    assert 'amount_ratio_to_user_avg' in feat_df.columns
    assert 'distance_prev_txn_km' in feat_df.columns
    assert 'txn_count_1h' in feat_df.columns


def test_model_training_and_shap():
    raw_df = generate_synthetic_transactions(num_records=200, random_seed=42)
    feat_df = compute_vectorized_features(raw_df)
    
    model, scaler, X_test, y_test, metadata = train_fraud_detection_model(feat_df)
    assert model is not None
    assert metadata['best_threshold'] > 0
    
    row = feat_df.iloc[[0]]
    shap_res = explain_single_transaction(row, model, scaler)
    assert 'fraud_probability' in shap_res
    assert 'top_risk_factors' in shap_res
    
    nl_res = generate_natural_language_explanation(shap_res)
    assert 'risk_level' in nl_res
    assert 'summary' in nl_res


def test_fastapi_endpoints():
    client = TestClient(app)
    
    # 1. Test metrics endpoint
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "total_transactions_ingested" in data
    assert data["total_transactions_ingested"] >= 500
    
    # 2. Test transactions list endpoint
    tx_res = client.get("/api/transactions?limit=10")
    assert tx_res.status_code == 200
    tx_items = tx_res.json()
    assert len(tx_items) > 0
    first_id = tx_items[0]["transaction_id"]
    
    # 3. Test explain transaction by ID endpoint
    exp_res = client.get(f"/api/explain/{first_id}")
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert "summary" in exp_data
    assert "top_risk_factors" in exp_data
    
    # 4. Test real-time predict endpoint
    payload = {
        "user_id": "USER_TEST",
        "amount": 3500.0,
        "merchant_category": "crypto_exchange",
        "device_type": "web_browser",
        "location_lat": 48.8566,
        "location_long": 2.3522,
        "card_present": 0,
        "time_since_prev_txn_sec": 45.0,
        "distance_prev_txn_km": 1500.0
    }
    pred_res = client.post("/api/predict", json=payload)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()
    assert "fraud_probability" in pred_data
    assert "explanation" in pred_data
    assert pred_data["latency_ms"] < 500
