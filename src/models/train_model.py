import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from src.data_pipeline.ingestion import DATA_DIR
from src.evaluation.cost_metrics import best_cost_threshold, savings

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models_artifacts')
MODEL_PATH = os.path.join(MODEL_DIR, 'fraud_lightgbm_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'feature_scaler.joblib')
META_PATH = os.path.join(MODEL_DIR, 'model_metadata.json')
TRAIN_BASELINE_PATH = os.path.join(MODEL_DIR, 'train_baseline.parquet')

FEATURE_COLUMNS = [
    'amount',
    'log_amount',
    'location_lat',
    'location_long',
    'card_present',
    'amount_ratio_to_user_avg',
    'amount_z_score',
    'robust_monetary_zscore',
    'time_since_prev_txn_sec',
    'recency_decay_score',
    'distance_prev_txn_km',
    'travel_speed_kmh',
    'txn_count_1h',
    'txn_count_24h',
    'spending_sum_24h',
    'is_high_risk_category',
    'is_night_transaction',
    'merchant_category_freq',
    'device_type_freq',
    'isolation_forest_anomaly_score'
]


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def train_fraud_detection_model(df: pd.DataFrame):
    """
    Trains LightGBM Classifier with cost-weighted loss scaling for financial fraud imbalance.
    Optimizes both F1 threshold and instance-dependent cost-optimal decision threshold (Baesens et al. 2022).
    Exports serialized model, feature scaler, and training baseline distribution for drift tracking.
    """
    ensure_model_dir()
    print("=" * 60)
    print("🤖 [ML Model Training] Training LightGBM Fraud Classifier (Baesens et al. 2022 Data Engineering)...")
    print("=" * 60)
    
    X = df[FEATURE_COLUMNS].copy()
    y = df['is_fraud'].astype(int)
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save training distribution baseline for PSI Drift Monitoring
    X_train.to_parquet(TRAIN_BASELINE_PATH, index=False)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index)
    
    # Calculate scale_pos_weight for severe class imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = float(num_neg / max(1, num_pos))
    
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Optimal threshold search based on F1 score and Cost Savings
    y_probs_test = model.predict_proba(X_test_scaled)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0.0
    
    for th in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_probs_test >= th).astype(int)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test == 0))
        fn = np.sum((y_pred == 0) & (y_test == 1))
        
        prec = tp / max(1, (tp + fp))
        rec = tp / max(1, (tp + fn))
        f1 = 2 * prec * rec / max(1e-5, (prec + rec))
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(th), 2)
            
    # Cost-optimal threshold search
    test_amounts = X_test['amount'].values
    cost_threshold, cost_savings = best_cost_threshold(y_test.values, y_probs_test, test_amounts, c_f=5.0)

    print(f"Optimal Fraud F1 Decision Threshold: {best_threshold} (F1 Score: {best_f1:.4f})")
    print(f"Cost-Optimal Decision Threshold: {cost_threshold:.4f} (Max Savings: {cost_savings*100:.2f}%)")
    
    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    metadata = {
        'feature_columns': FEATURE_COLUMNS,
        'best_threshold': best_threshold,
        'cost_optimal_threshold': round(float(cost_threshold), 4),
        'max_savings_pct': round(float(cost_savings * 100), 2),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'fraud_ratio_train': float(y_train.mean()),
        'scale_pos_weight': scale_pos_weight
    }
    
    with open(META_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"✅ [Model Training Complete] Saved model artifact to {MODEL_PATH}")
    return model, scaler, X_test, y_test, metadata


def load_trained_model():
    """
    Loads saved model, scaler, and metadata.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model artifacts not found. Please run training pipeline first.")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)
        
    return model, scaler, metadata
