import os
import time
import json
import uuid
import duckdb
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas import (
    TransactionInput, TransactionItem, PredictOutput,
    ExplainOutput, PipelineMetrics, ActionInput,
    DriftReportOutput, DriftFeatureReport, FairnessAuditItem
)
from src.data_pipeline.ingestion import DUCKDB_PATH, DATA_DIR
from src.data_pipeline.feature_engineering import compute_vectorized_features
from src.models.train_model import load_trained_model, FEATURE_COLUMNS, MODEL_DIR, TRAIN_BASELINE_PATH
from src.explainability.explainer import explain_single_transaction
from src.explainability.nl_explainer import generate_natural_language_explanation
from src.monitoring.drift_fairness import feature_drift_report

router = APIRouter(prefix="/api")

# Global state for analyst actions (in-memory overlay on DuckDB)
ANALYST_ACTIONS = {}


def get_feature_store_df() -> pd.DataFrame:
    if not os.path.exists(DUCKDB_PATH):
        raise HTTPException(status_code=500, detail="Data warehouse DuckDB file not found. Run pipeline first.")
    con = duckdb.connect(DUCKDB_PATH)
    df = con.execute("SELECT * FROM feature_store ORDER BY timestamp DESC").df()
    con.close()
    return df


@router.get("/transactions", response_model=List[TransactionItem])
def get_transactions(
    limit: int = Query(50, ge=1, le=500),
    risk_filter: Optional[str] = Query(None, description="Filter by risk level: CRITICAL, HIGH, ELEVATED, LOW")
):
    """
    Returns latest ingested transactions annotated with ML model risk scores.
    """
    df = get_feature_store_df()
    model, scaler, metadata = load_trained_model()
    
    # Run batch prediction for display
    X = df[FEATURE_COLUMNS].copy()
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLUMNS)
    probs = model.predict_proba(X_scaled)[:, 1]
    df['fraud_probability'] = np.round(probs, 4)
    
    items = []
    for idx, row in df.iterrows():
        prob = float(row['fraud_probability'])
        if prob >= 0.75:
            risk = 'CRITICAL'
        elif prob >= 0.40:
            risk = 'HIGH'
        elif prob >= 0.20:
            risk = 'ELEVATED'
        else:
            risk = 'LOW'
            
        if risk_filter and risk != risk_filter.upper():
            continue
            
        tx_id = str(row['transaction_id'])
        status = ANALYST_ACTIONS.get(tx_id, "PENDING")
        
        items.append({
            'transaction_id': tx_id,
            'timestamp': str(row['timestamp']),
            'user_id': str(row['user_id']),
            'amount': float(row['amount']),
            'merchant_category': str(row['merchant_category']),
            'device_type': str(row['device_type']),
            'location_lat': float(row['location_lat']),
            'location_long': float(row['location_long']),
            'card_present': int(row['card_present']),
            'is_fraud': int(row['is_fraud']),
            'fraud_probability': prob,
            'risk_level': risk,
            'status': status
        })
        if len(items) >= limit:
            break
            
    return items


@router.get("/explain/{transaction_id}", response_model=ExplainOutput)
def explain_transaction_by_id(transaction_id: str):
    """
    Generates detailed SHAP feature attributions and Natural Language explanation
    for a specific transaction ID.
    """
    df = get_feature_store_df()
    match = df[df['transaction_id'] == transaction_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Transaction ID {transaction_id} not found.")
        
    model, scaler, metadata = load_trained_model()
    row = match.iloc[[0]]
    
    shap_res = explain_single_transaction(row, model, scaler)
    nl_res = generate_natural_language_explanation(shap_res, row.to_dict(orient='records')[0])
    
    return {
        'transaction_id': transaction_id,
        'fraud_probability': shap_res['fraud_probability'],
        'risk_level': nl_res['risk_level'],
        'recommendation': nl_res['recommendation'],
        'summary': nl_res['summary'],
        'key_reasons': nl_res['key_reasons'],
        'top_risk_factors': shap_res['top_risk_factors'],
        'top_mitigating_factors': shap_res['top_mitigating_factors'],
        'all_attributions': shap_res['all_attributions']
    }


@router.post("/predict", response_model=PredictOutput)
def predict_realtime_transaction(tx_input: TransactionInput):
    """
    Real-time fraud scoring API endpoint with sub-50ms latency timer and XAI explanation.
    """
    t0 = time.time()
    model, scaler, metadata = load_trained_model()
    tx_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"
    
    # Build single-record DataFrame for pipeline transformation
    raw_dict = {
        'transaction_id': tx_id,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': tx_input.user_id,
        'amount': tx_input.amount,
        'merchant_category': tx_input.merchant_category,
        'device_type': tx_input.device_type,
        'location_lat': tx_input.location_lat,
        'location_long': tx_input.location_long,
        'ip_address': '192.168.1.100',
        'card_present': tx_input.card_present,
        'is_fraud': 0
    }
    raw_df = pd.DataFrame([raw_dict])
    
    # Compute engineered features
    feat_df = compute_vectorized_features(raw_df)
    
    # Override time_since_prev and distance if provided by caller
    if tx_input.time_since_prev_txn_sec is not None:
        feat_df['time_since_prev_txn_sec'] = tx_input.time_since_prev_txn_sec
    if tx_input.distance_prev_txn_km is not None:
        feat_df['distance_prev_txn_km'] = tx_input.distance_prev_txn_km
        hours = max(tx_input.time_since_prev_txn_sec / 3600.0, 1.0 / 3600.0)
        feat_df['travel_speed_kmh'] = round(tx_input.distance_prev_txn_km / hours, 2)
        
    shap_res = explain_single_transaction(feat_df, model, scaler)
    nl_res = generate_natural_language_explanation(shap_res, raw_dict)
    
    prob = shap_res['fraud_probability']
    threshold = metadata.get('best_threshold', 0.5)
    is_flagged = bool(prob >= threshold)
    
    latency = round((time.time() - t0) * 1000, 2)
    
    explanation_obj = {
        'transaction_id': tx_id,
        'fraud_probability': prob,
        'risk_level': nl_res['risk_level'],
        'recommendation': nl_res['recommendation'],
        'summary': nl_res['summary'],
        'key_reasons': nl_res['key_reasons'],
        'top_risk_factors': shap_res['top_risk_factors'],
        'top_mitigating_factors': shap_res['top_mitigating_factors'],
        'all_attributions': shap_res['all_attributions']
    }
    
    return {
        'transaction_id': tx_id,
        'fraud_probability': prob,
        'is_fraud_flagged': is_flagged,
        'risk_level': nl_res['risk_level'],
        'latency_ms': latency,
        'explanation': explanation_obj
    }


@router.post("/action/{transaction_id}")
def record_analyst_action(transaction_id: str, payload: ActionInput):
    """
    Records an analyst action (APPROVE, BLOCK, ESCALATE) for audit trail.
    """
    ANALYST_ACTIONS[transaction_id] = payload.action.upper()
    return {
        'status': 'success',
        'transaction_id': transaction_id,
        'action': payload.action.upper(),
        'notes': payload.notes
    }


@router.get("/metrics", response_model=PipelineMetrics)
def get_pipeline_metrics():
    """
    Returns pipeline throughput, DuckDB storage stats, model evaluation metrics,
    Baesens et al. (2022) cost savings, and fairness audit table.
    """
    df = get_feature_store_df()
    total_tx = len(df)
    fraud_cnt = int(df['is_fraud'].sum())
    fraud_rate = round(float(fraud_cnt / max(1, total_tx) * 100), 2)
    
    eval_metrics = {}
    metrics_path = os.path.join(MODEL_DIR, 'eval_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            eval_metrics = json.load(f)
            
    importances = eval_metrics.get('feature_importances', [])
    cost_metrics = eval_metrics.get('cost_metrics', None)
    fairness_audit = eval_metrics.get('fairness_audit', None)
    
    return {
        'total_transactions_ingested': total_tx,
        'fraud_count': fraud_cnt,
        'fraud_rate_pct': fraud_rate,
        'model_performance': eval_metrics,
        'feature_importances': importances,
        'cost_metrics': cost_metrics,
        'fairness_audit': fairness_audit
    }


@router.get("/drift", response_model=DriftReportOutput)
def get_feature_drift():
    """
    Population Stability Index (PSI) drift monitoring endpoint.
    Compares baseline training feature distribution with live feature store transactions.
    PSI < 0.1: STABLE, 0.1-0.25: MODERATE DRIFT, >0.25: CRITICAL DRIFT.
    """
    df = get_feature_store_df()
    
    if os.path.exists(TRAIN_BASELINE_PATH):
        train_df = pd.read_parquet(TRAIN_BASELINE_PATH)
    else:
        # Fallback split
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[split_idx:]
        
    live_df = df.iloc[:min(2000, len(df))]
    
    cols_to_check = [
        'amount', 'robust_monetary_zscore', 'txn_count_1h',
        'spending_sum_24h', 'distance_prev_txn_km', 'travel_speed_kmh',
        'recency_decay_score', 'isolation_forest_anomaly_score'
    ]
    cols_to_check = [c for c in cols_to_check if c in train_df.columns and c in live_df.columns]
    
    report_df = feature_drift_report(train_df, live_df, cols_to_check)
    
    drift_items = []
    has_critical = False
    has_moderate = False
    
    for _, row in report_df.iterrows():
        psi_val = float(row['psi'])
        if psi_val > 0.25:
            status = "CRITICAL DRIFT"
            has_critical = True
        elif psi_val >= 0.10:
            status = "MODERATE DRIFT"
            has_moderate = True
        else:
            status = "STABLE"
            
        drift_items.append({
            'feature': str(row['feature']),
            'psi': round(psi_val, 4),
            'drift_status': status
        })
        
    overall = "CRITICAL DRIFT DETECTED" if has_critical else ("MODERATE DRIFT DETECTED" if has_moderate else "ALL FEATURES STABLE")
    
    return {
        'overall_status': overall,
        'baseline_samples': len(train_df),
        'evaluated_samples': len(live_df),
        'feature_drift': drift_items
    }


@router.get("/fairness", response_model=List[FairnessAuditItem])
def get_fairness_audit():
    """
    Fairness and demographic/channel parity audit endpoint.
    Returns false positive rates (FPR) and recall across merchant categories and device channels.
    """
    metrics_path = os.path.join(MODEL_DIR, 'eval_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
            return data.get('fairness_audit', [])
    return []


@router.get("/ablation")
def get_ablation_results():
    """
    Returns benchmark table from the ablation study comparing
    Logistic Regression, Decision Tree, and LightGBM across feature sets & imbalance methods.
    """
    res_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', 'ablation_results.csv')
    if os.path.exists(res_path):
        df = pd.read_csv(res_path)
        return {
            'status': 'available',
            'total_experiments': len(df),
            'results': df.to_dict(orient='records')
        }
    return {
        'status': 'not_run',
        'message': 'Run python -m src.experiments.ablation to generate ablation benchmark.'
    }
