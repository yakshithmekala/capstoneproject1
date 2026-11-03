import numpy as np
import pandas as pd
import shap

from src.models.train_model import FEATURE_COLUMNS, load_trained_model

_EXPLAINER_CACHE = None


def get_tree_explainer(model):
    """
    Retrieves or initializes cached SHAP TreeExplainer.
    """
    global _EXPLAINER_CACHE
    if _EXPLAINER_CACHE is None:
        _EXPLAINER_CACHE = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE


def explain_single_transaction(feature_row: pd.DataFrame, model, scaler):
    """
    Calculates SHAP feature attributions for a single transaction.
    Returns structured explanation data formatted for API & UI.
    """
    explainer = get_tree_explainer(model)
    
    # Ensure feature order matches FEATURE_COLUMNS
    X_input = feature_row[FEATURE_COLUMNS].copy()
    X_scaled = pd.DataFrame(scaler.transform(X_input), columns=FEATURE_COLUMNS)
    
    shap_values = explainer.shap_values(X_scaled)
    
    # TreeExplainer for binary classification might return a list [shap_neg, shap_pos] or single array
    if isinstance(shap_values, list):
        vals = shap_values[1][0] # Positive class (fraud) SHAP values
        expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    else:
        vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        expected_val = explainer.expected_value
        
    prob = float(model.predict_proba(X_scaled)[:, 1][0])
    
    feature_attributions = []
    for col, shap_val in zip(FEATURE_COLUMNS, vals):
        actual_val = X_input[col].values[0]
        if isinstance(actual_val, (np.floating, float)):
            actual_val = round(float(actual_val), 4)
        elif isinstance(actual_val, (np.integer, int)):
            actual_val = int(actual_val)
            
        feature_attributions.append({
            'feature': col,
            'shap_value': round(float(shap_val), 4),
            'feature_value': actual_val,
            'impact': 'increases_risk' if shap_val > 0 else 'decreases_risk'
        })
        
    # Sort attributions by absolute SHAP impact
    feature_attributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    
    # Filter top positive drivers (risk factors) and top negative mitigators
    risk_factors = [f for f in feature_attributions if f['shap_value'] > 0]
    mitigating_factors = [f for f in feature_attributions if f['shap_value'] < 0]
    
    return {
        'fraud_probability': round(prob, 4),
        'base_value': round(float(expected_val), 4),
        'top_risk_factors': risk_factors[:5],
        'top_mitigating_factors': mitigating_factors[:5],
        'all_attributions': feature_attributions
    }
