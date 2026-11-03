import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

from src.models.train_model import FEATURE_COLUMNS, MODEL_DIR
from src.evaluation.cost_metrics import total_cost, savings, best_cost_threshold
from src.monitoring.drift_fairness import group_fpr_recall


def evaluate_fraud_model(
    model, scaler, X_test: pd.DataFrame, y_test: pd.Series,
    threshold: float = 0.5, full_df: pd.DataFrame = None, c_f: float = 5.0
):
    """
    Evaluates fraud classification performance across multiple dimensions:
    1. Statistical Classification Metrics (ROC-AUC, PR-AUC, F1, Precision, Recall)
    2. Instance-Dependent Cost Metrics (Baesens et al. 2022: Total Loss, Net Savings, Cost-Optimal Threshold)
    3. Fairness & Bias Audit (FPR and Recall by category and channel)
    """
    print("=" * 60)
    print("📊 [Model Evaluation] Calculating Model Metrics, Cost Savings & Fairness...")
    print("=" * 60)
    
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index)
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)
    
    y_true = np.asarray(y_test, dtype=int)
    amounts = np.asarray(X_test['amount'], dtype=float)
    
    # 1. Statistical ML Metrics
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(r_curve, p_curve)
    
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # 2. Instance-Dependent Cost Metrics (Baesens et al. 2022)
    # A missed fraud costs its amount A_i; each investigation alert costs administrative fee c_f
    loss_without_model = float(np.sum(y_true * amounts))
    model_cost = total_cost(y_true, y_preds, amounts, c_f=c_f)
    net_savings_val = max(0.0, loss_without_model - model_cost)
    savings_share = savings(y_true, y_preds, amounts, c_f=c_f)
    
    best_cost_th, best_cost_sav = best_cost_threshold(y_true, y_probs, amounts, c_f=c_f)
    
    # 3. Fairness Audit (Group FPR & Recall)
    fairness_records = []
    if full_df is not None and 'merchant_category' in full_df.columns:
        test_rows = full_df.loc[X_test.index]
        cat_fairness = group_fpr_recall(y_true, y_preds, test_rows['merchant_category'])
        for _, r in cat_fairness.iterrows():
            fairness_records.append({
                'attribute': 'merchant_category',
                'group': str(r['group']),
                'count': int(r['n']),
                'fpr': round(float(r['fpr']), 4),
                'recall': round(float(r['recall']), 4)
            })
        if 'device_type' in full_df.columns:
            dev_fairness = group_fpr_recall(y_true, y_preds, test_rows['device_type'])
            for _, r in dev_fairness.iterrows():
                fairness_records.append({
                    'attribute': 'device_type',
                    'group': str(r['group']),
                    'count': int(r['n']),
                    'fpr': round(float(r['fpr']), 4),
                    'recall': round(float(r['recall']), 4)
                })
    else:
        # Fallback fairness groups based on high risk category flag
        groups = np.where(X_test['is_high_risk_category'] == 1, 'High-Risk Categories', 'Standard Categories')
        grp_fair = group_fpr_recall(y_true, y_preds, groups)
        for _, r in grp_fair.iterrows():
            fairness_records.append({
                'attribute': 'risk_tier',
                'group': str(r['group']),
                'count': int(r['n']),
                'fpr': round(float(r['fpr']), 4),
                'recall': round(float(r['recall']), 4)
            })
    
    # 4. Feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    metrics = {
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(roc_auc), 4),
        'pr_auc': round(float(pr_auc), 4),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'decision_threshold': float(threshold),
        'cost_metrics': {
            'baseline_loss_without_model': round(loss_without_model, 2),
            'total_cost_with_model': round(model_cost, 2),
            'net_savings_amount': round(net_savings_val, 2),
            'savings_ratio_pct': round(float(savings_share * 100), 2),
            'cost_optimal_threshold': round(float(best_cost_th), 4),
            'cost_optimal_savings_pct': round(float(best_cost_sav * 100), 2),
            'alert_fee_c_f': c_f
        },
        'fairness_audit': fairness_records,
        'feature_importances': importance_df.to_dict(orient='records')
    }
    
    metrics_path = os.path.join(MODEL_DIR, 'eval_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Metrics: ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}")
    print(f"Cost Savings: Avoided {savings_share*100:.2f}% of fraud losses (Net Savings: ₹{net_savings_val:,.2f})")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Saved evaluation report to {metrics_path}")
    return metrics
