def generate_natural_language_explanation(shap_result: dict, raw_row: dict = None) -> dict:
    """
    Translates SHAP attribution scores and feature values into natural language bullet points
    and an executive summary paragraph for fraud investigators in the Indian Financial Sector.
    """
    prob = shap_result['fraud_probability']
    risk_factors = shap_result['top_risk_factors']
    
    if prob >= 0.75:
        risk_level = 'CRITICAL RISK'
        recommendation = 'AUTOMATIC BLOCK & URGENT UPI / BANK FREEZE'
    elif prob >= 0.40:
        risk_level = 'HIGH RISK'
        recommendation = 'HOLD FOR MANUAL REVIEW & OTP / BIOMETRIC VERIFICATION'
    elif prob >= 0.20:
        risk_level = 'ELEVATED RISK'
        recommendation = 'MONITOR ACCOUNT ACTIVITY'
    else:
        risk_level = 'LOW RISK'
        recommendation = 'APPROVE TRANSACTION'
        
    explanations = []
    
    feature_labels = {
        'amount': 'Transaction Amount (₹)',
        'log_amount': 'Log Transformed Amount',
        'amount_ratio_to_user_avg': 'Amount Spike vs Historical Average',
        'amount_z_score': 'Amount Anomaly Z-Score',
        'robust_monetary_zscore': 'Robust Monetary Z-Score (Median & MAD)',
        'recency_decay_score': 'Exponential Recency Decay Score',
        'distance_prev_txn_km': 'Geographic Location Jump Distance',
        'travel_speed_kmh': 'Implied Travel Velocity',
        'time_since_prev_txn_sec': 'Rapid Transaction Window',
        'txn_count_1h': '1-Hour Transaction Count',
        'txn_count_24h': '24-Hour Velocity Count',
        'spending_sum_24h': '24-Hour Total Spend Volume (₹)',
        'is_high_risk_category': 'High-Risk Merchant Category (Crypto/UPI P2P/Jewelry)',
        'is_night_transaction': 'Unusual Nighttime Activity Window',
        'card_present': 'Card / POS Present Status',
        'device_type_freq': 'Uncommon Device Signature',
        'isolation_forest_anomaly_score': 'Isolation Forest Unsupervised Anomaly Score'
    }
    
    for factor in risk_factors:
        feat = factor['feature']
        val = factor['feature_value']
        label = feature_labels.get(feat, feat)
        
        if feat == 'robust_monetary_zscore' and val > 2.5:
            explanations.append(f"Robust Monetary Anomaly (MAD Z-Score: {val}): Transaction amount deviates significantly from customer baseline median.")
        elif feat == 'recency_decay_score' and val > 0.8:
            explanations.append("High Recency Score: Transaction initiated almost immediately following prior user activity.")
        elif feat == 'isolation_forest_anomaly_score' and val > 0.5:
            explanations.append(f"Unsupervised Anomaly Metric (Isolation Forest score: {val}): Multi-dimensional deviation across velocity, amount, and location.")
        elif feat == 'amount_ratio_to_user_avg':
            explanations.append(f"Amount is {val}x higher than the account's 30-day baseline average.")
        elif feat == 'distance_prev_txn_km' and val > 50:
            explanations.append(f"Geographic anomaly detected: Transaction occurred {val} km away from prior active city location.")
        elif feat == 'travel_speed_kmh' and val > 100:
            explanations.append(f"Impossible travel speed detected: Physical location moved at an implied velocity of {val} km/h.")
        elif feat == 'time_since_prev_txn_sec' and val < 60:
            explanations.append(f"Rapid velocity trigger: Executed only {val} seconds after previous transaction.")
        elif feat == 'txn_count_1h' and val > 3:
            explanations.append(f"High 1-hour transaction frequency: {val} transactions attempted within 60 minutes.")
        elif feat == 'is_high_risk_category' and val == 1:
            explanations.append("Targeted high-risk sector (e.g. Crypto Exchange / UPI P2P / Jewelry & Gold).")
        elif feat == 'is_night_transaction' and val == 1:
            explanations.append("Transaction initiated during off-hours / late night timeframe.")
        elif feat == 'amount':
            explanations.append(f"Large absolute transaction value (₹{val:,.2f}).")
        else:
            explanations.append(f"{label} anomalous reading: {val}.")
            
    if not explanations:
        explanations.append("All key risk indicators remain within normal operational baselines.")
        
    summary_paragraph = (
        f"Transaction evaluated with a {prob*100:.1f}% risk probability ({risk_level}). "
        f"Recommended Action: {recommendation}. "
        f"Primary risk driver: {explanations[0]}"
    )
    
    return {
        'risk_level': risk_level,
        'risk_score_pct': round(prob * 100, 1),
        'recommendation': recommendation,
        'summary': summary_paragraph,
        'key_reasons': explanations
    }
