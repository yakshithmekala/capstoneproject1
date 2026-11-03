from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    user_id: str = Field(default="USER_0042", json_schema_extra={"example": "USER_0042"})
    amount: float = Field(default=25000.0, json_schema_extra={"example": 485000.0})
    merchant_category: str = Field(default="upi_p2p", json_schema_extra={"example": "crypto_exchange"})
    device_type: str = Field(default="mobile_app", json_schema_extra={"example": "web_browser"})
    location_lat: float = Field(default=19.0760, json_schema_extra={"example": 28.6139})
    location_long: float = Field(default=72.8777, json_schema_extra={"example": 77.2090})
    card_present: int = Field(default=0, json_schema_extra={"example": 0})
    time_since_prev_txn_sec: Optional[float] = Field(default=3600.0, json_schema_extra={"example": 15.0})
    distance_prev_txn_km: Optional[float] = Field(default=2.5, json_schema_extra={"example": 1450.0})


class TransactionItem(BaseModel):
    transaction_id: str
    timestamp: str
    user_id: str
    amount: float
    merchant_category: str
    device_type: str
    location_lat: float
    location_long: float
    card_present: int
    is_fraud: int
    fraud_probability: float
    risk_level: str
    status: str = "PENDING"


class ActionInput(BaseModel):
    action: str = Field(..., json_schema_extra={"example": "BLOCK"}) # APPROVE, BLOCK, ESCALATE
    notes: Optional[str] = None


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: Any
    impact: str


class ExplainOutput(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    recommendation: str
    summary: str
    key_reasons: List[str]
    top_risk_factors: List[RiskFactor]
    top_mitigating_factors: List[RiskFactor]
    all_attributions: List[RiskFactor]


class PredictOutput(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud_flagged: bool
    risk_level: str
    latency_ms: float
    explanation: ExplainOutput


class DriftFeatureReport(BaseModel):
    feature: str
    psi: float
    drift_status: str  # STABLE, MODERATE DRIFT, CRITICAL DRIFT


class DriftReportOutput(BaseModel):
    overall_status: str
    baseline_samples: int
    evaluated_samples: int
    feature_drift: List[DriftFeatureReport]


class FairnessAuditItem(BaseModel):
    attribute: str
    group: str
    count: int
    fpr: float
    recall: float


class PipelineMetrics(BaseModel):
    total_transactions_ingested: int
    fraud_count: int
    fraud_rate_pct: float
    model_performance: Dict[str, Any]
    feature_importances: List[Dict[str, Any]]
    cost_metrics: Optional[Dict[str, Any]] = None
    fairness_audit: Optional[List[FairnessAuditItem]] = None
