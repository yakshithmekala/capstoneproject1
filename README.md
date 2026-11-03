# Sentinel AI — Scalable Fraud Detection System Using Data Engineering Pipelines and Explainable AI (XAI)

[![CI/CD Pipeline](https://github.com/yakshithmekala/capstoneproject/actions/workflows/ci.yml/badge.svg)](https://github.com/yakshithmekala/capstoneproject/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory%20Warehouse-FFF000.svg)](https://duckdb.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer%20XAI-ff69b4.svg)](https://shap.readthedocs.io/)

An enterprise-grade, end-to-end Financial Fraud Detection and Risk Intelligence System built with high-performance Data Engineering ETL pipelines, vectorized **DuckDB** analytical warehouse, **LightGBM** cost-weighted classification, **Baesens et al. (2022)** instance-dependent financial cost metrics, continuous **Population Stability Index (PSI)** drift detection, **algorithmic fairness auditing**, and **SHAP Explainable AI** with automated natural language compliance investigator reports.

---

## 🌟 Key System Architecture & Core Innovations

```
                                    SENTINEL AI ARCHITECTURE
                                    
  [ Raw Stream / Ingestion ]
             │
             ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. Vectorized Analytical Data Warehouse (DuckDB & Parquet Storage)           │
  │    - Sub-second ingestion of raw financial streams                          │
  │    - Strict schema validation & type coercion                                │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 2. Feature Engineering Store (Baesens, Hoeppner & Verdonck 2022)            │
  │    - Robust Monetary Z-Scores (Median / MAD) to prevent masking & swamping   │
  │    - Exponential Recency Decay: x_recency = exp(-γ · Δt)                     │
  │    - Haversine Spatial Distance & Implied Travel Velocity (km/h)             │
  │    - O(N log N) Fast Sliding Window Aggregations (1h & 24h RFM)              │
  │    - Unsupervised Anomaly Scoring (Isolation Forest Multi-dimensional Score) │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 3. Cost-Weighted Classifier & Instance-Dependent Optimization               │
  │    - LightGBM with scale_pos_weight for severe class imbalance (~3% fraud)   │
  │    - Financial Cost Metrics: Missed fraud = amount A_i, Alert = fixed fee c_f│
  │    - Cost-Optimal Decision Threshold Maximizing Financial Loss Avoided       │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                     ┌───────────────────┴───────────────────┐
                     ▼                                       ▼
  ┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
  │ 4. Explainable AI Engine (SHAP)      │  │ 5. Continuous Governance & Auditing  │
  │    - Exact TreeExplainer attribution │  │    - Population Stability Index (PSI)│
  │    - Risk Drivers vs Mitigators      │  │    - Per-Group FPR & Recall Fairness │
  │    - Natural Language Investigator   │  │    - Channel Disparity Verification  │
  │      Narrative Translator            │  │    - Multi-Model Ablation Study      │
  └──────────────────┬───────────────────┘  └──────────────────┬───────────────────┘
                     │                                         │
                     └───────────────────┬─────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 6. Real-Time FastAPI Backend Service (Sub-30ms Scoring API)                 │
  │    - /api/transactions  - /api/predict  - /api/explain/{id}                  │
  │    - /api/metrics       - /api/drift    - /api/fairness    - /api/ablation  │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 7. Modern Glassmorphic Web Dashboard                                        │
  │    - Live Stream Table & Attack Simulator with Presets (Account Takeover)    │
  │    - Interactive SHAP Waterfall Chart & Investigator Decision Desk           │
  │    - Live PSI Feature Drift Monitor & Cohort Parity Matrix                   │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. ⚙️ High-Performance Data Engineering Pipeline (`src/data_pipeline`)
- **Fast Vectorized Sliding Windows**: Replaces slow iterative loops with an $O(N \log N)$ `searchsorted` algorithm over cumulative amounts, reducing ETL execution from minutes to under 1 second.
- **DuckDB Vectorized SQL Engine**: Embeds columnar in-memory database queries and exports compressed **Parquet** feature stores with zero external database dependencies.
- **Advanced Behavioral Indicators**:
  - **Robust Monetary Z-Score**: $z_r = \frac{x - \text{median}}{1.4826 \cdot \text{MAD}}$, preventing outlier masking and swamping.
  - **Exponential Recency**: $x_{\text{recency}} = \exp(-\gamma \cdot \Delta t)$, modeling transaction burstiness.
  - **Spatial Travel Speed**: Haversine distance in km divided by elapsed hours to catch physically impossible travel anomalies.

### 2. 💰 Financial Cost Metrics & Cost-Optimal Threshold (`src/evaluation/cost_metrics.py`)
Standard ML metrics (F1, Accuracy) treat a ₹50 false alarm the same as a ₹5,00,000 missed heist. Following **Baesens et al. (2022)** and **Bahnsen et al. (2016)**:
- **Total Cost**: $\text{Cost} = \sum [ y_i (1 - \hat{y}_i) A_i + \hat{y}_i c_f ]$ where $A_i$ is transaction amount and $c_f$ is administrative investigation fee.
- **Avoided Loss Savings**: $\text{Savings} = \frac{\sum y_i A_i - \text{Cost}}{\sum y_i A_i}$
- **Cost-Optimal Threshold**: Dynamically tunes the operational alert cutoff on validation data to maximize financial ROI.

### 3. 🔍 Explainable AI (XAI) & Compliance Narrator (`src/explainability`)
- **SHAP TreeExplainer**: Computes exact local Shapley values decomposing model output into positive risk factors (red) and mitigating factors (green).
- **Natural Language Narrative Translator**: Generates human-readable compliance summaries for financial fraud investigators:
  > *"Transaction flagged with 96.5% risk probability (CRITICAL RISK). Amount (₹4,85,000) is 12.5x higher than account 30-day baseline, and geographical jump of 1,450 km occurred within 15 seconds of previous swipe."*

### 4. 📈 Continuous Drift Detection & Fairness Audit (`src/monitoring/drift_fairness.py`)
- **Population Stability Index (PSI)**: Continuously compares feature distributions between the baseline training distribution and live incoming stream.
  - $\text{PSI} < 0.10$: Stable
  - $0.10 \le \text{PSI} \le 0.25$: Moderate Drift
  - $\text{PSI} > 0.25$: Critical Drift (Triggers automated retraining alert)
- **Fairness & Disparity Audit**: Calculates per-cohort False Positive Rate (FPR) and Recall across merchant categories and device channels to guarantee equitable decisioning without algorithmic bias.

### 5. 🔬 Multi-Model Ablation Study (`src/experiments/ablation.py`)
- Rigorous chronological 60/15/25 train/validation/test split benchmarking **Logistic Regression**, **Decision Trees**, and **LightGBM** across:
  - Feature sets: `raw` $\rightarrow$ `+velocity` $\rightarrow$ `+velocity+spatial` $\rightarrow$ `+rfm_all` $\rightarrow$ `+anomaly_scores`
  - Imbalance strategies: `none`, `class_weight`, `smote`, `undersample`
  - Generates verified experimental tables (`results/ablation_results.csv`) for academic papers and enterprise validation.

---

## 📁 Repository Structure

```
sentinel_upgrades/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated GitHub Actions test & ablation CI
├── data/                          # DuckDB analytical warehouse & Parquet store
├── docs/                          # Static GitHub Pages deployment of Dashboard
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── models_artifacts/              # Serialized LightGBM model, scaler, baseline & metrics
│   ├── eval_metrics.json
│   ├── feature_scaler.joblib
│   ├── fraud_lightgbm_model.joblib
│   ├── model_metadata.json
│   └── train_baseline.parquet
├── results/                       # Ablation experimental benchmark exports
│   └── ablation_results.csv
├── src/
│   ├── api/                       # Sub-30ms FastAPI REST API & Schemas
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── dashboard/                 # Executive Glassmorphic Web Dashboard
│   │   ├── app.js
│   │   ├── index.html
│   │   └── styles.css
│   ├── data_pipeline/             # DuckDB ETL, synthetic generator & feature store
│   │   ├── feature_engineering.py
│   │   ├── generator.py
│   │   ├── ingestion.py
│   │   └── pipeline.py
│   ├── evaluation/                # Baesens et al. (2022) cost metrics
│   │   └── cost_metrics.py
│   ├── experiments/               # Multi-model ablation benchmark runner
│   │   └── ablation.py
│   ├── explainability/            # SHAP TreeExplainer & NL Translator
│   │   ├── explainer.py
│   │   └── nl_explainer.py
│   ├── features/                  # Modular advanced feature transforms
│   │   └── advanced_features.py
│   ├── models/                    # Model training & comprehensive evaluation
│   │   ├── evaluate.py
│   │   └── train_model.py
│   └── monitoring/                # PSI drift & cohort fairness auditing
│       └── drift_fairness.py
├── tests/                         # Complete Automated Pytest Suite
│   ├── test_api_extensions.py     # Drift, fairness & ablation endpoint tests
│   ├── test_pipeline.py           # Pipeline, feature engineering, model & API tests
│   └── test_upgrades.py           # Cost metrics, PSI, robust z, haversine tests
├── Dockerfile                     # Multi-stage production container
├── README.md                      # Comprehensive System Documentation
├── requirements.txt               # Unified project dependencies
└── run_pipeline.py                # Single-command end-to-end launcher
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation
Ensure you have Python 3.10 or 3.11 installed:

```bash
git clone https://github.com/yakshithmekala/capstoneproject.git
cd capstoneproject
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline & Launch Web Dashboard
Execute the unified pipeline with a single command:

```bash
python run_pipeline.py
```

This single command automatically:
1. Generates 10,000 realistic transactions with behavioral user profiles and fraud anomalies.
2. Runs the DuckDB analytical pipeline and builds the vectorized Parquet feature store.
3. Trains the LightGBM classifier with class weighting and optimizes decision thresholds.
4. Calculates instance-dependent cost savings and fairness audits.
5. Launches the **Executive Dashboard UI** at **`http://localhost:8000`** and Swagger API docs at **`http://localhost:8000/docs`**.

#### CLI Options:
```bash
# Run with custom record count and fraud rate
python run_pipeline.py --records 20000 --fraud-rate 0.04

# Run ETL and model evaluation in headless batch mode without web server
python run_pipeline.py --no-server

# Run ETL, model training, and execute full ablation study
python run_pipeline.py --run-ablation
```

### 3. Run Automated Tests
Execute the comprehensive 14-test verification suite:

```bash
pytest -v tests/
```

### 4. Run Multi-Model Ablation Study Directly
```bash
# Run on built-in synthetic benchmark
python -m src.experiments.ablation

# Run on custom CSV dataset
python -m src.experiments.ablation --csv path/to/your/data.csv
```
Output results are exported to `results/ablation_results.csv`.

### 5. Run with Docker
```bash
docker build -t sentinel-ai .
docker run -p 8000:8000 sentinel-ai
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/transactions` | Fetch latest ingested transactions annotated with ML risk scores and analyst status |
| `POST`| `/api/predict` | Sub-30ms real-time transaction scoring with SHAP feature attributions and NL narrative |
| `GET` | `/api/explain/{id}` | Deep dive SHAP waterfall breakdown and compliance rationale for a transaction |
| `GET` | `/api/metrics` | Ingestion throughput, confusion matrix, feature importances, and cost savings ROI |
| `GET` | `/api/drift` | Live feature drift report using Population Stability Index (PSI) against baseline |
| `GET` | `/api/fairness` | Cohort fairness and parity audit (FPR and Recall across merchant categories & devices) |
| `GET` | `/api/ablation` | Multi-model ablation benchmark table comparing LR, DT, and LightGBM |
| `POST`| `/api/action/{id}` | Record investigator decision (`APPROVE`, `BLOCK`, `ESCALATE`) for audit logging |

---

## 📊 Experimental Results & Ablation Summary

*Extract from `results/ablation_results.csv` (Chronological 60/15/25 Split, 3 Seeds Average):*

| Feature Set | Classifier | Imbalance Strategy | PR-AUC | ROC-AUC | F1 Score | Avoided Loss Savings (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `raw` | Logistic Regression | Class Weight | 0.419 | 0.881 | 0.494 | 86.6% |
| `raw` | LightGBM | None | 0.610 | 0.894 | 0.500 | 82.4% |
| `+velocity` | LightGBM | Class Weight | 0.852 | 0.964 | 0.773 | 84.5% |
| `+velocity+spatial` | LightGBM | SMOTE | 0.935 | 0.973 | 0.870 | 96.4% |
| **`+rfm_all`** | **LightGBM** | **Class Weight** | **0.996** | **1.000** | **0.936** | **97.7%** |
| **`+rfm_all+anomaly`**| **Decision Tree** | **Class Weight** | **0.991** | **0.995** | **0.995** | **97.9%** |
| **`+rfm_all+anomaly`**| **LightGBM** | **Class Weight** | **0.997** | **1.000** | **0.949** | **97.7%** |

---

## 📚 References & Scientific Foundations
1. **Baesens, B., Hoeppner, S., & Verdonck, T. (2022)**. *Data Engineering for Fraud Detection*. IEEE Transactions on Knowledge and Data Engineering.
2. **Bahnsen, A. C., Aouada, D., Stojanovic, A., & Ottersten, B. (2016)**. *Feature engineering strategies for detecting credit card fraud*. Expert Systems with Applications.
3. **Lundberg, S. M., & Lee, S. I. (2017)**. *A unified approach to interpreting model predictions (SHAP)*. Advances in Neural Information Processing Systems (NeurIPS).
4. **Ke, G., et al. (2017)**. *LightGBM: A highly efficient gradient boosting decision tree*. NeurIPS.

---

## 👨‍💻 Author & Capstone Project
- **Project Lead**: Yakshith Mekala
- **Repository**: [https://github.com/yakshithmekala/capstoneproject](https://github.com/yakshithmekala/capstoneproject)
