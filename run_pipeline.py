import sys
import time
import os
import socket
import argparse

# Set UTF-8 encoding for stdout on Windows if possible
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.data_pipeline.pipeline import run_data_pipeline
from src.models.train_model import train_fraud_detection_model
from src.models.evaluate import evaluate_fraud_model


def find_available_port(preferred_port=8000):
    for port in [preferred_port, 8080, 8050, 8090, 8001]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return preferred_port


def main():
    parser = argparse.ArgumentParser(description="Sentinel AI — End-to-End Pipeline & Dashboard Server")
    parser.add_argument("--records", type=int, default=10000, help="Number of synthetic transactions to ingest")
    parser.add_argument("--fraud-rate", type=float, default=0.035, help="Simulated fraud rate (e.g. 0.035 for 3.5%)")
    parser.add_argument("--port", type=int, default=8000, help="Preferred web server port")
    parser.add_argument("--no-server", action="store_true", help="Run ETL and Model training only without launching web server")
    parser.add_argument("--run-ablation", action="store_true", help="Also execute full multi-model ablation study")
    args = parser.parse_args()

    print("""
  ================================================================================
  [SENTINEL.AI] — Scalable Fraud Detection & Explainable AI (XAI) System
  Data Engineering ETL | Vectorized DuckDB Store | LightGBM | SHAP | PSI Drift
  ================================================================================
    """)
    
    # 1. Execute Data Engineering Pipeline
    feature_df = run_data_pipeline(num_records=args.records, fraud_rate=args.fraud_rate, random_seed=42)
    
    # 2. Train LightGBM Model
    model, scaler, X_test, y_test, metadata = train_fraud_detection_model(feature_df)
    
    # 3. Evaluate Model, Cost Savings (Baesens et al. 2022) & Fairness Audit
    metrics = evaluate_fraud_model(
        model, scaler, X_test, y_test,
        threshold=metadata['best_threshold'],
        full_df=feature_df
    )
    
    # 4. Optional Ablation Study
    if args.run_ablation:
        print("\n" + "=" * 60)
        print("🔬 [Ablation Study] Running multi-model baseline & feature ablation...")
        print("=" * 60)
        from src.experiments.ablation import run as run_ablation
        run_ablation(feature_df)

    if args.no_server:
        print("\n✅ [Execution Complete] Pipeline completed in batch mode (--no-server flag).")
        return
        
    port = find_available_port(args.port)
    
    print("\n" + "=" * 70)
    print(f"🚀 Pipeline execution successful! Launching FastAPI Web Server on port {port}...")
    print(f"👉 Executive Dashboard UI : http://localhost:{port}")
    print(f"👉 Interactive OpenAPI Docs: http://localhost:{port}/docs")
    print(f"👉 Drift Monitoring API   : http://localhost:{port}/api/drift")
    print(f"👉 Performance Metrics API: http://localhost:{port}/api/metrics")
    print("=" * 70 + "\n")
    
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=port, reload=False)


if __name__ == '__main__':
    main()
