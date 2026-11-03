"""Ablation + baseline + imbalance study.
Usage:  python -m src.experiments.ablation                 # built-in synthetic data
        python -m src.experiments.ablation --csv my.csv    # your data (see COLUMNS below)
Required columns: account_id, timestamp, amount, lat, lon, merchant_cat, device_id, merchant_risk, is_fraud
Results -> results/ablation_results.csv"""
import argparse, os, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.evaluation.cost_metrics import savings, best_cost_threshold
from src.features.advanced_features import (add_recency, add_robust_z, add_window_features,
                                            add_spatial_features, anomaly_score_features)
warnings.filterwarnings("ignore")


def make_synthetic(n_users=400, n_tx=12000, fraud_rate=0.03, seed=7):
    rng = np.random.default_rng(seed)
    home = np.column_stack([rng.uniform(-35, 55, n_users), rng.uniform(-100, 100, n_users)])
    scale = rng.lognormal(3.6, 0.5, n_users)
    usual_dev = rng.integers(0, 3, n_users)
    t0 = pd.Timestamp("2025-01-01")
    uid = rng.integers(0, n_users, n_tx)
    ts = t0 + pd.to_timedelta(rng.uniform(0, 120 * 86400, n_tx), unit="s")
    amt = rng.lognormal(np.log(scale[uid]), 0.7)
    lat = home[uid, 0] + rng.normal(0, 0.3, n_tx); lon = home[uid, 1] + rng.normal(0, 0.3, n_tx)
    cat = rng.choice(["grocery", "fuel", "dining", "retail", "travel", "crypto", "wire", "luxury"], n_tx,
                     p=[.28, .15, .2, .2, .08, .03, .03, .03])
    dev = np.array([f"d{u}_{rng.integers(0, 3)}" for u in uid])
    df = pd.DataFrame(dict(account_id=uid, timestamp=ts, amount=amt, lat=lat, lon=lon, merchant_cat=cat, device_id=dev, is_fraud=0))
    # account-takeover bursts; 40% are "stealthy" (normal-looking amount/location) to keep the task hard
    n_att = int(n_tx * fraud_rate / 4)
    rows = []
    for _ in range(n_att):
        u = rng.integers(0, n_users); start = t0 + pd.Timedelta(seconds=float(rng.uniform(10 * 86400, 120 * 86400)))
        stealth = rng.random() < 0.4
        for k in range(4):
            rows.append(dict(account_id=u, timestamp=start + pd.Timedelta(minutes=float(rng.uniform(0, 25))),
                             amount=float(scale[u] * (rng.uniform(0.8, 2.0) if stealth else rng.uniform(4, 12))),
                             lat=float(home[u, 0] + (rng.normal(0, 0.3) if stealth else rng.uniform(8, 25))),
                             lon=float(home[u, 1] + (rng.normal(0, 0.3) if stealth else rng.uniform(8, 25))),
                             merchant_cat=str(rng.choice(["crypto", "wire", "luxury", "retail"] if not stealth else ["retail", "dining", "fuel"])),
                             device_id=f"new_{rng.integers(0, 10**6)}", is_fraud=1))
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df["merchant_risk"] = df.merchant_cat.isin(["crypto", "wire", "luxury"]).astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_features(df):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df.timestamp.dt.hour
    df = add_window_features(df, "account_id", "timestamp", "amount")
    df = add_spatial_features(df, "account_id", "timestamp", "lat", "lon")
    df = add_recency(df, "account_id", "timestamp", "device_id", out="recency_device")
    df = add_recency(df, "account_id", "timestamp", "merchant_cat", out="recency_merchant")
    df = add_robust_z(df, "account_id", "amount")
    return df.sort_values("timestamp").reset_index(drop=True)


RAW = ["amount", "hour", "lat", "lon", "merchant_risk"]
FEATURE_SETS = {
    "raw": RAW,
    "+velocity": RAW + ["txn_count_1h", "spend_24h"],
    "+velocity+spatial": RAW + ["txn_count_1h", "spend_24h", "dist_km", "speed_kmh"],
    "+rfm_all": RAW + ["txn_count_1h", "spend_24h", "dist_km", "speed_kmh", "amount_vs_30d_mean",
                       "recency_device", "recency_merchant", "amount_robust_z"],
}


def get_model(name, pos_weight=1.0, seed=0):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight={0: 1, 1: pos_weight}))
    if name == "DT":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight={0: 1, 1: pos_weight}, random_state=seed)
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, scale_pos_weight=pos_weight,
                          random_state=seed, verbose=-1)


def evaluate(y_va, s_va, a_va, y_te, s_te, a_te, c_f=5.0):
    t, _ = best_cost_threshold(y_va, s_va, a_va, c_f)
    pred = (s_te > t).astype(int)
    return dict(threshold=round(t, 4), precision=precision_score(y_te, pred, zero_division=0),
                recall=recall_score(y_te, pred), f1=f1_score(y_te, pred), pr_auc=average_precision_score(y_te, s_te),
                roc_auc=roc_auc_score(y_te, s_te), savings=savings(y_te, pred, a_te, c_f))


def run(df, seeds=(0, 1, 2), out_dir="results"):
    df = build_features(df)
    n = len(df); i1, i2 = int(n * 0.6), int(n * 0.75)       # chronological 60/15/25
    tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    pw = (tr.is_fraud == 0).sum() / max((tr.is_fraud == 1).sum(), 1)
    rows = []
    for fs_name, cols in FEATURE_SETS.items():
        Xtr, Xva, Xte = tr[cols].to_numpy(), va[cols].to_numpy(), te[cols].to_numpy()
        for model in ["LR", "DT", "LightGBM"]:
            for imb in ["none", "class_weight", "smote", "undersample"]:
                if model != "LightGBM" and imb in ("smote", "undersample"):
                    continue
                res = []
                for seed in seeds:
                    X, y = Xtr, tr.is_fraud.to_numpy()
                    w = pw if imb == "class_weight" else 1.0
                    if imb == "smote": X, y = SMOTE(sampling_strategy=0.11, random_state=seed).fit_resample(X, y)
                    if imb == "undersample": X, y = RandomUnderSampler(sampling_strategy=0.11, random_state=seed).fit_resample(X, y)
                    m = get_model(model, w, seed).fit(X, y)
                    res.append(evaluate(va.is_fraud, m.predict_proba(Xva)[:, 1], va.amount,
                                        te.is_fraud, m.predict_proba(Xte)[:, 1], te.amount))
                r = pd.DataFrame(res).mean().to_dict()
                r.update(feature_set=fs_name, model=model, imbalance=imb, seeds=len(seeds)); rows.append(r)
    # + unsupervised anomaly-score features (fit on train only)
    cols = FEATURE_SETS["+rfm_all"]
    a_tr, (a_va, a_te) = anomaly_score_features(tr[cols].to_numpy(), [va[cols].to_numpy(), te[cols].to_numpy()])
    X1, X2, X3 = np.hstack([tr[cols], a_tr]), np.hstack([va[cols], a_va]), np.hstack([te[cols], a_te])
    for model in ["LR", "DT", "LightGBM"]:
        m = get_model(model, pw).fit(X1, tr.is_fraud)
        r = evaluate(va.is_fraud, m.predict_proba(X2)[:, 1], va.amount, te.is_fraud, m.predict_proba(X3)[:, 1], te.amount)
        r.update(feature_set="+rfm_all+anomaly_scores", model=model, imbalance="class_weight", seeds=1); rows.append(r)
    out = pd.DataFrame(rows)[["feature_set", "model", "imbalance", "threshold", "precision", "recall", "f1", "pr_auc", "roc_auc", "savings", "seeds"]]
    os.makedirs(out_dir, exist_ok=True)
    out.round(4).to_csv(f"{out_dir}/ablation_results.csv", index=False)
    return out, (len(te), int(te.is_fraud.sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    data = pd.read_csv(a.csv, parse_dates=["timestamp"]) if a.csv else make_synthetic()
    res, (nte, fte) = run(data)
    print(f"Test set: {nte} transactions, {fte} frauds (chronological split)\n")
    print(res.round(3).to_string(index=False))
