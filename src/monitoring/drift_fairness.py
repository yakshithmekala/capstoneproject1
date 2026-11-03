import numpy as np
import pandas as pd


def psi(expected, actual, bins=10, eps=1e-6):
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate drift, >0.25 major drift."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + eps
    a = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift_report(train_df, live_df, cols):
    return pd.DataFrame({"feature": cols, "psi": [psi(train_df[c], live_df[c]) for c in cols]}).sort_values("psi", ascending=False)


def group_fpr_recall(y, yhat, groups):
    """False-positive rate and recall per group (fairness audit)."""
    df = pd.DataFrame({"y": y, "p": yhat, "g": groups})
    rows = []
    for g, d in df.groupby("g"):
        neg, pos = (d.y == 0).sum(), (d.y == 1).sum()
        rows.append({"group": g, "n": len(d), "fpr": ((d.p == 1) & (d.y == 0)).sum() / max(neg, 1),
                     "recall": ((d.p == 1) & (d.y == 1)).sum() / max(pos, 1)})
    return pd.DataFrame(rows)
