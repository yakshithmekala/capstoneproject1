"""Extra fraud features from Baesens, Hoeppner & Verdonck (2022).
All features use ONLY past transactions of the same account (no leakage)."""
import numpy as np
import pandas as pd

GAMMA_180 = -np.log(0.01) / 180  # recency = 0.01 after 180 days (~0.026)


def add_recency(df, id_col, time_col, key_col, gamma=GAMMA_180, out=None):
    """recency = exp(-gamma * delta_days) since last use of the same (account, key); 0 if first use."""
    df = df.sort_values(time_col).copy()
    prev = df.groupby([id_col, key_col])[time_col].shift(1)
    delta_days = (df[time_col] - prev).dt.total_seconds() / 86400
    df[out or f"recency_{key_col}"] = np.exp(-gamma * delta_days).fillna(0.0)
    return df


def robust_z(x, history):
    """Median/MAD z-score (MAD scaled by 1.4826). NaN if history has no spread."""
    h = np.asarray(history, dtype=float)
    med = np.median(h)
    mad = 1.4826 * np.median(np.abs(h - med))
    return np.nan if mad == 0 else (x - med) / mad


def classical_z(x, history):
    h = np.asarray(history, dtype=float)
    sd = h.std(ddof=1)
    return np.nan if sd == 0 else (x - h.mean()) / sd


def add_robust_z(df, id_col, amount_col, window=20, min_periods=5, out="amount_robust_z"):
    """Robust z of log(amount) vs the account's previous `window` transactions (current row excluded)."""
    df = df.copy()
    zs = np.zeros(len(df))
    pos_of = {ix: i for i, ix in enumerate(df.index)}
    for _, g in df.groupby(id_col, sort=False):
        la = np.log1p(g[amount_col].to_numpy(dtype=float))
        for k in range(len(la)):
            h = la[max(0, k - window):k]
            if len(h) >= min_periods:
                z = robust_z(la[k], h)
                zs[pos_of[g.index[k]]] = 0.0 if np.isnan(z) else float(np.clip(z, -20, 20))
    df[out] = zs
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def add_window_features(df, id_col, time_col, amount_col, short_h=1, long_h=24, base_days=30):
    """Velocity (count last 1h, spend last 24h) and amount / 30-day mean, per account (past rows only)."""
    df = df.sort_values([id_col, time_col]).copy()
    n = len(df)
    c1, s24, ratio = np.zeros(n), np.zeros(n), np.ones(n)
    pos = 0
    for _, g in df.groupby(id_col, sort=False):
        t = g[time_col].astype("int64").to_numpy() / 1e9
        a = g[amount_col].to_numpy(dtype=float)
        cs = np.concatenate([[0], np.cumsum(a)])
        idx = np.arange(len(g))
        lo1 = np.searchsorted(t, t - short_h * 3600, side="left")
        lo24 = np.searchsorted(t, t - long_h * 3600, side="left")
        lo30 = np.searchsorted(t, t - base_days * 86400, side="left")
        n30 = idx - lo30
        mean30 = np.where(n30 > 0, (cs[idx] - cs[lo30]) / np.maximum(n30, 1), np.nan)
        sl = slice(pos, pos + len(g))
        c1[sl] = idx - lo1
        s24[sl] = cs[idx] - cs[lo24]
        ratio[sl] = np.where(np.isnan(mean30), 1.0, a / mean30)
        pos += len(g)
    df["txn_count_1h"], df["spend_24h"], df["amount_vs_30d_mean"] = c1, s24, ratio
    return df


def add_spatial_features(df, id_col, time_col, lat_col, lon_col):
    """Distance to previous transaction, minutes elapsed, implied speed (impossible-travel signal)."""
    df = df.sort_values([id_col, time_col]).copy()
    g = df.groupby(id_col)
    plat, plon, pt = g[lat_col].shift(1), g[lon_col].shift(1), g[time_col].shift(1)
    d = haversine_km(plat, plon, df[lat_col], df[lon_col])
    mins = (df[time_col] - pt).dt.total_seconds() / 60
    df["dist_km"] = pd.Series(d, index=df.index).fillna(0.0)
    df["minutes_since_prev"] = mins.fillna(1e5)
    df["speed_kmh"] = (df["dist_km"] / (df["minutes_since_prev"].clip(lower=1) / 60)).clip(upper=5000)
    return df


def anomaly_score_features(X_train, X_other_list, contamination=0.03, seed=42):
    """Unsupervised outlyingness scores (higher = more suspicious). Fit on train only."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed).fit(X_train)
    lof = LocalOutlierFactor(n_neighbors=30, novelty=True).fit(X_train)
    def score(X):
        return np.column_stack([-iso.score_samples(X), -lof.score_samples(X)])
    return score(X_train), [score(X) for X in X_other_list]
