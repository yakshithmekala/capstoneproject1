import os
import math
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data_pipeline.ingestion import DUCKDB_PATH, DATA_DIR

FEATURE_STORE_PATH = os.path.join(DATA_DIR, 'feature_store.parquet')


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes Haversine distance between two coordinate pairs in kilometers.
    """
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def compute_robust_z_score(series: pd.Series) -> pd.Series:
    """
    Computes Robust Z-Score using Median and Median Absolute Deviation (MAD)
    as proposed by Baesens et al. (2022) to prevent masking & swamping effects.
    Formula: z_r = (x - median) / (1.4826 * MAD)
    """
    med = series.median()
    mad = 1.4826 * (series - med).abs().median()
    if mad < 1e-5:
        mad = 1e-5
    return ((series - med) / mad).clip(-20, 20).round(4)


def compute_exponential_recency(delta_seconds: pd.Series, half_life_days: float = 7.0) -> pd.Series:
    """
    Computes Exponential Recency Decay Feature:
    x_recency = exp(-gamma * delta_t)
    Where gamma = -log(0.01) / max_days
    """
    delta_days = delta_seconds / 86400.0
    gamma = -math.log(0.01) / (half_life_days * 30.0)
    recency = np.exp(-gamma * delta_days)
    return np.round(recency, 4)


def compute_vectorized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw transactions into high-dimensional feature vectors incorporating
    advanced Data Engineering methodologies from Baesens et al. (2022):
    - Robust Monetary Z-Scores (Median & MAD)
    - Exponential Recency Decay Features
    - Vectorized Fast Sliding Window RFM Velocity & Spending Totals (O(N log N))
    - Haversine Spatial Anomaly & Implied Travel Speed
    - Unsupervised Anomaly Detection (Isolation Forest Outlier Score)
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Stable sort by user and time
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # 1. User Baseline Mean & Median/MAD Statistics (Robust Monetary Features)
    user_stats = df.groupby('user_id')['amount'].agg(
        user_mean_amount='mean',
        user_median_amount='median',
        user_std_amount='std'
    ).reset_index()
    user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(10.0)
    
    df = df.merge(user_stats, on='user_id', how='left')
    
    # Amount Ratio to User Average
    df['amount_ratio_to_user_avg'] = (df['amount'] / (df['user_mean_amount'] + 1e-5)).round(4)
    df['amount_z_score'] = ((df['amount'] - df['user_mean_amount']) / (df['user_std_amount'] + 1e-5)).round(4)
    
    # Robust Monetary Z-Score (Baesens et al. Eq 14-15)
    df['robust_monetary_zscore'] = df.groupby('user_id')['amount'].transform(compute_robust_z_score)
    df['log_amount'] = np.round(np.log1p(df['amount']), 4)
    
    # 2. Time Delta, Distance & Exponential Recency
    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    df['prev_lat'] = df.groupby('user_id')['location_lat'].shift(1)
    df['prev_long'] = df.groupby('user_id')['location_long'].shift(1)
    
    df['prev_timestamp'] = df['prev_timestamp'].fillna(df['timestamp'])
    df['prev_lat'] = df['prev_lat'].fillna(df['location_lat'])
    df['prev_long'] = df['prev_long'].fillna(df['location_long'])
    
    df['time_since_prev_txn_sec'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
    df['recency_decay_score'] = compute_exponential_recency(df['time_since_prev_txn_sec'])
    
    # Haversine Distance (vectorized)
    dist = haversine_distance(
        df['prev_lat'].values, df['prev_long'].values,
        df['location_lat'].values, df['location_long'].values
    )
    df['distance_prev_txn_km'] = np.round(dist, 2)
    
    hours_since_prev = np.maximum(df['time_since_prev_txn_sec'] / 3600.0, 1.0 / 3600.0)
    df['travel_speed_kmh'] = np.round(np.clip(df['distance_prev_txn_km'] / hours_since_prev, 0, 5000), 2)
    
    # 3. Fast Vectorized Sliding Window Aggregations (1h count, 24h count, 24h spend)
    n = len(df)
    c1 = np.zeros(n, dtype=int)
    c24 = np.zeros(n, dtype=int)
    s24 = np.zeros(n, dtype=float)
    
    # Group-wise vectorized searchsorted runs in sub-second time
    pos = 0
    for _, g in df.groupby('user_id', sort=False):
        g_len = len(g)
        t = g['timestamp'].astype('int64').to_numpy() / 1e9  # seconds epoch
        a = g['amount'].to_numpy(dtype=float)
        cs = np.concatenate([[0.0], np.cumsum(a)])
        idx = np.arange(g_len)
        
        # 1-hour window = 3600s
        lo1 = np.searchsorted(t, t - 3600, side='left')
        # 24-hour window = 86400s
        lo24 = np.searchsorted(t, t - 86400, side='left')
        
        sl = slice(pos, pos + g_len)
        c1[sl] = (idx - lo1) + 1  # count inclusive of current
        c24[sl] = (idx - lo24) + 1
        s24[sl] = cs[idx + 1] - cs[lo24]
        pos += g_len
        
    df['txn_count_1h'] = c1
    df['txn_count_24h'] = c24
    df['spending_sum_24h'] = np.round(s24, 2)
    
    # 4. Categorical Flags & Frequency Ratios
    high_risk_categories = ['crypto_exchange', 'upi_p2p', 'jewelry_gold']
    df['is_high_risk_category'] = df['merchant_category'].isin(high_risk_categories).astype(int)
    df['is_night_transaction'] = (df['timestamp'].dt.hour.isin([23, 0, 1, 2, 3, 4, 5])).astype(int)
    
    cat_freq = df['merchant_category'].value_counts(normalize=True).to_dict()
    df['merchant_category_freq'] = df['merchant_category'].map(cat_freq).fillna(0.05).round(4)
    
    dev_freq = df['device_type'].value_counts(normalize=True).to_dict()
    df['device_type_freq'] = df['device_type'].map(dev_freq).fillna(0.25).round(4)
    
    # 5. Unsupervised Anomaly Score Feature (Isolation Forest as in Baesens et al. Sec 2.5)
    iso_features = ['amount', 'time_since_prev_txn_sec', 'distance_prev_txn_km', 'spending_sum_24h']
    iso = IsolationForest(n_estimators=50, contamination=0.035, random_state=42)
    iso.fit(df[iso_features])
    df['isolation_forest_anomaly_score'] = np.round(-iso.score_samples(df[iso_features]), 4)
    
    # Clean temporary columns
    drop_cols = ['prev_timestamp', 'prev_lat', 'prev_long', 'user_mean_amount', 'user_median_amount', 'user_std_amount']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


def build_and_save_feature_store(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes features and stores them into DuckDB feature store table and Parquet.
    """
    feature_df = compute_vectorized_features(raw_df)
    
    # Save to Parquet
    feature_df.to_parquet(FEATURE_STORE_PATH, index=False)
    
    # Store in DuckDB feature_store table
    con = duckdb.connect(DUCKDB_PATH)
    con.execute("CREATE OR REPLACE TABLE feature_store AS SELECT * FROM read_parquet(?)", [FEATURE_STORE_PATH])
    count = con.execute("SELECT count(*) FROM feature_store").fetchone()[0]
    con.close()
    
    print(f"[Feature Engineering] Built Baesens et al. (2022) feature store with {count} rows & {len(feature_df.columns)} features.")
    return feature_df
