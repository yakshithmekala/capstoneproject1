import numpy as np, pandas as pd
from src.evaluation.cost_metrics import savings, total_cost, best_cost_threshold
from src.features.advanced_features import robust_z, classical_z, add_recency, haversine_km
from src.monitoring.drift_fairness import psi, group_fpr_recall


def test_savings_matches_paper_formula():
    y = np.array([1, 1, 0, 0]); yhat = np.array([1, 0, 1, 0]); amt = np.array([100, 50, 20, 10])
    assert np.isclose(savings(y, yhat, amt, 5), (100 - 2 * 5) / 150)
    assert total_cost(y, yhat, amt, 5) == 50 + 10


def test_perfect_model_and_no_model():
    y = np.array([1, 0, 1]); amt = np.array([10., 5., 20.])
    assert savings(y, np.zeros(3), amt) == 0.0
    assert savings(y, y, amt, c_f=0) == 1.0


def test_robust_z_beats_classical_on_masked_outlier():
    hist = [191.31, 109.05, 47.06, 131.09, 36.12, 121.62, 63.17, 103.12, 152.77]
    allv = hist + [500]
    assert classical_z(500, allv) < 3          # masking: classical z misses it
    assert robust_z(500, hist) > 3             # robust z flags it


def test_recency_first_use_zero_and_decay():
    t0 = pd.Timestamp("2025-01-01")
    df = pd.DataFrame({"a": [1, 1, 1], "t": [t0, t0 + pd.Timedelta(days=1), t0 + pd.Timedelta(days=181)], "k": ["x"] * 3})
    r = add_recency(df, "a", "t", "k", out="r")["r"].tolist()
    assert r[0] == 0 and 0.9 < r[1] < 1 and r[2] < 0.011


def test_haversine_known_distance():
    assert abs(haversine_km(51.5074, -0.1278, 48.8566, 2.3522) - 344) < 5   # London-Paris


def test_psi_and_fairness():
    rng = np.random.default_rng(0); a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1 and psi(a, rng.normal(2, 1, 5000)) > 0.25
    out = group_fpr_recall([0, 0, 1, 1], [1, 0, 1, 0], ["A", "A", "B", "B"])
    assert out.set_index("group").loc["A", "fpr"] == 0.5
