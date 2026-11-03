"""Instance-dependent cost metrics (Hand et al. 2008; Bahnsen et al. 2016; Baesens et al. 2022).
A missed fraud costs its amount A_i; every alert costs a fixed administrative fee c_f."""
import numpy as np


def total_cost(y, yhat, amount, c_f=5.0):
    y, yhat, a = (np.asarray(v, dtype=float) for v in (y, yhat, amount))
    return float(np.sum(y * (1 - yhat) * a + yhat * c_f))


def savings(y, yhat, amount, c_f=5.0):
    """Share of fraud loss avoided vs. using no model: (sum y*yhat*A - yhat*c_f) / sum y*A."""
    y, a = np.asarray(y, dtype=float), np.asarray(amount, dtype=float)
    base = float(np.sum(y * a))
    if base == 0:
        return 0.0
    return (base - total_cost(y, yhat, a, c_f)) / base


def best_cost_threshold(y, scores, amount, c_f=5.0, grid=None):
    """Return (threshold, savings) maximising savings on a validation set."""
    scores = np.asarray(scores, dtype=float)
    grid = np.unique(np.quantile(scores, np.linspace(0.5, 0.9995, 200))) if grid is None else grid
    best_t, best_s = 0.5, -np.inf
    for t in grid:
        s = savings(y, (scores > t).astype(int), amount, c_f)
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t, best_s
