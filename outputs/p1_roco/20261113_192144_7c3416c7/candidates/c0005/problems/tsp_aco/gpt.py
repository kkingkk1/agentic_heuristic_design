import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge-selection priors for stochastic TSP solution sampling."""
    d = np.asarray(distance_matrix, dtype=float)

    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = d.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    x = np.nan_to_num(d, nan=np.inf, posinf=np.inf, neginf=0.0)
    x = np.maximum(x, 0.0)
    x = 0.5 * (x + x.T)
    np.fill_diagonal(x, 0.0)

    off_diag = ~np.eye(n, dtype=bool)
    positive = x[off_diag & (x > 0)]
    scale = float(np.median(positive)) if positive.size else 1.0
    scale = max(scale, np.finfo(float).eps)

    finite = np.isfinite(x)
    finite_off_diag = finite & off_diag
    if not np.any(finite_off_diag):
        return np.zeros_like(x)

    capped = x.copy()
    finite_values = capped[finite_off_diag]
    fallback = float(np.max(finite_values)) if finite_values.size else scale
    capped[~finite] = fallback * 10.0 + scale

    inv_distance = 1.0 / (1.0 + capped / scale)

    row_order = np.argsort(capped + np.eye(n) * np.inf, axis=1)
    ranks = np.empty_like(row_order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, row_order[i]] = rank_values
    rank_score = 1.0 / (1.0 + 0.5 * (ranks + ranks.T))

    nearest = np.partition(
        capped + np.eye(n) * np.inf,
        kth=min(2, n - 1),
        axis=1
    )[:, :min(2, n - 1)]
    local_scale = np.mean(nearest, axis=1)
    savings = local_scale[:, None] + local_scale[None, :] - capped
    savings = np.maximum(savings, 0.0)
    savings_scale = np.median(savings[finite_off_diag])
    savings_scale = max(float(savings_scale), np.finfo(float).eps)
    savings_score = 1.0 - np.exp(-savings / savings_scale)

    score = (
        np.power(inv_distance, 0.55)
        * np.power(rank_score, 0.30)
        * np.power(0.5 + 0.5 * savings_score, 0.15)
    )

    score[~finite_off_diag] = 0.0
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0 and np.isfinite(maximum):
        score /= maximum

    return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
