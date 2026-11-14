import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability indicators for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n <= 1:
        return np.zeros_like(distances, dtype=float)

    finite = np.isfinite(distances)
    if not np.any(finite):
        return np.zeros_like(distances, dtype=float)

    d = np.where(finite, distances, np.inf)
    d = np.maximum(d, 0.0)
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, np.inf)

    valid_values = d[np.isfinite(d)]
    fallback = float(np.max(valid_values)) if valid_values.size else 1.0
    d = np.where(np.isfinite(d), d, fallback * 10.0 + 1.0)

    row_median = np.median(d, axis=1)
    row_q75 = np.percentile(d, 75, axis=1)
    row_scale = np.maximum(row_q75 - row_median, np.finfo(float).eps)
    row_scale = np.maximum(row_scale, row_median * 0.05 + np.finfo(float).eps)

    scale_i = row_scale[:, None]
    scale_j = row_scale[None, :]
    local_scale = 0.5 * (scale_i + scale_j)

    distance_score = np.exp(-d / local_scale)

    order = np.argsort(d, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rank_values = np.arange(n, dtype=float)
    for i in range(n):
        ranks[i, order[i]] = rank_values

    rank_score = 1.0 - ranks / max(n - 1, 1)
    mutual_rank_score = np.sqrt(
        np.clip(rank_score, 0.0, 1.0)
        * np.clip(rank_score.T, 0.0, 1.0)
    )

    score = 0.65 * distance_score + 0.35 * mutual_rank_score
    score = np.where(np.isfinite(score), score, 0.0)
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
