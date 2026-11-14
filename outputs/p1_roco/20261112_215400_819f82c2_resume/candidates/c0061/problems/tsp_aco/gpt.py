import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Return edge-priority indicators for stochastic TSP solution sampling.
    Higher values indicate more promising directed edges.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distance_matrix.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)

    d = distance_matrix.copy()
    valid = np.isfinite(d) & (d >= 0.0)
    np.fill_diagonal(valid, False)

    finite_values = d[valid]
    positive_values = finite_values[finite_values > 0.0]

    if positive_values.size:
        scale = float(np.median(positive_values))
    elif finite_values.size:
        scale = max(float(np.median(finite_values)), 1.0)
    else:
        scale = 1.0

    scale = max(scale, np.finfo(float).eps)
    safe_d = np.where(valid, d, np.inf)

    normalized = np.clip(safe_d / scale, 0.0, 50.0)
    distance_score = np.exp(-1.15 * normalized)

    row_order = np.argsort(safe_d, axis=1, kind="stable")
    col_order = np.argsort(safe_d, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_indices = np.arange(n)[:, None]
    col_indices = np.arange(n)[None, :]

    row_rank[row_indices, row_order] = np.arange(n)[None, :]
    col_rank[col_order, col_indices] = np.arange(n)[:, None]

    rank_score = 1.0 / (1.0 + row_rank + col_rank)
    rank_score[~valid] = 0.0

    row_best = np.min(safe_d, axis=1)
    col_best = np.min(safe_d, axis=0)

    row_gap = np.clip((safe_d - row_best[:, None]) / scale, 0.0, 50.0)
    col_gap = np.clip((safe_d - col_best[None, :]) / scale, 0.0, 50.0)

    endpoint_compatibility = np.exp(-0.85 * (row_gap + col_gap))

    nearest_row = np.argmin(safe_d, axis=1)
    nearest_col = np.argmin(safe_d, axis=0)

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_row[i]
        if np.isfinite(safe_d[i, j]) and nearest_col[j] == i:
            mutual[i, j] = 1.0

    score = (
        distance_score
        * (0.45 + 0.55 * rank_score)
        * (0.62 + 0.38 * endpoint_compatibility)
        * (1.0 + 0.28 * mutual)
    )

    score[~valid] = 0.0
    np.fill_diagonal(score, 0.0)

    maximum = np.max(score)
    if np.isfinite(maximum) and maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
