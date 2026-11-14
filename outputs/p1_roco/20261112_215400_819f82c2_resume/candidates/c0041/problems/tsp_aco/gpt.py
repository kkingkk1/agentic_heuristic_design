import numpy as np


def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric edge-prior matrix for stochastic TSP sampling."""
    distances = np.asarray(distance_matrix, dtype=float)

    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    n = distances.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.zeros((1, 1), dtype=float)

    valid = np.isfinite(distances) & (distances >= 0.0)
    np.fill_diagonal(valid, False)

    finite_valid = distances[valid]
    positive_valid = finite_valid[finite_valid > 0.0]

    if positive_valid.size:
        global_scale = float(np.median(positive_valid))
    elif finite_valid.size:
        global_scale = max(float(np.median(finite_valid)), 1.0)
    else:
        global_scale = 1.0

    global_scale = max(global_scale, np.finfo(float).eps)

    safe = np.where(valid, np.maximum(distances, 0.0), np.inf)

    # Global distance desirability.
    distance_score = np.exp(
        -np.clip(safe / global_scale, 0.0, 50.0)
    )

    # Two-sided rank preference.
    row_order = np.argsort(safe, axis=1, kind="stable")
    col_order = np.argsort(safe, axis=0, kind="stable")

    row_rank = np.empty((n, n), dtype=float)
    col_rank = np.empty((n, n), dtype=float)

    row_rank[np.arange(n)[:, None], row_order] = np.arange(n)
    col_rank[col_order, np.arange(n)[None, :]] = np.arange(n)[:, None]

    rank_score = 1.0 / (1.0 + row_rank + col_rank)
    rank_score[~valid] = 0.0

    # Endpoint-relative compatibility using each endpoint's best edge.
    row_best = np.min(safe, axis=1)
    col_best = np.min(safe, axis=0)

    relative_gap = (
        np.maximum(safe - row_best[:, None], 0.0)
        + np.maximum(safe - col_best[None, :], 0.0)
    )
    local_score = np.exp(
        -np.clip(relative_gap / global_scale, 0.0, 50.0)
    )
    local_score[~valid] = 0.0

    # Mutual-nearest-neighbor structural bonus.
    nearest_from_row = np.argmin(safe, axis=1)
    nearest_from_col = np.argmin(safe, axis=0)

    mutual = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = nearest_from_row[i]
        if valid[i, j] and nearest_from_col[j] == i:
            mutual[i, j] = 1.0

    # Degree balancing prevents one endpoint from dominating sampling.
    attractiveness = np.where(valid, distance_score, 0.0)
    row_mass = attractiveness.sum(axis=1, keepdims=True)
    col_mass = attractiveness.sum(axis=0, keepdims=True)
    degree_balance = np.sqrt(
        (attractiveness / (row_mass + 1e-12))
        * (attractiveness / (col_mass + 1e-12))
    )

    score = (
        0.45 * distance_score
        + 0.25 * rank_score
        + 0.20 * local_score
        + 0.10 * mutual
    )
    score *= 0.75 + 0.25 * degree_balance
    score[~valid] = 0.0

    # Convert directed evidence into consistent undirected priors.
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 0.0)
    score[~np.isfinite(score)] = 0.0

    maximum = float(np.max(score))
    if maximum > 0.0:
        score /= maximum

    return score.astype(float, copy=False)
